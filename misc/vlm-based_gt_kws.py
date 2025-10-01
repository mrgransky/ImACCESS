from utils import *

# LLAVA 1.5x collection:
# model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "llava-hf/llava-1.5-13b-hf"
# model_id =  "llava-hf/bakLlava-v1-hf"

# LLaVa-NeXT (1.6x) collection:
# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
# model_id = "llava-hf/llama3-llava-next-8b-hf"

# Qwen 2.5x collection:
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

# does not fit into VRAM:
# model_id = "llava-hf/llava-v1.6-34b-hf"
# model_id = "llava-hf/llava-next-72b-hf"
# model_id = "llava-hf/llava-next-110b-hf"
# model_id = "Qwen/Qwen2.5-VL-72B-Instruct"

# debugging required:
# # model_id = "tiiuae/falcon-11B-vlm"
# # model_id = "utter-project/EuroVLM-1.7B-Preview"
# # model_id = "OpenGVLab/InternVL-Chat-V1-2"

print(f"{USER} HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub")
huggingface_hub.login(token=hf_tk)

VLM_INSTRUCTION_TEMPLATE = """Act as a meticulous historical archivist specializing in 20th century documentation.
Identify the five most prominent, factual and distinct **KEYWORDS** that capture the main action, object or event.
Exclude any explanatory text, comments, questions, or words about image quality, style, or temporal era.
**Return *only* these keywords as a clean, parseable Python list, e.g., ['keyword1', 'keyword2', ...].**
"""
# **Return *only* these keywords as a clean, parseable Python list, e.g., ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5'].**

def load_(model_id: str, device: str):
	print(f"[INFO] Loading model: {model_id} on {device}")
	config = tfs.AutoConfig.from_pretrained(model_id)
	print(f"[INFO] Model type: {config.model_type} Architectures: {config.architectures}")
	if config.architectures:
		cls_name = config.architectures[0]
		if hasattr(tfs, cls_name):
			model_cls = getattr(tfs, cls_name)

	processor = tfs.AutoProcessor.from_pretrained(
		model_id, 
		use_fast=True, 
		trust_remote_code=True, 
		cache_dir=cache_directory[USER]
	)
	print(type(processor))

	model = model_cls.from_pretrained(
		model_id,
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True,
		trust_remote_code=True,
		cache_dir=cache_directory[USER]
	)
	print(type(model))
	model.to(device)
	print(f"[INFO] Loaded {model.__class__.__name__} on {device}")
	return processor, model

def get_prompt(
		model_id: str, 
		processor: tfs.AutoProcessor, 
		img_path: str
	):
	if "-v1.6-" or "-next-" in model_id:
		conversation = [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": VLM_INSTRUCTION_TEMPLATE},
					{"type": "image", "image": img_path},
				],
			},
		]
		txt = processor.apply_chat_template(
			conversation,
			add_generation_prompt=True
		)
	elif "-1.5-" or "bakLlava" in model_id:
		txt = f"USER: <image>\n{VLM_INSTRUCTION_TEMPLATE}\nASSISTANT:"
	elif "Qwen" in model_id:
		messages = [
			{
				"role": "user",
				"content": [
					{"type": "image", "image": img_path},
					{"type": "text", "text": VLM_INSTRUCTION_TEMPLATE},
				],
			},
		]	
		txt = processor.apply_chat_template(
			messages, 
			tokenize=False, 
			add_generation_prompt=True,
		)
	else:
		raise ValueError(f"Unknown model ID: {model_id}")
	return txt

def get_vlm_response(model_id: str, raw_response: str, verbose: bool=False):
	if "Qwen" in model_id:
		return _qwen_vlm_(raw_response, verbose=verbose)
	elif "llava" in model_id:
		return _llava_vlm_(raw_response, verbose=verbose)
	else:
		raise NotImplementedError(f"VLM response parsing not implemented for {model_id}")

def _qwen_vlm_(response: str, verbose: bool=False) -> Optional[list]:
	if verbose:
		print(f"\n[DEBUG] Raw VLM output:\n{response}")
	if not isinstance(response, str):
		if verbose: print("[ERROR] VLM output is not a string.")
		return None
	
	# Find all lists in the response (could be multiple!)
	all_matches = re.findall(r"\[[^\[\]]+\]", response, re.DOTALL)

	if verbose: print(f"[DEBUG] Found {len(all_matches)} Python lists: {all_matches}")
	# Choose the last match **as the answer**
	if all_matches:
		list_str = all_matches[-1]
		if verbose: print("[DEBUG] Extracted last list:", list_str)
		try:
			keywords = ast.literal_eval(list_str)
			if verbose: print("[DEBUG] Parsed Python list:", keywords)
			if isinstance(keywords, list):
				result = [str(k).strip() for k in keywords if isinstance(k, str)]
				if verbose: print("[INFO] Final parsed keywords:", result)
				return result
			else:
				if verbose: print("[ERROR] Parsed output is not a list.")
		except Exception as e:
			if verbose: print("[ERROR] ast.literal_eval failed:", e)

	# Fallback: Try extracting single quoted items in assistant block (not recommended)
	after_assistant = response.split("assistant")[-1].strip()
	candidates = re.findall(r"'([^']+)'", after_assistant)

	if verbose: print("[DEBUG] Regex candidates:", candidates)
	if candidates:
		result = [str(x).strip() for x in candidates]
		if verbose: print("[INFO] Final parsed fallback keywords: ", result)
		return result

	# Final fallback
	raw_split = [x.strip(" ,'\"]") for x in after_assistant.split(",") if x.strip()]

	if verbose: print("[DEBUG] Comma split candidates:", raw_split)
	if len(raw_split) > 1:
		if verbose: print("[INFO] Final last-resort keywords:", raw_split)
		return raw_split

	if verbose:
		print("[ERROR] Unable to parse any keywords from VLM output.")

	return None

def _llava_vlm_(response: str, verbose: bool = False) -> Optional[list]:
		if verbose:
				print(f"\n[DEBUG] Raw VLM output:\n{response}")
		if not isinstance(response, str):
				if verbose:
						print("[ERROR] VLM output is not a string.")
				return None

		# --- Step 1: Try to locate 'ASSISTANT:' and extract the following Python list ---
		assistant_split = response.split("ASSISTANT:")
		if verbose:
				print(f"[DEBUG] Split on 'ASSISTANT:': {len(assistant_split)} parts.")
		
		# If ASSISTANT: is found, focus parsing on the answer text after it
		if len(assistant_split) > 1:
				answer_part = assistant_split[-1].strip()
				if verbose:
						print(f"[DEBUG] Text after 'ASSISTANT:': {answer_part}")
				# Try to find a list in the assistant's part
				list_matches = re.findall(r"\[[^\[\]]+\]", answer_part)
				if verbose:
						print(f"[DEBUG] Found {len(list_matches)} python-like lists after ASSISTANT.")
				if list_matches:
						list_str = list_matches[0]  # In LLaVa, usually only one list follows
						if verbose:
								print(f"[DEBUG] Extracted list: {list_str}")
						try:
								keywords = ast.literal_eval(list_str)
								if verbose:
										print(f"[DEBUG] Parsed Python list: {keywords}")
								if isinstance(keywords, list):
										result = [str(k).strip() for k in keywords if isinstance(k, str)]
										if verbose:
												print(f"[INFO] Final parsed keywords: {result}")
										return result
						except Exception as e:
								if verbose:
										print("[ERROR] ast.literal_eval failed:", e)
				# Fallback 1: extract quoted strings
				candidates = re.findall(r"'([^']+)'", answer_part)
				if verbose:
						print(f"[DEBUG] Regex candidates:", candidates)
				if candidates:
						result = [str(x).strip() for x in candidates]
						if verbose:
								print("[INFO] Final parsed fallback (quotes) keywords: ", result)
						return result
				# Fallback 2: comma splitting
				raw_split = [x.strip(" ,'\"]") for x in answer_part.split(",") if x.strip()]
				if verbose:
						print(f"[DEBUG] Comma split candidates:", raw_split)
				if len(raw_split) > 1:
						if verbose:
								print("[INFO] Final last-resort (comma) keywords:", raw_split)
						return raw_split
		
		# -- If parsing above fails --
		if verbose:
				print("[ERROR] Unable to parse any keywords from VLM output (LLaVa model).")
		return None

def query_local_vlm(
		model: tfs.PreTrainedModel, 
		processor: tfs.AutoProcessor, 
		img_path: str,
		text: str,
		device: str,
		max_new_tokens: int,
		verbose: bool=False,
	):
	try:
		img = Image.open(img_path)
	except FileNotFoundError:
		try:
			response = requests.get(img_path)
			response.raise_for_status()
			img = Image.open(io.BytesIO(response.content))
		except requests.exceptions.RequestException as e:
			print(f"ERROR: failed to load image from {img_path} => {e}")
			return
	img = img.convert("RGB")
	if verbose: print(f"IMG: {type(img)} {img.size} {img.mode}")

	model_id = getattr(model.config, '_name_or_path', None)
	if model_id is None:
		model_id = getattr(model, 'name_or_path', 'unknown_model')

	inputs = processor(
		images=img,
		text=text,
		padding=True,
		return_tensors="pt"
	).to(device)
	with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
		output = model.generate(
			**inputs, 
			max_new_tokens=max_new_tokens,
			use_cache=True,
		)
	vlm_response = processor.decode(output[0], skip_special_tokens=True)
	if verbose:
		print(f"\nVLM response:\n{vlm_response}")
	vlm_response_parsed = get_vlm_response(
		model_id=model_id, 
		raw_response=vlm_response, 
		verbose=verbose,
	)
	return vlm_response_parsed

def get_vlm_based_labels_inefficient(
		model_id: str,
		device: str,
		image_paths: Union[str, List[str]],
		batch_size: int,
		max_new_tokens: int,
		verbose: bool = False,
	) -> List[List[str]]:

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		if verbose:
			print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	processor, model = load_(model_id, device)

	all_keywords = []
	for i, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing images"):
		text = get_prompt(
			model_id=model_id, 
			processor=processor,
			img_path=img_path,
		)
		keywords = query_local_vlm(
			model=model, 
			processor=processor,
			img_path=img_path, 
			text=text,
			device=device,
			max_new_tokens=max_new_tokens,
			verbose=verbose,
		)
		all_keywords.append(keywords)
	return all_keywords

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="VLLM-based keyword extraction for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, help="Path to the metadata CSV file")
	parser.add_argument("--image_path", '-i', type=str, help="img path [or URL]")
	parser.add_argument("--model_id", '-m', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace model ID")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=16, help="Batch size for processing")
	parser.add_argument("--max_new_tokens", '-mnt', type=int, default=64, help="Batch size for processing")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")

	args = parser.parse_args()
	args.device = torch.device(args.device)
	print(args)

	if args.csv_file:
		df = pd.read_csv(
			filepath_or_buffer=args.csv_file, 
			on_bad_lines='skip', 
			dtype=dtypes, 
			low_memory=False,
		)
		if 'img_path' not in df.columns:
			raise ValueError("CSV file must have 'img_path' column")
		img_paths = df['img_path'].tolist()
		print(f"Loaded {len(img_paths)} images from {args.csv_file}")
		output_csv = args.csv_file.replace(".csv", "_vlm_keywords.csv")
	elif args.image_path:
		img_paths = [args.image_path]
	else:
		raise ValueError("Either --csv_file or --image_path must be provided")

	keywords = get_vlm_based_labels_inefficient(
		model_id=args.model_id,
		device=args.device,
		image_paths=img_paths,
		batch_size=args.batch_size,
		max_new_tokens=args.max_new_tokens,
		verbose=args.verbose,
	)
	print(f"{len(keywords)} Extracted keywords: {keywords}")

	if args.csv_file:
		df['vlm_keywords'] = keywords
		df.to_csv(output_csv, index=False)
		try:
			df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")

if __name__ == "__main__":
	main()