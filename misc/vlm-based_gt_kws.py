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
**Return *only* these keywords as a clean, parseable Python list, e.g., ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5'].**
"""

def get_vlm_response(model_id: str, raw_vlm_response: str, verbose: bool = False):
	if "Qwen" in model_id:
		return _qwen_vlm_response(raw_vlm_response, verbose=verbose)
	else:
		raise NotImplementedError(f"VLM response parsing not implemented for {model_id}")

def _qwen_vlm_response(vlm_response: str, verbose: bool = True) -> list:
	"""
	Parse VLM response to get a clean Python list of keywords.
	Handles various formatting issues and provides verbose debugging output.
	Args:
			vlm_response (str): Raw response string from VLM API.
			verbose (bool): If True, show all steps and warnings.
	Returns:
			list: List of parsed keywords.
	"""
	# Initial raw output
	if verbose:
			print("="*60)
			print("[DEBUG] Raw VLM output:\n", vlm_response)
	if not isinstance(vlm_response, str):
			if verbose:
					print("[ERROR] VLM output is not a string.")
			return []
	# Find the start of the keywords list
	# 1. Try direct Python list string parsing
	list_match = re.search(r"\[(.*?)\]", vlm_response, re.DOTALL)
	if verbose:
			print("[DEBUG] Regex match:", "Found" if list_match else "Not found")
	if list_match:
			list_str = list_match.group(0)  # the "[...]" part
			if verbose:
					print("[DEBUG] Extracted list portion:", list_str)
			try:
					parsed_keywords = ast.literal_eval(list_str)
					if verbose:
							print("[DEBUG] Parsed Python list:", parsed_keywords)
					# Ensure it's a list of strings
					if isinstance(parsed_keywords, list):
							result = [str(x).strip() for x in parsed_keywords if isinstance(x, str)]
							print("[INFO] Final parsed keywords:", result)
							print("="*60)
							return result
					else:
							if verbose:
									print("[ERROR] Parsed output is not a list.")
			except Exception as e:
					if verbose:
							print("[ERROR] ast.literal_eval failed:", e)
	# 2. As fallback, try extracting keywords separated by commas or newline after "assistant"
	if verbose:
			print("[DEBUG] Trying fallback extraction.")
	after_assistant = vlm_response.split("assistant")[-1].strip()
	candidates = re.findall(r"'([^']+)'", after_assistant)
	if verbose:
			print("[DEBUG] Regex candidates:", candidates)
	if candidates:
			result = [str(x).strip() for x in candidates]
			print("[INFO] Final parsed fallback keywords:", result)
			print("="*60)
			return result
	# 3. Last fallback: split by commas and validate
	raw_split = [x.strip(" ,'\"]") for x in after_assistant.split(",") if x.strip()]
	if verbose:
			print("[DEBUG] Comma split candidates:", raw_split)
	if len(raw_split) > 1:
			print("[INFO] Final last-resort keywords:", raw_split)
			print("="*60)
			return raw_split
	if verbose:
			print("[ERROR] Unable to parse any keywords from VLM output.")
			print("="*60)
	return None

def query_local_vlm(
		model: tfs.PreTrainedModel, 
		processor: tfs.AutoProcessor, 
		img_path: str,
		text: str,
		device: str,
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

	print(f"IMG: {type(img)} {img.size} {img.mode}")
	model_id = getattr(model.config, '_name_or_path', None)
	print(f"Model ID: {model_id}")
	inputs = processor(
		images=img,
		text=text,
		padding=True,
		return_tensors="pt"
	).to(device)

	# autoregressively complete prompt
	output = model.generate(**inputs, max_new_tokens=128)
	print("Output:")
	print(processor.decode(output[0], skip_special_tokens=True))
	vlm_response = processor.decode(output[0], skip_special_tokens=True)
	# vlm_response_parsed = _qwen_vlm_response(vlm_response, verbose=True)
	vlm_response_parsed = get_vlm_response(
		model_id=model_id, 
		raw_vlm_response=vlm_response, 
		verbose=True
	)
	return vlm_response_parsed

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

def get_prompt(model_id: str, processor: tfs.AutoProcessor, img_path: str):
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

def get_vlm_based_labels_inefficient(
		model_id: str,
		device: str,
		image_paths: Union[str, List[str]],
		batch_size: int = 64,
		verbose: bool = False,
	) -> List[List[str]]:
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		if verbose:
			print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	processor, model = load_(model_id, device)

	# text = get_prompt(
	# 	model_id=model_id, 
	# 	processor=processor,
	# 	img_path=image_path
	# )

	for i, img_path in enumerate(image_paths):
		text = get_prompt(
			model_id=model_id, 
			processor=processor,
			img_path=image_paths[i],
		)
		query_local_vlm(
			model=model, 
			processor=processor,
			img_path=image_paths[i], 
			text=text,
			device=device
		)

def get_vlm_based_labels_efficient():
	pass

def main():
	parser = argparse.ArgumentParser(description="VLLM-based keyword extraction for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, help="Path to the metadata CSV file")
	parser.add_argument('--image_path', '-i',type=str, required=True, help='img path [or URL]')
	parser.add_argument("--model_id", '-m', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="HuggingFace model ID")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
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
	elif args.image_path:
		img_paths = [args.image_path]
	else:
		raise ValueError("Either --csv_file or --image_path must be provided")

	get_vlm_based_labels_inefficient(
		model_id=args.model_id,
		device=args.device,
		image_path=args.image_path,
		batch_size=1,
		verbose=True,
	)

if __name__ == "__main__":
	main()