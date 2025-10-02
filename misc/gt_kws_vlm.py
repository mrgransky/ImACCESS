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
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct" # only fits Puhti and Mahti

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
Identify up to {k} most prominent, factual and distinct **KEYWORDS** that capture the main action, object or event.
Exclude any explanatory text, comments, questions, or words about image quality, style, or temporal era.
**Return **ONLY** a clean Python list of keywords.
"""

def _load_vlm_(model_id: str, device: str, verbose: bool=False):
	if verbose:
		print(f"[INFO] Loading model: {model_id} on {device}")
	config = tfs.AutoConfig.from_pretrained(model_id)
	if verbose:
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

	model = model_cls.from_pretrained(
		model_id,
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True,
		trust_remote_code=True,
		cache_dir=cache_directory[USER]
	)

	if verbose:
		print(f"[INFO] Processor: {processor.__class__.__name__} {type(processor)}")
		print(f"[INFO] Model: {model.__class__.__name__} {type(model)}")

	model.to(device)
	return processor, model

def get_prompt(
		model_id: str, 
		processor: tfs.AutoProcessor, 
		img_path: str,
		max_kws: int,
	):
	if "-1.5-" in model_id or "bakLlava" in model_id:
		txt = f"USER: <image>\n{VLM_INSTRUCTION_TEMPLATE.format(k=max_kws)}\nASSISTANT:"
	else:
		messages = [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": VLM_INSTRUCTION_TEMPLATE.format(k=max_kws)},
					{"type": "image", "image": img_path},
				],
			},
		]	
		txt = processor.apply_chat_template(
			messages, 
			tokenize=False, 
			add_generation_prompt=True,
		)
	return txt

def get_vlm_response(model_id: str, raw_response: str, verbose: bool=False):
	if "Qwen" in model_id:
		return _qwen_vlm_(raw_response, verbose=verbose)
	elif "llava" in model_id:
		return _llava_vlm_(raw_response, verbose=verbose)
	else:
		raise NotImplementedError(f"VLM response parsing not implemented for {model_id}")

def _qwen_vlm_(response: str, verbose: bool=False) -> Optional[List[str]]:
		"""Parse keywords from VLM response, handling various output formats."""
		
		# Temporal patterns to filter out
		TEMPORAL_PATTERNS = [
				r"\b\d{4}\b",              # 1900, 1944, 2020
				r"\b\d{3,4}s\b",           # 1940s, 1800s
				r"\bcentur(?:y|ies)\b",    # century, centuries
				r"\bdecade(?:s)?\b",       # decade, decades
				r"\bseason(?:s)?\b",       # season, seasons
				r"\b(month|months|day|days|year|years)\b",  # explicit time units
				r"\bworld war\s*[ivx]+\b",  # World War II, World War I (roman numerals)
		]
		
		def is_temporal(s: str) -> bool:
				"""Check if string contains temporal expressions."""
				ss = s.lower()
				for p in TEMPORAL_PATTERNS:
						if re.search(p, ss):
								if verbose:
										print(f"[DEBUG] Temporal filter matched '{s}' with pattern '{p}'")
								return True
				return False
		
		def clean_item_text(s: str) -> str:
				"""Clean individual item: remove numbering, markdown, quotes, brackets."""
				original = s
				
				# Remove leading numbering like "1. " or "1) "
				s = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", s)
				
				# Strip markdown emphasis
				s = re.sub(r"\*+", "", s)
				
				# Normalize smart quotes/apostrophes to standard ASCII
				s = s.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
				
				# Remove leading bracket if leaked
				s = s.lstrip("[")
				
				# Strip wrapping quotes
				if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in "'\""):
						s = s[1:-1]
				
				s = s.strip()
				
				# Optional: trim possessive 's (e.g., "Desmond's" -> "Desmond")
				# Comment out the next line if you want to keep possessives
				s = re.sub(r"(?<=\w)'s\b", "", s)
				
				# Collapse whitespace
				s = re.sub(r"\s+", " ", s)
				
				if verbose and original != s:
						print(f"[DEBUG] Cleaned item: '{original}' -> '{s}'")
				return s
		
		def split_items_relaxed(list_like: str) -> List[str]:
				"""Split list-like string by commas, handling unbalanced brackets/quotes."""
				if verbose:
						print(f"[DEBUG] split_items_relaxed input: {repr(list_like)}")
				
				inner = list_like.strip()
				if inner.startswith("["):
						inner = inner[1:]
						if verbose: print(f"[DEBUG] Removed leading '[', now: {repr(inner)}")
				if inner.endswith("]"):
						inner = inner[:-1]
						if verbose: print(f"[DEBUG] Removed trailing ']', now: {repr(inner)}")
				
				parts = [p.strip() for p in inner.split(",")]
				if verbose:
						print(f"[DEBUG] Split by comma into {len(parts)} parts: {parts}")
				
				items = []
				for idx, p in enumerate(parts):
						if not p or p in {"'", '"'}:
								if verbose: print(f"[DEBUG] Part {idx} is empty or just a quote, skipping: {repr(p)}")
								continue
						# Handle dangling quotes
						if (p.startswith("'") and not p.endswith("'")) or (p.startswith('"') and not p.endswith('"')):
								if verbose: print(f"[DEBUG] Part {idx} has dangling quote, stripping: {repr(p)}")
								p = p.strip("'\"")
						cleaned = clean_item_text(p)
						if cleaned:
								items.append(cleaned)
								if verbose: print(f"[DEBUG] Part {idx} added: '{cleaned}'")
						else:
								if verbose: print(f"[DEBUG] Part {idx} cleaned to empty, skipping")
				
				if verbose:
						print(f"[DEBUG] split_items_relaxed output: {items}")
				return [x for x in items if x]
		
		def dedupe_preserve_order(items: List[str], limit: int = 5) -> List[str]:
				"""Remove duplicates while preserving order, cap at limit."""
				if verbose:
						print(f"[DEBUG] dedupe_preserve_order input ({len(items)} items): {items}")
				
				seen, out = set(), []
				for idx, x in enumerate(items):
						if x and x not in seen:
								out.append(x)
								seen.add(x)
								if verbose: print(f"[DEBUG] Item {idx} '{x}' added (unique)")
						elif x in seen:
								if verbose: print(f"[DEBUG] Item {idx} '{x}' skipped (duplicate)")
						if len(out) >= limit:
								if verbose: print(f"[DEBUG] Reached limit of {limit}, stopping")
								break
				
				if verbose:
						print(f"[DEBUG] dedupe_preserve_order output ({len(out)} items): {out}")
				return out
		
		# Main parsing logic
		if verbose:
				print(f"\n{'='*80}")
				print(f"[DEBUG] Raw VLM output ({len(response)} chars):\n{response}")
				print(f"{'='*80}\n")
		
		if not isinstance(response, str):
				if verbose: print("[ERROR] VLM output is not a string.")
				return None

		# 1) Try to capture balanced lists
		all_matches = re.findall(r"\[[^\[\]]+\]", response, re.DOTALL)
		if verbose:
				print(f"[DEBUG] Regex r\"\\[[^\\[\\]]+\\]\" found {len(all_matches)} balanced lists")
				for idx, m in enumerate(all_matches):
						print(f"[DEBUG]   Match {idx}: {repr(m[:100])}{'...' if len(m) > 100 else ''}")

		# Detect numbered singletons like [1. thing] repeated across lines
		numbered_singletons = []
		for idx, m in enumerate(all_matches):
				inner = m[1:-1].strip()
				if re.match(r"^\d+\s*[\.\)]\s*", inner) and "," not in inner:
						numbered_singletons.append(inner)
						if verbose:
								print(f"[DEBUG] Match {idx} is a numbered singleton: {repr(inner)}")
		
		if verbose:
				print(f"\n[DEBUG] Numbered singleton segments: {len(numbered_singletons)}")
		
		if numbered_singletons and len(numbered_singletons) >= max(2, len(all_matches)//2):
				if verbose:
						print(f"[DEBUG] Processing {len(numbered_singletons)} numbered singletons (threshold met)")
				items = [clean_item_text(s) for s in numbered_singletons]
				if verbose:
						print(f"[DEBUG] After cleaning: {items}")
				items = [i for i in items if i and not is_temporal(i)]
				if verbose:
						print(f"[DEBUG] After temporal filter: {items}")
				result = dedupe_preserve_order(items, limit=5)
				if result:
						if verbose: print(f"\n[INFO] ✓ Final parsed keywords (from numbered singletons): {result}\n")
						return result
				else:
						if verbose: print(f"[DEBUG] No valid items after deduplication")

		primary = None
		if all_matches:
				# Pick the longest as primary
				primary = max(all_matches, key=len)
				if verbose:
						print(f"\n[DEBUG] Selected primary list (longest, {len(primary)} chars): {repr(primary[:200])}{'...' if len(primary) > 200 else ''}")

				# If it's a numbered list inside a single pair of brackets
				if re.search(r"\d+\s*[\.\)]\s*", primary):
						if verbose: print(f"[DEBUG] Primary contains numbered format (e.g., '1. item')")
						cleaned = re.sub(r"^\s*\[|\]\s*$", "", primary)
						if verbose: print(f"[DEBUG] After removing outer brackets: {repr(cleaned[:200])}")
						parts = [clean_item_text(p) for p in cleaned.split(",")]
						if verbose: print(f"[DEBUG] Split into {len(parts)} parts: {parts}")
						items = [p for p in parts if p and not is_temporal(p)]
						if verbose: print(f"[DEBUG] After temporal filter: {items}")
						result = dedupe_preserve_order(items, limit=5)
						if result:
								if verbose: print(f"\n[INFO] ✓ Final parsed keywords (numbered list): {result}\n")
								return result
						else:
								if verbose: print(f"[DEBUG] No valid items after deduplication")

				# Try literal_eval for proper lists (may fail if apostrophes unescaped)
				if verbose: print(f"\n[DEBUG] Attempting ast.literal_eval on primary...")
				try:
					parsed = ast.literal_eval(primary)
					if verbose: print(f"[DEBUG] ✓ ast.literal_eval succeeded: {parsed}")
					if isinstance(parsed, list):
						if verbose: print(f"[DEBUG] Parsed result is a list with {len(parsed)} items")
						items = [clean_item_text(str(k)) for k in parsed if isinstance(k, (str, int, float))]
						if verbose: print(f"[DEBUG] After cleaning: {items}")
						items = [i for i in items if i and not is_temporal(i)]
						if verbose: print(f"[DEBUG] After temporal filter: {items}")
						result = dedupe_preserve_order(items, limit=5)
						if result:
							if verbose: print(f"\n[INFO] ✓ Final parsed keywords (literal list): {result}\n")
							return result
						else:
							if verbose: print(f"[DEBUG] No valid items after deduplication")
					else:
						if verbose: print(f"[DEBUG] Parsed result is not a list: {type(parsed)}")
				except Exception as e:
					if verbose: print(f"[ERROR] ast.literal_eval failed: {type(e).__name__}: {e}")
					if verbose: print(f"[DEBUG] Falling back to relaxed split...")
					# Unescaped apostrophes or minor issues: relaxed split
					items = split_items_relaxed(primary)
					if verbose: print(f"[DEBUG] Relaxed split returned {len(items)} items: {items}")
					items = [i for i in items if i and not is_temporal(i)]
					if verbose: print(f"[DEBUG] After temporal filter: {items}")
					result = dedupe_preserve_order(items, limit=5)
					if result:
						if verbose: print(f"\n[INFO] ✓ Final parsed keywords (relaxed primary): {result}\n")
						return result
					else:
						if verbose: print(f"[DEBUG] No valid items after deduplication")

		# 2) If no balanced list found (truncated/unbalanced case): recover from 'assistant' block
		if verbose: print(f"\n[DEBUG] Attempting recovery from 'assistant' block...")
		after_assistant = response.split("assistant")[-1].strip()
		if verbose:
			print(f"[DEBUG] Content after 'assistant' ({len(after_assistant)} chars): {repr(after_assistant[:200])}{'...' if len(after_assistant) > 200 else ''}")
		
		idx = after_assistant.find("[")
		if idx != -1:
				if verbose: print(f"[DEBUG] Found '[' at position {idx}")
				blob = after_assistant[idx:]
				open_count = blob.count("[")
				close_count = blob.count("]")
				if verbose: print(f"[DEBUG] Blob has {open_count} '[' and {close_count} ']'")
				# If it looks truncated (e.g., ends with a quote), try to close it
				if open_count > close_count:
						if verbose: print(f"[DEBUG] Unbalanced brackets detected, adding ']'")
						blob = blob + "]"
				if verbose: print(f"[DEBUG] Recovered blob: {repr(blob[:200])}{'...' if len(blob) > 200 else ''}")
				items = split_items_relaxed(blob)
				if verbose: print(f"[DEBUG] Relaxed split returned {len(items)} items: {items}")
				items = [i for i in items if i and not is_temporal(i)]
				if verbose: print(f"[DEBUG] After temporal filter: {items}")
				result = dedupe_preserve_order(items, limit=5)
				if result:
						if verbose: print(f"\n[INFO] ✓ Final parsed keywords (recovered blob): {result}\n")
						return result
				else:
						if verbose: print(f"[DEBUG] No valid items after deduplication")
		else:
				if verbose: print(f"[DEBUG] No '[' found in content after 'assistant'")

		# 3) Last resort: comma-split after assistant
		if verbose: print(f"\n[DEBUG] Last resort: comma-splitting after 'assistant'...")
		comma_parts = [clean_item_text(x) for x in after_assistant.split(",") if x.strip()]
		if verbose: print(f"[DEBUG] Comma split produced {len(comma_parts)} parts: {comma_parts}")
		comma_parts = [i for i in comma_parts if i and not is_temporal(i)]
		if verbose: print(f"[DEBUG] After temporal filter: {comma_parts}")
		
		if len(comma_parts) > 1:
				result = dedupe_preserve_order(comma_parts, limit=5)
				if verbose: print(f"\n[INFO] ✓ Final last-resort keywords: {result}\n")
				return result
		else:
				if verbose: print(f"[DEBUG] Not enough comma parts ({len(comma_parts)} <= 1)")

		if verbose:
				print(f"\n[ERROR] ✗ Unable to parse any keywords from VLM output.\n")
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
		max_generated_tks: int,
		verbose: bool=False,
	):
	try:
		img = Image.open(img_path)
	except Exception as e:
		if verbose: print(f"Failed to load image from {img_path} => {e}")
		try:
			response = requests.get(img_path)
			response.raise_for_status()
			img_content = io.BytesIO(response.content) # Convert response content to BytesIO object
			img = Image.open(img_content)
		except requests.exceptions.RequestException as e:
			if verbose: print(f"ERROR: failed to load image from {img_path} => {e}")
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
			max_new_tokens=max_generated_tks,
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

def get_vlm_based_labels(
		model_id: str,
		device: str,
		batch_size: int,
		max_kws: int,
		max_generated_tks: int,
		csv_file: str=None,
		image_path: str=None,
		verbose: bool = False,
	) -> List[List[str]]:

	if csv_file:
		output_csv = csv_file.replace(".csv", "_vlm_keywords.csv")

	if csv_file and image_path:
		raise ValueError("Only one of csv_file or image_path must be provided")

	if csv_file and os.path.exists(output_csv):
		df = pd.read_csv(
			filepath_or_buffer=output_csv,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
		if 'vlm_keywords' in df.columns:
			if verbose: print(f"Found existing VLM keywords in {output_csv}")
			return df['vlm_keywords'].tolist()

	if csv_file:
		df = pd.read_csv(
			filepath_or_buffer=csv_file,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
		if 'img_path' not in df.columns:
			raise ValueError("CSV file must have 'img_path' column")
		image_paths = df['img_path'].tolist()
		if verbose:
			print(f"Loaded {len(image_paths)} images from {csv_file}")
	elif image_path:
		image_paths = [image_path]
		if verbose:
			print(f"Loaded 1 image from {image_path}")
	else:
		raise ValueError("Either csv_file or image_path must be provided")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		if verbose:
			print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	processor, model = _load_vlm_(model_id, device, verbose=verbose)

	all_keywords = []
	for i, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing images"):
		if verbose: print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
		text = get_prompt(
			model_id=model_id, 
			processor=processor,
			img_path=img_path,
			max_kws=max_kws,
		)
		if verbose: print(f"Prompt:\n{text}")
		keywords = query_local_vlm(
			model=model, 
			processor=processor,
			img_path=img_path, 
			text=text,
			device=device,
			max_generated_tks=max_generated_tks,
			verbose=verbose,
		)
		all_keywords.append(keywords)

	if csv_file:
		df['vlm_keywords'] = all_keywords
		df.to_csv(output_csv, index=False)
		try:
			df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")
		if verbose:
			print(f"Saved {len(all_keywords)} keywords to {output_csv}")
			print(f"Done! dataframe: {df.shape} {list(df.columns)}")

	return all_keywords

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="VLLM-based keyword extraction for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, default=None, help="Path to the metadata CSV file")
	parser.add_argument("--image_path", '-i', type=str, default=None, help="img path [or URL]")
	parser.add_argument("--model_id", '-m', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace Vision-Language model ID")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device('cuda:0' or 'cpu')")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=16, help="Batch size for processing")
	parser.add_argument("--max_keywords", '-mkw', type=int, default=5, help="Max number of keywords to extract")
	parser.add_argument("--max_generated_tks", '-mgt', type=int, default=64, help="Batch size for processing")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")

	args = parser.parse_args()
	args.device = torch.device(args.device)
	print(args)

	keywords = get_vlm_based_labels(
		model_id=args.model_id,
		device=args.device,
		csv_file=args.csv_file,
		image_path=args.image_path,
		batch_size=args.batch_size,
		max_kws=args.max_keywords,
		max_generated_tks=args.max_generated_tks,
		verbose=args.verbose,
	)
	if args.verbose:
		print(f"{len(keywords)} Extracted keywords")
		for i, kw in enumerate(keywords):
			print(f"{i:06d}. {len(kw)} {kw}")

if __name__ == "__main__":
	main()