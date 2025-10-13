from utils import *

# llama:
# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


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

EXP_BACKOFF = 2  # seconds

VLM_INSTRUCTION_TEMPLATE = """Act as a meticulous historical archivist specializing in 20th century documentation.
Identify up to {k} most prominent, factual and distinct **KEYWORDS** that capture the main action, object, or event.

**CRITICAL RULES**:
- Return **ONLY** a clean, valid and parsable **Python LIST** with a maximum of {k} keywords.
- **ZERO HALLUCINATION POLICY**: You should not invent or infer specifics that lack clear verification from the visual content. When in doubt, omit rather than fabricate.
- **ABSOLUTELY NO** additional explanatory text, code blocks, terms containing numbers, comments, tags, thoughts, questions, or explanations before or after the Python list.
- **STRICTLY EXCLUDE TEMPORAL KEYWORDS** such as dates, times, time periods, seasons, months, days, years, decades, centuries, or any time-related phrases.
- **STRICTLY EXCLUDE** vague, generic, or historical keywords.
- **STRICTLY EXCLUDE** image quality, type, format, or style as keywords.
- Exclude numerical words, special characters, stopwords, or abbreviations.
- Exclude meaningless, repeating or synonym-duplicate keywords.
- The **Python LIST** must be the **VERY LAST THING** in your response.
"""

def _load_vlm_(
		model_id: str,
		device: str,
		use_quantization: bool = False,
		quantization_bits: int = 8,          # 4 or 8
		verbose: bool = False,
	) -> Tuple[tfs.PreTrainedTokenizerBase, torch.nn.Module]:

	if verbose:
			print("\n[DEBUG] ------------------- HARDWARE INFO -------------------")
			print(f"[DEBUG] torch version          : {torch.__version__}")
			print(f"[DEBUG] CUDA available?       : {torch.cuda.is_available()}")
			if torch.cuda.is_available():
					cur = torch.cuda.current_device()
					print(f"[DEBUG] Current CUDA device   : {cur} "
								f"({torch.cuda.get_device_name(cur)})")
					major, minor = torch.cuda.get_device_capability(cur)
					print(f"[DEBUG] Compute capability    : {major}.{minor}")
					print(f"[DEBUG] BF16 support?         : {torch.cuda.is_bf16_supported()}")
					print(f"[DEBUG] CUDA memory allocated: "
								f"{torch.cuda.memory_allocated(cur)//(1024**2)} MiB")
					print(f"[DEBUG] CUDA memory reserved : "
								f"{torch.cuda.memory_reserved(cur)//(1024**2)} MiB")
			else:
					print("[DEBUG] Running on CPU only")
			print("[DEBUG] ----------------------------------------------------\n")

	if verbose:
			print(f"[INFO] Loading configuration for model_id='{model_id}'")

	config = tfs.AutoConfig.from_pretrained(model_id, trust_remote_code=True,)

	if verbose:
			print("[INFO] Config summary")
			print(f"   • model_type        : {config.model_type}")
			print(f"   • architectures     : {config.architectures}")
			print(f"   • torch_dtype (if set) : {config.torch_dtype}")
			print()

	model_cls = None

	if config.architectures:
		cls_name = config.architectures[0]          # first architecture listed
		if hasattr(tfs, cls_name):
			model_cls = getattr(tfs, cls_name)
	if model_cls is None:
		raise ValueError(f"Unable to locate model class for architecture(s): {config.architectures}")

	if verbose:
			print(f"[INFO] Resolved model class → {model_cls.__name__}\n")

	def _optimal_dtype(m_id: str, dev: str) -> torch.dtype:
			bf16_ok = (
					torch.cuda.is_available()
					and torch.cuda.is_bf16_supported()
					and dev != "cpu"
			)
			if "Qwen" in m_id:
					return torch.bfloat16 if bf16_ok else torch.float16
			if "llava" in m_id.lower():
					return torch.float16
			if "falcon" in m_id.lower():
					return torch.bfloat16 if bf16_ok else torch.float16
			# default fallback
			return torch.bfloat16 if bf16_ok else torch.float16
	dtype = _optimal_dtype(model_id, device)

	if verbose:
			print("[INFO] Dtype selection")
			print(f"   • BF16 supported on this device? : {torch.cuda.is_bf16_supported()}")
			print(f"   • Chosen torch dtype            : {dtype}\n")
	def _optimal_attn_impl(m_id: str, dev: str) -> str:
			if not torch.cuda.is_available() or dev == "cpu":
					return "eager"
			# try importing flash_attn and check compute capability
			try:
					import flash_attn  # noqa: F401
					major, _ = torch.cuda.get_device_capability()
					flash_ok = major >= 8               # SM80+ required
			except Exception:
					flash_ok = False
			if flash_ok:
					if "Qwen" in m_id:
							return "flash_attention_2"
					if "llava" in m_id.lower():
							return "flash_attention_2"
					return "flash_attention_2"
			return "eager"
	attn_impl = _optimal_attn_impl(model_id, device)

	if verbose:
			print("[INFO] Attention implementation")
			print(f"   • Selected implementation : {attn_impl}\n")
	quantization_config = None

	if use_quantization:
			if quantization_bits == 8:
					quantization_config = tfs.BitsAndBytesConfig(
						load_in_8bit=True,
						bnb_8bit_compute_dtype=torch.bfloat16,
						llm_int8_enable_fp32_cpu_offload=True,
					)
			elif quantization_bits == 4:
					quantization_config = tfs.BitsAndBytesConfig(
						load_in_4bit=True,
						bnb_4bit_quant_type="nf4",
						bnb_4bit_compute_dtype=torch.bfloat16,
						bnb_4bit_use_double_quant=True,
					)
			else:
					raise ValueError("quantization_bits must be 4 or 8")
			if verbose:
					print("[INFO] Quantisation enabled")
					print(f"   • Bits                : {quantization_bits}")
					print(f"   • Config object type  : {type(quantization_config).__name__}")
					print()

	if verbose:
		print("[INFO] Loading processor / tokenizer …")
	processor = tfs.AutoProcessor.from_pretrained(
		model_id,
		use_fast=True,
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)

	if verbose:
		print(f"[INFO] Processor {processor.__class__.__name__} loaded")
	
	model_kwargs: Dict[str, Any] = {
		"low_cpu_mem_usage": True,
		"trust_remote_code": True,
		"cache_dir": cache_directory[USER],
		"attn_implementation": attn_impl,
		"dtype": dtype,
	}

	if use_quantization:
		model_kwargs["quantization_config"] = quantization_config
		model_kwargs["device_map"] = "auto"

	if verbose:
		print("[INFO] Model loading kwargs")
		for k, v in model_kwargs.items():
			if k == "quantization_config":
				print(f"   • {k}: {type(v).__name__}")
			else:
				print(f"   • {k}: {v}")
		print()

	if verbose and torch.cuda.is_available():
			cur = torch.cuda.current_device()
			print("[DEBUG] CUDA memory BEFORE model load")
			print(f"   • allocated : {torch.cuda.memory_allocated(cur)//(1024**2)} MiB")
			print(f"   • reserved  : {torch.cuda.memory_reserved(cur)//(1024**2)} MiB\n")

	if verbose:
			print("[INFO] Calling `from_pretrained` …")

	model = model_cls.from_pretrained(model_id, **model_kwargs)

	if verbose:
			print("[INFO] Model loaded successfully")
			print(f"   • Model class          : {model.__class__.__name__}")
			# first parameter dtype gives a quick hint
			first_param = next(model.parameters())
			print(f"   • First parameter dtype: {first_param.dtype}")
			# Parameter count + naive FP16 memory estimate
			total_params = sum(p.numel() for p in model.parameters())
			approx_fp16_gb = total_params * 2 / (1024 ** 3)   # 2 bytes per fp16 value
			print(f"   • Total parameters    : {total_params:,}")
			print(f"   • Approx. fp16 RAM    : {approx_fp16_gb:.2f} GiB (if stored as fp16)")
			# Show the resolved device map (for both quantised & non‑quantised)
			if hasattr(model, "hf_device_map"):
					dm = model.hf_device_map   # type: ignore[attr-defined]
					print("[INFO] Final device map (model.hf_device_map):")
					# pretty‑print the dict
					for k in sorted(dm):
							print(f"   '{k}': {repr(dm[k])}")
			else:
					print("[INFO] No `hf_device_map` attribute – model resides on a single device")
			print()

	if not use_quantization:
		if verbose:
			print(f"[INFO] Moving model to device '{device}' (full‑precision path)…")
		model.to(device)
		if verbose and torch.cuda.is_available():
			cur = torch.cuda.current_device()
			print("[DEBUG] CUDA memory AFTER model.to()")
			print(f"   • allocated : {torch.cuda.memory_allocated(cur)//(1024**2)} MiB")
			print(f"   • reserved  : {torch.cuda.memory_reserved(cur)//(1024**2)} MiB\n")
	else:
		if verbose:
			print("[INFO] Quantisation path – placement handled by `device_map='auto'`")
			if torch.cuda.is_available():
				gpu_params = sum(1 for p in model.parameters() if p.device.type == "cuda")
				cpu_params = sum(1 for p in model.parameters() if p.device.type == "cpu")
				print(f"   • Parameters on GPU : {gpu_params}")
				print(f"   • Parameters on CPU  : {cpu_params}\n")

	return processor, model

def prepare_prompts_and_images(
		unique_inputs, 
		max_kws, 
		num_threads, 
		verbose=False
	):
	if verbose:
		print(f"[PREP] Preparing prompts and verifying {len(unique_inputs)} images...")
	prep_start = time.time()
	process = psutil.Process()
	base_prompt = VLM_INSTRUCTION_TEMPLATE.format(k=max_kws)
	valid_paths, unique_prompts = [], []
	def verify_path(img_path):
		if img_path is None or not os.path.exists(str(img_path)):
			return None
		try:
			with Image.open(img_path) as img:
				img.verify()
			return img_path
		except Exception:
			return None
	# Parallel verification
	with ThreadPoolExecutor(max_workers=num_threads) as ex:
		verified = list(tqdm(ex.map(verify_path, unique_inputs), total=len(unique_inputs), desc="Verifying images"))
	for v in verified:
		valid_paths.append(v)
		unique_prompts.append(base_prompt if v else None)
	mem_gb = process.memory_info().rss / (1024 ** 3)
	if verbose:
		print(f"[PREP] Completed in {time.time() - prep_start:.2f}s | Memory: {mem_gb:.2f} GB | {sum(v is not None for v in valid_paths)} valid images")
	
	gc.collect()
	
	return valid_paths, unique_prompts

def get_prompt(
		processor: tfs.AutoProcessor, 
		img_path: str,
		max_kws: int,
	):
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

class SafeLogitsProcessor(tfs.LogitsProcessor):
	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
		# Replace NaN/Inf with very negative number so softmax ~ 0
		mask_nonfinite = ~torch.isfinite(scores)
		if mask_nonfinite.any():
				scores = scores.clone()
				scores[mask_nonfinite] = -1e9
		# Optional clamp to avoid extreme magnitudes
		scores = torch.clamp(scores, min=-1e4, max=1e4)
		return scores

def query_local_vlm(
		model: tfs.PreTrainedModel, 
		processor: tfs.AutoProcessor, 
		img_path: str,
		text: str,
		device,
		max_generated_tks: int,
		verbose: bool=False,
	):
	# ========== Entry mem ==========
	if verbose and torch.cuda.is_available():
		mem_alloc = torch.cuda.memory_allocated(device) / (1024**3)
		mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
		mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		print("\n" + "="*80)
		print(f"[ENTRY] GPU {device} Memory: {mem_alloc:.2f}GB allocated | {mem_reserved:.2f}GB reserved | {mem_total:.2f}GB total")
		print(f"[ENTRY] Free: {mem_total - mem_alloc:.2f}GB")
		print("="*80)

	# ========== Load image ==========
	if verbose:
		print(f"[LOAD] Loading image: {img_path}")
	try:
		img = Image.open(img_path)
	except Exception as e:
		if verbose: print(f"[ERROR] Failed local open => {e}, retry via URL")
		try:
			r = requests.get(img_path, timeout=10)
			r.raise_for_status()
			img = Image.open(io.BytesIO(r.content))
		except Exception as e2:
			if verbose: print(f"[ERROR] URL fetch failed => {e2}")
			return None
	img = img.convert("RGB")
	if verbose:
		arr = np.array(img)
		print(f"[IMAGE] Type: {type(img)} Size: {img.size} Mode: {img.mode}")
		print(f"[IMAGE] Shape: {arr.shape} Dtype: {arr.dtype} Min/Max: {arr.min()}/{arr.max()}")
		print(f"[IMAGE] NaN: {np.isnan(arr).any()} Inf: {np.isinf(arr).any()} Size: {arr.nbytes/(1024**2):.2f}MB")
	if img.size[0] == 0 or img.size[1] == 0:
		if verbose: print("[ERROR] Invalid image size")
		return None

	model_id = getattr(model.config, "_name_or_path", None) or getattr(model, "name_or_path", "unknown_model")
	if verbose:
		print(f"[MODEL] ID: {model_id}")
	
	# ========== Preprocess ==========
	if verbose:
		if torch.cuda.is_available():
			print(f"[PREPROCESS] GPU {device} mem before processor: {torch.cuda.memory_allocated(device)/(1024**3):.2f}GB")

	try:
		inputs = processor(
			images=img, 
			text=text, 
			padding=True, 
			return_tensors="pt"
		).to(device, non_blocking=True)
	except Exception as e:
		if verbose: print(f"[ERROR] Processor failed: {e}")
		return None

	if verbose:
		print(f"[PREPROCESS] Keys: {list(inputs.keys())}")
		for k, v in inputs.items():
			if isinstance(v, torch.Tensor):
				info = f"shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
				extra = ""
				if torch.is_floating_point(v):
					extra = f" min={v.min().item():.4f} max={v.max().item():.4f} nan={torch.isnan(v).any().item()} inf={torch.isinf(v).any().item()}"
				print(f"[PREPROCESS]   {k}: {info}{extra}")

	if "pixel_values" not in inputs:
		if verbose: print("[ERROR] 'pixel_values' missing")
		return None
	
	# ========== Move to device ==========
	if verbose and torch.cuda.is_available():
		print(f"[DEVICE] GPU {device} mem before .to(): {torch.cuda.memory_allocated(device)/(1024**3):.2f}GB")
	try:
		inputs = inputs.to(device, non_blocking=True)
	except Exception as e:
		if verbose:
			print(f"[ERROR] inputs.to(device) failed: {e}")
			if torch.cuda.is_available():
				print("[ERROR] After device-side assert, CUDA context is invalid; need process restart.")
		return None
	if verbose and torch.cuda.is_available():
		print(f"[DEVICE] GPU {device} mem after .to(): {torch.cuda.memory_allocated(device)/(1024**3):.2f}GB")
		for k, v in inputs.items():
			if isinstance(v, torch.Tensor):
				print(f"[DEVICE]   {k}: device={v.device}, contiguous={v.is_contiguous()}")
	
	# ========== Generation config & safety ==========
	gen_cfg = getattr(model, "generation_config", None)
	tok = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
	pad_token_id = getattr(gen_cfg, "pad_token_id", None) if gen_cfg else None
	eos_token_id = getattr(gen_cfg, "eos_token_id", None) if gen_cfg else None
	if tok:
		if tok.pad_token_id is None and tok.eos_token_id is not None:
			tok.pad_token_id = tok.eos_token_id
			pad_token_id = tok.pad_token_id

	if gen_cfg:
		if gen_cfg.pad_token_id is None and pad_token_id is not None:
			gen_cfg.pad_token_id = pad_token_id
		if gen_cfg.eos_token_id is None and tok and tok.eos_token_id is not None:
			gen_cfg.eos_token_id = tok.eos_token_id
		# Force deterministic non-sampling
		gen_cfg.do_sample = False
		gen_cfg.num_beams = 1
		gen_cfg.temperature = None  # Remove temperature to avoid warning

	if verbose:
		print(f"[GENERATE] max_new_tokens={max_generated_tks}")
		print(f"[GENERATE] pad_token_id={pad_token_id} eos_token_id={eos_token_id}")
		if torch.cuda.is_available():
			print(f"[GENERATE] GPU {device} mem before: {torch.cuda.memory_allocated(device)/(1024**3):.2f}GB")

	logits_processors = tfs.LogitsProcessorList([SafeLogitsProcessor()])

	# ========== Generate (no autocast) ==========
	try:
		with torch.inference_mode():
			output = model.generate(
				**inputs,
				max_new_tokens=max_generated_tks,
				use_cache=True,
				do_sample=False,
				temperature=None,
				top_k=None,
				top_p=None,
				pad_token_id=getattr(model.generation_config, "pad_token_id", pad_token_id),
				eos_token_id=getattr(model.generation_config, "eos_token_id", eos_token_id),
				logits_processor=logits_processors,
			)
		if verbose:
			print(f"[GENERATE] Done. Output shape={tuple(output.shape)} dtype={output.dtype}")
	except RuntimeError as e:
		if verbose:
			print(f"[ERROR] RuntimeError in generate: {e}")
			if "device-side assert" in str(e).lower():
				print("[ERROR] CUDA device-side assert: context is now invalid; best to skip and continue in a fresh process.")
			if torch.cuda.is_available():
				try:
					torch.cuda.synchronize(device)
				except:
					pass
				print(f"[ERROR] GPU {device} mem at error: {torch.cuda.memory_allocated(device)/(1024**3):.2f}GB")
		return None
	except Exception as e:
		if verbose: print(f"[ERROR] Unexpected error in generate: {type(e).__name__}: {e}")
		return None

	vlm_response = processor.decode(output[0], skip_special_tokens=True)
	if verbose:
		tks_breakdown = get_conversation_token_breakdown(vlm_response, model_id)
		print(f"\n")
		print(f"[RESPONSE] Token Count Breakdown: {tks_breakdown}")
		print(f"{vlm_response}")
		print(f"\n")

	# ========== Memory post ==========
	if torch.cuda.is_available():
		alloc = torch.cuda.memory_allocated(device) / (1024**3)
		resrv = torch.cuda.memory_reserved(device) / (1024**3)
		total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		if verbose:
			print(f"[MEMORY] Post-gen: {alloc:.2f}GB allocated | {resrv:.2f}GB reserved | {total:.2f}GB total")
		if alloc / (total + 1e-6) > 0.9:
			if verbose: print("[MEMORY] >90%, empty_cache()")
			torch.cuda.empty_cache()
	
	# ========== Parse ==========
	try:
		parsed = get_vlm_response(model_id=model_id, raw_response=vlm_response, verbose=verbose)
		if verbose:
			print(f"[PARSE] {parsed}")
		return parsed
	except Exception as e:
		if verbose: print(f"[ERROR] Parsing failed: {e}")
		return None

def get_vlm_based_labels_debug(
		model_id: str,
		device: str,
		batch_size: int,
		max_kws: int,
		max_generated_tks: int,
		csv_file: str=None,
		image_path: str=None,
		use_quantization: bool = False,
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

	processor, model = _load_vlm_(
		model_id=model_id, 
		device=device,
		use_quantization=use_quantization,
		verbose=verbose
	)

	all_keywords = []
	for i, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing images"):
		if verbose: print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
		text = get_prompt(
			processor=processor,
			img_path=img_path,
			max_kws=max_kws,
		)
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

	print(any(kw is not None for kw in all_keywords))

	if csv_file and any(kw is not None for kw in all_keywords):
		df['vlm_keywords'] = all_keywords
		df.to_csv(output_csv, index=False)
		try:
			df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")
		if verbose:
			print(f"Saved {len(all_keywords)} keywords to {output_csv}")
			print(f"Done! dataframe: {df.shape} {list(df.columns)}")
	elif csv_file:
		if verbose:
			print("Skipping file save: all VLM keywords are None")

	return all_keywords

def get_vlm_based_labels_opt(
		model_id: str,
		device: str,
		batch_size: int,
		num_workers: int,
		max_generated_tks: int,
		max_kws: int,
		csv_file: str,
		do_dedup: bool = True,
		max_retries: int = 2,
		use_quantization: bool = False,
		verbose: bool = False,
	) -> List[Optional[List[str]]]:

	if verbose:
		print(f"\n{'='*100}")
		print(f"[INIT] Starting OPTIMIZED batch VLM processing")
		print(f"[INIT] Model: {model_id}")
		print(f"[INIT] Batch size: {batch_size}")
		print(f"[INIT] Device: {device}")
		print(f"{'='*100}\n")
	st_t = time.time()
	
	# ========== Check existing results ==========
	check_start = time.time()
	output_csv = csv_file.replace(".csv", "_vlm_keywords.csv")
	if os.path.exists(output_csv):
		df = pd.read_csv(
			filepath_or_buffer=output_csv,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
		if 'vlm_keywords' in df.columns:
			if verbose: 
				print(f"[EXISTING] Found existing results in {output_csv} ({time.time() - check_start:.2f}s)")
			return df['vlm_keywords'].tolist()
	if verbose:
		print(f"[CHECK] Existing results check: {time.time() - check_start:.2f}s")

	# ========== Load data ==========
	load_start = time.time()
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
		print(f"[DATA] Loaded {len(image_paths)} image paths from CSV ({time.time() - load_start:.2f}s)")

	# Store original inputs for later reference
	original_inputs = image_paths
	if len(original_inputs) == 0:
		return None

	# ========== Load model ==========
	model_start = time.time()
	processor, model = _load_vlm_(
		model_id=model_id, 
		device=device,
		use_quantization=use_quantization,
		verbose=verbose
	)
	if verbose:
		print(f"[MODEL] model & processor loaded in {time.time() - model_start:.2f}s")

	if verbose:
		print(f"[INPUT] Validating inputs...")
		st_t = time.time()
		valid_count = sum(1 for x in original_inputs if x is not None and os.path.exists(str(x)))
		null_count = len(original_inputs) - valid_count
		print(f"📊 Input stats: {len(original_inputs)} total, {valid_count} valid, {null_count} null")
		print(f"[INPUT] Input validation: {time.time() - st_t:.2f}s")

	# ========== Deduplication ==========
	dedup_start = time.time()
	if verbose:
		print(f"[DEDUP] Deduplicating inputs...")
	if do_dedup:
		unique_map: Dict[str, int] = {}
		unique_inputs = []
		original_to_unique_idx = []
		for img_path in original_inputs:
			if img_path is None or not os.path.exists(str(img_path)):
				key = "__NULL__"
			else:
				key = str(img_path)
			if key in unique_map:
				original_to_unique_idx.append(unique_map[key])
			else:
				idx = len(unique_inputs)
				unique_map[key] = idx
				unique_inputs.append(None if key == "__NULL__" else key)
				original_to_unique_idx.append(idx)
	else:
		unique_inputs = []
		for img_path in original_inputs:
			if img_path is None or not os.path.exists(str(img_path)):
				unique_inputs.append(None)
			else:
				unique_inputs.append(str(img_path))
		original_to_unique_idx = list(range(len(unique_inputs)))
	if verbose:
		print(f"[DEDUP] Deduplication: {time.time() - dedup_start:.2f}s ({len(original_inputs)} → {len(unique_inputs)} unique)")

	# ========== Prepare prompts and images ==========
	unique_images, unique_prompts = prepare_prompts_and_images(
		unique_inputs=unique_inputs,
		max_kws=max_kws,
		num_threads=num_workers,
		verbose=verbose
	)

	# ========== Sequential processing ==========
	# Will hold parsed results for unique inputs
	unique_results: List[Optional[List[str]]] = [None] * len(unique_prompts)
	if verbose:
		print(f"[PROCESS] Generating valid indices for {len(unique_inputs)} unique images")
	process_start = time.time()
	valid_indices = [
		i
		for i, (p, img) in enumerate(zip(unique_prompts, unique_images)) 
		if p is not None and img is not None
	]
	if valid_indices:
		if verbose:
			print(f" ├─ Found {len(valid_indices)} Valid indices in {time.time() - process_start:.2f}s")
			print(f" ├─ Sequential processing with optimizations...")
		
		generation_time = 0
		parsing_time = 0
		
		for idx in tqdm(valid_indices, desc="Processing images"):
			img_path = unique_inputs[idx]
			img = unique_images[idx]
				
			# 🔄 RETRY LOGIC for individual images
			for attempt in range(max_retries + 1):
				try:
					if attempt > 0 and verbose:
						print(f"🔄 Retry attempt {attempt + 1}/{max_retries + 1} for image {idx + 1}")
					# ========== Prepare inputs ==========
					is_chat_model = hasattr(processor, "apply_chat_template")

					if is_chat_model:
							messages = [
									{
											"role": "user",
											"content": [
													{"type": "text", "text": VLM_INSTRUCTION_TEMPLATE.format(k=max_kws)},
													{"type": "image", "image": img},
											],
									}
							]
							chat_prompt = processor.apply_chat_template(
									messages,
									tokenize=False,
									add_generation_prompt=True,
							)
							single_inputs = processor(
									images=img,
									text=chat_prompt,
									padding=True,
									return_tensors="pt"
							).to(device)
					else:
							single_inputs = processor(
									images=img,
									text=VLM_INSTRUCTION_TEMPLATE.format(k=max_kws),
									padding=True,
									return_tensors="pt"
							).to(device)

					# ========== Generate response ==========
					gen_start = time.time()
					with torch.inference_mode():
						outputs = model.generate(
							**single_inputs,
							max_new_tokens=max_generated_tks,
							use_cache=True,
							temperature=None,
							top_k=None,
							top_p=None,
							do_sample=False,
							pad_token_id=getattr(model.generation_config, "pad_token_id", None),
							eos_token_id=getattr(model.generation_config, "eos_token_id", None),
						)
					generation_time += time.time() - gen_start
					
					# Decode response
					response = processor.decode(outputs[0], skip_special_tokens=True)
					if verbose:
						print(f"✅ Image {idx + 1} generation successful")
					
					# ========== Parse the response ==========
					parse_start = time.time()
					try:
						parsed = get_vlm_response(
							model_id=model_id,
							raw_response=response,
							verbose=verbose,
						)
						unique_results[idx] = parsed
						parsing_time += time.time() - parse_start
						if verbose and parsed:
							print(f"✅ Parsed keywords: {parsed}")
					except Exception as e:
						parsing_time += time.time() - parse_start
						if verbose:
							print(f"⚠️ Parsing error for image {idx + 1}: {e}")
						unique_results[idx] = None
					break  # Break retry loop on success	
				except Exception as e:
					if verbose:
						print(f"❌ Image {idx + 1} attempt {attempt + 1} failed: {e}")
					
					if attempt < max_retries:
						# Exponential backoff
						sleep_time = EXP_BACKOFF ** attempt
						if verbose:
							print(f"⏳ Waiting {sleep_time}s before retry...")
						time.sleep(sleep_time)
						torch.cuda.empty_cache() if torch.cuda.is_available() else None
					else:
						# Final attempt failed
						if verbose:
							print(f"💥 Image {idx + 1} failed after {max_retries + 1} attempts")
						unique_results[idx] = None
			
			# Clean up after each image
			if 'single_inputs' in locals():
				del single_inputs
			if 'outputs' in locals():
				del outputs
			if 'response' in locals():
				del response
			
			# Memory management - clear cache every few images
			if idx % 50 == 0 and torch.cuda.is_available():
				torch.cuda.empty_cache()
				gc.collect()
	
	if verbose:
		print(f"[PROCESS] Sequential processing: {time.time() - process_start:.2f}s")
		print(f"  ├─ Generation time: {generation_time:.2f}s ({generation_time/len(valid_indices):.3f}s/img)")
		print(f"  └─ Parsing time: {parsing_time:.2f}s ({parsing_time/len(valid_indices):.3f}s/img)")
	
	# ========== Map results back ==========
	map_start = time.time()
	results = []
	for orig_i, uniq_idx in enumerate(original_to_unique_idx):
		results.append(unique_results[uniq_idx])
	if verbose:
		print(f"[MAP] Result mapping: {time.time() - map_start:.2f}s")
	
	# ========== Final statistics ==========
	stats_start = time.time()
	if verbose:
		n_ok = sum(1 for r in results if r is not None)
		n_null = sum(
			1 
			for i, inp in enumerate(original_inputs) 
			if inp is None or not os.path.exists(str(inp))
		)
		n_failed = len(results) - n_ok - n_null
		success_rate = (n_ok / (len(results) - n_null)) * 100 if (len(results) - n_null) > 0 else 0
		
		print(
			f"📊 Final results: {n_ok}/{len(results)-n_null} successful ({success_rate:.1f}%), "
			f"{n_null} null inputs, {n_failed} failed"
		)
		print(f"[STATS] Statistics calculation: {time.time() - stats_start:.2f}s")
	
	# ========== Cleanup ==========
	cleanup_start = time.time()
	del model, processor
	torch.cuda.empty_cache() if torch.cuda.is_available() else None
	if verbose:
		print(f"[CLEANUP] Model cleanup: {time.time() - cleanup_start:.2f}s")

	# ========== Save results ==========
	save_start = time.time()
	if csv_file:
		df['vlm_keywords'] = results
		df.to_csv(output_csv, index=False)
		try:
			df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")
		if verbose:
			print(f"[SAVE] Saved {len(results)} keywords to {output_csv} ({time.time() - save_start:.2f}s)")
			print(f"[SAVE] DataFrame: {df.shape}, columns: {list(df.columns)}")

	if verbose:
		print(f"[FINAL] Total time: {time.time() - st_t:.2f} sec")
		print(f"{'='*100}")

	return results

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="VLLM-based keyword extraction for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, default=None, help="Path to the metadata CSV file")
	parser.add_argument("--image_path", '-i', type=str, default=None, help="img path [or URL]")
	parser.add_argument("--model_id", '-vlm', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace Vision-Language model ID")
	parser.add_argument("--device", '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device('cuda:0' or 'cpu')")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=64, help="Batch size for processing")
	parser.add_argument("--max_keywords", '-mkw', type=int, default=5, help="Max number of keywords to extract")
	parser.add_argument("--max_generated_tks", '-mgt', type=int, default=64, help="Batch size for processing")
	parser.add_argument("--use_quantization", '-q', action='store_true', help="Use quantization")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	parser.add_argument("--debug", '-d', action='store_true', help="Debug mode")

	args = parser.parse_args()
	args.device = torch.device(args.device)
	print(args)

	if args.verbose and torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(args.device)
		total_mem = torch.cuda.get_device_properties(args.device).total_memory / (1024**3)  # Convert to GB
		print(f"Available GPU(s) = {torch.cuda.device_count()}")
		print(f"GPU: {torch.cuda.get_device_name(args.device)}")
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))
		print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
		print(f"CUDA Version: {torch.version.cuda}")

	if args.debug or args.image_path:
		keywords = get_vlm_based_labels_debug(
			model_id=args.model_id,
			device=args.device,
			csv_file=args.csv_file,
			image_path=args.image_path,
			batch_size=args.batch_size,
			max_kws=args.max_keywords,
			max_generated_tks=args.max_generated_tks,
			use_quantization=args.use_quantization,
			verbose=args.verbose,
		)
	else:
		keywords = get_vlm_based_labels_opt(
			model_id=args.model_id,
			device=args.device,
			csv_file=args.csv_file,
			num_workers=args.num_workers,
			batch_size=args.batch_size,
			max_kws=args.max_keywords,
			max_generated_tks=args.max_generated_tks,
			use_quantization=args.use_quantization,
			verbose=args.verbose,
		)

	if args.verbose:
		print(f"{len(keywords)} Extracted keywords")
		for i, kw in enumerate(keywords):
			print(f"{i:06d}. {kw}")

if __name__ == "__main__":
	main()
