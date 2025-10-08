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

TEMPERATURE = 1e-8

print(f"{USER} HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub")
huggingface_hub.login(token=hf_tk)

VLM_INSTRUCTION_TEMPLATE = """Act as a meticulous historical archivist specializing in 20th century documentation.
Identify up to {k} most prominent, factual and distinct **KEYWORDS** that capture the main action, object, or event.

**CRITICAL RULES**:
- Return **ONLY** a clean, valid and parsable Python list with a maximum of {k} keywords.
- **ZERO HALLUCINATION POLICY**: Do NOT invent or assume details that cannot be confidently verified from the visual content. When in doubt, exclude rather than invent.
- **ABSOLUTELY NO** additional explanatory text, code blocks, terms containing numbers, comments, tags, thoughts, questions, or explanations before or after the Python list.
- **STRICTLY EXCLUDE TEMPORAL EXPRESSIONS**: No dates, times, time periods, seasons, months, days, years, decades, centuries, or any time-related phrases (e.g., "early evening", "night", "daytime", "morning", "20th century", ""1950s", "weekend", "May 25th", "July 10").
- **STRICTLY EXCLUDE VAGUE CONTEXT WORDS**: No generic historical or contextual terms (e.g., "warzone", "war", "historical", "vintage", "archive", "wartime", "industrial").
- **STRICTLY EXCLUDE GENERIC IMAGE DESCRIPTORS**: No terms describing the image type, format, or genre (e.g., "Black and White Photography", "Historical Document", "War Photography", "Photograph", "Image", "Photo", "Archive", "Documentation").
- Exclude numerical words, special characters, stopwords, or abbreviations.
- Exclude meaningless, repeating or synonym-duplicate keywords.
- The Python list must be the **VERY LAST THING** in your response.
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

	tokenizer = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
	if tokenizer is not None:
		tokenizer.padding_side = 'left'
		if verbose:
			print(f"[INFO] Set tokenizer padding_side to 'left'")

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
		max_kws: int,
		max_generated_tks: int,
		csv_file: str = None,
		image_path: str = None,
		do_dedup: bool = True,
		checkpoint_interval: int = 1000,
		max_retries: int = 2,
		verbose: bool = False,
	) -> List[Optional[List[str]]]:
	"""
	Optimized VLM-based keyword extraction adopting the LLM batch processing pattern:
	- Smart deduplication of image paths
	- True batch processing with retry logic
	- Checkpoint/resume capability
	- Hybrid fallback for failed batches
	"""
	
	def _save_checkpoint(df, output_csv, temp_suffix="_checkpoint"):
		"""Save intermediate results."""
		checkpoint_path = output_csv.replace(".csv", f"{temp_suffix}.csv")
		try:
			df.to_csv(checkpoint_path, index=False)
			if verbose:
				print(f"[CHECKPOINT] Saved to {checkpoint_path}")
			return True
		except Exception as e:
			if verbose:
				print(f"[CHECKPOINT ERROR] {e}")
			return False
	
	def _load_checkpoint(output_csv, temp_suffix="_checkpoint"):
		"""Load checkpoint if exists."""
		checkpoint_path = output_csv.replace(".csv", f"{temp_suffix}.csv")
		if os.path.exists(checkpoint_path):
			try:
				df = pd.read_csv(checkpoint_path, on_bad_lines='skip', dtype=dtypes, low_memory=False)
				if verbose:
					print(f"[CHECKPOINT] Loaded from {checkpoint_path}")
				return df
			except Exception as e:
				if verbose:
					print(f"[CHECKPOINT ERROR] {e}")
		return None
	
	def _load_single_image(img_path, verbose_inner=False):
		"""Load a single image with error handling."""
		try:
			if img_path.startswith('http://') or img_path.startswith('https://'):
				r = requests.get(img_path, timeout=10)
				r.raise_for_status()
				img = Image.open(io.BytesIO(r.content))
			else:
				img = Image.open(img_path)
			
			img = img.convert("RGB")
			if img.size[0] > 0 and img.size[1] > 0:
				return img
			elif verbose_inner:
				print(f"[LOAD] Invalid image size: {img_path}")
		except Exception as e:
			if verbose_inner:
				print(f"[LOAD ERROR] {img_path}: {e}")
		return None
	
	def _batch_process_images(model, processor, img_paths, prompts, device, max_new_tokens, verbose_inner=False):
		"""Process a batch of images through VLM."""
		if not img_paths:
			return []
		
		try:
			# Load all images for this batch
			images = []
			valid_indices = []
			for idx, img_path in enumerate(img_paths):
				img = _load_single_image(img_path, verbose_inner=verbose_inner)
				if img is not None:
					images.append(img)
					valid_indices.append(idx)
			
			if not images:
				if verbose_inner:
					print(f"[BATCH] No valid images loaded")
				return [None] * len(img_paths)
			
			# Get prompts for valid images
			valid_prompts = [prompts[i] for i in valid_indices]
			
			if verbose_inner:
				print(f"[BATCH] Processing {len(images)} images")
			
			# Batch preprocessing
			if len(images) == 1:
				inputs = processor(images=images[0], text=valid_prompts[0], padding=True, return_tensors="pt")
			else:
				inputs = processor(images=images, text=valid_prompts, padding=True, return_tensors="pt")
			
			inputs = inputs.to(device, non_blocking=True)
			
			# Generation config
			gen_cfg = getattr(model, "generation_config", None)
			tok = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
			pad_token_id = None
			eos_token_id = None
			
			if gen_cfg:
				pad_token_id = getattr(gen_cfg, "pad_token_id", None)
				eos_token_id = getattr(gen_cfg, "eos_token_id", None)
			
			if tok:
				if tok.pad_token_id is None and tok.eos_token_id is not None:
					tok.pad_token_id = tok.eos_token_id
				pad_token_id = tok.pad_token_id
				if eos_token_id is None:
					eos_token_id = tok.eos_token_id
			
			logits_processors = tfs.LogitsProcessorList([SafeLogitsProcessor()])
			
			# Generate
			with torch.inference_mode():
				outputs = model.generate(
					**inputs,
					max_new_tokens=max_new_tokens,
					use_cache=True,
					do_sample=False,
					temperature=None,
					top_k=None,
					top_p=None,
					pad_token_id=pad_token_id,
					eos_token_id=eos_token_id,
					logits_processor=logits_processors,
				)
			
			# Decode batch
			responses = []
			for i in range(outputs.shape[0]):
				response = processor.decode(outputs[i], skip_special_tokens=True)
				responses.append(response)
			
			# Map responses back to original batch order
			batch_results = [None] * len(img_paths)
			for local_idx, global_idx in enumerate(valid_indices):
				batch_results[global_idx] = responses[local_idx]
			
			# Clear GPU memory
			del inputs, outputs
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			
			return batch_results
			
		except RuntimeError as e:
			if verbose_inner:
				print(f"[BATCH ERROR] RuntimeError: {e}")
				if "out of memory" in str(e).lower():
					print("[BATCH] OOM detected")
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			return None
		except Exception as e:
			if verbose_inner:
				print(f"[BATCH ERROR] {type(e).__name__}: {e}")
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			return None
	
	# ===== Main Function Logic =====
	
	if csv_file and image_path:
		raise ValueError("Only one of csv_file or image_path must be provided")
	
	# Setup paths and check for existing results
	if csv_file:
		output_csv = csv_file.replace(".csv", "_vlm_keywords.csv")
		
		# Check if already processed
		if os.path.exists(output_csv):
			df = pd.read_csv(output_csv, on_bad_lines='skip', dtype=dtypes, low_memory=False)
			if 'vlm_keywords' in df.columns:
				if verbose:
					print(f"[EXISTING] Found results in {output_csv}")
				return df['vlm_keywords'].tolist()
		
		# Try to load checkpoint
		checkpoint_df = _load_checkpoint(output_csv)
		if checkpoint_df is not None and 'vlm_keywords' in checkpoint_df.columns:
			df = checkpoint_df
		else:
			df = pd.read_csv(csv_file, on_bad_lines='skip', dtype=dtypes, low_memory=False)
			df['vlm_keywords'] = None
		
		if 'img_path' not in df.columns:
			raise ValueError("CSV file must have 'img_path' column")
		
		image_paths = df['img_path'].tolist()
		
		if verbose:
			total = len(image_paths)
			processed = df['vlm_keywords'].notna().sum() if 'vlm_keywords' in df.columns else 0
			print(f"[SETUP] Total: {total} | Already processed: {processed}")
	
	elif image_path:
		image_paths = [image_path]
		df = None
		output_csv = None
		if verbose:
			print(f"[SETUP] Single image: {image_path}")
	else:
		raise ValueError("Either csv_file or image_path must be provided")
	
	# GPU info
	if torch.cuda.is_available() and verbose:
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		print(f"[GPU] {gpu_name} | {total_mem:.2f}GB VRAM")
	
	# Load model
	if verbose:
		print(f"[MODEL] Loading {model_id}...")
	processor, model = _load_vlm_(model_id, device, verbose=verbose)
	
	# Set padding side for decoder-only models
	tokenizer = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
	if tokenizer is not None:
		tokenizer.padding_side = 'left'
		if verbose:
			print(f"[MODEL] Set tokenizer padding_side to 'left'")
	
	# Prepare inputs
	inputs = image_paths
	if len(inputs) == 0:
		return None
	
	if verbose:
		valid_count = sum(1 for x in inputs if x is not None and str(x).strip() not in ("", "nan", "None"))
		null_count = len(inputs) - valid_count
		print(f"📊 Input stats: {len(inputs)} total, {valid_count} valid, {null_count} null")
	
	# 🔧 NULL-SAFE DEDUPLICATION
	if do_dedup:
		unique_map: Dict[str, int] = {}
		unique_inputs = []
		original_to_unique_idx = []
		
		for img_path in inputs:
			if img_path is None or str(img_path).strip() in ("", "nan", "None"):
				key = "__NULL__"
			else:
				key = str(img_path).strip()
			
			if key in unique_map:
				original_to_unique_idx.append(unique_map[key])
			else:
				idx = len(unique_inputs)
				unique_map[key] = idx
				unique_inputs.append(None if key == "__NULL__" else key)
				original_to_unique_idx.append(idx)
		
		if verbose:
			print(f"📊 Deduplication: {len(inputs)} -> {len(unique_inputs)} unique images")
	else:
		unique_inputs = []
		for img_path in inputs:
			if img_path is None or str(img_path).strip() in ("", "nan", "None"):
				unique_inputs.append(None)
			else:
				unique_inputs.append(str(img_path).strip())
		original_to_unique_idx = list(range(len(unique_inputs)))
	
	# Prepare prompts for unique inputs
	unique_prompts = []
	for img_path in unique_inputs:
		if img_path is None:
			unique_prompts.append(None)
		else:
			prompt = get_prompt(model_id=model_id, processor=processor, img_path=img_path, max_kws=max_kws)
			unique_prompts.append(prompt)
	
	# Will hold parsed results for unique inputs
	unique_results: List[Optional[List[str]]] = [None] * len(unique_prompts)
	
	# 🔄 BATCH PROCESSING WITH RETRY LOGIC
	valid_indices = [i for i, p in enumerate(unique_prompts) if p is not None]
	
	if valid_indices:
		if verbose:
			print(f"🔄 Processing {len(valid_indices)} unique images in batches of {batch_size}...")
		
		# Group valid indices into batches
		batches = []
		for i in range(0, len(valid_indices), batch_size):
			batch_indices = valid_indices[i:i + batch_size]
			batch_img_paths = [unique_inputs[idx] for idx in batch_indices]
			batch_prompts = [unique_prompts[idx] for idx in batch_indices]
			batches.append((batch_indices, batch_img_paths, batch_prompts))
		
		for batch_num, (batch_indices, batch_img_paths, batch_prompts) in enumerate(tqdm(batches, desc="Processing batches")):
			if verbose:
				print(f"📦 Batch {batch_num + 1}/{len(batches)} with {len(batch_img_paths)} items")
			
			success = False
			
			# 🔄 RETRY LOGIC
			for attempt in range(max_retries + 1):
				try:
					if attempt > 0 and verbose:
						print(f"🔄 Retry attempt {attempt + 1}/{max_retries + 1} for batch {batch_num + 1}")
					
					# Process batch
					batch_responses = _batch_process_images(
						model=model,
						processor=processor,
						img_paths=batch_img_paths,
						prompts=batch_prompts,
						device=device,
						max_new_tokens=max_generated_tks,
						verbose_inner=(verbose and attempt > 0)
					)
					
					if batch_responses is None:
						raise RuntimeError("Batch processing returned None (likely OOM)")
					
					if verbose:
						print(f"✅ Batch {batch_num + 1} generation successful")
					
					# Parse each response
					for i, response in enumerate(batch_responses):
						idx = batch_indices[i]
						if response is not None:
							try:
								parsed = get_vlm_response(model_id=model_id, raw_response=response, verbose=False)
								unique_results[idx] = parsed
							except Exception as e:
								if verbose:
									print(f"⚠️ Parsing error for batch index {idx}: {e}")
								unique_results[idx] = None
						else:
							unique_results[idx] = None
					
					success = True
					break  # Break retry loop on success
				
				except Exception as e:
					if verbose:
						print(f"❌ Batch {batch_num + 1} attempt {attempt + 1} failed: {e}")
					
					if attempt < max_retries:
						# Exponential backoff
						sleep_time = 2 ** attempt
						if verbose:
							print(f"⏳ Waiting {sleep_time}s before retry...")
						time.sleep(sleep_time)
						if torch.cuda.is_available():
							torch.cuda.empty_cache()
					else:
						# Final attempt failed
						if verbose:
							print(f"💥 Batch {batch_num + 1} failed after {max_retries + 1} attempts")
						# Mark all items in this batch as failed (will be retried individually)
						for idx in batch_indices:
							unique_results[idx] = None
			
			# Checkpoint saving
			if df is not None and (batch_num + 1) % (checkpoint_interval // batch_size) == 0:
				# Map unique_results back to original order for checkpoint
				temp_results = [unique_results[original_to_unique_idx[i]] for i in range(len(inputs))]
				if 'vlm_keywords' not in df.columns:
					df['vlm_keywords'] = None
				df['vlm_keywords'] = temp_results
				_save_checkpoint(df, output_csv)
			
			# Clean up
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
	
	# 🔄 HYBRID FALLBACK: Retry failed items individually
	failed_indices = [
		i 
		for i, result in enumerate(unique_results) 
		if result is None and unique_inputs[i] is not None
	]
	
	if failed_indices and verbose:
		print(f"🔄 Retrying {len(failed_indices)} failed items individually...")
	
	for idx in tqdm(failed_indices, desc="Individual retries", disable=not verbose):
		img_path = unique_inputs[idx]
		prompt = unique_prompts[idx]
		
		if verbose and idx == failed_indices[0]:
			print(f"🔄 Retrying individual item {idx}: {img_path}")
		
		try:
			# Load single image
			img = _load_single_image(img_path, verbose_inner=False)
			if img is None:
				unique_results[idx] = None
				continue
			
			# Process single image (same as query_local_vlm but inline)
			inputs_single = processor(images=img, text=prompt, padding=True, return_tensors="pt")
			inputs_single = inputs_single.to(device, non_blocking=True)
			
			gen_cfg = getattr(model, "generation_config", None)
			tok = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
			pad_token_id = None
			eos_token_id = None
			
			if gen_cfg:
				pad_token_id = getattr(gen_cfg, "pad_token_id", None)
				eos_token_id = getattr(gen_cfg, "eos_token_id", None)
			
			if tok:
				if tok.pad_token_id is None and tok.eos_token_id is not None:
					tok.pad_token_id = tok.eos_token_id
				pad_token_id = tok.pad_token_id
				if eos_token_id is None:
					eos_token_id = tok.eos_token_id
			
			logits_processors = tfs.LogitsProcessorList([SafeLogitsProcessor()])
			
			with torch.inference_mode():
				output = model.generate(
					**inputs_single,
					max_new_tokens=max_generated_tks,
					use_cache=True,
					do_sample=False,
					temperature=None,
					top_k=None,
					top_p=None,
					pad_token_id=pad_token_id,
					eos_token_id=eos_token_id,
					logits_processor=logits_processors,
				)
			
			response = processor.decode(output[0], skip_special_tokens=True)
			
			# Parse response
			parsed = get_vlm_response(model_id=model_id, raw_response=response, verbose=False)
			unique_results[idx] = parsed
			
			# Clear memory
			del inputs_single, output
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			
			if verbose and parsed and idx == failed_indices[0]:
				print(f"✅ Individual retry successful: {parsed}")
		
		except Exception as e:
			if verbose and idx == failed_indices[0]:
				print(f"💥 Individual retry error for item {idx}: {e}")
			unique_results[idx] = None
	
	# Map unique_results back to original order
	results = [unique_results[original_to_unique_idx[i]] for i in range(len(inputs))]
	
	# Final statistics
	if verbose:
		n_ok = sum(1 for r in results if r is not None)
		n_null = sum(
			1 
			for i, inp in enumerate(inputs) 
			if inp is None or str(inp).strip() in ("", "nan", "None")
		)
		n_failed = len(results) - n_ok - n_null
		success_rate = (n_ok / (len(results) - n_null)) * 100 if (len(results) - n_null) > 0 else 0
		
		print(
			f"📊 Final results: {n_ok}/{len(results)-n_null} successful ({success_rate:.1f}%), "
			f"{n_null} null inputs, {n_failed} failed"
		)
	
	# Clean up model and processor
	del model, processor
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	
	# Save results
	if df is not None:
		try:
			df['vlm_keywords'] = results
			df.to_csv(output_csv, index=False)
			if verbose:
				print(f"[SAVE] Results saved to {output_csv}")
			
			try:
				df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
			except:
				pass
			
			# Clean up checkpoint
			checkpoint_path = output_csv.replace(".csv", "_checkpoint.csv")
			if os.path.exists(checkpoint_path):
				os.remove(checkpoint_path)
				if verbose:
					print(f"[CLEANUP] Removed checkpoint file")
		except Exception as e:
			if verbose:
				print(f"[SAVE ERROR] {e}")
	
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
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	parser.add_argument("--debug", '-d', action='store_true', help="Debug mode")

	args = parser.parse_args()
	args.device = torch.device(args.device)
	print(args)

	if args.debug:
		keywords = get_vlm_based_labels_debug(
			model_id=args.model_id,
			device=args.device,
			csv_file=args.csv_file,
			image_path=args.image_path,
			batch_size=args.batch_size,
			max_kws=args.max_keywords,
			max_generated_tks=args.max_generated_tks,
			verbose=args.verbose,
		)
	else:
		keywords = get_vlm_based_labels_opt(
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
			print(f"{i:06d}. {kw}")

if __name__ == "__main__":
	main()