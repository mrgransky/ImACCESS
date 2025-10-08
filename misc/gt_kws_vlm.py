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
		inputs = processor(images=img, text=text, padding=True, return_tensors="pt")
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
				verbose: bool = False,
				checkpoint_interval: int = 1000,
				max_retries: int = 2,
		) -> List[List[str]]:
		if verbose:
			print(f"Optimized VLM Keyword Extraction".center(160, "-"))
		
		def _save_checkpoint(df, output_csv, idx, temp_suffix="_checkpoint"):
				"""Save intermediate results to avoid losing progress."""
				checkpoint_path = output_csv.replace(".csv", f"{temp_suffix}.csv")
				try:
						df.to_csv(checkpoint_path, index=False)
						if verbose:
								print(f"[CHECKPOINT] Saved at index {idx} to {checkpoint_path}")
						return True
				except Exception as e:
						if verbose:
								print(f"[CHECKPOINT ERROR] Failed to save: {e}")
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
										print(f"[CHECKPOINT ERROR] Failed to load: {e}")
				return None
		
		def _process_single_image(model, processor, img_path, text, device, max_new_tokens, verbose_inner=verbose):
				"""Process a single image - fallback for OOM situations."""
				try:
						# Load image
						if img_path.startswith('http://') or img_path.startswith('https://'):
							r = requests.get(img_path, timeout=10)
							r.raise_for_status()
							img = Image.open(io.BytesIO(r.content))
						else:
							img = Image.open(img_path)
						
						img = img.convert("RGB")
						if img.size[0] == 0 or img.size[1] == 0:
							if verbose_inner:
								print(f"[SINGLE] Invalid image size: {img_path}")
							return None
						
						# Process
						inputs = processor(images=img, text=text, padding=True, return_tensors="pt")
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
							output = model.generate(
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
						
						response = processor.decode(output[0], skip_special_tokens=True)
						
						# Clear memory
						del inputs, output
						if torch.cuda.is_available():
								torch.cuda.empty_cache()
						
						return response
						
				except Exception as e:
						if verbose_inner:
								print(f"[SINGLE ERROR] {img_path}: {type(e).__name__}: {e}")
						if torch.cuda.is_available():
								torch.cuda.empty_cache()
						return None
		
		def _batch_load_images(img_paths, verbose_inner=verbose):
				"""Load multiple images with error handling."""
				images = []
				valid_indices = []
				
				for idx, img_path in enumerate(img_paths):
						try:
								if img_path.startswith('http://') or img_path.startswith('https://'):
										r = requests.get(img_path, timeout=10)
										r.raise_for_status()
										img = Image.open(io.BytesIO(r.content))
								else:
										img = Image.open(img_path)
								
								img = img.convert("RGB")
								if img.size[0] > 0 and img.size[1] > 0:
										images.append(img)
										valid_indices.append(idx)
								elif verbose_inner:
										print(f"[BATCH_LOAD] Skipped invalid size: {img_path}")
						except Exception as e:
								if verbose_inner:
										print(f"[BATCH_LOAD ERROR] {img_path}: {e}")
				
				return images, valid_indices
		
		def _batch_process_vlm(model, processor, images, texts, device, max_new_tokens, verbose_inner=verbose):
				"""Process a batch of images through VLM."""
				if not images:
						return []
				
				try:
						if verbose_inner:
								print(f"[BATCH_PROCESS] Processing {len(images)} images")
						
						# Batch preprocessing
						if len(images) == 1:
								inputs = processor(images=images[0], text=texts[0], padding=True, return_tensors="pt")
						else:
								inputs = processor(images=images, text=texts, padding=True, return_tensors="pt")
						
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
						
						# Clear GPU memory
						del inputs, outputs
						if torch.cuda.is_available():
								torch.cuda.empty_cache()
						
						return responses
						
				except RuntimeError as e:
						if verbose_inner:
								print(f"[BATCH_PROCESS ERROR] RuntimeError: {e}")
						if torch.cuda.is_available():
								torch.cuda.empty_cache()
						return None  # Signal OOM or other runtime error
				except Exception as e:
						if verbose_inner:
								print(f"[BATCH_PROCESS ERROR] Unexpected: {type(e).__name__}: {e}")
						if torch.cuda.is_available():
								torch.cuda.empty_cache()
						return None
		
		# ===== Main Function Logic =====
		if csv_file and image_path:
			raise ValueError("Only one of csv_file or image_path must be provided")
		
		# Setup output paths
		if csv_file:
			output_csv = csv_file.replace(".csv", "_vlm_keywords.csv")
			
			# Try to load checkpoint first
			checkpoint_df = _load_checkpoint(output_csv)
			if checkpoint_df is not None:
				if 'vlm_keywords' in checkpoint_df.columns:
					if verbose:
						processed = checkpoint_df['vlm_keywords'].notna().sum()
						total = len(checkpoint_df)
						print(f"[RESUME] Checkpoint: {processed}/{total} processed")
					df = checkpoint_df
				else:
					df = pd.read_csv(csv_file, on_bad_lines='skip', dtype=dtypes, low_memory=False)
					df['vlm_keywords'] = None
			else:
				# Check if final output already exists
				if os.path.exists(output_csv):
					df = pd.read_csv(output_csv, on_bad_lines='skip', dtype=dtypes, low_memory=False)
					if 'vlm_keywords' in df.columns:
						processed = df['vlm_keywords'].notna().sum()
						if processed == len(df):
							if verbose:
								print(f"[COMPLETE] All {processed} images already processed")
							return df['vlm_keywords'].tolist()
						elif verbose:
							print(f"[RESUME] Partial: {processed}/{len(df)} processed")
				else:
					df = pd.read_csv(csv_file, on_bad_lines='skip', dtype=dtypes, low_memory=False)
					df['vlm_keywords'] = None
			
			if 'img_path' not in df.columns:
				raise ValueError("CSV file must have 'img_path' column")
			
			image_paths = df['img_path'].tolist()
			
			if 'vlm_keywords' not in df.columns:
				df['vlm_keywords'] = None
			
			if verbose:
				total = len(image_paths)
				already_processed = df['vlm_keywords'].notna().sum()
				remaining = total - already_processed
				print(f"[SETUP] Total: {total} | Processed: {already_processed} | Remaining: {remaining}")
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
		
		# Initialize results storage
		all_keywords = [None] * len(image_paths)
		
		# Determine which indices to process
		if df is not None:
			indices_to_process = [i for i, kw in enumerate(df['vlm_keywords']) if pd.isna(kw)]
			if verbose:
				print(f"[PROCESS] {len(indices_to_process)} images to process")
		else:
			indices_to_process = list(range(len(image_paths)))
		
		# Adaptive batch processing
		current_batch_size = min(batch_size, len(indices_to_process))
		successful_count = 0
		failed_count = 0
		oom_count = 0
		
		# Use tqdm for progress tracking
		pbar = tqdm(total=len(indices_to_process), desc="Processing images")
		
		idx = 0
		while idx < len(indices_to_process):
			# Determine current batch
			batch_end = min(idx + current_batch_size, len(indices_to_process))
			batch_indices = indices_to_process[idx:batch_end]
			batch_img_paths = [image_paths[i] for i in batch_indices]
			
			if verbose and idx % 100 == 0:
				print(f"\n[BATCH] Processing {idx}-{batch_end} (batch_size={current_batch_size})")
			
			# Try batch processing first (if batch_size > 1)
			batch_success = False
			if current_batch_size > 1:
				# Load images
				batch_images, valid_batch_indices = _batch_load_images(batch_img_paths, verbose_inner=verbose)
				
				if batch_images:
					# Create prompts
					batch_texts = [
						get_prompt(model_id=model_id, processor=processor, img_path=batch_img_paths[i], max_kws=max_kws)
						for i in valid_batch_indices
					]
					
					# Try batch processing
					batch_responses = _batch_process_vlm(
						model=model,
						processor=processor,
						images=batch_images,
						texts=batch_texts,
						device=device,
						max_new_tokens=max_generated_tks,
						verbose_inner=verbose
					)
					
					if batch_responses:
						# Success! Parse responses
						for local_idx, response in tqdm(enumerate(batch_responses), total=len(batch_responses), desc="Parsing batch responses"):
							global_idx = batch_indices[valid_batch_indices[local_idx]]
							
							try:
								parsed_keywords = get_vlm_response(model_id=model_id, raw_response=response, verbose=verbose)
								
								if df is not None:
									df.at[global_idx, 'vlm_keywords'] = str(parsed_keywords) if parsed_keywords else None
								else:
									all_keywords[global_idx] = parsed_keywords
								
								successful_count += 1
							except Exception as e:
								if verbose:
									print(f"[PARSE ERROR] Index {global_idx}: {e}")
								failed_count += 1
						batch_success = True
						pbar.update(len(batch_indices))
						idx = batch_end
					else:
						# Batch failed (likely OOM)
						if verbose:
							print(f"[OOM] Batch size {current_batch_size} failed, reducing to {max(1, current_batch_size // 2)}")
						oom_count += 1
						current_batch_size = max(1, current_batch_size // 2)
						# Don't increment idx, retry with smaller batch
			
			# Fallback: process one-by-one if batch failed or batch_size is 1
			if not batch_success:
				for batch_idx, global_idx in tqdm(enumerate(batch_indices), total=len(batch_indices), desc="Processing batch images"):
					img_path = image_paths[global_idx]
					
					# Create prompt
					text = get_prompt(model_id=model_id, processor=processor, img_path=img_path, max_kws=max_kws)
					
					# Process single image
					response = _process_single_image(
						model=model,
						processor=processor,
						img_path=img_path,
						text=text,
						device=device,
						max_new_tokens=max_generated_tks,
						verbose_inner=verbose,
					)
					
					if response:
						try:
							parsed_keywords = get_vlm_response(model_id=model_id, raw_response=response, verbose=verbose)
							
							if df is not None:
								df.at[global_idx, 'vlm_keywords'] = str(parsed_keywords) if parsed_keywords else None
							else:
								all_keywords[global_idx] = parsed_keywords
							
							successful_count += 1
						except Exception as e:
							if verbose:
								print(f"[PARSE ERROR] Index {global_idx}: {e}")
							failed_count += 1
					else:
						failed_count += 1
					
					pbar.update(1)
				
				idx = batch_end
				
				# Try to increase batch size after successful single-image processing
				if current_batch_size == 1 and oom_count == 0:
					current_batch_size = min(batch_size, 2)
			
			# Checkpoint saving
			if df is not None and (idx % checkpoint_interval == 0 or idx >= len(indices_to_process)):
				_save_checkpoint(df, output_csv, idx)
			
			# Periodic memory cleanup
			if idx % 50 == 0 and torch.cuda.is_available():
				torch.cuda.empty_cache()
		
		pbar.close()
		
		if df is not None:
			if verbose:
				processed_count = df['vlm_keywords'].notna().sum()
				print(f"\n[COMPLETE] Processed: {processed_count}/{len(df)}")
				print(f"[COMPLETE] Successful: {successful_count} | Failed: {failed_count} | OOM events: {oom_count}")
			# Final save
			try:
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
			
			return df['vlm_keywords'].tolist()
		else:
			return all_keywords

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
