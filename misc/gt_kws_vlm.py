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

	# tokenizer = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
	# if tokenizer is not None:
	# 	tokenizer.padding_side = 'left'
	# 	if verbose:
	# 		print(f"[INFO] Set tokenizer padding_side to 'left'")

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
			max_generated_tks: int,
			max_kws: int,
			csv_file: str,
			do_dedup: bool = True,
			max_retries: int = 2,
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
		
		output_csv = csv_file.replace(".csv", "_vlm_keywords.csv")

		# Check for existing results
		if os.path.exists(output_csv):
			df = pd.read_csv(
				filepath_or_buffer=output_csv,
				on_bad_lines='skip',
				dtype=dtypes,
				low_memory=False,
			)
			if 'vlm_keywords' in df.columns:
				if verbose: 
					print(f"[EXISTING] Found existing results in {output_csv}")
				return df['vlm_keywords'].tolist()

		# Load data
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
			print(f"[DATA] Loaded {len(image_paths)} image paths from CSV")

		# Store original inputs for later reference
		original_inputs = image_paths
		if len(original_inputs) == 0:
			return None

		# Load model once
		processor, model = _load_vlm_(model_id, device, verbose=verbose)

		if verbose:
			valid_count = sum(1 for x in original_inputs if x is not None and os.path.exists(str(x)))
			null_count = len(original_inputs) - valid_count
			print(f"📊 Input stats: {len(original_inputs)} total, {valid_count} valid, {null_count} null")

		# 🔧 NULL-SAFE DEDUPLICATION
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

		# Generate prompts and load images for unique inputs
		unique_prompts = []
		unique_images = []
		for img_path in unique_inputs:
			if img_path is None:
				unique_prompts.append(None)
				unique_images.append(None)
			else:
				try:
					img = Image.open(img_path).convert("RGB")
					prompt = get_prompt(
						processor=processor,
						img_path=img_path,
						max_kws=max_kws,
					)
					unique_prompts.append(prompt)
					unique_images.append(img)
				except Exception as e:
					if verbose:
						print(f"❌ Failed to load image {img_path}: {e}")
					unique_prompts.append(None)
					unique_images.append(None)

		# Will hold parsed results for unique inputs
		unique_results: List[Optional[List[str]]] = [None] * len(unique_prompts)
		
		# 🔄 SEQUENTIAL PROCESSING WITH BATCH-LIKE OPTIMIZATIONS
		# For VLM models, sequential processing often gives better quality than batch processing
		valid_indices = [
			i
			for i, (p, img) in enumerate(zip(unique_prompts, unique_images)) 
			if p is not None and img is not None
		]
		
		if valid_indices:
			if verbose:
				print(f"🔄 Processing {len(valid_indices)} unique images sequentially with optimizations...")
			
			for idx in tqdm(valid_indices, desc="Processing images"):
				img_path = unique_inputs[idx]
				prompt = unique_prompts[idx]
				img = unique_images[idx]
								
				success = False
				last_error = None
				
				# 🔄 RETRY LOGIC for individual images
				for attempt in range(max_retries + 1):
					try:
						if attempt > 0 and verbose:
							print(f"🔄 Retry attempt {attempt + 1}/{max_retries + 1} for image {idx + 1}")
						# Process single image (not batch)
						single_inputs = processor(
							images=img,
							text=prompt,
							padding=True,
							return_tensors="pt"
						).to(device, non_blocking=True)									
						# Generate response
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
						# Decode and parse response
						response = processor.decode(outputs[0], skip_special_tokens=True)
						if verbose:
							print(f"✅ Image {idx + 1} generation successful")									
						# Parse the response
						try:
							parsed = get_vlm_response(
								model_id=model_id,
								raw_response=response,
								verbose=verbose,
							)
							unique_results[idx] = parsed
							if verbose and parsed:
								print(f"✅ Parsed keywords: {parsed}")
						except Exception as e:
							if verbose:
								print(f"⚠️ Parsing error for image {idx + 1}: {e}")
							unique_results[idx] = None
						success = True
						break  # Break retry loop on success	
					except Exception as e:
						last_error = e
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
		
		# 🔄 HYBRID FALLBACK: Retry failed items with different approach
		failed_indices = [
			i
			for i, result in enumerate(unique_results)
			if result is None and unique_inputs[i] is not None
		]
		
		if failed_indices and verbose:
			print(f"🔄 Retrying {len(failed_indices)} failed items with fallback approach...")
		
		for idx in failed_indices:
			img_path = unique_inputs[idx]
			if verbose:
				print(f"🔄 Retrying failed item {idx}: {os.path.basename(img_path)}")
			try:
				# Use the original query_local_vlm function as fallback
				prompt = get_prompt(
					processor=processor,
					img_path=img_path,
					max_kws=max_kws,
				)
				individual_result = query_local_vlm(
					model=model,
					processor=processor,
					img_path=img_path,
					text=prompt,
					device=device,
					max_generated_tks=max_generated_tks,
					verbose=verbose,
				)
				unique_results[idx] = individual_result
				if verbose and individual_result:
					print(f"✅ Fallback retry successful: {individual_result}")
				elif verbose:
					print(f"❌ Fallback retry failed for item {idx}")
			except Exception as e:
				if verbose:
					print(f"💥 Fallback retry error for item {idx}: {e}")
				unique_results[idx] = None
		
		# Map unique_results back to original order
		results = []
		for orig_i, uniq_idx in enumerate(original_to_unique_idx):
			results.append(unique_results[uniq_idx])
		
		# Final statistics
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
		
		# Clean up model and processor
		del model, processor
		torch.cuda.empty_cache() if torch.cuda.is_available() else None

		if csv_file:
			df['vlm_keywords'] = results
			df.to_csv(output_csv, index=False)
			try:
				df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
			except Exception as e:
				print(f"Failed to write Excel file: {e}")
			if verbose:
				print(f"[SAVE] Saved {len(results)} keywords to {output_csv}")
				print(f"[SAVE] DataFrame: {df.shape}, columns: {list(df.columns)}")

		if verbose:
			print(f"[FINAL] Total time: {time.time() - st_t:.2f} sec")

		return results

def get_vlm_based_labels_opt_x2(
		model_id: str,
		device: str,
		batch_size: int,
		max_generated_tks: int,
		max_kws: int,
		csv_file: str = None,
		image_path: str = None,
		do_dedup: bool = True,
		max_retries: int = 2,
		verbose: bool = False,
	) -> List[Optional[List[str]]]:
	"""
	TRULY OPTIMIZED VLM batch processing with:
	- Real batch processing (processes multiple images simultaneously)
	- Smart handling of variable-length outputs in batches
	- Deduplication to avoid redundant processing
	- Retry logic with exponential backoff
	- Fallback to single-image processing for failed batches
	"""
	if verbose:
		print(f"\n{'='*100}")
		print(f"[INIT] Starting OPTIMIZED batch VLM processing")
		print(f"[INIT] Model: {model_id}")
		print(f"[INIT] Batch size: {batch_size}")
		print(f"[INIT] Device: {device}")
		print(f"{'='*100}\n")

	st_t = time.time()
	if csv_file:
		output_csv = csv_file.replace(".csv", "_vlm_keywords.csv")

	if csv_file and image_path:
		raise ValueError("Only one of csv_file or image_path must be provided")

	# Check for existing results
	if csv_file and os.path.exists(output_csv):
		df = pd.read_csv(
			filepath_or_buffer=output_csv,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
		if 'vlm_keywords' in df.columns:
			if verbose: 
				print(f"[EXISTING] Found existing results in {output_csv}")
			return df['vlm_keywords'].tolist()

	# Load data
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
			print(f"[DATA] Loaded {len(image_paths)} image paths from CSV")
	elif image_path:
		image_paths = [image_path]
		if verbose:
			print(f"[DATA] Single image mode: {image_path}")
	else:
		raise ValueError("Either csv_file or image_path must be provided")

	original_inputs = image_paths
	if len(original_inputs) == 0:
		return None

	# Load model
	if verbose:
		print(f"[MODEL] Loading VLM model and processor...")
	processor, model = _load_vlm_(model_id, device, verbose=verbose)
	
	# Set padding for batch processing
	tokenizer = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
	if tokenizer is not None:
		tokenizer.padding_side = 'left'
		if verbose:
			print(f"[MODEL] Set padding_side='left' for decoder-only model")
	
	# CRITICAL: Qwen2-VL has issues with batch processing due to left-padding
	# Force batch_size=1 for Qwen models to ensure quality
	if "Qwen" in model_id or "qwen" in model_id.lower():
		if batch_size > 1:
			if verbose:
				print(f"[MODEL] ⚠️ Qwen models have known issues with batch processing")
				print(f"[MODEL] Forcing batch_size=1 for quality (original: {batch_size})")
			batch_size = 1

	if verbose:
		valid_count = sum(1 for x in original_inputs if x is not None and os.path.exists(str(x)))
		print(f"[VALIDATION] {valid_count}/{len(original_inputs)} valid paths")

	# 🔧 DEDUPLICATION
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
		
		if verbose:
			dedup_saved = len(original_inputs) - len(unique_inputs)
			print(f"[DEDUP] {len(original_inputs)} → {len(unique_inputs)} unique ({dedup_saved} duplicates)")
	else:
		unique_inputs = [x if x is not None and os.path.exists(str(x)) else None for x in original_inputs]
		original_to_unique_idx = list(range(len(unique_inputs)))

	# Preload all images and prompts
	if verbose:
		print(f"[PRELOAD] Loading images and generating prompts...")
	
	unique_prompts = []
	unique_images = []
	load_failures = 0
	
	for img_path in unique_inputs:
		if img_path is None:
			unique_prompts.append(None)
			unique_images.append(None)
		else:
			try:
				img = Image.open(img_path).convert("RGB")
				prompt = get_prompt(
					processor=processor,
					img_path=img_path,
					max_kws=max_kws,
				)
				unique_prompts.append(prompt)
				unique_images.append(img)
			except Exception as e:
				if verbose:
					print(f"[PRELOAD] Failed to load {img_path}: {e}")
				unique_prompts.append(None)
				unique_images.append(None)
				load_failures += 1
	
	if verbose:
		valid_loaded = sum(1 for p, img in zip(unique_prompts, unique_images) if p is not None and img is not None)
		print(f"[PRELOAD] Loaded {valid_loaded}/{len(unique_inputs)} images ({load_failures} failures)")

	# Results storage
	unique_results: List[Optional[List[str]]] = [None] * len(unique_prompts)
	
	# 🔄 TRUE BATCH PROCESSING
	valid_indices = [
		i for i, (p, img) in enumerate(zip(unique_prompts, unique_images)) 
		if p is not None and img is not None
	]
	
	if valid_indices:
		num_batches = (len(valid_indices) + batch_size - 1) // batch_size
		if verbose:
			print(f"\n[BATCH] Processing {len(valid_indices)} images in {num_batches} batches of {batch_size}")
		
		# Group into batches
		batches = []
		for i in range(0, len(valid_indices), batch_size):
			batch_indices = valid_indices[i:i + batch_size]
			batch_prompts = [unique_prompts[idx] for idx in batch_indices]
			batch_images = [unique_images[idx] for idx in batch_indices]
			batches.append((batch_indices, batch_prompts, batch_images))
		
		successful_batches = 0
		failed_batches = 0
		
		for batch_num, (batch_indices, batch_prompts, batch_images) in enumerate(tqdm(batches, desc="Processing batches", disable=not verbose)):
			if verbose:
				print(f"\n[BATCH {batch_num+1}/{num_batches}] Processing {len(batch_indices)} images: {batch_indices}")
			
			batch_success = False
			last_error = None
			current_batch_size = len(batch_indices)
			
			# Try batch processing with adaptive size reduction on OOM
			for attempt in range(max_retries + 1):
				try:
					if attempt > 0:
						# On OOM retry, try reducing batch size
						if last_error and "out of memory" in str(last_error).lower():
							new_size = max(1, current_batch_size // 2)
							if new_size < current_batch_size and new_size > 0:
								if verbose:
									print(f"[BATCH {batch_num+1}/{num_batches}] OOM detected, reducing batch size {current_batch_size} → {new_size}")
								# Split this batch into smaller sub-batches
								sub_batches = []
								for j in range(0, len(batch_indices), new_size):
									sub_indices = batch_indices[j:j+new_size]
									sub_prompts = [unique_prompts[idx] for idx in sub_indices]
									sub_images = [unique_images[idx] for idx in sub_indices]
									sub_batches.append((sub_indices, sub_prompts, sub_images))
								
								# Process sub-batches
								sub_success_count = 0
								for sub_idx, (sub_indices, sub_prompts, sub_images) in enumerate(sub_batches):
									try:
										if verbose:
											print(f"[BATCH {batch_num+1}/{num_batches}] Sub-batch {sub_idx+1}/{len(sub_batches)}: {len(sub_indices)} images")
										
										sub_inputs = processor(
											images=sub_images,
											text=sub_prompts,
											padding=True,
											return_tensors="pt"
										).to(device, non_blocking=True)
										
										with torch.inference_mode():
											sub_outputs = model.generate(
												**sub_inputs,
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
										
										# Decode sub-batch
										sub_input_ids = sub_inputs.get('input_ids')
										for i in range(len(sub_indices)):
											if sub_input_ids.dim() > 1:
												input_len = sub_input_ids[i].shape[0]
											else:
												input_len = sub_input_ids.shape[0]
											
											if sub_outputs[i].shape[0] > input_len:
												generated_ids = sub_outputs[i][input_len:]
												response = processor.decode(generated_ids, skip_special_tokens=True)
											else:
												response = processor.decode(sub_outputs[i], skip_special_tokens=True)
											
											idx = sub_indices[i]
											try:
												parsed = get_vlm_response(model_id=model_id, raw_response=response, verbose=False)
												unique_results[idx] = parsed
												if parsed is not None:
													sub_success_count += 1
											except Exception as e:
												if verbose:
													print(f"[BATCH {batch_num+1}/{num_batches}] Parse error: {e}")
												unique_results[idx] = None
										
										# Cleanup sub-batch
										del sub_inputs, sub_outputs
										if torch.cuda.is_available():
											torch.cuda.empty_cache()
									
									except Exception as sub_e:
										if verbose:
											print(f"[BATCH {batch_num+1}/{num_batches}] Sub-batch {sub_idx+1} failed: {sub_e}")
										# Mark sub-batch as failed (will retry individually later)
										for idx in sub_indices:
											unique_results[idx] = None
								
								if sub_success_count > 0:
									if verbose:
										print(f"[BATCH {batch_num+1}/{num_batches}] ✓ Sub-batch processing: {sub_success_count}/{len(batch_indices)} successful")
									successful_batches += 1
									batch_success = True
									break
								else:
									# Sub-batches failed, will fall through to regular retry
									if verbose:
										print(f"[BATCH {batch_num+1}/{num_batches}] Sub-batches all failed")
						
						if verbose and not batch_success:
							print(f"[BATCH {batch_num+1}/{num_batches}] Retry {attempt+1}/{max_retries+1}")
					
					if batch_success:
						break  # Already succeeded via sub-batching
					
					# Process batch - REAL BATCH PROCESSING HERE
					batch_inputs = processor(
						images=batch_images,
						text=batch_prompts,
						padding=True,
						return_tensors="pt"
					).to(device, non_blocking=True)
					
					if verbose:
						input_shapes = {k: v.shape for k, v in batch_inputs.items() if isinstance(v, torch.Tensor)}
						print(f"[BATCH {batch_num+1}/{num_batches}] Input shapes: {input_shapes}")
					
					# Generation config
					gen_cfg = getattr(model, "generation_config", None)
					pad_token_id = getattr(gen_cfg, "pad_token_id", None) if gen_cfg else None
					eos_token_id = getattr(gen_cfg, "eos_token_id", None) if gen_cfg else None
					
					if tokenizer:
						if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
							tokenizer.pad_token_id = tokenizer.eos_token_id
						pad_token_id = tokenizer.pad_token_id
						if eos_token_id is None:
							eos_token_id = tokenizer.eos_token_id
					
					logits_processors = tfs.LogitsProcessorList([SafeLogitsProcessor()])
					
					# Generate - THIS IS TRUE BATCH GENERATION
					with torch.inference_mode():
						outputs = model.generate(
							**batch_inputs,
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
					
					if verbose:
						print(f"[BATCH {batch_num+1}/{num_batches}] Generated outputs: {outputs.shape}")
					
					# Decode each item in batch
					input_ids = batch_inputs.get('input_ids')
					batch_responses = []
					
					for i in range(len(batch_indices)):
						# Get input length for this item
						if input_ids.dim() > 1:
							input_len = input_ids[i].shape[0]
						else:
							input_len = input_ids.shape[0]
						
						# Decode ONLY the generated tokens (critical for correct output)
						if outputs[i].shape[0] > input_len:
							generated_ids = outputs[i][input_len:]
							response = processor.decode(generated_ids, skip_special_tokens=True)
						else:
							response = processor.decode(outputs[i], skip_special_tokens=True)
						
						batch_responses.append(response)
						
						if verbose and i == 0:  # Show first item as sample
							print(f"[BATCH {batch_num+1}/{num_batches}] Sample decode: input_len={input_len}, output_len={outputs[i].shape[0]}")
							print(f"[BATCH {batch_num+1}/{num_batches}] Sample response: {response[:150]}...")
					
					# Parse all responses
					parse_success = 0
					for i, response in enumerate(batch_responses):
						idx = batch_indices[i]
						try:
							parsed = get_vlm_response(
								model_id=model_id,
								raw_response=response,
								verbose=False,
							)
							unique_results[idx] = parsed
							if parsed is not None:
								parse_success += 1
						except Exception as e:
							if verbose:
								print(f"[BATCH {batch_num+1}/{num_batches}] Parse error for item {i}: {e}")
							unique_results[idx] = None
					
					if verbose:
						print(f"[BATCH {batch_num+1}/{num_batches}] ✓ Success: {parse_success}/{len(batch_indices)} parsed")
					
					successful_batches += 1
					batch_success = True
					break  # Success, exit retry loop
					
				except Exception as e:
					last_error = e
					if verbose:
						print(f"[BATCH {batch_num+1}/{num_batches}] ✗ Attempt {attempt+1} failed: {type(e).__name__}: {e}")
					
					if attempt < max_retries:
						sleep_time = EXP_BACKOFF ** attempt
						if verbose:
							print(f"[BATCH {batch_num+1}/{num_batches}] Waiting {sleep_time}s...")
						time.sleep(sleep_time)
						if torch.cuda.is_available():
							torch.cuda.empty_cache()
			
			# If batch failed after all retries, mark for individual retry
			if not batch_success:
				failed_batches += 1
				if verbose:
					print(f"[BATCH {batch_num+1}/{num_batches}] 💥 Failed after {max_retries+1} attempts")
				for idx in batch_indices:
					unique_results[idx] = None
			
			# Cleanup
			if 'batch_inputs' in locals():
				del batch_inputs
			if 'outputs' in locals():
				del outputs
			if 'batch_responses' in locals():
				del batch_responses
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		
		if verbose:
			print(f"\n[BATCH] Completed: {successful_batches}/{num_batches} batches successful, {failed_batches} failed")
	
	# 🔄 FALLBACK: Individual retry for failed items
	failed_indices = [
		i for i, result in enumerate(unique_results) 
		if result is None and unique_inputs[i] is not None
	]
	
	if failed_indices:
		if verbose:
			print(f"\n[FALLBACK] Retrying {len(failed_indices)} failed items individually...")
		
		for idx in tqdm(failed_indices, desc="Individual retries", disable=not verbose):
			img_path = unique_inputs[idx]
			try:
				prompt = get_prompt(
					processor=processor,
					img_path=img_path,
					max_kws=max_kws,
				)
				
				result = query_local_vlm(
					model=model,
					processor=processor,
					img_path=img_path,
					text=prompt,
					device=device,
					max_generated_tks=max_generated_tks,
					verbose=False,
				)
				unique_results[idx] = result
				
			except Exception as e:
				if verbose:
					print(f"[FALLBACK] Item {idx} failed: {e}")
				unique_results[idx] = None
		
		if verbose:
			fallback_success = sum(1 for i in failed_indices if unique_results[i] is not None)
			print(f"[FALLBACK] Recovered {fallback_success}/{len(failed_indices)} items")
	
	# Map back to original order
	results = [unique_results[original_to_unique_idx[i]] for i in range(len(original_inputs))]
	
	# Statistics
	if verbose:
		n_ok = sum(1 for r in results if r is not None)
		n_null = sum(1 for i, inp in enumerate(original_inputs) if inp is None or not os.path.exists(str(inp)))
		n_failed = len(results) - n_ok - n_null
		success_rate = (n_ok / (len(results) - n_null)) * 100 if (len(results) - n_null) > 0 else 0
		
		print(f"\n{'='*100}")
		print(f"[FINAL] Results: {n_ok}/{len(results)-n_null} successful ({success_rate:.1f}%)")
		print(f"[FINAL] Failed: {n_failed}, Null: {n_null}")
		print(f"{'='*100}\n")
	
	# Cleanup
	del model, processor
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	# Save
	if csv_file:
		df['vlm_keywords'] = results
		df.to_csv(output_csv, index=False)
		try:
			df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
		except Exception as e:
			if verbose:
				print(f"[SAVE] Excel save failed: {e}")
		if verbose:
			print(f"[SAVE] Saved to {output_csv}")
			print(f"[SAVE] DataFrame: {df.shape}, columns: {list(df.columns)}")

	if verbose:
		print(f"[FINAL] Total time: {time.time() - st_t:.2f} sec")

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

	if args.debug or args.image_path:
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