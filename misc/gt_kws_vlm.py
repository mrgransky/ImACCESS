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

# Qwen 2.5x VL collection:
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct" # only fits Puhti and Mahti

# Qwen 3 VL collection:
# Qwen/Qwen3-VL-2B-Instruct
# Qwen/Qwen3-VL-4B-Instruct
# Qwen/Qwen3-VL-8B-Instruct # only fits Puhti and Mahti
# Qwen/Qwen3-VL-32B-Instruct # multiple gpus required
# Qwen/Qwen3-VL-30B-A3B-Instruct # multiple gpus required

# does not fit into VRAM:
# model_id = "llava-hf/llava-v1.6-34b-hf"
# model_id = "llava-hf/llava-next-72b-hf"
# model_id = "llava-hf/llava-next-110b-hf"
# model_id = "Qwen/Qwen2.5-VL-72B-Instruct"

# debugging required:
# # model_id = "tiiuae/falcon-11B-vlm"
# # model_id = "utter-project/EuroVLM-1.7B-Preview"
# # model_id = "OpenGVLab/InternVL-Chat-V1-2"

EXP_BACKOFF = 2  # seconds
IMG_MAX_RES = 512
VLM_INSTRUCTION_TEMPLATE = """You function as a historical archivist whose expertise lies in the 20th century.
Identify no more than {k} highly prominent, factual, and distinct **KEYWORDS** that capture the visually observable actions, objects, or occurrences in the image - **completely ignoring all text**.

**CRITICAL RULES**:
- Return **ONLY** a clean, valid, and parsable **Python LIST** with a maximum of {k} keywords - fewer is acceptable only if the image is too simple.
- **PRIORITIZE MEANINGFUL PHRASES**: Opt for multi-word n-grams such as NOUN PHRASES and NAMED ENTITIES over single terms only if they convey a more distinct meaning.
- **ZERO HALLUCINATION POLICY**: Do not invent or infer specifics that lack clear verification from the visual content. When in doubt, omit rather than fabricate.
- **ABSOLUTELY NO** additional explanatory text, code blocks, comments, tags, thoughts, questions, or explanations before or after the **Python LIST**.
- **ABSOLUTELY NO** TEXT EXTRACTION / OCR: Do NOT read, transcribe, quote, paraphrase, or use any visible text from the image (including captions, labels, dates, signage, stamped annotations, handwritten notes, or serial numbers).
- **STRICTLY EXCLUDE** image quality, type, format, or style as keywords.
- **STRICTLY EXCLUDE ALL TEMPORAL EXPRESSIONS**: any time-related phrases such as dates, seasons, decades, centuries is STRICTLY FORBIDDEN.
- **STRICTLY EXCLUDE** vague, generic, or historical keywords.
- Exclude meaningless abbreviations, numerical words, special characters, or stopwords.
- **ABSOLUTELY NO** synonymous, duplicate, identical or misspelled keywords.
- The parsable **Python LIST** must be the **VERY LAST THING** in your response."""

def _load_vlm_(
		model_id: str,
		use_quantization: bool = False,
		quantization_bits: int = 8,
		verbose: bool = False,
) -> Tuple[tfs.PreTrainedTokenizerBase, torch.nn.Module]:
	"""
	Load a Vision-Language Model (VLM) with optimal settings.
	
	Args:
		model_id: HuggingFace model identifier
		use_quantization: Whether to use quantization
		quantization_bits: Quantization bits (4 or 8)
		verbose: Enable verbose logging
	
	Returns:
		Tuple of (processor, model)
	"""

	# ========== Version and CUDA info ==========
	n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
	if verbose:
		print(f"[VERSIONS] torch : {torch.__version__} transformers: {tfs.__version__}")
		print(f"[INFO] CUDA available?        : {torch.cuda.is_available()} {n_gpus} GPU(s) available: {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")

		if torch.cuda.is_available():
			cur = torch.cuda.current_device()
			major, minor = torch.cuda.get_device_capability(cur)
			print(f"[INFO] Compute capability     : {major}.{minor}")
			print(f"[INFO] BF16 support?          : {torch.cuda.is_bf16_supported()}")
			print(f"[INFO] CUDA memory allocated  : {torch.cuda.memory_allocated(cur)//(1024**2)} MiB")
			print(f"[INFO] CUDA memory reserved   : {torch.cuda.memory_reserved(cur)//(1024**2)} MiB")
		else:
			print("[INFO] Running on CPU only")

	# ========== HuggingFace login ==========
	if verbose:
		print(f"{USER} HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub")
	try:
		huggingface_hub.login(token=hf_tk)
	except Exception as e:
		print(f"<!> Failed to login to HuggingFace Hub: {e}")
		raise e

	# ========== Load config ==========
	config = tfs.AutoConfig.from_pretrained(model_id, trust_remote_code=True)

	if verbose:
		print("[INFO] Config summary")
		print(f"   • model_type        : {config.model_type}")
		print(f"   • architectures     : {config.architectures}")
		print(f"   • dtype (if set)    : {config.dtype}")
		print()

	# ========== Determine model class ==========
	model_cls = None

	if config.architectures:
		cls_name = config.architectures[0]
		if hasattr(tfs, cls_name):
			model_cls = getattr(tfs, cls_name)
	
	if model_cls is None:
		raise ValueError(f"Unable to locate model class for architecture(s): {config.architectures}")

	# ========== Optimal dtype selection ==========
	def _optimal_dtype(m_id: str) -> torch.dtype:
		"""
		Select optimal dtype for the given model.
		
		Qwen3-VL MoE models have dtype issues with bfloat16 in the router,
		so we force float16 for them.
		"""
		
		bf16_ok = (
			torch.cuda.is_available()
			and torch.cuda.is_bf16_supported()
		)
		
		m_id_lower = m_id.lower()
		
		# Qwen3-VL MoE: force float16 to avoid scatter() dtype mismatch
		if "qwen3-vl" in m_id_lower or "qwen3_vl" in m_id_lower:
			return torch.float16
		
		# Other Qwen models: bf16 if available
		if "qwen" in m_id_lower:
			return torch.bfloat16 if bf16_ok else torch.float16
		
		# LLaVA: float16
		if "llava" in m_id_lower:
			return torch.float16
		
		# Falcon: bf16 if available
		if "falcon" in m_id_lower:
			return torch.bfloat16 if bf16_ok else torch.float16
		
		# Default: bf16 if available, else fp16
		return torch.bfloat16 if bf16_ok else torch.float16

	dtype = _optimal_dtype(model_id)

	if verbose:
		print("[INFO] Dtype selection")
		print(f"   • BF16 supported on this device? : {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
		print(f"   • Chosen torch dtype             : {dtype}")

	# ========== Optimal attention implementation ==========
	def _optimal_attn_impl(m_id: str) -> str:
		"""
		Select optimal attention implementation.
		
		Flash Attention 2 requires:
		- CUDA available
		- flash_attn package installed
		- Compute capability >= 8.0 (Ampere or newer)
		"""
		if not torch.cuda.is_available():
			return "eager"
		
		try:
			import flash_attn
			major, _ = torch.cuda.get_device_capability()
			flash_ok = major >= 8
		except Exception as e:
			if verbose:
				print(f"[INFO] Flash Attention not available: {e}")
			flash_ok = False
		
		if flash_ok:
			m_id_lower = m_id.lower()
			if "qwen" in m_id_lower or "llava" in m_id_lower:
				return "flash_attention_2"
			return "flash_attention_2"

		return "eager"
	
	attn_impl = _optimal_attn_impl(model_id)

	if verbose:
		print("[INFO] Attention implementation")
		print(f"   • Selected implementation : {attn_impl}\n")

	# ========== Quantization config ==========
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

	# ========== Processor loading ==========
	processor = tfs.AutoProcessor.from_pretrained(
		model_id,
		use_fast=True,
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)

	if verbose:
		print(f"\n[INFO] {processor.__class__.__name__} {type(processor)}")
		print(f"{processor.__call__.__code__.co_varnames}")
		print(f"\t• apply_chat_template: {hasattr(processor, 'apply_chat_template')}")
		print(f"\t• tokenizer          : {hasattr(processor, 'tokenizer')}")
		print(f"\t• text_tokenizer     : {hasattr(processor, 'text_tokenizer')}")
	
	# ========== Extract tokenizer from processor ==========
	if hasattr(processor, "tokenizer"):
		tokenizer = processor.tokenizer
	elif hasattr(processor, "text_tokenizer"):
		tokenizer = processor.text_tokenizer
	else:
		raise ValueError("Unable to locate tokenizer in processor")

	if hasattr(tokenizer, "padding_side") and tokenizer.padding_side is not None:
		tokenizer.padding_side = "left"

	if verbose:
		print(f"   • tokenizer type          : {type(tokenizer)}")
		print(f"   • tokenizer vocab size    : {tokenizer.vocab_size}")
		print(f"   • tokenizer pad token     : {tokenizer.pad_token}")
		print(f"   • tokenizer eos token     : {tokenizer.eos_token}")
		print(f"   • tokenizer padding side  : {tokenizer.padding_side}")
	
	
	# ========== Model loading kwargs ==========
	model_kwargs: Dict[str, Any] = {
		"low_cpu_mem_usage": True,
		"trust_remote_code": True,
		"cache_dir": cache_directory[USER],
		"attn_implementation": attn_impl,
		"dtype": dtype,
	}
	model_kwargs["device_map"] = "auto"
	if use_quantization:
		model_kwargs["quantization_config"] = quantization_config

	if verbose:
		print(f"[INFO] {model_cls.__name__} loading kwargs")
		print(f"   • {n_gpus} GPU(s)")
		for k, v in model_kwargs.items():
			if k == "quantization_config":
				print(f"   • {k}: {type(v).__name__}")
			else:
				print(f"   • {k}: {v}")
		print()

	if verbose and torch.cuda.is_available():
		cur = torch.cuda.current_device()
		print(f"[DEBUG] cuda:{cur} memory BEFORE model load")
		print(f"   • allocated : {torch.cuda.memory_allocated(cur)//(1024**2)} MiB")
		print(f"   • reserved  : {torch.cuda.memory_reserved(cur)//(1024**2)} MiB\n")

	# ========== Load model ==========
	if verbose:
		print(f"[INFO] Calling pretrained {model_cls.__name__} {model_id}")

	try:
		model = model_cls.from_pretrained(model_id, **model_kwargs)
	except Exception as e:
		if verbose:
			print(f"[ERROR] Error loading model: {e}")
		raise e
	
	model.eval()

	# ========== Model info ==========
	if verbose:
		print(f"\n[INFO] {model.__class__.__name__} {type(model)}")
		first_param = next(model.parameters())
		print(f"   • First parameter dtype: {first_param.dtype}")
		total_params = sum(p.numel() for p in model.parameters())
		approx_fp16_gb = total_params * 2 / (1024 ** 3)
		print(f"   • Total parameters    : {total_params:,}")
		print(f"   • Approx. fp16 RAM    : {approx_fp16_gb:.2f} GiB (if stored as fp16)")

		if hasattr(model, "hf_device_map"):
			dm = model.hf_device_map
			print(f"[INFO] device_map='auto' (model.hf_device_map):\n{json.dumps(dm, indent=2, ensure_ascii=False)}")
		# else:
		# 	print(f"[INFO] No `hf_device_map` attribute => model resides on a single device: {device}")

		if hasattr(model, 'generation_config'):
			print(f"{model.generation_config}")

	if verbose:
		print(f"[INFO] Model placement handled by device_map='auto'")
		if torch.cuda.is_available():
			gpu_params = sum(1 for p in model.parameters() if p.device.type == "cuda")
			cpu_params = sum(1 for p in model.parameters() if p.device.type == "cpu")
			print(f"   • Parameters on GPU  : {gpu_params}")
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

def parse_vlm_response(model_id: str, raw_response: str, verbose: bool=False):
	if verbose: 
		print(f"[DEBUG] Parsing VLM response for {model_id}")
		print(f"[RESPONSE]\n{raw_response}\n")
	
	if "Qwen" in model_id:
		return _qwen_vlm_(raw_response, verbose=verbose)
	elif "llava" in model_id:
		return _llava_vlm_(raw_response, verbose=verbose)
	else:
		raise NotImplementedError(f"VLM response parsing not implemented for {model_id}")

def _qwen_vlm_(response: str, verbose: bool = False) -> Optional[List[str]]:
	if not isinstance(response, str):
		if verbose:
			print("[ERROR] VLM output is not a string.")
		return None
		
	# Step 1: Find all balanced bracket expressions [...] in the response
	list_pattern = r"\[[^\[\]]+\]"
	matches = re.findall(list_pattern, response, re.DOTALL)
	
	if verbose:
		print(f"[DEBUG] Found {len(matches)} list-like pattern(s)")
		for i, m in enumerate(matches):
			print(f"[DEBUG]   Match {i}: {m}")
		print()
	
	if not matches:
		if verbose:
			print("[ERROR] No list pattern found in response")
		return None
	
	# Step 2: Pick the longest match (most likely to be the main list)
	primary = max(matches, key=len)
	
	if verbose:
		print(f"[DEBUG] Selected primary list ({len(primary)} chars):")
		print(f"[DEBUG]   {primary}")
		print()
	
	# Step 3: Try to parse it as a Python literal
	try:
		parsed = ast.literal_eval(primary)
		if verbose:
			print(f"[DEBUG] ✓ ast.literal_eval succeeded")
			print(f"[DEBUG]   Type: {type(parsed)}")
			print(f"[DEBUG]   Content: {parsed}")
			print()
	except Exception as e:
		if verbose:
			print(f"[ERROR] ast.literal_eval failed: {type(e).__name__}: {e}")
		return None
	
	# Step 4: Validate it's a list
	if not isinstance(parsed, list):
		if verbose:
			print(f"[ERROR] Parsed result is not a list (got {type(parsed)})")
		return None
	
	# Step 5: Convert all items to strings and clean
	keywords = []
	for i, item in enumerate(parsed):
		# Convert to string
		s = str(item).strip()
		
		if verbose:
			print(f"[DEBUG] Item {i}: {repr(item)} -> {repr(s)}")
		
		# Skip empty strings
		if not s:
			if verbose:
				print(f"[DEBUG]   → Skipped (empty)")
			continue
		
		keywords.append(s)
		
		if verbose:
			print(f"[DEBUG]   → Added")
	
	if verbose:
		print()
		print(f"[DEBUG] Extracted {len(keywords)} keyword(s): {keywords}")
		print()
	
	# Step 6: Remove duplicates while preserving order
	seen = set()
	unique_keywords = []
	
	for i, kw in enumerate(keywords):
		if kw not in seen:
			unique_keywords.append(kw)
			seen.add(kw)
			if verbose:
				print(f"[DEBUG] Dedup {i}: '{kw}' → kept (unique)")
		else:
			if verbose:
				print(f"[DEBUG] Dedup {i}: '{kw}' → skipped (duplicate)")
	
	if verbose:
		print()
		print(f"[FINAL] Returning {len(unique_keywords)} unique keyword(s): {unique_keywords}")
		print()
	
	return unique_keywords if unique_keywords else None

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

def get_vlm_based_labels_single(
		model_id: str,
		device: str,
		image_path: str,
		max_generated_tks: int,
		max_kws: int,
		use_quantization: bool = False,
		verbose: bool = False,
):

	# ========== Load image ==========
	if verbose:
		print(f"[LOAD] Loading image: {image_path}")

	try:
		img = Image.open(image_path)
	except Exception as e:
		if verbose: print(f"{e}\n=> retry via URL")
		try:
			r = requests.get(image_path, timeout=10)
			r.raise_for_status()
			img = Image.open(io.BytesIO(r.content))
		except Exception as e2:
			if verbose: print(f"[ERROR] URL fetch failed => {e2}")
			return None

	img = img.convert("RGB")

	if verbose:
		print(f"\n[IMAGE] {image_path}")
		print(f"[IMAGE] {type(img)} {img.size} {img.mode}")

		arr = np.array(img)
		print(
			f"[IMAGE] {type(arr)} {arr.shape} {arr.dtype} (Min, Max): ({arr.min()}, {arr.max()}) "
			f"NaN: {np.isnan(arr).any()} Inf: {np.isinf(arr).any()} Size: {arr.nbytes/(1024**2):.1f} MB\n")

	if img.size[0] == 0 or img.size[1] == 0:
		if verbose: print("[ERROR] Invalid image size")
		return None

	# load model and processor
	processor, model = _load_vlm_(
		model_id=model_id, 
		use_quantization=use_quantization,
		verbose=verbose
	)

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
	).to(model.device)

	if verbose:
		print(f"[INPUT] Pixel: {single_inputs.pixel_values.shape} {single_inputs.pixel_values.dtype} {single_inputs.pixel_values.device}")

	if single_inputs.pixel_values.numel() == 0:
		raise ValueError(f"Pixel values of {image_path} are empty: {single_inputs.pixel_values.shape}")

	gen_kwargs = dict(max_new_tokens=max_generated_tks, use_cache=True,)
	# Use model’s built-in defaults unless the user overrides
	if hasattr(model, "generation_config"):
		gen_config = model.generation_config
		gen_kwargs["temperature"] = getattr(gen_config, "temperature", 1e-6)
		gen_kwargs["do_sample"] = getattr(gen_config, "do_sample", True)
	else:
		gen_kwargs.update(dict(temperature=1e-6, do_sample=True))

	if verbose:
		print(f"[GEN CONFIG] Using generation parameters:")
		print(json.dumps(gen_kwargs, indent=2, ensure_ascii=False))

	# ========== Generate response ==========
	with torch.no_grad():
		with torch.amp.autocast(
			device_type=device.type, 
			enabled=torch.cuda.is_available(), 
			dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
		):
			outputs = model.generate(**single_inputs, **gen_kwargs)	

	# Decode response
	response = processor.decode(outputs[0], skip_special_tokens=True)

	# ========== Parse the response ==========
	try:
		parsed = parse_vlm_response(
			model_id=model_id,
			raw_response=response,
			verbose=verbose,
		)
		if verbose and parsed: print(f"✅ Parsed keywords: {parsed}")
	except Exception as e:
		if verbose: print(f"⚠️ Parsing error for image {image_path} {e}")
		return None

	return [parsed]

def get_vlm_based_labels_debug(
		model_id: str,
		device: str,
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
		print(f"[INIT] VLM-based keyword generation [DEBUG MODE]")
		print(f"[INIT] Model: {model_id}")
		print(f"[INIT] Device: {device}")
		print(f"{'='*100}\n")
	st_t = time.time()
	
	# ========== Check existing results ==========
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
				print(f"[EXISTING] Found existing results! {type(df)} {df.shape} {list(df.columns)}")
			return df['vlm_keywords'].tolist()

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

	generation_time = 0
	parsing_time = 0

	if not valid_indices:
		if verbose: print(f" <!> No valid indices found after deduplication => exiting")
		return None
		
	for idx in tqdm(valid_indices, desc="Processing images"):
		img_path = unique_images[idx]
		for attempt in range(max_retries + 1):
			try:
				if attempt > 0 and verbose:
					print(f"🔄 Retry attempt {attempt + 1}/{max_retries + 1} for image {idx + 1}")

				# ========== Prepare inputs ==========
				with Image.open(img_path).convert("RGB") as img_obj:
					img = img_obj.copy()
					img.thumbnail((IMG_MAX_RES, IMG_MAX_RES), Image.LANCZOS)
				
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
				).to(model.device)

				if verbose:
					print(f"[INPUT] Pixel: {single_inputs.pixel_values.shape} {single_inputs.pixel_values.dtype} {single_inputs.pixel_values.device}")

				if single_inputs.pixel_values.numel() == 0:
					raise ValueError(f"Pixel values of {img_path} are empty: {single_inputs.pixel_values.shape}")
				
				gen_kwargs = dict(max_new_tokens=max_generated_tks, use_cache=True,)
				# Use model’s built-in defaults unless the user overrides
				if hasattr(model, "generation_config"):
					gen_config = model.generation_config
					gen_kwargs["temperature"] = getattr(gen_config, "temperature", 1e-6)
					gen_kwargs["do_sample"] = getattr(gen_config, "do_sample", True)
				else:
					gen_kwargs.update(dict(temperature=1e-6, do_sample=True))
				if verbose:
					print(f"\n[GEN CONFIG] Using generation parameters:")
					for k, v in gen_kwargs.items():
						print(f"   • {k}: {v}")

				# ========== Generate response ==========
				with torch.no_grad():
					with torch.amp.autocast(
						device_type=device.type, 
						enabled=torch.cuda.is_available(), 
						dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
					):
						outputs = model.generate(**single_inputs, **gen_kwargs)
				
				# Decode response
				response = processor.decode(outputs[0], skip_special_tokens=True)
				
				# ========== Parse the response ==========
				parse_start = time.time()
				try:
					parsed = parse_vlm_response(
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
					print(f"❌ Image {idx + 1} {img_path} attempt {attempt + 1} failed:\n{e}\n")
				
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
		
		# Memory management - clear cache
		if idx % 250 == 0 and torch.cuda.is_available():
			torch.cuda.synchronize()
			torch.cuda.empty_cache()
			gc.collect()
			if verbose:
				print(f"\t>>> Memory cleared after image[{idx}] {img_path}")
	
	if verbose:
		print(f"[PROCESS] Sequential processing: {time.time() - process_start:.2f}s")
		print(f"  ├─ Generation time: {generation_time:.2f}s ({generation_time/len(valid_indices):.3f}s/img)")
		print(f"  └─ Parsing time: {parsing_time:.2f}s ({parsing_time/len(valid_indices):.3f}s/img)")
	
	# ========== Map results back ==========
	map_start = time.time()
	results = [
		[el.lower() for el in unique_results[uniq_idx]] if unique_results[uniq_idx] else None
		for uniq_idx in original_to_unique_idx
	]
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

		print(f"📊 Final results statistics")		
		print(f"  ├─ {n_ok}/{len(results)-n_null} successful ({success_rate:.1f}%)")
		print(f"  └─ {n_null} null inputs, {n_failed} failed")
		print(f"[STATS] Statistics calculation: {time.time() - stats_start:.2f}s")
	
	# ========== Cleanup ==========
	cleanup_start = time.time()
	del model, processor
	torch.cuda.empty_cache() if torch.cuda.is_available() else None
	if verbose:
		print(f"[CLEANUP] Model cleanup & cache cleanup: {time.time() - cleanup_start:.2f}s")

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
			print(f"  ├─ {type(df)} {df.shape}")
			print(f"  └─ {list(df.columns)}")

	if verbose:
		print(f"[FINAL] Total VLM time: {time.time() - st_t:.2f} sec")
		print(f"{'='*100}")

	return results

def get_vlm_based_labels_opt(
				model_id: str,
				device: str,
				batch_size: int,
				num_workers: int,
				max_generated_tks: int,
				max_kws: int,
				csv_file: str,
				do_dedup: bool = True,
				use_quantization: bool = False,
				verbose: bool = False,
):
		"""
		Optimised VLM-based keyword extraction:

			- Verifies/filters image paths
			- Deduplicates per-image if requested
			- Batches images and generates responses with a VLM
			- NEW: Parses VLM responses in parallel per batch via ThreadPoolExecutor
			- Falls back to sequential per-image processing on batch failure
			- Maps results back to original order and writes *_vlm_keywords.csv / .xlsx
		"""

		process = psutil.Process()
		t0 = time.time()

		# ========== Check existing results ==========
		output_csv = csv_file.replace(".csv", "_vlm_keywords.csv")
		if os.path.exists(output_csv):
				if verbose:
						print(f"[EXISTING] Found existing results at {output_csv}")
				df = pd.read_csv(
						filepath_or_buffer=output_csv,
						on_bad_lines='skip',
						dtype=dtypes,
						low_memory=False,
				)
				if 'vlm_keywords' in df.columns:
						if verbose:
								print(f"[EXISTING] Found existing results! {type(df)} {df.shape} {list(df.columns)}")
						return df['vlm_keywords'].tolist()

		# ========== Load data ==========
		load_start = time.time()
		df = pd.read_csv(csv_file, on_bad_lines="skip", low_memory=False)
		if "img_path" not in df.columns:
				raise ValueError("CSV file must have 'img_path' column")

		image_paths = [p if isinstance(p, str) and os.path.exists(p) else None for p in df["img_path"]]
		n_total = len(image_paths)
		if verbose:
				print(f"[DATA] Loaded {len(image_paths)} image paths from CSV ({time.time() - load_start:.2f}s)")

		# ========== Load model ==========
		processor, model = _load_vlm_(
			model_id=model_id,
			use_quantization=use_quantization,
			verbose=verbose,
		)
		if verbose and torch.cuda.is_available():
				print(f"[MODEL] Loaded {model_id} | {device} | {torch.cuda.get_device_name(device)}")

		# ========== Prepare generation kwargs ==========
		gen_kwargs = dict(max_new_tokens=max_generated_tks, use_cache=True)
		if hasattr(model, "generation_config"):
				gen_config = model.generation_config
				gen_kwargs["temperature"] = getattr(gen_config, "temperature", 1e-6)
				gen_kwargs["do_sample"] = getattr(gen_config, "do_sample", True)
		else:
				gen_kwargs.update(dict(temperature=1e-6, do_sample=True))
		if verbose:
				print(f"\n[GEN CONFIG] Using generation parameters:")
				for k, v in gen_kwargs.items():
						print(f"   • {k}: {v}")

		# ========== Prepare inputs (dedup + verification) ==========
		if do_dedup:
				uniq_map: Dict[str, int] = {}
				uniq_inputs: List[Optional[str]] = []
				orig_to_uniq: List[int] = []

				for path in image_paths:
						key = str(path) if path else "__NULL__"
						if key not in uniq_map:
								uniq_map[key] = len(uniq_inputs)
								uniq_inputs.append(path if path else None)
						orig_to_uniq.append(uniq_map[key])
		else:
				uniq_inputs = image_paths
				orig_to_uniq = list(range(n_total))

		def verify(p: Optional[str]) -> Optional[str]:
			if p is None or not os.path.exists(p):
				return None
			try:
				with Image.open(p) as im:
					im.verify()
				return p
			except Exception:
				return None

		with ThreadPoolExecutor(max_workers=num_workers) as ex:
			verified = list(tqdm(ex.map(verify, uniq_inputs), total=len(uniq_inputs), desc="Verifying images", ncols=100,))

		valid_indices = [i for i, v in enumerate(verified) if v is not None]
		total_batches = math.ceil(len(valid_indices) / batch_size)
		base_prompt = VLM_INSTRUCTION_TEMPLATE.format(k=max_kws)
		results: List[Optional[List[str]]] = [None] * len(uniq_inputs)

		if verbose:
			print(f"[INFO] {len(valid_indices)} valid unique images → {total_batches} batches of {batch_size}")
			print(f"[INFO] Memory[in-use]: {process.memory_info().rss / (1024**3):.2f} GB")

		# ========== Helper: parallel parsing for one batch ==========
		def _parse_batch_parallel(
				decoded_responses: List[str],
				batch_indices: List[int],
				model_id_: str,
				verbose_: bool,
		) -> Dict[int, Optional[List[str]]]:
				"""
				Parse a list of decoded VLM responses in parallel.
				Returns a mapping: {unique_index: parsed_keywords_or_None}
				"""
				out: Dict[int, Optional[List[str]]] = {}

				def _parse_one(local_idx: int) -> Tuple[int, Optional[List[str]]]:
					uniq_index = batch_indices[local_idx]
					resp = decoded_responses[local_idx]
					try:
						parsed = parse_vlm_response(
							model_id=model_id_,
							raw_response=resp,
							verbose=verbose_,  # keep parallel logs quiet
						)
						return uniq_index, parsed
					except Exception as e:
						if verbose_:
							print(f"⚠️ Parsing error for unique index {uniq_index}: {e}")
						return uniq_index, None

				if not decoded_responses:
					return out

				with ThreadPoolExecutor(max_workers=num_workers) as ex:
					futures = {ex.submit(_parse_one, i): i for i in range(len(decoded_responses))}
					for fut in as_completed(futures):
						uniq_index, parsed = fut.result()
						out[uniq_index] = parsed

				return out

		# ========== Process batches ==========
		for b in tqdm(range(total_batches), desc="Processing (visual) batches", ncols=100):
				batch_indices = valid_indices[b * batch_size:(b + 1) * batch_size]

				def load_img(p: str) -> Optional[Image.Image]:
					try:
						with Image.open(p).convert("RGB") as im:
							im.thumbnail((IMG_MAX_RES, IMG_MAX_RES))
							return im.copy()
					except Exception as e:
						print(f"Error loading image {p}: {e}")
						return None

				with ThreadPoolExecutor(max_workers=num_workers) as ex:
					imgs = list(ex.map(load_img, [verified[i] for i in batch_indices]))

				valid_pairs = [(i, img) for i, img in zip(batch_indices, imgs) if img is not None]
				if not valid_pairs:
						if verbose:
								print(f" [batch {b}]: No valid images in batch => skipping")
						continue

				# Build per-sample messages
				messages = [
						[
								{
										"role": "user",
										"content": [
												{"type": "text", "text": base_prompt},
												{"type": "image", "image": img},
										],
								}
						]
						for _, img in valid_pairs
				]

				try:
						# Build chat templates and process batch
						chat_texts = [
								processor.apply_chat_template(
										m, tokenize=False, add_generation_prompt=True
								)
								for m in messages
						]

						inputs = processor(
								text=chat_texts,
								images=[img for _, img in valid_pairs],
								return_tensors="pt",
								padding=True,
						).to(model.device)

						# Generate response
						with torch.no_grad():
								with torch.amp.autocast(
										device_type=device.type,
										enabled=torch.cuda.is_available(),
										dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
								):
										outputs = model.generate(**inputs, **gen_kwargs)

						decoded = processor.batch_decode(outputs, skip_special_tokens=True)

						# if verbose:
						#     print(f"\n[batch {b}] Decoded responses: {type(decoded)} {len(decoded)}\n")

						# NEW: parallel parsing for this batch
						parsed_dict = _parse_batch_parallel(
								decoded_responses=decoded,
								batch_indices=[i for i, _ in valid_pairs],
								model_id_=model_id,
								verbose_=verbose,
						)

						for uniq_idx, parsed in parsed_dict.items():
								results[uniq_idx] = parsed

				except Exception as e_batch:
						print(f"\n[BATCH {b}]: {e_batch}\n")
						if torch.cuda.is_available():
								torch.cuda.empty_cache()
						gc.collect()
						if verbose:
								print(f"\tFalling back to sequential processing for {len(valid_pairs)} images in this batch.")

						# process each image sequentially
						for uniq_idx, img in tqdm(valid_pairs, desc="Processing batch images [sequential]", ncols=100):
								try:
										single_message = [
												{
														"role": "user",
														"content": [
																{"type": "text", "text": base_prompt},
																{"type": "image", "image": img},
														],
												}
										]
										chat = processor.apply_chat_template(
												single_message,
												tokenize=False,
												add_generation_prompt=True,
										)
										single_inputs = processor(
												text=[chat],
												images=[img],
												return_tensors="pt",
										).to(model.device)

										if single_inputs.pixel_values.numel() == 0:
												raise ValueError(
														f"Pixel values of {uniq_idx} are empty: {single_inputs.pixel_values.shape}"
												)

										with torch.no_grad():
												with torch.amp.autocast(
														device_type=device.type,
														enabled=torch.cuda.is_available(),
														dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
												):
														out = model.generate(**single_inputs, **gen_kwargs)

										decoded_single = processor.decode(out[0], skip_special_tokens=True)
										results[uniq_idx] = parse_vlm_response(
												model_id=model_id,
												raw_response=decoded_single,
												verbose=verbose,
										)
								except Exception as e_fallback:
										print(f"\n[Fallback ❌] image {uniq_idx}:\n{e_fallback}\n")
										results[uniq_idx] = None

				# Clean up batch tensors immediately after use
				for name in ("inputs", "outputs", "decoded"):
						if name in locals():
								try:
										del locals()[name]
								except Exception:
										pass

				# if verbose and torch.cuda.is_available():
				# 		print(f"[MEM] Batch {b}: {process.memory_info().rss / (1024**3):.2f}GB in-use")
				# 		print(f"[MEM] Batch {b}: {torch.cuda.memory_allocated() / (1024**3):.2f}GB allocated, "
				# 					f"{torch.cuda.memory_reserved() / (1024**3):.2f}GB reserved")
				# 		print("deleted inputs, outputs, decoded")

				# Periodic memory cleanup (every 10 batches)
				if b % 10 == 0:
						if torch.cuda.is_available():
								torch.cuda.empty_cache()
						gc.collect()
						# if verbose and torch.cuda.is_available():
						# 		mem_allocated = torch.cuda.memory_allocated() / (1024**3)
						# 		mem_reserved = torch.cuda.memory_reserved() / (1024**3)
						# 		print(f"[MEM] Batch {b}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

		# ========== Map back to original ordering ==========
		final = [results[i] for i in orig_to_uniq]
		out_csv = csv_file.replace(".csv", "_vlm_keywords.csv")
		df["vlm_keywords"] = final

		# Save results
		df.to_csv(out_csv, index=False)
		try:
			df.to_excel(out_csv.replace('.csv', '.xlsx'), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")

		elapsed = time.time() - t0
		if verbose:
			n_ok = sum(1 for r in final if r)
			print(f"[STATS] ✅ Success {n_ok}/{len(final)}")
			print(f"[TIME] {elapsed/3600:.2f}h | avg {len(final)/elapsed:.2f}/s")
			print(f"[SAVE] Results written to: {out_csv}")

		return final

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="VLLM-based keyword extraction for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, default=None, help="Path to the metadata CSV file")
	parser.add_argument("--image_path", '-i', type=str, default=None, help="img path [or URL]")
	parser.add_argument("--model_id", '-vlm', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace Vision-Language model ID")
	parser.add_argument("--device", '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device('cuda:0' or 'cpu')")
	parser.add_argument("--num_workers", '-nw', type=int, default=12, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=32, help="Batch size for processing")
	parser.add_argument("--max_keywords", '-mkw', type=int, default=5, help="Max number of keywords to extract")
	parser.add_argument("--max_generated_tks", '-mgt', type=int, default=256, help="Batch size for processing")
	parser.add_argument("--use_quantization", '-q', action='store_true', help="Use quantization")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	parser.add_argument("--debug", '-d', action='store_true', help="Debug mode")

	args = parser.parse_args()
	set_seeds(seed=42, debug=args.debug)
	args.device = torch.device(args.device)
	args.num_workers = min(args.num_workers, os.cpu_count())
	print(args)

	if args.verbose and torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(args.device)
		total_mem = torch.cuda.get_device_properties(args.device).total_memory / (1024**3) # GB
		print(f"Available GPU(s) = {torch.cuda.device_count()}")
		print(f"GPU: {gpu_name} {total_mem:.2f} GB VRAM")
		print(f"\t• CUDA: {torch.version.cuda} Compute Capability: {torch.cuda.get_device_capability(args.device)}")


	if args.image_path:
		keywords = get_vlm_based_labels_single(
			model_id=args.model_id,
			device=args.device,
			image_path=args.image_path,
			max_kws=args.max_keywords,
			max_generated_tks=args.max_generated_tks,
			use_quantization=args.use_quantization,
			verbose=args.verbose,
		)
	elif args.debug:
		keywords = get_vlm_based_labels_debug(
			model_id=args.model_id,
			device=args.device,
			num_workers=args.num_workers,
			csv_file=args.csv_file,
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

	if args.verbose and keywords:
		print(f"{len(keywords)} Extracted keywords")
		for i, kw in enumerate(keywords):
			print(f"{i:06d}. {kw}")

if __name__ == "__main__":
	main()