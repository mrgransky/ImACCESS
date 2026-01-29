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

process = psutil.Process(os.getpid())
EXP_BACKOFF = 2  # seconds
IMG_MAX_RES = 512

VLM_INSTRUCTION_TEMPLATE = """You function as a historical archivist whose expertise lies in the 20th century and whose task is to produce **general-purpose, reusable semantic labels** suitable for **multi-label classification**.

Extract no more than {k} **PROMINENT, FACTUAL, and DISTINCT KEYWORDS** that capture the visually observable actions, objects, or occurrences in the image - **completely ignoring all text**.

**CRITICAL RULES**:
- Return **ONLY** a standardized, valid, and parsable **Python LIST** with **AT MOST {k} KEYWORDS**. 
	Fewer keywords are **preferred** if the caption lacks distinct reusable concepts.

- **PRIORITIZE MEANINGFUL PHRASES**: 
	Each keyword must be **semantically atomic**, denoting exactly **one clearly observable concept**.
	When possible, prefer **concrete visual descriptors** over abstract or umbrella terms, **without making inferences**.

- **ZERO HALLUCINATION POLICY**: 
	Do not invent or infer specifics that lack clear verification from the visual content. 
	If a keyword cannot be confidently justified by what is visible, omit it.

- **STRICTLY EXCLUDE**:
	- verbs, possessive cases, abbreviations, shortened words or acronyms as standalone keywords.
	- keywords that start or end with prepositions or conjunctions.
	- dates, times, hours, minutes, calendar references, time periods, seasons, months, days, years, decades, centuries, or **ANY** time-related content.
	- explanatory texts, code blocks, punctuations, or tags before or after the **Python LIST**.
	- TEXT EXTRACTION / OCR: do NOT read, transcribe, quote, paraphrase, or rely on any visible text in the image.
	- image quality, format, medium, or stylistic attributes.
	- **overly vague or generic terms** (e.g., abstract container nouns that add little semantic value).

- The clean, valid, and parsable **Python LIST** must be the **VERY LAST THING** in your response."""

def verify(p: str):
	if p is None or not os.path.exists(p):
		return None
	try:
		with open(p, 'rb') as f:
			header = f.read(3)
			if header == b'\xff\xd8\xff':  # Valid JPEG SOI marker
				return p
		return None
	except Exception:
		return None

def _load_vlm_(
	model_id: str,
	use_quantization: bool = False,
	quantization_bits: int = 8,
	force_multi_gpu: bool = False,
	verbose: bool = False,
) -> Tuple[tfs.PreTrainedTokenizerBase, torch.nn.Module]:
	"""
	Load a Vision-Language Model (VLM) with optimal settings.
	
	Implements intelligent device strategy:
	1. For small models (<20GB): Single GPU for speed
	2. For large models (>=20GB): Multi-GPU distribution
	3. Adaptive VRAM buffering based on GPU size
	4. Quantization-aware memory allocation
	5. Avoids disk offloading at all costs
	
	Args:
		model_id: HuggingFace model identifier
		use_quantization: Whether to use quantization
		quantization_bits: Quantization bits (4 or 8)
		force_multi_gpu: Force multi-GPU distribution (for large models)
		verbose: Enable verbose logging
	
	Returns:
		Tuple of (processor, model)
	"""
	if verbose:
		print(f"\n{'='*110}")
		print(f"[MODEL] Loading {model_id} on cache_dir: {cache_directory.get(USER)}")

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
	try:
		if verbose:
			print(f"[INFO] Logging in to HuggingFace Hub...")
		huggingface_hub.login(token=hf_tk)
	except Exception as e:
		print(f"<!> Failed to login to HuggingFace Hub: {e}")
		raise e
	
	# ========== Load config ==========
	config = tfs.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
	if verbose:
		print(f"[INFO] {model_id} Config summary")
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
		
	dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
	if verbose:
		print(f"[INFO] {model_id} Dtype selection: {dtype}")
	
	# ========== Optimal attention implementation ==========
	def _optimal_attn_impl(m_id: str) -> str:
		"""Select best available attention implementation."""
		if not torch.cuda.is_available():
			return "eager"
		
		flash_ok = False
		try:
			import flash_attn
			major, _ = torch.cuda.get_device_capability()
			flash_ok = major >= 8
		except Exception as e:
			if verbose:
				print(f"[WARN] Flash Attention unavailable: {type(e).__name__}")
			traceback.print_exc()
		
		if flash_ok:
			return "flash_attention_2"

		# torch >= 2.0.0 has SDPA
		if torch.__version__ >= "2.0.0":
			if verbose:
				print(f"[INFO] Using SDPA attention (torch {torch.__version__})")
			return "sdpa"

		return "eager"
	
	attn_impl = _optimal_attn_impl(model_id)
	if verbose:
		print(f"[INFO] {model_id} Attention implementation: {attn_impl}")
	
	# ========== Quantization config ==========
	quantization_config = None
	if use_quantization:
		if quantization_bits == 8:
			quantization_config = tfs.BitsAndBytesConfig(
				load_in_8bit=True,
				bnb_8bit_compute_dtype=dtype,
				llm_int8_enable_fp32_cpu_offload=False, # avoid offloading to CPU
			)
		elif quantization_bits == 4:
			quantization_config = tfs.BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_compute_dtype=torch.bfloat16,
				bnb_4bit_use_double_quant=True,
			)
		else:
			raise ValueError(f"quantization_bits must be 4 or 8, got {quantization_bits}")
		
		if verbose:
			print(f"[INFO] {model_id} Quantization enabled")
			print(f"\t• Bits                : {quantization_bits}")
			print(f"\t• Config object type  : {type(quantization_config).__name__}")
	
	# ========== Processor loading ==========
	processor = tfs.AutoProcessor.from_pretrained(
		model_id,
		use_fast=True,
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
		quantization_config=quantization_config,
	)
	if verbose:
		print(f"[INFO] {model_id} Processor: {processor.__class__.__name__}")
	
	# Extract tokenizer
	if hasattr(processor, "tokenizer"):
		tokenizer = processor.tokenizer
	elif hasattr(processor, "text_tokenizer"):
		tokenizer = processor.text_tokenizer
	else:
		raise ValueError("Unable to locate tokenizer in processor")
	if hasattr(tokenizer, "padding_side") and tokenizer.padding_side is not None:
		tokenizer.padding_side = "left"
	
	def get_estimated_gb_size(m_id: str) -> float:
		info = huggingface_hub.model_info(m_id, token=hf_tk)
		try:
			if hasattr(info, "safetensors") and info.safetensors:
				total_bytes = info.safetensors.total
				if total_bytes > 0:
					size_gb = total_bytes / (1024 ** 3)
					return size_gb
		except Exception as e:
			print(f"<!> Failed to estimate model size from safetensors: {e}")
			raise e

	estimated_size_gb = get_estimated_gb_size(model_id)
	
	if verbose:
		print(f"[INFO] {model_id} Estimated size: {estimated_size_gb:.2f} GB (fp16)")
	
	# ========== Dynamic Device Strategy with Adaptive VRAM Buffering ==========
	max_memory = {}
	
	if n_gpus > 0:
		total_vram_available = 0
		gpu_vram = []
		
		for i in range(n_gpus):
			props = torch.cuda.get_device_properties(i)
			if verbose:
				print(f"GPU {i}: {props}")
			vram_gb = props.total_memory / (1024**3)
			gpu_vram.append(vram_gb)
			total_vram_available += vram_gb
		
		# ADAPTIVE BUFFER: Scale based on GPU size
		# Small GPUs (<10GB): 0.7GB buffer
		# Medium GPUs (10-20GB): 2GB buffer
		# Large GPUs (>20GB): 4GB buffer
		if gpu_vram[0] < 10:
			vram_buffer_gb = 0.7
		elif gpu_vram[0] < 20:
			vram_buffer_gb = 2.0
		else:
			vram_buffer_gb = 4.0
		if verbose:
			print(f"[INFO] VRAM buffer: {vram_buffer_gb:.2f} GB")

		# For quantization, reduce buffer further
		if use_quantization:
			vram_buffer_gb = max(0.5, vram_buffer_gb * 0.5)
			if verbose:
				print(f"[INFO] Quantization enabled - reducing VRAM buffer to {vram_buffer_gb:.1f} GB")
				
		# Adjust estimated size for quantization
		adjusted_size = estimated_size_gb
		if use_quantization:
			if quantization_bits == 8:
				adjusted_size = estimated_size_gb * 0.5
			elif quantization_bits == 4:
				adjusted_size = estimated_size_gb * 0.25
			
			if verbose:
				print(f"[INFO] Adjusted size for {quantization_bits}-bit quantization: {adjusted_size:.1f} GB")
		
		# ========== PRE-FLIGHT VRAM VALIDATION ==========
		# Account for overhead: model weights + activations + gradients + overhead
		# Rule of thumb: need X.Xx model size for inference (2x for training)
		INFERENCE_OVERHEAD_MULTIPLIER = 1.5
		required_vram = adjusted_size * INFERENCE_OVERHEAD_MULTIPLIER
		usable_vram = total_vram_available - (n_gpus * vram_buffer_gb)

		if verbose:
			print(f"\n[VRAM CHECK] Pre-flight validation:")
			print(f"\t• Estimated Model size (fp16): {adjusted_size:.1f} GB (with {INFERENCE_OVERHEAD_MULTIPLIER}x overhead): {required_vram:.1f} GB")
			print(f"\t• Available VRAM (total):      {total_vram_available:.1f} GB")
			print(f"\t• Available VRAM (usable):     {usable_vram:.1f} GB ({n_gpus}x GPU(s), {vram_buffer_gb:.1f} GB buffer per GPU)")

		# Check if model will fit
		if required_vram > usable_vram:
			print("\n" + "="*80)
			print("❌ INSUFFICIENT VRAM ERROR")
			print("="*80)
			print(f"\nModel: {model_id}")
			print(f"Estimated Model size: {adjusted_size:.1f} GB")
			print(f"Required VRAM (with overhead): {required_vram:.1f} GB")
			print(f"Available VRAM: {usable_vram:.1f} GB ({n_gpus}x GPUs)")
			print(f"\nDeficit: {required_vram - usable_vram:.1f} GB SHORT")
			print("\nSOLUTIONS:")
			
			if not use_quantization:
				print("\n1. ✅ ENABLE QUANTIZATION (Recommended):")
				print("   use_quantization=True, quantization_bits=8")
				quant8_size = estimated_size_gb * 0.5
				quant8_required = quant8_size * INFERENCE_OVERHEAD_MULTIPLIER
				quant8_fits = "✅ YES" if quant8_required < usable_vram else "❌ NO, try 4-bit"
				print(f"   → Reduces size to ~{quant8_size:.1f} GB")
				print(f"   → Required VRAM: ~{quant8_required:.1f} GB")
				print(f"   → Will fit: {quant8_fits}")
				
				print("\n2. ⚠️  ENABLE 4-BIT QUANTIZATION (More aggressive):")
				print("   use_quantization=True, quantization_bits=4")
				quant4_size = estimated_size_gb * 0.25
				quant4_required = quant4_size * INFERENCE_OVERHEAD_MULTIPLIER
				print(f"   → Reduces size to ~{quant4_size:.1f} GB")
				print(f"   → Required VRAM: ~{quant4_required:.1f} GB")
				print(f"   → Will fit: ✅ YES")
			else:
				if quantization_bits == 8:
					print("\n1. ⚠️  TRY 4-BIT QUANTIZATION:")
					print("   quantization_bits=4")
					quant4_size = estimated_size_gb * 0.25
					quant4_required = quant4_size * INFERENCE_OVERHEAD_MULTIPLIER
					print(f"   → Reduces size to ~{quant4_size:.1f} GB")
					print(f"   → Required VRAM: ~{quant4_required:.1f} GB")
				
				print("\n2. 🔄 USE LARGER GPU:")
				print("   • A100 80GB available")
				print("   • Switch to Mahti gpusmall partition")
				
				print("\n3. 📉 USE SMALLER MODEL:")
				print("   • Qwen3-VL-8B-Instruct (~16 GB)")
				print("   • Qwen2.5-VL-7B-Instruct (~14 GB)")
			
			print("\n" + "="*80 + "\n")
			
			raise RuntimeError(
				f"Model requires {required_vram:.2f} GB but only {usable_vram:.2f} GB available. "
				f"Enable quantization or use larger GPU."
			)

		if verbose:
			print(f"[VRAM] PASSED: Model will fit!")

		# Decision: Single GPU vs Multi GPU
		single_gpu_capacity = gpu_vram[0] - vram_buffer_gb
		if verbose:
			print(f"\t• Single GPU capacity: {single_gpu_capacity:.1f} GB (GPU VRAM: {gpu_vram[0]:.1f} GB - {vram_buffer_gb:.1f} GB buffer)")
		is_large_model = adjusted_size >= 20
		if verbose:
			print(f"\t• is {model_id} Large? ({adjusted_size:.1f} > 20GB) : {is_large_model}")
		use_single_gpu = (
			not force_multi_gpu and
			not is_large_model and  # Don't use single GPU for large models
			adjusted_size < single_gpu_capacity * 0.8 and  # 80% safety margin
			(n_gpus == 1 or adjusted_size < 20)
		)
		
		if use_single_gpu:
			max_memory[0] = f"{max(1, single_gpu_capacity):.0f}GB"
			strategy_desc = f"Single GPU (GPU 0, limit: {max_memory[0]})"
		else:
			# Multi-GPU distribution [Model Parallelism]
			for i in range(n_gpus):
				# Smaller buffer on secondary GPUs
				if gpu_vram[i] < 10:
					buffer = vram_buffer_gb if i == 0 else 0.5
				else:
					buffer = vram_buffer_gb if i == 0 else 2
				
				if use_quantization:
					buffer = buffer * 0.5
				
				max_memory[i] = f"{max(1, gpu_vram[i] - buffer):.0f}GB"
			
			total_usable = sum(float(v.replace('GB', '')) for v in max_memory.values())
			strategy_desc = f"{model_id} is too large ({adjusted_size:.2f} GB + {INFERENCE_OVERHEAD_MULTIPLIER:.1f}x overhead = {required_vram:.2f} GB) to fit in a single GPU ({single_gpu_capacity:.2f} GB) => Multi-GPU [Model Parallelism] ({n_gpus} Available GPUs, {total_usable:.0f}GB total)"
			
			if verbose:
				print(f"[INFO] Using multi-GPU strategy:")
				print(f"• Estimated model size: {estimated_size_gb:.1f} GB (fp16)")
				if use_quantization:
					print(f"• Adjusted for quantization: {adjusted_size:.1f} GB")
				print(f"• Single GPU capacity: {single_gpu_capacity:.1f} GB")
				print(f"• Total VRAM: {total_vram_available:.1f} GB")
				if force_multi_gpu:
					print(f"• Reason: force_multi_gpu=True")
	else:
		strategy_desc = "CPU (no GPUs)"
	
	if verbose:
		print(f"\n[INFO] {strategy_desc}")
		if max_memory:
			print(f"Max memory per GPU:")
			for gpu_id, limit in max_memory.items():
				print(f"\tGPU {gpu_id}: {limit}")
	
	# ========== Base Model Loading Kwargs ==========
	base_model_kwargs: Dict[str, Any] = {
		"low_cpu_mem_usage": True,
		"trust_remote_code": True,
		"cache_dir": cache_directory[USER],
		"attn_implementation": attn_impl,
		"dtype": dtype,
	}
	
	if use_quantization:
		base_model_kwargs["quantization_config"] = quantization_config
	
	# ========== Load Model ==========
	model = None
	try:
		if n_gpus > 0:
			model = model_cls.from_pretrained(
				model_id,
				**base_model_kwargs,
				device_map="auto",
				max_memory=max_memory,
			)
		else:
			model = model_cls.from_pretrained(
				model_id,
				**base_model_kwargs,
			)	
	except Exception as e:
		if verbose:
			print(f"[ERROR] Failed to load model: {e}")
		raise e
	
	model.eval()
	
	# ========== Model Info & Verification ==========
	if verbose:
		print(f"\n[MODEL] {model_id} {model.__class__.__name__}")
		try:
			first_param = next(model.parameters())
			print(f"   • First parameter dtype: {first_param.dtype}")
			print(f"   • First parameter device: {first_param.device}")
		except StopIteration:
			pass
		
		total_params = sum(p.numel() for p in model.parameters())
		approx_fp16_gb = total_params * 2 / (1024 ** 3)
		approx_fp8_gb = total_params * 1 / (1024 ** 3)
		approx_fp4_gb = total_params * 0.5 / (1024 ** 3)
		
		print(f"   • Total parameters: {total_params:,}")
		print(f"   • Actual model size (fp16): {approx_fp16_gb:.2f} GB")
		if use_quantization:
			if quantization_bits == 8:
				print(f"   • Actual model size (int8): {approx_fp8_gb:.2f} GB")
			elif quantization_bits == 4:
				print(f"   • Actual model size (int4): {approx_fp4_gb:.2f} GB")
		
		# Validate estimation
		estimation_error = abs(estimated_size_gb - approx_fp16_gb) / approx_fp16_gb * 100
		if estimation_error > 50:
			print(f"   ⚠️  WARNING: Size estimation was off by {estimation_error:.0f}%!")
			print(f"      Estimated: {estimated_size_gb:.1f} GB, Actual: {approx_fp16_gb:.1f} GB")
		
		if hasattr(model, "hf_device_map"):
			dm = model.hf_device_map
			
			# Check for disk offloading
			disk_layers = [k for k, v in dm.items() if v == "disk"]
			cpu_layers = [k for k, v in dm.items() if v == "cpu"]
			
			if disk_layers:
				print(f"\n{'='*70}")
				print(f"❌ CRITICAL WARNING: {len(disk_layers)} layers on DISK!")
				print(f"{'='*70}")
				print(f"This will cause 100-1000x slowdown!")
				print(f"\nSOLUTIONS:")
				print(f"  1. Use quantization: use_quantization=True, quantization_bits=8")
				print(f"  2. Force multi-GPU: force_multi_gpu=True")
				print(f"  3. Use smaller model variant")
				print(f"  4. Use 4-bit quantization for even more memory savings")
				print(f"{'='*70}\n")
			
			if cpu_layers:
				print(f"\n⚠️  WARNING: {len(cpu_layers)} layers on CPU (slower than GPU)")
			
			# Count GPU distribution
			gpu_counts = {}
			for layer_name, device in dm.items():
				if isinstance(device, int):
					gpu_counts[device] = gpu_counts.get(device, 0) + 1
			
			if gpu_counts:
				print(f"\n[INFO] GPU Distribution:")
				total_gpu_layers = sum(gpu_counts.values())
				for gpu_id in sorted(gpu_counts.keys()):
					count = gpu_counts[gpu_id]
					pct = count / total_gpu_layers * 100 if total_gpu_layers > 0 else 0
					print(f"   • GPU {gpu_id}: {count} layers ({pct:.1f}%)")
			
			# Show device map only if there are issues
			if disk_layers or cpu_layers:
				print(f"\n[INFO] Device map (showing problematic layers):")
				for k, v in dm.items():
					if v in ["disk", "cpu"]:
						print(f"   {k}: {v}")
			elif not disk_layers and not cpu_layers:
				print(f"\n✅ All layers on GPU - optimal performance!")
		print(f"[MODEL] Loading of {model_id} complete!")
		print(f"{'='*110}\n")

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

def get_vlm_based_labels_single(
	model_id: str,
	image_path: str,
	max_generated_tks: int,
	max_kws: int,
	img_resized_shape: int = 512,
	use_quantization: bool = False,
	verbose: bool = False,
):

	# ========== Load image ==========
	if verbose:
		print(f"[LOAD] {image_path}")

	img = None
	try:
		img = Image.open(image_path)
	except Exception as e:
		if verbose: 
			print(f"{e}\n=> retry via URL")

		try:
			r = requests.get(image_path, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, stream=True)
			r.raise_for_status()

			img = Image.open(io.BytesIO(r.content))

		except Exception as e2:
			if verbose: print(f"[ERROR] URL fetch failed => {e2}")
			return None

	if img is None:
		if verbose: 
			print(f"[ERROR] Failed to load image: {image_path}")
		return None

	try:
		img = img.convert("RGB")
	except Exception as e:
		if verbose: 
			print(f"[ERROR] Failed to convert image to RGB: {e}")
		return None

	try:
		img_copy = img.copy()
		img_copy.thumbnail((img_resized_shape, img_resized_shape), resample=Image.Resampling.LANCZOS)
		img = img_copy
	except Exception as e:
		if verbose: 
			print(f"[ERROR] Failed to resize image: {e}")
		return None

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
	
	input_single = processor(
		images=img,
		text=chat_prompt,
		padding=True,
		return_tensors="pt"
	).to(next(model.parameters()).device)

	if verbose:
		print(f"[INPUT] Pixel: {input_single.pixel_values.shape} {input_single.pixel_values.dtype} {input_single.pixel_values.device}")

	if input_single.pixel_values.numel() == 0:
		raise ValueError(f"Pixel values of {image_path} are empty: {input_single.pixel_values.shape}")

	gen_kwargs = dict(
		max_new_tokens=max_generated_tks, 
		use_cache=True,
		eos_token_id=processor.tokenizer.eos_token_id,
		pad_token_id=processor.tokenizer.pad_token_id,
	)
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
	tt = time.time()
	with torch.no_grad():
		outputs = model.generate(**input_single, **gen_kwargs)
	generation_time = time.time() - tt

	if verbose:
		print(f"[RESPONSE] {type(outputs)} {outputs.shape}")
		breakdown = get_token_breakdown(input_single, outputs)
		print(f"   • Generation time:   {generation_time:.2f}s")
		print(f"   • Generation ratio:  {breakdown['generated_tokens'] / breakdown['input_tokens']:.2%}")
		print(f"   • Time per token:    {generation_time / breakdown['generated_tokens']:.3f}s")
		print(f"   • Tokens per second: {breakdown['generated_tokens'] / generation_time:.1f}")

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

	# ========== Initialize =========
	num_workers = min(os.cpu_count(), num_workers)
	if verbose:
		print(f"\n{'='*100}")
		print(f"[INIT] VLM-based keyword generation [DEBUG MODE]")
		print(f"[INIT] Model: {model_id}")
		print(f"[INIT] Device: {device}")
		print(f"[INIT] Num workers: {num_workers}")
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
				
				input_single = processor(
					images=img,
					text=chat_prompt,
					padding=True,
					return_tensors="pt"
				).to(next(model.parameters()).device)

				if verbose:
					print(f"[INPUT] Pixel: {input_single.pixel_values.shape} {input_single.pixel_values.dtype} {input_single.pixel_values.device}")

				if input_single.pixel_values.numel() == 0:
					raise ValueError(f"Pixel values of {img_path} are empty: {input_single.pixel_values.shape}")
				
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
					print(json.dumps(gen_kwargs, indent=2, ensure_ascii=False))

				# ========== Generate response ==========
				gen_start = time.time()
				with torch.no_grad():
					outputs = model.generate(**input_single, **gen_kwargs)
				
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
		if 'input_single' in locals():
			del input_single
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

def get_vlm_based_labels(
	model_id: str,
	device: str,
	batch_size: int,
	num_workers: int,
	max_generated_tks: int,
	max_kws: int,
	csv_file: str,
	mem_cleanup_th: int=95,
	do_dedup: bool=True,
	use_quantization: bool=False,
	verbose: bool=False,
):
	t0 = time.time()
	output_csv = csv_file.replace(".csv", "_vlm_keywords.csv")
	base_prompt = VLM_INSTRUCTION_TEMPLATE.format(k=max_kws)

	try:
		df = pd.read_csv(
			filepath_or_buffer=output_csv,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
			usecols = ['vlm_keywords'],
		)
		return df['vlm_keywords'].tolist()
	except Exception as e:
		print(f"<!> {e} Generating from scratch...")
	
	# ========== Load data ==========
	if verbose:
		print(f"[PREP] Loading data (col: img_path) from {csv_file}...")
	try:
		df = pd.read_csv(
			filepath_or_buffer=csv_file,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
			usecols = ['doc_url', 'img_path'],
		)
	except Exception as e:
		raise ValueError(f"Error loading CSV file: {e}")

	image_paths = [p if isinstance(p, str) and os.path.exists(p) else None for p in df["img_path"]]
	n_total = len(image_paths)
	if verbose:
		print(f"[DATA] Loaded {type(df)} {df.shape} with {n_total} image paths from CSV ({time.time() - t0:.2f}s)")

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
	
	if verbose:
		print(f"[INIT] Deduplication: {len(uniq_inputs)} unique images")
	results: List[Optional[List[str]]] = [None] * len(uniq_inputs)

	with ThreadPoolExecutor(max_workers=num_workers) as ex:
		verified_paths = list(
			tqdm(
				ex.map(verify, uniq_inputs), 
				total=len(uniq_inputs), 
				desc=f"parallel image verification (nw: {num_workers})"
			)
		)
	valid_indices = [i for i, v in enumerate(verified_paths) if v is not None]

	if verbose:
		print(f"[INIT] Verification: {len(valid_indices)} verified images")

	# # MEMORY INTENSIVE! (DO NOT USE)
	# valid_imgs = [Image.open(p).convert("RGB") for p in verified_paths if p is not None]
	# print(len(valid_imgs), len(valid_indices), len(verified_paths))
	# print(type(valid_imgs[0]), valid_imgs[0].size, valid_imgs[0].mode)

	# ========== Load model ==========
	processor, model = _load_vlm_(
		model_id=model_id,
		use_quantization=use_quantization,
		verbose=verbose,
	)
	# ========== Prepare generation kwargs ==========
	gen_kwargs = dict(
		max_new_tokens=max_generated_tks, 
		use_cache=True,
		eos_token_id=processor.tokenizer.eos_token_id,
		pad_token_id=processor.tokenizer.pad_token_id,
	)
	
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
	
	# ========== Process batches ==========
	def _load_(p: str) -> Optional[Image.Image]:
		try:
			with Image.open(p) as im:
				im = im.convert("RGB")
				return im.copy()
		except Exception as e:
			print(f"Error loading image {p}: {e}")
			return None

	# num_workers = min(8, num_workers)
	if verbose:
		print(f"[INIT] BATCHED PARALLEL OPTIMIZED VLM processing with {num_workers} workers")

	total_batches = math.ceil(len(valid_indices) / batch_size)
	if verbose:
		print(f"[INFO] {len(valid_indices)} valid unique images → {total_batches} batches of {batch_size}")

	for b in tqdm(range(total_batches), desc="Processing (visual) batches", ncols=120):
		batch_indices = valid_indices[b * batch_size:(b + 1) * batch_size]
		batch_paths = [verified_paths[i] for i in batch_indices]

		with ThreadPoolExecutor(max_workers=num_workers) as ex:
			batch_imgs = list(ex.map(_load_, batch_paths))
		
		valid_pairs = [
			(i, img)
			for i, img in zip(batch_indices, batch_imgs)
			if img
		]
		
		if not valid_pairs:
			if verbose:
				print(f"\n[BATCH {b}]: No valid images in batch => skipping")
			continue
		else:
			if verbose:
				print(f"\n[BATCH {b}] contains {len(valid_pairs)} valid images.")

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
				processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
				for m in messages
			]
			if verbose:
				print(f"\n[BATCH {b}] Chat templates built: {type(chat_texts)} {len(chat_texts)} => Processing batch inputs in {next(model.parameters()).device}...")
			inputs = processor(
				text=chat_texts,
				images=[img for _, img in valid_pairs],
				return_tensors="pt",
				padding=True,
			).to(next(model.parameters()).device)

			if verbose:
				print(f"\n[BATCH {b}] Generating responses for {len(valid_pairs)} images [takes a while]...")

			# Generate response
			tt = time.time()
			with torch.no_grad():
				outputs = model.generate(**inputs, **gen_kwargs)
			generation_time = time.time() - tt

			if verbose: 
				print(f"\n[BATCH {b}]")
				print(f"[RESPONSE] {type(outputs)} {outputs.shape}")
				breakdown = get_token_breakdown(inputs, outputs)
				print(f"   • Generation time:   {generation_time:.2f}s")
				print(f"   • Generation ratio:  {breakdown['generated_tokens'] / breakdown['input_tokens']:.2%}")
				print(f"   • Time per token:    {generation_time / breakdown['generated_tokens']:.3f}s")
				print(f"   • Tokens per second: {breakdown['generated_tokens'] / generation_time:.1f}")

			decoded = processor.batch_decode(outputs, skip_special_tokens=True)

			if verbose:
				print(f"\n[BATCH {b}] Decoded responses: {type(decoded)} {len(decoded)}\n")

			# Sequential parsing
			for (idx, _), resp in zip(valid_pairs, decoded):
				try:
					results[idx] = parse_vlm_response(
						model_id=model_id,
						raw_response=resp,
						verbose=verbose,
					)
				except Exception as e:
					if verbose:
						print(f"[WARN] Parse error for idx {idx}: {e}")
					results[idx] = None

			# Clean up batch tensors immediately after use
			try:
				del inputs, outputs, decoded, valid_pairs, messages, chat_texts
			except NameError:
				pass

		except Exception as e_batch:
			print(f"\n[BATCH {b}]: {e_batch}\n")

			if verbose:
				print(f"Cleaning up after batch failure...")
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			gc.collect()

			if verbose:
				print(f"\tFalling back to SEQUENTIAL processing for {len(valid_pairs)} images in this batch.")
			for uniq_idx, img in tqdm(valid_pairs, desc="Processing batch images [SEQUENTIAL]", ncols=150):
				if verbose:
					print(f"\n[Fallback] Processing image {uniq_idx}: {type(img)} {img.size} {img.mode}\n")

				single_message = [
					{
						"role": "user",
						"content": [
							{"type": "text", "text": base_prompt},
							{"type": "image", "image": img},
						],
					}
				]

				try:
					chat_single = processor.apply_chat_template(
						single_message,
						tokenize=False,
						add_generation_prompt=True,
					)

					input_single = processor(
						text=[chat_single],
						images=[img],
						return_tensors="pt",
						padding=True,
					).to(next(model.parameters()).device)
					
					if verbose:
						print(f"[INPUT] Pixel: {input_single.pixel_values.shape} {input_single.pixel_values.dtype} {input_single.pixel_values.device}")

					if input_single.pixel_values.numel() == 0:
						raise ValueError(f"Pixel values of {uniq_idx} are empty: {input_single.pixel_values.shape}")

					with torch.no_grad():
						out_single = model.generate(**input_single, **gen_kwargs)

					decoded_single = processor.decode(out_single[0], skip_special_tokens=True)

					if verbose:
						print(f"\n[Sequential Fallback] image {uniq_idx}:\n{type(decoded_single)} {len(decoded_single)}\n")

					results[uniq_idx] = parse_vlm_response(
						model_id=model_id,
						raw_response=decoded_single,
						verbose=verbose,
					)
				except Exception as e_fallback:
					print(f"\n[Sequential Fallback] image {uniq_idx}:\n{e_fallback}\nNO keywords extracted.\n")
					results[uniq_idx] = None
				
				if verbose:
					print(f"\n[BATCH {b} Sequential Fallback] Deleting sequential fallback tensors for image: {uniq_idx}...")
				try:
					del single_message, chat_single, input_single, out_single, decoded_single
				except NameError:
					pass
				
				if verbose:
					print(f"\n[BATCH {b} Sequential Fallback] Clearing cache for image: {uniq_idx}...")
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
				gc.collect()

		# memory management
		need_cleanup = False
		memory_consumed_percent = 0
		for device_idx in range(torch.cuda.device_count()):
			mem_total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3) 
			mem_allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
			mem_reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)	
			mem_usage_pct = (mem_reserved / mem_total) * 100 if mem_total > 0 else 0
			if verbose:
				print(
					f"[MEM] Batch {b} (GPU {device_idx}): {mem_usage_pct:.2f}% usage: "
					f"{mem_allocated:.2f}GB alloc / {mem_reserved:.2f}GB reserved (Total: {mem_total:.1f}GB)"
				)
			if mem_usage_pct > mem_cleanup_th: 
				need_cleanup = True
				memory_consumed_percent += mem_usage_pct

		if need_cleanup:
			print(f"\n[WARN] High memory usage ({memory_consumed_percent:.1f}% > {mem_cleanup_th}%) => Clearing cache...")
			torch.cuda.empty_cache() # clears all GPUs
			gc.collect()

	# ========== Map back to original ordering ==========
	final = [results[i] for i in orig_to_uniq]
	df["vlm_keywords"] = final

	df.to_csv(output_csv, index=False)
	try:
		df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	elapsed = time.time() - t0
	if verbose:
		n_ok = sum(1 for r in final if r)
		print(f"[STATS] ✅ Success {n_ok}/{len(final)}")
		print(f"[TIME] {elapsed/3600:.2f}h | avg {len(final)/elapsed:.2f}/s")
		print(f"[SAVE] Results written to: {output_csv}")
	return final

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="VLLM-instruct-based keyword annotation for Historical Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, default=None, help="Path to the metadata CSV file")
	parser.add_argument("--image_path", '-i', type=str, default=None, help="img path [or URL]")
	parser.add_argument("--model_id", '-vlm', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace Vision-Language model ID")
	parser.add_argument("--device", '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device('cuda:0' or 'cpu')")
	parser.add_argument("--num_workers", '-nw', type=int, default=12, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=32, help="Batch size for processing")
	parser.add_argument("--max_keywords", '-mkw', type=int, default=3, help="Max number of keywords to extract")
	parser.add_argument("--max_generated_tks", '-mgt', type=int, default=64, help="Batch size for processing")
	parser.add_argument("--use_quantization", '-q', action='store_true', help="Use quantization")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	parser.add_argument("--debug", '-d', action='store_true', help="Debug mode")

	args = parser.parse_args()
	set_seeds(seed=42, debug=args.debug)
	args.device = torch.device(args.device)
	args.num_workers = min(args.num_workers, os.cpu_count())
	print(args)

	if args.image_path:
		keywords = get_vlm_based_labels_single(
			model_id=args.model_id,
			image_path=args.image_path,
			max_kws=args.max_keywords,
			img_resized_shape=1024,
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
		keywords = get_vlm_based_labels(
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
		print(f"{len(keywords)} {type(keywords)} Extracted keywords")
		# for i, kw in enumerate(keywords):
		# 	print(f"{i:06d}. {kw}")

if __name__ == "__main__":
	main()