import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

print(f"sys.path: {sys.path}")

from utils import *
from nlp_utils import get_enriched_description

# how to run:
# local:
# one sample:
# python stage1_vlm_cot.py -i /home/farid/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/images/SLASH76SLASHjlm_item_94084.jpg -c "The Defence. Norwegian refugees in the spring of 1940, on the border in Gäddede. Tasks: Ingvar Holmström, Lund, 1985." -vlm "Qwen/Qwen3.5-4B" -qb 4 -v

# csv input:
# python stage1_vlm_cot.py -csv /home/farid/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/test.csv -vlm "Qwen/Qwen3.5-4B" -qb 4 -v

# with nohup:
# nohup python -u stage1_vlm_cot.py -csv /home/farid/datasets/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata_multi_label.csv -vlm "Qwen/Qwen3.5-4B" -qb 4 -bs 4 -v > logs/ww2_vlm_cot.log 2>&1 &

# HPC:
# one sample:
# $ python stage1_vlm_cot.py -i /scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/images/SLASH76SLASHjlm_item_94084.jpg -c "The Defence. Norwegian refugees in the spring of 1940, on the border in Gäddede. Tasks: Ingvar Holmström, Lund, 1985." -vlm "Qwen/Qwen3.6-27B" -v

PROMPT_TEMPLATE = """Given an image and its caption, extract no more than {k} prominent concepts, then categorize them into three lists of keywords.
The extracted keywords must be semantically atomic, visually grounded, and broad with absolute maximum degree of breadth.

Forbidden keywords:
  - Generic terms (e.g., 'World War I', 'post war era', 'Post-war', 'aftermath of World War II', 'war', 'battle').
  - Dates, times, years, decades, seasonal periods, or any temporal references (e.g., 'winter', 'May 12, 1964', 'September 1919', '1950s era').
  - Quantities, counts, measurements, or numerical expressions (e.g., '1 1/2 ton truck', '1 kilovolt', '7.3mm', '3 Dodge trucks').
  - Identifiers, serial numbers, brands, or models.
  - Names of places, buildings, or structures (e.g., Plaza de Santiago, St. Louis Cathedral).
  - Continents, countries, states, provinces, cities, towns, islands, regions, or roads.
  - Nationalities, ethnicities, or religions.
  - Individual people's names or honorifics (e.g., A. A. Robinson, A. Philip Randolph, Barbara Briggs, Mr. Terry Duce, Allan M. Hardy, Josef Dietrich, Mrs. Howard Russell). 
  - Family relationship terms (e.g., 'mother', 'father', 'son', 'uncle').
  - Roman numerals, fractions, or ordinal numeral keywords (e.g., IV, VIII, fourth, 1st, 115th).
  - Abbreviations, acronyms, phrasal verbs, or descriptive clauses.
  - Image types or characteristics (e.g., photograph, image, black and white photograph).

Output format:
  - text_concepts: Keywords derived STRICTLY from the caption.
  - visual_concepts: Keywords derived STRICTLY from the pixel data.
  - fused_concepts: Keywords inferred from BOTH modalities. In case the modalities are essentially disjoint (e.g., text says "aircraft" but image shows "ships"), return an empty list [] and refrain from forcing a fusion.

Return ONLY a valid JSON object with standarized, valid and parsable **Python** lists without any additional text:
{{
"text_concepts": [],
"visual_concepts": [],
"fused_concepts":[]
}}

Caption: {caption}"""

def is_empty_concepts(concepts: Optional[Dict[str, Any]]) -> bool:
	if not concepts or not isinstance(concepts, dict):
		return True
	return (
		not concepts.get("text_concepts") and
		not concepts.get("visual_concepts") and
		not concepts.get("fused_concepts")
	)

def flush_jsonl_state(jsonl_path: str, state: Dict[int, Dict[str, Any]], verbose: bool = False):
	"""
	Atomically rewrite JSONL so there is exactly one row per id.
	"""
	tmp_path = f"{jsonl_path}.tmp"

	with open(tmp_path, "w", encoding="utf-8") as f:
		for sid in sorted(state.keys(), key=str):
			f.write(json.dumps(state[sid], ensure_ascii=False) + "\n")

	os.replace(tmp_path, jsonl_path)

	if verbose:
		print(f"[SAVE] Compacted JSONL written to: {jsonl_path} ({len(state)} rows)")

def load_jsonl_state(jsonl_path: str, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
	state: Dict[str, Dict[str, Any]] = {}
	if not os.path.exists(jsonl_path):
		if verbose:
			print(f"[WARN] {jsonl_path} not found => Return empty state dict.")
		return state

	with open(jsonl_path, "r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				rec = json.loads(line)
				sid = rec.get("id")
				if sid is None:
					continue
				# Skip legacy int-keyed rows
				if isinstance(sid, int) or (isinstance(sid, str) and sid.lstrip('-').isdigit()):
					if verbose:
						print(f"[WARN] Skipping legacy int-keyed row: {sid}")
					continue
				
				# Non-empty wins: only overwrite if incoming record is non-empty
				# or if the key has never been seen before
				incoming_concepts = rec.get("vlm_cot_raw", {})
				if sid not in state:
					state[sid] = rec
				elif not is_empty_concepts(incoming_concepts):
					state[sid] = rec  # upgrade empty → non-empty
				# else: keep existing non-empty, discard incoming empty
			except Exception as e:
				if verbose:
					print(f"[WARN] Skipping malformed JSONL line {line_no}: {e}")
				continue
	
	if verbose:
		n_empty = sum(1 for r in state.values() if is_empty_concepts(r.get("vlm_cot_raw", {})))
		print(f"[LOAD] {len(state)} unique ids | {n_empty} empty (will retry)")
	
	return state

def _load_vlm_(
	model_id: str,
	quantization_bits: Optional[int] = None,  # None => no quantization; 4 or 8 => quantize
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
		quantization_bits: None (no quant), or 4 / 8 for bitsandbytes quantization
		force_multi_gpu: Force multi-GPU distribution (for large models)
		verbose: Enable verbose logging
	
	Returns:
		Tuple of (processor, model)
	"""
	if quantization_bits is not None and quantization_bits not in (4, 8):
		raise ValueError(f"quantization_bits must be None, 4, or 8 (got {quantization_bits})")
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
	def _optimal_attn_impl() -> str:
		if not torch.cuda.is_available():
			return "eager"
		
		# model config for FlashAttention dimension limits
		max_head_dim = getattr(config, "head_dim", 0)
		if hasattr(config, "text_config"):
			max_head_dim = max(max_head_dim, getattr(config.text_config, "head_dim", 0))
			max_head_dim = max(max_head_dim, getattr(config.text_config, "global_head_dim", 0))

		major, minor = torch.cuda.get_device_capability()
		compute_cap = major + minor / 10
		
		if compute_cap >= 8.0:
			# Only use Flash Attention 2 if the head dimensions are supported
			if max_head_dim <= 256:
				try:
					import flash_attn
					if verbose: print(f"[INFO] Flash Attention 2 available (compute {compute_cap})")
					return "flash_attention_2"
				except ImportError:
					if verbose: print(f"[WARN] Flash Attention 2 not installed")
			else:
				if verbose: print(f"[INFO] Bypassing Flash Attention 2: max head_dim ({max_head_dim}) > 256")
		
		# Fallback to SDPA (which handles >256 dimensions automatically)
		if compute_cap >= 7.0 and torch.__version__ >= "2.0.0":
			if verbose: print(f"[INFO] Using SDPA attention (compute {compute_cap}, PyTorch {torch.__version__})")
			return "sdpa"		
		
		return "eager"

	attn_impl = _optimal_attn_impl()

	if verbose:
		print(f"[INFO] {model_id} Attention implementation: {attn_impl}")
	
	# ========== Quantization config ==========
	quantization_config = None
	if quantization_bits is not None:
		if quantization_bits == 8:
			quantization_config = tfs.BitsAndBytesConfig(
				load_in_8bit=True,
				bnb_8bit_compute_dtype=dtype,
				llm_int8_enable_fp32_cpu_offload=False,
			)
		elif quantization_bits == 4:
			quantization_config = tfs.BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_compute_dtype=torch.bfloat16,
				bnb_4bit_use_double_quant=True,
			)
		else:
			raise ValueError(f"quantization_bits must be 4, 8, or None, got {quantization_bits}")
		
		if verbose:
			print(f"[INFO] {model_id} Quantization enabled: {quantization_bits}-bit")
	
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

	@functools.lru_cache(maxsize=8)
	def get_estimated_gb_size(model_id: str) -> float:
		try:
			info = huggingface_hub.model_info(model_id, token=hf_tk, files_metadata=True)
		except Exception as e:
			raise ValueError(f"Failed to fetch model info for {model_id}: {e}")

		# print("="*100)
		# print(type(info))
		# print(info)
		# print("="*100)

		disk_bytes = 0
		param_count = None

		# 1. Sum actual file sizes (most reliable when available)
		if info.siblings:
			for s in info.siblings:
				if s.size and (s.rfilename.endswith(".safetensors") or s.rfilename.endswith(".bin")):
					disk_bytes += s.size

		# 2. Try safetensors metadata (parameter count)
		if hasattr(info, "safetensors") and info.safetensors:
			safet = info.safetensors
			if isinstance(safet, dict):
				param_count = safet.get("total")
			elif hasattr(safet, "total"):
				param_count = safet.total

		# 3. Choose best source and apply realistic multiplier
		if disk_bytes > 0:
			# print(f"disk_bytes: {disk_bytes}")
			# Disk size already in target dtype → small overhead (1%) (alignment, buffers)
			est_gb = (disk_bytes * 1.01) / (1024 ** 3)

			return est_gb

		if param_count:
			# print(f"param_count: {param_count}")
			# fp16/bf16 = 2 bytes/param + 18–25% overhead
			est_bytes = param_count * 2.0 * 1.22
			est_gb = est_bytes / (1024 ** 3)

			return est_gb

		raise ValueError(
			f"No usable size info for {model_id}. "
			"No file sizes, parameter count, or safetensors metadata available."
		)

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
		if gpu_vram[0] < 10: vram_buffer_gb = 0.7
		elif gpu_vram[0] < 20: vram_buffer_gb = 2.0
		else: vram_buffer_gb = 3.5

		if verbose:
			print(f"[INFO] VRAM buffer: {vram_buffer_gb:.2f} GB")

		# For quantization, reduce buffer further
		if quantization_bits is not None:
			vram_buffer_gb = max(0.5, vram_buffer_gb * 0.5)
			if verbose:
				print(f"[INFO] Quantization enabled - reducing VRAM buffer to {vram_buffer_gb:.1f} GB")
				
		# Adjust estimated size for quantization
		adjusted_size = estimated_size_gb
		if quantization_bits is not None:
			if quantization_bits == 8:
				adjusted_size = estimated_size_gb * 0.5
			elif quantization_bits == 4:
				adjusted_size = estimated_size_gb * 0.25
			
			if verbose:
				print(f"[INFO] Adjusted size for {quantization_bits}-bit quantization: {adjusted_size:.1f} GB")
		
		# ========== PRE-FLIGHT VRAM VALIDATION ==========
		# Account for overhead: model weights + activations + gradients + overhead
		# Rule of thumb: need X.Xx model size for inference (2x for training)
		INFERENCE_OVERHEAD_MULTIPLIER = 1.2
		required_vram = adjusted_size * INFERENCE_OVERHEAD_MULTIPLIER
		usable_vram = total_vram_available - (n_gpus * vram_buffer_gb)

		if verbose:
			print(f"\n[VRAM CHECK] Pre-flight validation:")
			print(f"\t• Estimated Model size (fp16): {adjusted_size:.2f} GB (with {INFERENCE_OVERHEAD_MULTIPLIER}x overhead): {required_vram:.2f} GB")
			print(f"\t• Available VRAM (total):      {total_vram_available:.2f} GB")
			print(f"\t• Available VRAM (usable):     {usable_vram:.2f} GB ({n_gpus}x GPU(s), {vram_buffer_gb:.2f} GB buffer per GPU)")

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
			
			if not quantization_bits:
				print("\n1. ✅ ENABLE QUANTIZATION (Recommended):")
				print("   quantization_bits=8")
				quant8_size = estimated_size_gb * 0.5
				quant8_required = quant8_size * INFERENCE_OVERHEAD_MULTIPLIER
				quant8_fits = "✅ YES" if quant8_required < usable_vram else "❌ NO, try 4-bit"
				print(f"   → Reduces size to ~{quant8_size:.1f} GB")
				print(f"   → Required VRAM: ~{quant8_required:.1f} GB")
				print(f"   → Will fit: {quant8_fits}")
				
				print("\n2. ⚠️  ENABLE 4-BIT QUANTIZATION (More aggressive):")
				print("   quantization_bits=4")
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
		is_large_model = adjusted_size >= 20

		if verbose:
			print(f"\t• Single GPU capacity: {single_gpu_capacity:.2f} GB (GPU VRAM: {gpu_vram[0]:.2f} GB - {vram_buffer_gb:.2f} GB buffer)")
			print(f"\t• is {model_id} Large? (adjusted_size: {adjusted_size:.2f} > 20GB) : {is_large_model}")

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
				
				if quantization_bits is not None:
					buffer = buffer * 0.5
				
				max_memory[i] = f"{max(1, gpu_vram[i] - buffer):.0f}GB"
			
			total_usable = sum(float(v.replace('GB', '')) for v in max_memory.values())
			strategy_desc = f"{model_id} is too large ({adjusted_size:.2f} GB + {INFERENCE_OVERHEAD_MULTIPLIER:.1f}x overhead = {required_vram:.2f} GB) to fit in a single GPU ({single_gpu_capacity:.2f} GB) => Multi-GPU [Model Parallelism] ({n_gpus} Available GPUs, {total_usable:.0f}GB total)"
			
			if verbose:
				print(f"[INFO] Using multi-GPU strategy:")
				print(f"• Estimated model size: {estimated_size_gb:.1f} GB (fp16)")
				if quantization_bits is not None:
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
		"torch_dtype": dtype,  # <-- use torch_dtype (not "dtype")
	}
	
	if quantization_bits is not None:
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
		if quantization_bits is not None:
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
				print(f"  1. Use quantization: ex) quantization_bits=8")
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
				print(f"\nAll layers on GPU - optimal performance!")
		print(f"[MODEL] Loading of {model_id} complete!")

		print(f"{'='*110}\n")
		print(model.config)
		print(f"{'='*110}\n")

	return processor, model

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

def parse_vlm_response(
	model_id: str, 
	response: str, 
	verbose: bool = False
) -> Optional[Dict[str, Any]]:
	if verbose:
		print(f"\nPARSING VLM: {model_id} RESPONSE")
	
	if not response or not isinstance(response, str):
		if verbose:
			print(f"ERROR: Invalid response input.")
			print(f"  Type: {type(response)}")
			print(f"  Value: {response}")
		return None
	
	response = response.strip()
	
	if verbose:
		print(f"\nRaw response (Total length: {len(response)} characters):")
		print(f"{response}")
	
	# Step 1a: Remove "assistant" prefix if present
	if response.lower().startswith("assistant"):
		response = response[len("assistant"):].strip()
		if verbose:
			print(f"[STEP 1a] Removed 'assistant' prefix")
	
	# Step 1b: Remove markdown code fences
	response = re.sub(r'```json\s*', '', response, flags=re.IGNORECASE)
	response = re.sub(r'```\s*', '', response)
	if verbose:
		print(f"[STEP 1b] After removing markdown fences: {len(response)} characters")

	# Step 3: Extract the LAST valid JSON object (STRING-AWARE bracket matching)
	start_idx = response.rfind('{')
	
	if verbose:
		print(f"\n[STEP 3] JSON extraction:")
		print(f"  Last left curly bracket found at index: {start_idx}")
	
	if start_idx == -1:
		if verbose:
			print("  ERROR: No JSON object found in response.")
		return None
	
	# String-aware depth counter to ignore braces inside quotes
	depth = 0
	in_string = False
	escape_next = False
	json_str = None
	
	for i in range(start_idx, len(response)):
			char = response[i]
			
			if escape_next:
					escape_next = False
					continue
					
			if char == '\\':
					escape_next = True
					continue
					
			if char == '"':
					in_string = not in_string
					continue
					
			if not in_string:
					if char == '{':
							depth += 1
					elif char == '}':
							depth -= 1
							if depth == 0:
									json_str = response[start_idx:i+1]
									break
	
	if not json_str:
			if verbose:
					print(f"  ERROR: Unmatched JSON braces in response.")
			return None
	
	if verbose:
		print(f"  Extracted JSON string ({len(json_str)} chars):")
		print(f"{json_str}")
	
	# Step 4: Clean common JSON issues (trailing commas)
	json_str = re.sub(r',\s*}', '}', json_str)
	json_str = re.sub(r',\s*]', ']', json_str)

	try:
		parsed = json.loads(json_str)
		
		if verbose:
			print(f"\n[STEP 4] JSON parsed successfully: {type(parsed)} ({len(parsed)} items) {parsed}")
		
		if not isinstance(parsed, dict):
			if verbose:
				print(f"  ERROR: Parsed JSON is not an object/dict. Got: {type(parsed)}")
			return None
				
		# Ensure required keys exist and are clean lists
		required_keys = ["text_concepts", "visual_concepts", "fused_concepts"]
		
		if verbose:
			print(f"\n[STEP 5] Validating required keys: {required_keys}")
		
		for key in required_keys:
			if key not in parsed:
				if verbose:
					print(f"  ⚠ Missing {key} - adding empty list for safety ([])")
				parsed[key] = []
			else:
				# Safely convert to list and filter out None/empty/whitespace
				raw_list = parsed[key] if isinstance(parsed[key], list) else [parsed[key]]
				cleaned = [
					str(item).strip() for item in raw_list 
					if item is not None and str(item).strip()
				]
				if verbose:
					print(f"{key} {len(cleaned)}/{len(raw_list)} valid")
				
				parsed[key] = cleaned
		
		if verbose:
			print(f"[RESULT] text: {len(parsed['text_concepts'])} visual: {len(parsed['visual_concepts'])} fused: {len(parsed['fused_concepts'])}")

		return parsed
	except json.JSONDecodeError as e:
		if verbose:
			print(f"[ERROR] Failed to parse JSON {type(e).__name__}: {e} at line {e.lineno}, column {e.colno}")
			print(f"{json_str}")
		return None

def get_vlm_cot_labels_single(
	model_id: str,
	image_path: str,
	max_generated_tks: int,
	max_kws: int,
	caption: str,
	img_resized_shape: int = 512,
	quantization_bits: Optional[int]=None,
	verbose: bool = False,
):
	# ========== Load image ==========
	if verbose:
		print(f"[LOADING] {image_path}")
		print(f"[CAPTION] {caption}")

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
		if verbose: 
			print("[ERROR] Invalid image size")
		return None

	# load model and processor
	processor, model = _load_vlm_(
		model_id=model_id, 
		quantization_bits=quantization_bits,
		verbose=verbose
	)

	messages = [
		{"role": "system", "content": "You are an expert historical archivist specializing in multi-label annotation for 20th-century conflict photography."},
		{
			"role": "user",
			"content": [
				{"type": "text", "text": PROMPT_TEMPLATE.format(caption=caption, k=max_kws)},
				{"type": "image", "image": img},
			],
		}
	]
	
	chat_prompt = processor.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
		enable_thinking=False,
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
		generated_tokens.append(breakdown['generated_tokens'])
		print(f"   • Generation time:   {generation_time:.2f}s")
		print(f"   • Generation ratio:  {breakdown['generated_tokens'] / breakdown['input_tokens']:.2%}")
		print(f"   • Time per token:    {generation_time / breakdown['generated_tokens']:.3f}s")
		print(f"   • Tokens per second: {breakdown['generated_tokens'] / generation_time:.1f}")
		print("-"*100)

	# Decode response
	response = processor.decode(outputs[0], skip_special_tokens=True)
	# response = processor.decode(
	# 	outputs[0][input_single["input_ids"].shape[1]:],
	# 	skip_special_tokens=True
	# )	

	parsed = parse_vlm_response(model_id=model_id, response=response, verbose=verbose)

	if verbose:
		print(f"Parsed Response: {type(parsed)} {len(parsed)} {parsed.keys()}")
		print(json.dumps(parsed, indent=2, ensure_ascii=False))

	return [parsed]

def get_vlm_cot_labels(
	model_id: str,
	batch_size: int,
	num_workers: int,
	max_generated_tks: int,
	max_kws: int,
	csv_file: str,
	mem_cleanup_th: int=95,
	do_dedup: bool=True,
	quantization_bits: Optional[int]=None,
	verbose: bool=False,
):
	t0 = time.time()
	output_csv = csv_file.replace(".csv", "_vlm_cot.csv")
	output_jsonl = csv_file.replace(".csv", "_vlm_cot.jsonl")

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
		print(f"[PREP] Loading data (col: enriched_document_description) from {csv_file}...")
	wanted_cols = {
		'doc_url',
		'title',
		'description',
		'keywords', # SMU dataset
		'img_path',
		'enriched_document_description',
	}

	try:
		df = pd.read_csv(
			filepath_or_buffer=csv_file,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
			usecols = lambda c: c in wanted_cols, # automatically skips missing cols
		)
	except Exception as e:
		raise ValueError(f"Error loading CSV file: {e}")
	
	if verbose:
		print(f"[LOADED] {type(df)} {df.shape} {list(df.columns)}")

	# regenerate enriched_document_description
	df = get_enriched_description(df=df, check_english=True, verbose=verbose)
	if verbose:
		print(f"[READY] {type(df)} {df.shape} {list(df.columns)} ({time.time() - t0:.2f}s)")
		print(df.info(verbose=True, memory_usage=True))
		print(df.head())
	
	doc_urls = [url if isinstance(url, str) else None for url in df["doc_url"]]
	image_paths = [p if isinstance(p, str) and os.path.exists(p) else None for p in df["img_path"]]
	descriptions = [desc  if desc and isinstance(desc, str) else None for desc in df["enriched_document_description"]]
	
	assert len(image_paths) == len(descriptions), f"Length mismatch: {len(image_paths)} != {len(descriptions)}"
	n_total = len(image_paths)
	
	if verbose:
		print(f"\n[LOAD] {n_total} image paths from {csv_file} ({time.time() - t0:.2f}s)")
		print(f"IMAGES: {len(image_paths)} | DESCRIPTIONS: {len(descriptions)} | URLS: {len(doc_urls)}")
		print("-"*100)

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
		print(f"[DEDUP] {len(uniq_inputs)} unique images after deduplication ({len(image_paths)} total)")

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
		print(f"[INIT] {len(valid_indices)} verified images")

	# ========== JSONL Resume Logic (1 row per id, atomic rewrite) ==========
	# Load JSONL into a dict keyed by id. Last occurrence wins for any duplicates
	# from previous crash-restart cycles. Empty-concept entries are NOT restored
	# into processed_ids so they will be retried and overwritten in-place.
	jsonl_state: Dict[str, Dict[str, Any]] = load_jsonl_state(output_jsonl, verbose=verbose)

	processed_ids: set[int] = set()
	retry_ids: set[int] = set()

	# Build reverse map: doc_url → uniq_idx
	url_to_idx: Dict[str, int] = {
		(doc_urls[i] or f"__unknown_{i}__"): i
		for i in range(len(uniq_inputs))
	}

	for url_key, rec in jsonl_state.items():
		idx = url_to_idx.get(url_key)
		if idx is None:
			continue
		concepts = rec.get("vlm_cot_raw", {})
		if is_empty_concepts(concepts):
			retry_ids.add(idx)
		else:
			results[idx] = concepts
			processed_ids.add(idx)

	if verbose:
		n_unseen = len([i for i in valid_indices if (doc_urls[i] or f"__unknown_{i}__") not in jsonl_state])
		print(f"[RESUME] JSONL loaded: {len(jsonl_state)} unique ids found")
		print(f"         ✅ Restored (non-empty) : {len(processed_ids)}")
		print(f"         🔄 Will retry (empty)   : {len(retry_ids)}")
		print(f"         🆕 Never seen           : {n_unseen}")

	# Only skip ids with confirmed non-empty results
	valid_indices = [i for i in valid_indices if i not in processed_ids]
	if verbose:
		print(f"[RESUME] {len(valid_indices)} images remaining to process")
	# ========== End Resume Logic ==========

	# # MEMORY INTENSIVE! (DO NOT USE)
	# valid_imgs = [Image.open(p).convert("RGB") for p in verified_paths if p is not None]
	# print(len(valid_imgs), len(valid_indices), len(verified_paths))
	# print(type(valid_imgs[0]), valid_imgs[0].size, valid_imgs[0].mode)

	# ========== Load model ==========
	processor, model = _load_vlm_(
		model_id=model_id,
		quantization_bits=quantization_bits,
		verbose=verbose,
	)

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
		print(f"[GEN CONFIG] {gen_kwargs}")
	
	# ========== Process batches ==========
	def _load_(p: str) -> Optional[Image.Image]:
		try:
			with Image.open(p) as im:
				im = im.convert("RGB")
				return im.copy()
		except Exception as e:
			print(f"Error loading image {p}: {e}")
			return None

	total_batches = math.ceil(len(valid_indices) / batch_size)

	if verbose:
		print(
			f"[INIT] BATCHED PARALLEL OPTIMIZED VLM (nw: {num_workers}) "
			f"{len(valid_indices)} valid unique images → {total_batches} batches of {batch_size}"
		)

	batch_max_tokens = []
	all_image_tokens = []

	for b in range(total_batches):
		print(f"\n[BATCH] {b}/{total_batches}")
		batch_indices = valid_indices[b * batch_size:(b + 1) * batch_size]
		batch_paths = [verified_paths[i] for i in batch_indices]
		batch_descs = [descriptions[i] if descriptions[i] else "No caption available." for i in batch_indices]

		# Parallel Image Loading
		with ThreadPoolExecutor(max_workers=num_workers) as ex:
			batch_imgs = list(ex.map(_load_, batch_paths))
		
		valid_pairs = [
			(i, img, desc)
			for i, img, desc in zip(batch_indices, batch_imgs, batch_descs)
			if img
		]
		
		if not valid_pairs:
			if verbose:
				print(f"[BATCH {b}]: No valid images in batch => skipping")
			continue
		else:
			if verbose:
				print(f"[BATCH {b}] {len(valid_pairs)} valid images")

		# Build per-sample messages
		messages = [
			[
				{"role": "system", "content": "You are an expert historical archivist specializing in multi-label annotation for 20th-century conflict photography."},
				{
					"role": "user",
					"content": [
						{"type": "text", "text": PROMPT_TEMPLATE.format(caption=desc, k=max_kws)},
						{"type": "image", "image": img},
					],
				}
			]
			for idx, img, desc in valid_pairs
		]
		
		try:
			# Build chat templates and process batch
			chat_texts = [
				processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False,)
				for m in messages
			]
			if verbose:
				print(f"[BATCH {b}] Chat templates built: {type(chat_texts)} {len(chat_texts)} => Processing batch inputs in {next(model.parameters()).device}...")

			inputs = processor(
				text=chat_texts,
				images=[img for _, img,_ in valid_pairs],
				return_tensors="pt",
				padding=True,
			).to(next(model.parameters()).device)

			if verbose:
				print(f"[BATCH {b}] Generating responses for {len(valid_pairs)} images [takes a while]...")

			# Generate response
			tt = time.time()
			with torch.no_grad():
				outputs = model.generate(**inputs, **gen_kwargs)
			generation_time = time.time() - tt

			breakdown = get_token_breakdown(inputs, outputs)

			# Determine input length (handle both 1D and 2D input_ids safely)
			input_len = inputs.input_ids.shape[1] if inputs.input_ids.dim() > 1 else inputs.input_ids.shape[0]
			
			# 1. Batch-level stats (Maximum generated tokens in this batch due to padding)
			batch_max_generated = outputs.shape[1] - input_len
			batch_max_tokens.append(batch_max_generated)
			
			# 2. Per-image stats (Actual generated tokens per image, excluding padding)
			generated_sequences = outputs[:, input_len:]
			
			pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
			eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
			
			if pad_token_id is not None:
					# Accurately count non-padding tokens for each image in the batch
					per_image_counts = (generated_sequences != pad_token_id).sum(dim=1).cpu().tolist()
			elif eos_token_id is not None:
					# Count up to the first EOS token if no pad token is defined
					per_image_counts = []
					for seq in generated_sequences:
							eos_mask = (seq == eos_token_id)
							if eos_mask.any():
									# argmax finds the first True value (first EOS token)
									per_image_counts.append(eos_mask.float().argmax().item() + 1) 
							else:
									per_image_counts.append(seq.shape[0])
			else:
					# Fallback: assume no padding was applied
					per_image_counts = [generated_sequences.shape[1]] * generated_sequences.shape[0]
					
			all_image_tokens.extend(per_image_counts)


			if verbose: 
				print(f"[BATCH {b}]")
				print(f"   • Inputs:            {type(inputs)} {type(inputs.input_ids)} {inputs.input_ids.shape}")
				print(f"   • Outputs:           {type(outputs)} {outputs.shape}")
				print(f"   • Generation time:   {generation_time:.2f}s")
				print(f"   • Time per token:    {generation_time / breakdown['generated_tokens']:.3f}s")
				print(f"   • Tokens per second: {breakdown['generated_tokens'] / generation_time:.1f}")

			decoded = processor.batch_decode(outputs, skip_special_tokens=True)

			# input_len = inputs["input_ids"].shape[1]
			# decoded = processor.batch_decode(
			# 	outputs[:, input_len:],  # slice off input tokens
			# 	skip_special_tokens=True
			# )

			if verbose:
				print(f"[BATCH {b}] Decoded responses: {type(decoded)} {len(decoded)}: {decoded}")

			# Sequential parsing
			for (idx, _, _), resp in zip(valid_pairs, decoded):
				try:
					parsed = parse_vlm_response(
						model_id=model_id,
						response=resp,
						verbose=verbose,
					)
					results[idx] = parsed
					
					# Update in-memory state (flushed atomically to disk after the batch)
					doc_url = doc_urls[idx] or f"__unknown_{idx}__"
					jsonl_state[doc_url] = {
						"id": doc_url,
						"vlm_cot_raw": parsed if parsed else {
							"text_concepts": [], "visual_concepts": [], "fused_concepts": []
						}
					}

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
			
			for uniq_idx, img, desc in tqdm(valid_pairs, desc="Processing batch images [SEQUENTIAL]", ncols=150):
				if verbose:
					print(f"\n[Fallback] Processing image {uniq_idx}: {type(img)} {img.size} {img.mode}\n")

				single_message = [
					{"role": "system", "content": "You are an expert historical archivist specializing in multi-label annotation for 20th-century conflict photography."},
					{
						"role": "user",
						"content": [
							{"type": "text", "text": PROMPT_TEMPLATE.format(caption=desc, k=max_kws)},
							{"type": "image", "image": img},
						],
					}
				]

				try:
					chat_single = processor.apply_chat_template(
						single_message,
						tokenize=False,
						add_generation_prompt=True,
						enable_thinking=False,
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

					parsed = parse_vlm_response(
						model_id=model_id,
						response=decoded_single,
						verbose=verbose,
					)
					results[uniq_idx] = parsed
					doc_url = doc_urls[uniq_idx] or f"__unknown_{idx}__"
					jsonl_state[doc_url] = {
						"id": doc_url,
						"vlm_cot_raw": parsed if parsed else {
							"text_concepts": [], "visual_concepts": [], "fused_concepts": []
						}
					}
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
		print("="*100)

		# Atomically rewrite JSONL: exactly one row per id, no duplicates
		flush_jsonl_state(output_jsonl, jsonl_state, verbose=verbose)

	# Final flush to ensure any last-batch updates are persisted
	flush_jsonl_state(output_jsonl, jsonl_state, verbose=verbose)

	# Map back to original ordering
	final = [results[i] for i in orig_to_uniq]
	df["vlm_cot_labels"] = final

	if verbose:
		print(df.head())
		print(df.info(verbose=True, memory_usage=True))
		print(f'vlm_cot_labels column contains {df["vlm_cot_labels"].isna().sum()} None(s) (failed).')
		print("-"*100)

	# df.to_csv(output_csv, index=False)
	# try:
	# 	df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
	# except Exception as e:
	# 	print(f"Failed to write Excel file: {e}")
	# elapsed = time.time() - t0
	if verbose:
		# print(f"[SAVE] Results written to: {output_csv}")
		n_ok = sum(1 for r in final if r)
		print(f"[SUCCESS] {n_ok}/{len(final)}")
		
		print(f"{len(batch_max_tokens)} Total Batches: {batch_max_tokens}")
		if batch_max_tokens:
			print(f"  Batch Max Tokens (padded) -> Min: {min(batch_max_tokens)}, Max: {max(batch_max_tokens)}, Avg: {np.mean(batch_max_tokens):.2f}")

		print(f"{len(all_image_tokens)} Total Images: {all_image_tokens}")
		if all_image_tokens:
			print(f"  Actual Image Tokens       -> Min: {min(all_image_tokens)}, Max: {max(all_image_tokens)}, Avg: {np.mean(all_image_tokens):.2f}")

		# Count how many hit the max token limit
		hits_max = sum(1 for t in all_image_tokens if t >= max_generated_tks)
		print(f"  Hit max tokens ({max_generated_tks}): {hits_max}/{len(all_image_tokens)} ({hits_max/len(all_image_tokens)*100:.1f}%)")

		print("-"*100)
	
	return final

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="VLLM-instruct-based keyword annotation for Historical Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, default=None, help="Path to the metadata CSV file")
	parser.add_argument("--image_path", '-i', type=str, default=None, help="img path [or URL]")
	parser.add_argument("--caption", '-c', type=str, default=None, help="caption")
	parser.add_argument("--model_id", '-vlm', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace Vision-Language model ID")
	parser.add_argument("--device", '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device('cuda:0' or 'cpu')")
	parser.add_argument("--num_workers", '-nw', type=int, default=8, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=2, help="Batch size for processing")
	parser.add_argument("--max_keywords", '-mkw', type=int, default=3, help="Max number of keywords to extract")
	parser.add_argument("--max_generated_tks", '-mgt', type=int, default=256, help="Max number of generated tokens")
	parser.add_argument("--quantization_bits", '-qb', type=int, default=None, help="Quantization bits")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")

	args = parser.parse_args()
	set_seeds(seed=42)
	args.device = torch.device(args.device)
	args.num_workers = min(args.num_workers, os.cpu_count())
	print(args)

	if not args.image_path and not args.csv_file:
		raise ValueError("Either --image_path or --csv_file must be provided")

	if args.image_path:
		keywords = get_vlm_cot_labels_single(
			model_id=args.model_id,
			image_path=args.image_path,
			max_kws=args.max_keywords,
			caption=args.caption,
			img_resized_shape=1024,
			max_generated_tks=args.max_generated_tks,
			quantization_bits=args.quantization_bits,
			verbose=args.verbose,
		)
	else:
		keywords = get_vlm_cot_labels(
			model_id=args.model_id,
			csv_file=args.csv_file,
			num_workers=args.num_workers,
			batch_size=args.batch_size,
			max_kws=args.max_keywords,
			max_generated_tks=args.max_generated_tks,
			quantization_bits=args.quantization_bits,
			verbose=args.verbose,
		)

	if args.verbose and keywords:
		print(f"{len(keywords)} {type(keywords)} Extracted keywords")
		# for i, kw in enumerate(keywords):
		# 	print(f"{i:06d}. {kw}")

if __name__ == "__main__":
	main()