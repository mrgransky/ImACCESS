from utils import *

# basic models:
# model_id = "google/gemma-1.1-2b-it"
# model_id = "google/gemma-1.1-7b-it"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.1-405B-Instruct"
# model_id = "meta-llama/Llama-3.2-1B-Instruct" # default for local
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.3-70B-Instruct"

# better models:
# model_id = "Qwen/Qwen3-4B-Instruct-2507"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# model_id = "NousResearch/Hermes-2-Pro-Mistral-7B"
# model_id = "allenai/Olmo-3-7B-Instruct"
# model_id = "google/flan-t5-xxl"

# does not fit into VRAM:
# Qwen/Qwen3-30B-A3B-Instruct-2507 # multi-gpu required

# not useful for instruction tuning:
# model_id = "microsoft/DialoGPT-large"
# model_id = "gpt2-xl"

if not hasattr(tfs.utils, "LossKwargs"):
	class LossKwargs(TypedDict, total=False):
		"""
		Compatibility shim for older Phi models 
		expecting LossKwargs in transformers.utils.
		Acts as a stub TypedDict with no required keys.
		"""
		pass
	tfs.utils.LossKwargs = LossKwargs

if not hasattr(tfs.utils, "FlashAttentionKwargs"):
	class FlashAttentionKwargs(TypedDict, total=False):
		"""Stub TypedDict for models expecting FlashAttentionKwargs in transformers.utils"""
		pass
	tfs.utils.FlashAttentionKwargs = FlashAttentionKwargs

TEMPERATURE = 1e-8
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt

# STOPWORDS = set(nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())) # all languages
STOPWORDS = set(nltk.corpus.stopwords.words('english')) # english only
# custom_stopwords_list = requests.get("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/refs/heads/master/stopwords-en.txt").content
# stopwords = set(custom_stopwords_list.decode().splitlines())
with open('meaningless_words.txt', 'r') as file_:
	custom_stopwords_list=[line.strip().lower() for line in file_]
stopwords = set(custom_stopwords_list)
STOPWORDS.update(stopwords)
# print(f"Successfully loaded {len(STOPWORDS)} stopwords")

LLM_INSTRUCTION_TEMPLATE = """<s>[INST]
You function as a historical archivist whose expertise lies in the 20th century.
Given the caption below, extract no more than {k} highly prominent, factual, and distinct **KEYWORDS** that convey the primary actions, objects, or occurrences.

{caption}

**CRITICAL RULES**:
- Return **ONLY** a clean, valid, and parsable **Python LIST** with **AT MOST {k} KEYWORDS** - fewer is expected if the text is either short or lacks distinct concepts.
- **PRIORITIZE MEANINGFUL PHRASES**: Opt for multi-word n-grams such as NOUN PHRASES and NAMED ENTITIES over single terms only if they convey more distinct meanings.
- Extracted **KEYWORDS** must be self-contained and grammatically complete phrases that explicitly appear in the text. If you are uncertain, in doubt, or unsure, omit the keyword rather than guessing.
- **STRICTLY EXCLUDE NUMERICAL CONTENT** such as measurements, units, or quantitative terms.
- **STRICTLY EXCLUDE MEDIA DESCRIPTORS** such as generic photography, image, picture, or media terms.
- **STRICTLY EXCLUDE TEMPORAL EXPRESSIONS** such as specific times, calendar dates, seasonal periods, or extended historical eras.
- **ABSOLUTELY NO** synonymous, duplicate, identical or misspelled keywords.
- **STRICTLY EXCLUDE ALL** explanatory texts, code blocks, punctuations, or tags before or after the **Python LIST**.
- **ABSOLUTELY NO** keywords that start or end with prepositions or conjunctions.
- Exclude meaningless abbreviations, numerical words, special characters, or stopwords.
- The clean, valid, and parsable **Python LIST** must be the **VERY LAST THING** in your response.
[/INST]"""

def _load_llm_(
	model_id: str,
	use_quantization: bool = False,
	quantization_bits: int = 8,
	force_multi_gpu: bool = False,
	verbose: bool = False,
) -> Tuple[tfs.PreTrainedTokenizerBase, torch.nn.Module]:
	"""
	Load a Large Language Model with optimal device placement.
	
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
		Tuple of (tokenizer, model)
	"""
	if verbose:
		print(f"[MODEL] Loading {model_id}...")

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
	use_auto_model = False
	
	if config.architectures:
		cls_name = config.architectures[0]
		if hasattr(tfs, cls_name):
			model_cls = getattr(tfs, cls_name)
			if verbose:
				print(f"[INFO] Resolved model class from transformers → {model_cls.__name__}\n")
		else:
			# Custom architecture - will use AutoModelForCausalLM
			use_auto_model = True
			if verbose:
				print(f"[INFO] Custom architecture detected: {cls_name}")
				print(f"[INFO] Will use AutoModelForCausalLM with trust_remote_code=True\n")
	else:
		# No architecture specified - use AutoModelForCausalLM
		use_auto_model = True
		if verbose:
			print(f"[INFO] No architecture specified in config")
			print(f"[INFO] Will use AutoModelForCausalLM\n")

	# ========== Quantization config ==========
	quantization_config = None
	if use_quantization:
		if quantization_bits == 8:
			quantization_config = tfs.BitsAndBytesConfig(
				load_in_8bit=True,
				bnb_8bit_compute_dtype=torch.bfloat16,
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
			raise ValueError("quantization_bits must be 4 or 8")
		
		if verbose:
			print(f"[INFO] Quantization enabled")
			print(f"   • Bits                : {quantization_bits}")
			print(f"   • Config object type  : {type(quantization_config).__name__}")
			print()
	
	# ========== Tokenizer loading ==========
	tokenizer = None
	try:
		tokenizer = tfs.AutoTokenizer.from_pretrained(
			model_id,
			use_fast=True,
			trust_remote_code=True,
			cache_dir=cache_directory[USER],
		)
	except (KeyError, ValueError, OSError) as exc:
		if verbose:
			print(f"[WARN] AutoTokenizer failed: {exc}")
			print("[INFO] Trying specific tokenizer classes...")
		
		fallback_exc = None
		candidate_tokenizer_classes = [
			getattr(tfs, "MistralTokenizer", None),
			getattr(tfs, "MistralTokenizerFast", None),
			getattr(tfs, "LlamaTokenizer", None),
			getattr(tfs, "LlamaTokenizerFast", None),
		]
		
		candidate_tokenizer_classes = [cls for cls in candidate_tokenizer_classes if cls is not None]
		
		for TokCls in candidate_tokenizer_classes:
			try:
				if verbose:
					print(f"[DEBUG] Trying {TokCls.__name__}...")
				
				tokenizer = TokCls.from_pretrained(
					model_id,
					trust_remote_code=True,
					cache_dir=cache_directory[USER],
				)
				
				if verbose:
					print(f"[SUCCESS] Loaded tokenizer using {TokCls.__name__}")
				break
			except Exception as e:
				fallback_exc = e
				if verbose:
					print(f"[DEBUG] {TokCls.__name__} failed: {e}")
				continue

		if tokenizer is None:
			if verbose:
				print("[INFO] All specific tokenizer classes failed, trying AutoTokenizer with use_fast=False...")
			try:
				tokenizer = tfs.AutoTokenizer.from_pretrained(
					model_id,
					use_fast=False,
					trust_remote_code=True,
					cache_dir=cache_directory[USER],
				)
				if verbose:
					print("[SUCCESS] Loaded tokenizer using AutoTokenizer with use_fast=False")
			except Exception as final_exc:
				raise RuntimeError(
					f"Failed to load tokenizer for '{model_id}'. "
					f"AutoTokenizer error: {exc}. "
					f"Fallback errors: {fallback_exc}. "
					f"Final attempt error: {final_exc}"
				) from final_exc

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id
	
	if hasattr(tokenizer, "padding_side") and tokenizer.padding_side is not None:
		tokenizer.padding_side = "left"

	if verbose:
		# print all tokenizer attributes
		print(f"[TOKENIZER] {tokenizer.__class__.__name__}")
		for attr in dir(tokenizer):
			print(f"{attr}")

		print(f"   • vocab size        : {tokenizer.vocab_size:>20,}")
		print(f"   • padding side      : {tokenizer.padding_side:>20}")
		print()
	
	# ========== Dynamic Device Strategy with Adaptive VRAM Buffering ==========
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
		print(f"[INFO] Estimated model size: ~{estimated_size_gb:.1f} GB (fp16)")

	# ========== Model loading kwargs ==========
	max_memory = {}
	
	if n_gpus > 0:
		total_vram_available = 0
		gpu_vram = []
		
		for i in range(n_gpus):
			props = torch.cuda.get_device_properties(i)
			vram_gb = props.total_memory / (1024**3)
			gpu_vram.append(vram_gb)
			total_vram_available += vram_gb
		
		# ADAPTIVE BUFFER: Scale based on GPU size
		# Small GPUs (<10GB): 1GB buffer
		# Medium GPUs (10-20GB): 2GB buffer
		# Large GPUs (>20GB): 4GB buffer
		if gpu_vram[0] < 10:
			vram_buffer_gb = 1.0  # Small GPU (6GB, 8GB)
		elif gpu_vram[0] < 20:
			vram_buffer_gb = 2.0  # Medium GPU (16GB)
		else:
			vram_buffer_gb = 4.0  # Large GPU (32GB, 40GB)
		
		# For quantization, we need even less buffer (quantized models are memory efficient)
		if use_quantization:
			vram_buffer_gb = max(0.5, vram_buffer_gb * 0.5)  # Half the buffer for quantized
			if verbose:
				print(f"[INFO] Quantization enabled - reducing VRAM buffer to {vram_buffer_gb:.1f}GB")
		
		# Decision: Single GPU vs Multi GPU
		single_gpu_capacity = gpu_vram[0] - vram_buffer_gb
		
		# Adjust estimated size for quantization
		adjusted_size = estimated_size_gb
		if use_quantization:
			if quantization_bits == 8:
				adjusted_size = estimated_size_gb * 0.5  # 8-bit is ~50% of fp16
			elif quantization_bits == 4:
				adjusted_size = estimated_size_gb * 0.25  # 4-bit is ~25% of fp16
			
			if verbose:
				print(f"[INFO] Adjusted size for {quantization_bits}-bit quantization: {adjusted_size:.1f} GB")
		
		use_single_gpu = (
			not force_multi_gpu and
			adjusted_size < single_gpu_capacity * 0.8 and  # 80% safety margin
			(n_gpus == 1 or adjusted_size < 20)
		)
		
		if use_single_gpu:
			max_memory[0] = f"{max(1, single_gpu_capacity):.0f}GB"  # At least 1GB
			strategy_desc = f"Single GPU (GPU 0, limit: {max_memory[0]})"
		else:
			# Multi-GPU distribution
			for i in range(n_gpus):
				# Smaller buffer on secondary GPUs
				if gpu_vram[i] < 10:
					buffer = vram_buffer_gb if i == 0 else 0.5
				else:
					buffer = vram_buffer_gb if i == 0 else 2
				
				if use_quantization:
					buffer = buffer * 0.5  # Even smaller for quantized
				
				max_memory[i] = f"{max(1, gpu_vram[i] - buffer):.0f}GB"
			
			total_usable = sum(float(v.replace('GB', '')) for v in max_memory.values())
			strategy_desc = f"Multi-GPU ({n_gpus} GPUs, {total_usable:.0f}GB total)"
			
			if verbose:
				print(f"[INFO] Using multi-GPU strategy:")
				print(f"   • Estimated model size: {estimated_size_gb:.1f} GB (fp16)")
				if use_quantization:
					print(f"   • Adjusted for quantization: {adjusted_size:.1f} GB")
				print(f"   • Single GPU capacity: {single_gpu_capacity:.1f} GB")
				print(f"   • Total VRAM: {total_vram_available:.1f} GB")
				if force_multi_gpu:
					print(f"   • Reason: force_multi_gpu=True")
	else:
		strategy_desc = "CPU (no GPUs)"
	
	if verbose:
		print(f"[INFO] Loading strategy: {strategy_desc}")
		if max_memory:
			print(f"   • Max memory per GPU:")
			for gpu_id, limit in max_memory.items():
				print(f"      - GPU {gpu_id}: {limit}")
		print()
	
	# ========== Model loading kwargs ==========
	model_kwargs: Dict[str, Any] = {
		"low_cpu_mem_usage": True,
		"trust_remote_code": True,
		"cache_dir": cache_directory[USER],
		"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
	}
	
	if use_quantization:
		model_kwargs["quantization_config"] = quantization_config
	
	if n_gpus > 0:
		model_kwargs["device_map"] = "auto"
		model_kwargs["max_memory"] = max_memory

	if verbose:
		model_loader_name = "AutoModelForCausalLM" if use_auto_model else model_cls.__name__
		print(f"[INFO] Model loading kwargs for {model_loader_name}:")
		for k, v in model_kwargs.items():
			if k == "quantization_config":
				print(f"   • {k}: {type(v).__name__}")
			elif k == "max_memory":
				print(f"   • {k}: {v}")
			else:
				print(f"   • {k}: {v}")

		print()

		if torch.cuda.is_available():
			cur = torch.cuda.current_device()
			print("[DEBUG] CUDA memory BEFORE model load")
			print(f"   • allocated : {torch.cuda.memory_allocated(cur)//(1024**2)} MiB")
			print(f"   • reserved  : {torch.cuda.memory_reserved(cur)//(1024**2)} MiB\n")

	try:
		if use_auto_model:
			model = tfs.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
		else:
			model = model_cls.from_pretrained(model_id, **model_kwargs)
	except Exception as e:
		if verbose:
			print(f"[ERROR] Error loading model: {e}")
		raise e

	model.eval()

	# ========== Model Info & Verification ==========
	if verbose:
		# print all model attributes
		print(f"[MODEL] {model_id} {model.__class__.__name__}")
		for attr in dir(model):
			print(f"{attr}")

	if verbose:
		print(f"\n[MODEL] {model_id} {model.__class__.__name__}")
		print(f"\n[DEBUG] Model generate signature: {inspect.signature(model.generate)}")

		custom_methods = [m for m in dir(model) if 'generate' in m.lower()]
		print(f"\n[DEBUG] Available generate methods: {custom_methods}")
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
			
			# Show abbreviated device map for large models
			if not disk_layers and not cpu_layers:
				print(f"\n✅ All layers on GPU - optimal performance!")
			elif not disk_layers:
				print(f"\n[INFO] Device map summary:")
				items = list(dm.items())
				for k, v in items:
					print(f"   {k}: {v}")

		if hasattr(model, 'generation_config'):
			print(f"\n[GENERATION CONFIG]")
			gen_cfg = model.generation_config
			print(f"   • max_length: {getattr(gen_cfg, 'max_length', 'N/A')}")
			print(f"   • temperature: {getattr(gen_cfg, 'temperature', 'N/A')}")
			print(f"   • top_p: {getattr(gen_cfg, 'top_p', 'N/A')}")
			print(f"   • top_k: {getattr(gen_cfg, 'top_k', 'N/A')}")
			print(f"   • do_sample: {getattr(gen_cfg, 'do_sample', 'N/A')}")

	return tokenizer, model

def get_prompt(tokenizer: tfs.PreTrainedTokenizer, description: str, max_kws: int):
	messages = [
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": LLM_INSTRUCTION_TEMPLATE.format(k=max_kws, caption=description.strip())},
	]
	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True
	)
	return text

def parse_llm_response(
	model_id: str, 
	input_prompt: str, 
	raw_llm_response: str, 
	max_kws: int, 
	verbose: bool = False
):
	if verbose: 
		print(f"[DEBUG] Parsing LLM response for {model_id}")
		print(f"[RESPONSE]\n{raw_llm_response}\n")

	llm_response: Optional[str] = None

	# response differs significantly between models
	if "meta-llama" in model_id:
		llm_response = _llama_llm_response(model_id, input_prompt, raw_llm_response, max_kws, verbose)
	elif "Qwen" in model_id:
		llm_response = _qwen_llm_response(model_id, input_prompt, raw_llm_response, max_kws, verbose)
	elif "microsoft" in model_id:
		llm_response = _microsoft_llm_response(model_id, input_prompt, raw_llm_response, max_kws, verbose)
	elif "mistralai" in model_id:
		llm_response = _mistral_llm_response(model_id, input_prompt, raw_llm_response, max_kws, verbose)
	elif "NousResearch" in model_id:
		llm_response = _nousresearch_llm_response(model_id, input_prompt, raw_llm_response, max_kws, verbose)
	elif "google" in model_id:
		llm_response = _google_llm_response(model_id, input_prompt, raw_llm_response, verbose)
	else:
		raise NotImplementedError(f"Model {model_id} not implemented")

	return llm_response

def _google_llm_response(model_id: str, input_prompt: str, llm_response: str, max_kws: int, verbose: bool = False) -> Optional[List[str]]:
		print(f"Handling Google response [model_id: {model_id}]...")
		print(f"Raw response (repr): {repr(llm_response)}")
		
		# Find all potential list-like structures
		list_matches = re.findall(r"\[.*?\]", llm_response, re.DOTALL)
		print(f"All bracketed matches: {list_matches}")
		
		# Look for a list with three quoted strings
		list_match = re.search(
				r"\[\s*['\"][^'\"]*['\"](?:\s*,\s*['\"][^'\"]*['\"]){2}\s*\]",
				llm_response, re.DOTALL
		)
		
		if list_match:
				final_list_str = list_match.group(0)
				print(f"Found potential list: '{final_list_str}'")
		else:
				print("Error: Could not find a valid list in the response.")
				# attempt to extract comma-separated keywords
				match = re.search(r"([\w\s\-]+(?:,\s*[\w\s\-]+){2,})", llm_response, re.DOTALL)
				if match:
						keywords = [kw.strip() for kw in match.group(1).split(',')]
						keywords = [re.sub(r'[\d#]', '', kw).strip() for kw in keywords if kw.strip()]
						processed_keywords = []
						for kw in keywords[:3]:
								cleaned_keyword = re.sub(r'\s+', ' ', kw)
								if cleaned_keyword and cleaned_keyword not in processed_keywords:
										processed_keywords.append(cleaned_keyword)
						if len(processed_keywords) >= 3:
								print(f"Fallback extracted {len(processed_keywords)} keywords: {processed_keywords}")
								return processed_keywords[:3]
				print("Error: No valid list or keywords found.")
				return None
		
		# Clean the string (handle smart quotes and normalize)
		cleaned_string = final_list_str.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
		print(f"Cleaned string: '{cleaned_string}'")
		
		# Parse the string into a Python list
		try:
				keywords_list = ast.literal_eval(cleaned_string)
				# Validate: must be a list of strings
				if not (isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list)):
						print("Error: Extracted string is not a valid list of strings.")
						return None
				
				# Post-process: remove numbers, special characters, and duplicates
				processed_keywords = []
				for keyword in keywords_list:
						# Remove numbers and special characters
						cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
						cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)
						# Optionally exclude abbreviations (e.g., single letters or common historical abbreviations)
						if len(cleaned_keyword) > 2 or cleaned_keyword.lower() not in {'u.s.', 'wwii', 'raf', 'nato', 'mt.'}:
								if cleaned_keyword and cleaned_keyword not in processed_keywords:
										processed_keywords.append(cleaned_keyword)
				
				if len(processed_keywords) > max_kws:
					processed_keywords = processed_keywords[:max_kws]
				
				print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
				return processed_keywords
		
		except Exception as e:
				print(f"Error parsing the list: {e}")
				print(f"Problematic string: '{cleaned_string}'")
				return None

def _microsoft_llm_response(model_id: str, input_prompt: str, llm_response: str, max_kws: int, verbose: bool = False):
		print(f"Handling Microsoft response model_id: {model_id}...")
		
		# The model output is at the end after the [/INST] tag
		# Split by lines and look for the content after the last [/INST]
		lines = llm_response.strip().split('\n')
		
		# Find the content after the last [/INST] tag
		model_output = None
		for i, line in enumerate(lines):
				if '[/INST]' in line:
						# Get everything after this line
						model_output = '\n'.join(lines[i+1:]).strip()
						break
		
		if not model_output:
				print("Error: Could not find model output after [/INST] tag.")
				return None
		
		print(f"Model output: {model_output}")
		
		# Look for the list in the model output
		match = re.search(r"(\[.*?\])", model_output, re.DOTALL)
		
		if not match:
				print("Error: Could not find a list in the Microsoft response.")
				return None
				
		final_list_str = match.group(1)
		print(f"Found list string: {final_list_str}")
		
		# Clean the string - replace single quotes with double quotes for JSON
		cleaned_string = final_list_str.replace("'", '"')
		
		# Replace smart quotes with standard straight quotes
		cleaned_string = cleaned_string.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
		
		# Remove any apostrophes inside words (like "don't" -> "dont")
		cleaned_string = re.sub(r'(\w)\'(\w)', r'\1\2', cleaned_string)

		if cleaned_string == "[]":
				print("Model returned an empty list.")
				return []

		try:
				# Use json.loads to parse the JSON-like string
				keywords_list = json.loads(cleaned_string)
				
				# Ensure the parsed result is a list of strings
				if not (isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list)):
						print("Error: Extracted string is not a valid list of strings.")
						return None
				
				# Post-process to enforce rules
				processed_keywords = []
				for keyword in keywords_list:
						# Remove numbers and special characters
						cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
						cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)
						
						if cleaned_keyword and cleaned_keyword not in processed_keywords:
								processed_keywords.append(cleaned_keyword)
								
				if len(processed_keywords) > max_kws:
						processed_keywords = processed_keywords[:max_kws]
						
				if not processed_keywords:
						print("Error: No valid keywords found after processing.")
						return None
						
				print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
				return processed_keywords
				
		except json.JSONDecodeError as e:
				print(f"Error parsing the list with JSON: {e}")
				print(f"Problematic string: {cleaned_string}")
				
				# try ast.literal_eval if JSON fails
				try:
						import ast
						keywords_list = ast.literal_eval(final_list_str)
						if isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list):
								print(f"Using ast fallback: {keywords_list}")
								return keywords_list
				except:
						pass
						
				return None
				
		except Exception as e:
				print(f"An unexpected error occurred: {e}")
				return None

def _mistral_llm_response(
		model_id: str,
		input_prompt: str,
		llm_response: str,
		max_kws: int,
		verbose: bool = False
) -> Optional[List[str]]:
		"""
		Extract and clean keywords from Mistral LLM response.

		Returns:
				List[str]: Cleaned keywords, or None if extraction fails
		"""

		# ------------------------------------------------------------------
		# Helper: remove redundant subphrases like "Division" if
		# "motorized Troops Division" is also present.
		# Only removes single-token keywords that appear as a whole token
		# inside another, longer keyword.
		# ------------------------------------------------------------------
		def dedupe_redundant_subphrases(keywords: List[str], verbose: bool = False) -> List[str]:
				if not keywords:
						return keywords

				lowered = [k.lower() for k in keywords]
				keep: List[str] = []

				if verbose:
						print("\n[DEBUG] Running dedupe_redundant_subphrases()")
						print(f"[DEBUG]   Input keywords: {keywords}")

				for i, (kw, kw_l) in enumerate(zip(keywords, lowered)):
						tokens = kw_l.split()
						redundant = False

						# Only consider single-token keywords for now
						if len(tokens) == 1:
								for j, other_l in enumerate(lowered):
										if j == i:
												continue
										other_tokens = other_l.split()
										# If this single token appears as a whole token
										# inside another, longer keyword, treat as redundant.
										if kw_l in other_tokens and len(other_tokens) > 1:
												redundant = True
												if verbose:
														print(f"[DEBUG]   '{kw}' → REJECTED as redundant (sub-token of '{keywords[j]}')")
												break

						if not redundant:
								keep.append(kw)

				if verbose:
						print(f"[DEBUG]   Output keywords after redundancy dedupe: {keep}\n")

				return keep

		# ------------------------------------------------------------------
		# Step 1: Find the list line
		# ------------------------------------------------------------------
		lines = llm_response.strip().split('\n')

		if verbose:
				print(f"[DEBUG] Split response into {len(lines)} lines")

		list_line = None
		for i, line in enumerate(lines):
				line_stripped = line.strip()
				if line_stripped.startswith('[') and line_stripped.endswith(']'):
						list_line = line_stripped
						if verbose:
								print(f"[DEBUG] Found list at line {i}: {list_line}")
						break

		if not list_line:
				if verbose:
						print("[ERROR] Could not find a list pattern [...] in response")
						print("[DEBUG] Last few lines inspected:")
						for j, line in enumerate(lines[-5:]):  # show last 5 lines
								print(f"  {j}: {repr(line[:100])}")
				return None

		# ------------------------------------------------------------------
		# Step 2: Normalize quotes
		# ------------------------------------------------------------------
		quote_map = str.maketrans(
				{
						'“': '"',
						'”': '"',
						'‘': "'",
						'’': "'",
				}
		)
		cleaned_string = list_line.translate(quote_map)

		if verbose:
				print(f"[DEBUG] After quote normalization: {cleaned_string}")

		# ------------------------------------------------------------------
		# Step 3: Handle empty list
		# ------------------------------------------------------------------
		if cleaned_string == "[]":
				if verbose:
						print("[INFO] Model returned empty list []")
				return []

		# ------------------------------------------------------------------
		# Step 4: Parse with ast.literal_eval
		# ------------------------------------------------------------------
		try:
				keywords_list = ast.literal_eval(cleaned_string)
				if verbose:
						print(f"[DEBUG] ✓ ast.literal_eval succeeded")
						print(f"[DEBUG]   Type: {type(keywords_list)}")
						print(f"[DEBUG]   Raw content: {keywords_list}")
		except Exception as e:
				if verbose:
						print(f"[ERROR] ast.literal_eval failed: {type(e).__name__}: {e}")
						print(f"[DEBUG] Failed string: {cleaned_string}")
				return None

		# ------------------------------------------------------------------
		# Step 5: Validate it's a list of strings
		# ------------------------------------------------------------------
		if not isinstance(keywords_list, list):
				if verbose:
						print(f"[ERROR] Parsed result is not a list (got {type(keywords_list)})")
				return None

		if not all(isinstance(item, str) for item in keywords_list):
				if verbose:
						print(f"[ERROR] List contains non-string items")
						print(f"[DEBUG] Item types: {[type(x) for x in keywords_list]}")
				return None

		if verbose:
				print(f"[DEBUG] Validated as list of {len(keywords_list)} strings")

		# ------------------------------------------------------------------
		# Step 6: Process keywords (numeric filter, cleaning, dedupe)
		# ------------------------------------------------------------------
		processed_keywords: List[str] = []

		for i, keyword in enumerate(keywords_list):
				original = keyword

				# Reject keywords that are ONLY digits/special chars
				# But preserve keywords like "MG 42", "StuG III", "3rd Infantry"
				if re.fullmatch(r'[\d\s\-#]+', keyword.strip()):
						if verbose:
								print(f"[DEBUG] Item {i}: '{original}' → REJECTED (purely numeric/special)")
						continue

				# Clean: collapse whitespace, strip leading/trailing whitespace
				# Do NOT strip digits that are part of the keyword
				cleaned_keyword = re.sub(r'\s+', ' ', keyword).strip()

				# Minimum length check
				if len(cleaned_keyword) < 2:
						if verbose:
								print(f"[DEBUG] Item {i}: '{original}' → '{cleaned_keyword}' → REJECTED (too short)")
						continue

				# Check for duplicates (case-insensitive)
				if cleaned_keyword.lower() in [k.lower() for k in processed_keywords]:
						if verbose:
								print(f"[DEBUG] Item {i}: '{original}' → '{cleaned_keyword}' → REJECTED (duplicate)")
						continue

				processed_keywords.append(cleaned_keyword)

				if verbose:
						if original != cleaned_keyword:
								print(f"[DEBUG] Item {i}: '{original}' → '{cleaned_keyword}' → KEPT (cleaned)")
						else:
								print(f"[DEBUG] Item {i}: '{original}' → KEPT (unchanged)")

				# Early stop if we've reached max_kws
				if len(processed_keywords) >= max_kws:
						if verbose:
								print(f"[DEBUG] Reached max_kws={max_kws}, stopping further processing")
						break

		# ------------------------------------------------------------------
		# Step 7: Redundancy reduction (subphrase dedupe)
		# ------------------------------------------------------------------
		if not processed_keywords:
				if verbose:
						print("[ERROR] No valid keywords remaining after initial processing")
				return None

		if verbose:
				print(f"\n[DEBUG] Keywords before redundancy dedupe: {processed_keywords}")

		deduped_keywords = dedupe_redundant_subphrases(processed_keywords, verbose=verbose)

		# Safety: enforce max_kws again after dedupe (though it shouldn't increase)
		final_keywords = deduped_keywords[:max_kws]

		if not final_keywords:
				if verbose:
						print("[ERROR] No valid keywords remaining after redundancy dedupe")
				return None

		if verbose:
				print(f"\n[SUCCESS] Extracted {len(final_keywords)} keyword(s): {final_keywords}\n")

		return final_keywords

def _qwen_llm_response(
	model_id: str, 
	input_prompt: str, 
	llm_response: str, 
	max_kws: int, 
	verbose: bool = False
) -> Optional[List[str]]:
	# Step 1: Find the [/INST] tag
	inst_end_match = re.search(r'\[/INST\]', llm_response)
	
	if not inst_end_match:
		if verbose:
			print("[ERROR] No [/INST] tag found in response")
		return None
	
	inst_end_pos = inst_end_match.end()
	response_content = llm_response[inst_end_pos:].strip()
	
	if verbose:
		print(f"[STEP 1] Found [/INST] at position {inst_end_pos}")
		print(f"[STEP 1] Content after [/INST]:\n{response_content}")
	
	# Step 2: Extract the Python list (first occurrence)
	# Matches: ['item1', "item's 2"] with proper quote pairing (single/double)
	list_patterns = [
			# Pattern 1: Double-quoted strings (can contain single quotes/apostrophes)
			r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]',
			# Pattern 2: Single-quoted strings (can contain double quotes)
			r"\[\s*'[^']*'(?:\s*,\s*'[^']*')*\s*\]",
		]
		
	list_str = None
	for pattern in list_patterns:
		list_match = re.search(pattern, response_content)
		if list_match:
			list_str = list_match.group(0)
			if verbose:
				print(f"[STEP 2] Matched with pattern: {pattern[:50]}...")
			break
	
	if not list_str:
		if verbose:
			print("[ERROR] No Python list found in response content")
			print(f"[DEBUG] Response content: {response_content[:200]}")
		return None
	
	if verbose:
		print(f"[STEP 2] Extracted list string: {list_str}\n")	

	# Step 3: Parse the list
	try:
		keywords_list = ast.literal_eval(list_str)
		
		if not isinstance(keywords_list, list):
			if verbose:
				print(f"[ERROR] Parsed result is not a list: {type(keywords_list)}")
			return None
		
		if verbose:
			print(f"[STEP 3] Successfully parsed list with {len(keywords_list)} items:")
			for i, kw in enumerate(keywords_list, 1):
				print(f"  [{i}] {repr(kw)}")
			print()
	except Exception as e:
		if verbose:
			print(f"[ERROR] Failed to parse list: {e}")
			print(f"[ERROR] Problematic string: {list_str}")
		return None
	
	# Step 4: Post-process keywords
	if verbose:
		print(f"[STEP 4] Post-processing keywords (max={max_kws})...")
	
	processed = []
	seen = set()
	
	for idx, kw in enumerate(keywords_list, 1):
		if verbose:
			print(f"\n  Processing [{idx}/{len(keywords_list)}]: {repr(kw)}")
		
		# Check if empty
		if not kw or not str(kw).strip():
			if verbose:
				print(f"    ✗ Skipped: empty/whitespace")
			continue
		
		# Normalize
		cleaned = re.sub(r'\s+', ' ', str(kw).strip())
		
		if verbose:
			print(f"    → Cleaned: {repr(cleaned)}")
		
		# Check length
		if len(cleaned) < 2:
			if verbose:
				print(f"    ✗ Skipped: too short (len={len(cleaned)})")
			continue
		
		# Check stopwords
		if cleaned.lower() in STOPWORDS:
			if verbose:
				print(f"    ✗ Skipped: stopword")
			continue
		
		# Check for duplicates (case-insensitive)
		normalized = cleaned.lower()
		if normalized in seen:
			if verbose:
				print(f"    ✗ Skipped: duplicate")
			continue
		
		# Accept keyword
		seen.add(normalized)
		processed.append(cleaned)
		
		if verbose:
			print(f"    ✓ Accepted (total: {len(processed)})")
		
		# Check max limit
		if len(processed) >= max_kws:
			if verbose:
				print(f"\n  [LIMIT] Reached max_kws={max_kws}, stopping")
			break
	
	# Step 5: Return results
	if verbose:
		print(f"[RESULT] Final keywords ({len(processed)}/{len(keywords_list)} kept):")
		for i, kw in enumerate(processed, 1):
			print(f"  [{i}] {kw}")
	
	return processed if processed else None

def _nousresearch_llm_response(model_id: str, input_prompt: str, llm_response: str, max_kws: int, verbose: bool = False):
		print(f"Handling NousResearch response model_id: {model_id}...")
		print(f"Raw response (repr): {repr(llm_response)}")  # Debug hidden characters
		
		# Strip code block markers (```python
		cleaned_response = re.sub(r'```python\n|```', '', llm_response)
		print(f"Cleaned response (repr): {repr(cleaned_response)}")  # Debug
		
		# Look for a list with three quoted strings after [/INST]
		list_match = re.search(
				r"\[/INST\][\s\S]*?(\[\s*['\"][^'\"]*['\"](?:\s*,\s*['\"][^'\"]*['\"]){2}\s*\])",
				cleaned_response, re.DOTALL
		)
		
		if list_match:
				potential_list = list_match.group(1)
				print(f"Found potential list: '{potential_list}'")
		else:
				print("Error: Could not find any complete list patterns after [/INST].")
				# try any three-item list
				list_match = re.search(
						r"\[\s*['\"][^'\"]*['\"](?:\s*,\s*['\"][^'\"]*['\"]){2}\s*\]",
						cleaned_response, re.DOTALL
				)
				if list_match:
						potential_list = list_match.group(0)
						print(f"Fallback list: '{potential_list}'")
				else:
						print("Error: No list found in response.")
						# Debug all bracketed matches
						matches = re.findall(r"\[.*?\]", cleaned_response, re.DOTALL)
						print(f"All bracketed matches: {matches}")
						return None
		
		# Clean the string - replace smart quotes and normalize
		cleaned_string = potential_list.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
		print(f"Cleaned string: '{cleaned_string}'")
		
		# Use ast.literal_eval to parse the string into a Python list
		try:
				keywords_list = ast.literal_eval(cleaned_string)
				# Validate: must be a list of strings
				if not (isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list)):
						print("Error: Extracted string is not a valid list of strings.")
						return None
				
				# Post-process: remove numbers, special characters, and duplicates
				processed_keywords = []
				for keyword in keywords_list:
						cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
						cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)
						if cleaned_keyword and cleaned_keyword not in processed_keywords:
								processed_keywords.append(cleaned_keyword)
				
				if len(processed_keywords) > max_kws:
					processed_keywords = processed_keywords[:max_kws]
				if not processed_keywords:
					if verbose:
						print("Error: No valid keywords found after processing.")
					return None
				if verbose:
					print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
				return processed_keywords
		
		except Exception as e:
				print(f"Error parsing the list: {e}")
				print(f"Problematic string: '{cleaned_string}'")
				# Fallback: extract quoted strings
				try:
						manual_matches = re.findall(r"['\"]([^'\"]+)['\"]", cleaned_string)
						if manual_matches:
								print(f"Using fallback extraction: {manual_matches}")
								return manual_matches[:3]
						return None
				except Exception as e:
						print(f"Fallback extraction failed: {e}")
						return None

def _llama_llm_response(model_id: str, input_prompt: str, llm_response: str, max_kws: int, verbose: bool = True):
		def _normalize_text(s: str) -> str:
				"""Normalize text for comparison (case folding, unicode normalization)."""
				s = unicodedata.normalize("NFKD", s or "")
				s = "".join(ch for ch in s if not unicodedata.combining(ch))
				return s.lower()

		def _token_clean(s: str) -> str:
				"""Clean and normalize whitespace."""
				return re.sub(r"\s+", " ", (s or "").strip())

		def _has_letter(s: str) -> bool:
				"""Check if string contains letters."""
				return bool(re.search(r"[A-Za-z]", s or ""))

		def _is_punct_only(s: str) -> bool:
				"""Check if string contains only punctuation."""
				return bool(s) and bool(re.fullmatch(r"[\W_]+", s))

		def _looks_like_abbrev(s: str) -> bool:
				"""Heuristic to detect real abbreviations, not normal capitalized words."""
				# Skip if it's a normal multi-word phrase
				if ' ' in s:
						return False
						
				# Real abbreviation patterns
				if re.search(r"[.&/]", s):
						return True
						
				# Very short ALL-CAPS (2-3 chars) like "USA", "UK", "SP"
				if len(s) <= 3 and s.isalpha() and s.isupper():
						return True
						
				# Mixed case words are not abbreviations
				if not s.isupper():
						return False
						
				# For longer ALL-CAPS words, only flag if they look like acronyms
				# (all caps with no vowels or very short)
				if len(s) >= 4 and s.isupper():
						# Check if it has vowels - if no vowels, likely acronym
						if not re.search(r'[AEIOUaeiou]', s):
								return True
						# Otherwise, it's probably just a capitalized normal word
						return False
						
				return False

		# Temporal words to exclude
		TEMPORAL_WORDS = {
				"morning", "evening", "night", "noon", "midnight", "today", "yesterday", "tomorrow",
				"spring", "summer", "autumn", "fall", "winter", "weekend", "weekday",
				"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
				"january", "february", "march", "april", "may", "june", "july", "august", 
				"september", "october", "november", "december", "century", "centuries", 
				"year", "years", "month", "months", "day", "days", "decade", "decades", "time", "times"
		}

		def _is_temporal_word(s: str) -> bool:
				"""Check if word is temporal."""
				return _normalize_text(s) in TEMPORAL_WORDS

		def _is_valid_form(s: str) -> bool:
				"""Validate keyword form."""
				if not s:
						return False
				if len(s) < 2:
						return False
				if _is_punct_only(s):
						return False
				if re.search(r"\d", s):
						return False
				if not _has_letter(s):
						return False
				if _is_temporal_word(s):
						return False
						
				tokens = [t for t in re.split(r"\s+", s) if t]
				if tokens and all(_normalize_text(t) in STOPWORDS for t in tokens):
						return False
						
				if _looks_like_abbrev(s):
						return False
						
				return True

		def _appears_in_description(candidate: str, description_text: str) -> bool:
				"""Check if keyword appears in original description with flexible matching."""
				if not description_text:
						return True
						
				cand_norm = _normalize_text(candidate)
				desc_norm = _normalize_text(description_text)
				
				# Direct substring match
				if cand_norm in desc_norm:
						return True
						
				# Word-level matching for multi-word keywords
				cand_words = cand_norm.split()
				desc_words = desc_norm.split()
				
				# Check if all words in candidate appear in description
				if all(any(cand_word in desc_word for desc_word in desc_words) for cand_word in cand_words):
						return True
						
				# Check for partial matches
				for desc_word in desc_words:
						if any(cand_word in desc_word for cand_word in cand_words):
								return True
								
				return False

		# ---------- Response Extraction Utilities ----------
		def _after_last_inst(text: str) -> str:
				"""Extract content after last [/INST] tag."""
				matches = list(re.finditer(r"\[/INST\]", text or ""))
				if not matches:
						return (text or "").strip()
				return text[matches[-1].end():].strip()

		def _before_last_inst(text: str) -> str:
				"""Extract content before last [/INST] tag."""
				matches = list(re.finditer(r"\[/INST\]", text or ""))
				if not matches:
						return (text or "").strip()
				return text[:matches[-1].start()].strip()

		def _strip_codeblocks(s: str) -> str:
				"""Remove code blocks from text."""
				s = re.sub(r"```.*?```", "", s or "", flags=re.DOTALL)
				s = re.sub(r"`[^`]*`", "", s)
				return s

		# ---------- Parsing Strategies ----------
		def _parse_python_lists(s: str):
				"""Find and parse Python list literals."""
				pattern = r"\[\s*(?:(['\"])(?:(?:(?!\1).)*)\1\s*(?:,\s*(['\"])(?:(?:(?!\2).)*)\2\s*)*)?\]"
				for m in re.finditer(pattern, s or "", flags=re.DOTALL):
						yield s[m.start():m.end()]

		def _parse_bullet_lists(s: str) -> List[str]:
				"""Parse various bullet list formats."""
				items = []
				for line in (s or "").splitlines():
						line = line.strip()
						if not line:
								continue
						
						# Match various bullet formats: -, *, •, numbers
						bullet_match = re.match(r"^([-*•\u2022\u2023\u2043]|\d+[\.\)])\s+(.+)", line)
						if bullet_match:
								item = bullet_match.group(2).strip()
								# Clean up any trailing explanations
								item = re.sub(r"\s*[-–—].*$", "", item)
								item = re.sub(r"\s*[.:].*$", "", item)
								# Remove markdown formatting
								item = re.sub(r"\*\*", "", item)
								item = re.sub(r"\*", "", item)
								if item and len(item) > 1:
										items.append(item)
				return items

		def _postprocess_keywords(candidates: List[str], description_text: str) -> List[str]:
				"""Post-process and validate keywords."""
				out = []
				seen = set()
				
				for kw in candidates:
						kw = _token_clean(kw)
						
						if not _is_valid_form(kw):
								continue
						if not _appears_in_description(kw, description_text):
								continue
						
						key = _normalize_text(kw)
						if key in seen:
								continue
								
						seen.add(key)
						out.append(kw)
						
						if len(out) >= max_kws:
								break
								
				return out

		def _extract_description(text: str) -> str:
				"""Extract the original description from prompt."""
				pre = _before_last_inst(text or "")
				
				# Try to find the description before rules
				patterns = [
						r"Given the description below[^:\n]*[:\n]\s*(.*?)\n\s*\*\*Rules",
						r"Given the description below[^:\n]*[:\n]\s*(.*)",
				]
				
				for pat in patterns:
						m = re.search(pat, pre, flags=re.DOTALL | re.IGNORECASE)
						if m:
								candidate = m.group(1).strip()
								if len(candidate) >= 15:
										return candidate
				
				# Fallback: take last paragraph before rules
				parts = re.split(r"\*\*Rules\*\*", pre, flags=re.IGNORECASE)
				head = parts[0] if parts else pre
				paras = [p.strip() for p in re.split(r"\n{2,}", head) if p.strip()]
				if paras:
						return paras[-1]
						
				return ""

		# ---------- Main Parsing Logic ----------
		if verbose:
				print("=" * 100)
				print(f"Processing Llama response from: {model_id}")
				print("=" * 100)

		# Extract description for validation
		desc_for_validation = input_prompt or ""
		if (not desc_for_validation) or ("[INST]" in desc_for_validation) or ("**Rules**" in desc_for_validation):
				extracted_desc = _extract_description(llm_response or "")
				if extracted_desc:
						desc_for_validation = extracted_desc
						if verbose:
								print(f"Recovered description: {desc_for_validation}")
				else:
						if verbose:
								print("Using provided description as-is.")

		# Get content after last [/INST] tag
		content_after = _after_last_inst(llm_response or "")
		if verbose:
				print(f"Content after [/INST]: {repr(content_after[:200])}...")

		# Clean content
		content_clean = _strip_codeblocks(content_after)

		# Strategy 1: Python List Literals
		if verbose:
				print("\n[STRATEGY 1] Searching for Python lists...")
		
		list_candidates = list(_parse_python_lists(content_after))
		for list_str in reversed(list_candidates):
				try:
						cleaned = list_str.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
						parsed = ast.literal_eval(cleaned)
						if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
								result = _postprocess_keywords(parsed, desc_for_validation)
								if result:
										if verbose:
												print(f"✓ Found Python list: {result}")
										return result
				except Exception as e:
						if verbose:
								print(f"  List parse failed: {e}")

		# Strategy 2: Bullet Lists
		if verbose:
				print("\n[STRATEGY 2] Searching for bullet lists...")
		
		bullets = _parse_bullet_lists(content_clean)
		if bullets:
				if verbose:
						print(f"  Found {len(bullets)} bullet items")
				result = _postprocess_keywords(bullets, desc_for_validation)
				if result:
						if verbose:
								print(f"✓ Extracted from bullets: {result}")
						return result

		if verbose:
				print("\n❌ No valid keywords extracted")
		
		return None

def query_local_llm(
		model: tfs.PreTrainedModel,
		tokenizer: tfs.PreTrainedTokenizer, 
		text: str, 
		device: str,
		max_generated_tks: int,
		max_kws: int,
		verbose: bool = False,
	) -> List[str]:

	start_time = time.time()
	
	if not isinstance(text, str) or not text.strip():
		return None

	keywords: Optional[List[str]] = None
	prompt = get_prompt(tokenizer=tokenizer, description=text, max_kws=max_kws)

	model_id = getattr(model.config, '_name_or_path', None)
	if model_id is None:
		model_id = getattr(model, 'name_or_path', 'unknown_model')

	tokenization_start = time.time()
	try:
		inputs = tokenizer(
			prompt,
			return_tensors="pt", 
			truncation=True, 
			max_length=4096, 
			padding=True
		)
		if device != 'cpu':
			inputs = {k: v.to(device) for k, v in inputs.items()}

		if "token_type_ids" in inputs and not hasattr(model.config, "type_vocab_size"):
			inputs.pop("token_type_ids")
		
		tokenization_time = time.time() - tokenization_start
		if verbose: 
			print(f"Tokenization: {tokenization_time:.5f}s")

		generation_start = time.time()
		with torch.no_grad():
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
			):
				outputs = model.generate(
					**inputs,
					max_new_tokens=max_generated_tks,
					temperature=TEMPERATURE,
					top_p=TOP_P,
					do_sample=TEMPERATURE > 0.0,
					pad_token_id=tokenizer.pad_token_id,
					eos_token_id=tokenizer.eos_token_id,
					use_cache=True,
				)
		generation_time = time.time() - generation_start
		
		if verbose:
			print(f"Model generation: {generation_time:.5f}s")

		decoding_start = time.time()
		raw_llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
		
		if verbose: 
			print(f"Decoding: {time.time() - decoding_start:.5f}s")
	except Exception as e:
		print(f"<!> Error {e}")
		return None

	if verbose:
		output_tokens = get_conversation_token_breakdown(raw_llm_response, model_id)
		print(f"[INFO] Token breakdown: {output_tokens}")
	
	parsing_start = time.time()
	keywords = parse_llm_response(
		model_id=model_id, 
		input_prompt=prompt, 
		raw_llm_response=raw_llm_response,
		max_kws=max_kws,
		verbose=verbose,
	)
	parsing_time = time.time() - parsing_start
	if verbose: print(f"Response parsing: {parsing_time:.5f}s")

	# ⏱️ FILTERING TIMING
	filtering_start = time.time()
	if keywords:
		filtered_keywords = [
			kw 
			for kw in keywords 
			if kw not in re.sub(r'[^\w\s]', '', LLM_INSTRUCTION_TEMPLATE).split() # remove punctuation and split
		]
		if not filtered_keywords:
			return None
		keywords = filtered_keywords
	filtering_time = time.time() - filtering_start
	if verbose: print(f"⏱️ Keyword filtering: {filtering_time:.5f}s")

	total_time = time.time() - start_time
	if verbose: print(f"TOTAL execution time: {total_time:.2f}s")
	
	return keywords

def get_llm_based_labels_debug(
	model_id: str, 
	device: str, 
	max_generated_tks: int,
	max_kws: int,
	csv_file: str=None,
	description: str=None,
	use_quantization: bool = False,
	verbose: bool = False,
) -> List[List[str]]:

	if csv_file and description:
		raise ValueError("Only one of csv_file or description must be provided")

	if csv_file:
		df = pd.read_csv(
			filepath_or_buffer=csv_file, 
			on_bad_lines='skip', 
			dtype=dtypes, 
			low_memory=False,
		)
		if 'enriched_document_description' not in df.columns:
			raise ValueError("CSV file must have 'enriched_document_description' column")
		if verbose: print(f"Loading descriptions from {csv_file} ...")
		descriptions = df['enriched_document_description'].tolist()
	elif description:
		descriptions = [description]
	else:
		raise ValueError("Either csv_file or description must be provided")
	
	if verbose: 
		print(f"{'-'*100}\nLoaded {len(descriptions)} {type(descriptions)} description(s)")

	if len(descriptions) == 0:
		print("No descriptions to process. Exiting...")
		return None

	tokenizer, model = _load_llm_(
		model_id=model_id, 
		use_quantization=use_quantization,
		verbose=verbose
	)

	all_keywords = list()
	for i, desc in tqdm(enumerate(descriptions), total=len(descriptions), desc="Processing descriptions"):
		if verbose: print(f"Processing description {i+1}/{len(descriptions)}: {desc}")
		if pd.notna(desc) and str(desc).strip():
			desc_str = str(desc).strip()
			kws = query_local_llm(
				model=model, 
				tokenizer=tokenizer, 
				text=desc_str,
				device= device,
				max_generated_tks=max_generated_tks,
				max_kws=min(max_kws, len(desc_str.split())),
				verbose=verbose,
			)
			all_keywords.append(kws)
		else:
			if verbose: print(f"Skipping empty description {i+1}/{len(descriptions)}")
			all_keywords.append(None)

	if csv_file:
		output_csv = csv_file.replace(".csv", "_llm_keywords.csv")
		df['llm_keywords'] = all_keywords
		df.to_csv(output_csv, index=False)
		try:
			df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")
		if verbose:
			print(f"Saved {len(all_keywords)} keywords to {output_csv}")
			print(f"Done! dataframe: {df.shape} {list(df.columns)}")

	return all_keywords

def get_llm_based_labels(
	model_id: str,
	device: str,
	batch_size: int,
	max_generated_tks: int,
	max_kws: int,
	csv_file: str,
	num_workers: int,
	do_dedup: bool = True,
	max_retries: int = 2,
	use_quantization: bool = False,
	verbose: bool = False,
) -> List[Optional[List[str]]]:

	output_csv = csv_file.replace(".csv", "_llm_keywords.csv")
	if os.path.exists(output_csv):
		if verbose:
			print(f"[EXISTING] Found existing results at {output_csv}")
		df = pd.read_csv(
			filepath_or_buffer=output_csv,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
		if 'llm_keywords' in df.columns:
			if verbose:
				print(f"[EXISTING] Found existing LLM keywords! {type(df)} {df.shape} {list(df.columns)}")
			return df['llm_keywords'].tolist()
	
	num_workers = min(os.cpu_count(), num_workers)
	if verbose:
		print(f"[INIT] Starting OPTIMIZED batch LLM processing with {num_workers} workers")
		print(f"[INIT] Model: {model_id}")
		print(f"[INIT] Batch size: {batch_size}")
		print(f"[INIT] Device: {device}")

	st_t = time.time()
		
	try:
		df = pd.read_csv(
			filepath_or_buffer=csv_file,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
	except pd.errors.ParserError as e:
		if verbose:
			print(f"CSV parsing error, trying with python engine: {e}")
		df = pd.read_csv(
			filepath_or_buffer=csv_file,
			on_bad_lines='skip',
			dtype=dtypes,
			engine='python',
		)

	if 'enriched_document_description' not in df.columns:
		raise ValueError("CSV file must have 'enriched_document_description' column")
	
	descriptions = df['enriched_document_description'].tolist()
	if verbose:
		print(f"<> Loaded {type(df)} {df.shape} with {len(descriptions)} descriptions")
	
	inputs = descriptions
	if len(inputs) == 0:
		return None
	
	# Load tokenizer and model
	
	tokenizer, model = _load_llm_(
		model_id=model_id,
		use_quantization=use_quantization,
		verbose=verbose,
	)
	if verbose:
		valid_count = sum(
			1 for x in inputs
			if x is not None and str(x).strip() not in ("", "nan", "None")
		)
		null_count = len(inputs) - valid_count
		print(f"📊 Input stats: {type(inputs)} {len(inputs)} total, {valid_count} valid, {null_count} null")
	
	# NULL-SAFE DEDUPLICATION
	
	if do_dedup:
			unique_map: Dict[str, int] = {}
			unique_inputs: List[Optional[str]] = []
			original_to_unique_idx: List[int] = []
			for s in inputs:
					if s is None or str(s).strip() in ("", "nan", "None"):
							key = "__NULL__"
					else:
							key = str(s).strip()
					if key in unique_map:
							original_to_unique_idx.append(unique_map[key])
					else:
							idx = len(unique_inputs)
							unique_map[key] = idx
							unique_inputs.append(None if key == "__NULL__" else key)
							original_to_unique_idx.append(idx)
	else:
			unique_inputs = []
			for s in inputs:
					if s is None or str(s).strip() in ("", "nan", "None"):
							unique_inputs.append(None)
					else:
							unique_inputs.append(str(s).strip())
			original_to_unique_idx = list(range(len(unique_inputs)))
	
	# Build prompts
	unique_prompts: List[Optional[str]] = []
	for s in unique_inputs:
		if s is None:
			unique_prompts.append(None)
		else:
			if verbose:
				print(f"Generating prompt for text with len={len(s.split()):<10}max_kws={min(max_kws, len(s.split()))}")
			prompt = get_prompt(
				tokenizer=tokenizer,
				description=s,
				max_kws=min(max_kws, len(s.split())),
			)
			unique_prompts.append(prompt)
	
	unique_results: List[Optional[List[str]]] = [None] * len(unique_prompts)
	
	valid_indices = [i for i, p in enumerate(unique_prompts) if p is not None]
	if not valid_indices:
		if verbose:
			print(f" <!> No valid prompts found after deduplication => exiting")
		return None
	total_batches = math.ceil(len(valid_indices) / batch_size)

	if verbose:
		print(
			f"Processing {len(valid_indices)} unique prompts "
			f"in batches of {batch_size} samples => {total_batches} batches"
		)
	
	def _parse_batch_parallel(
		decoded_batch: List[str],
		batch_indices: List[int],
		batch_prompts: List[str],
		model_id_: str,
		max_kws_: int,
		verbose_: bool,
	) -> Dict[int, Optional[List[str]]]:
		results_dict: Dict[int, Optional[List[str]]] = {}
		
		def _parse_one(local_i: int) -> Tuple[int, Optional[List[str]]]:
			idx = batch_indices[local_i]
			try:
				parsed = parse_llm_response(
					model_id=model_id_,
					input_prompt=batch_prompts[local_i],
					raw_llm_response=decoded_batch[local_i],
					max_kws=max_kws_,
					verbose=verbose_,
				)
				return idx, parsed
			except Exception as e:
				if verbose_:
					print(f"⚠️ Parsing error for batch index {idx}: {e}")
				return idx, None
		
		with ThreadPoolExecutor(max_workers=num_workers) as executor:
			futures = {executor.submit(_parse_one, i): i for i in range(len(decoded_batch))}
			for future in as_completed(futures):
				idx, parsed = future.result()
				results_dict[idx] = parsed
		
		return results_dict

	# Batching: generate + parse
	batches: List[Tuple[List[int], List[str]]] = []
	for i in tqdm(range(0, len(valid_indices), batch_size), desc="Batching prompts", ncols=100):
		batch_indices = valid_indices[i:i + batch_size]
		batch_prompts = [unique_prompts[idx] for idx in batch_indices]
		batches.append((batch_indices, batch_prompts))
	
	if verbose:
		print(f"Batched {len(batches)} prompts into {len(batches)} batches of {batch_size} samples")

	for batch_num, (batch_indices, batch_prompts) in enumerate(tqdm(batches, desc="Processing (textual) batches", ncols=100)):
		# Retry whole batch on failure (e.g., OOM or generation error)
		for attempt in range(max_retries + 1):
			if attempt > 0 and verbose:
				print(f"🔄 Retry attempt {attempt + 1}/{max_retries + 1} for batch {batch_num + 1}")
			try:
				tokenized = tokenizer(
					batch_prompts,
					return_tensors="pt",
					truncation=True,
					max_length=4096,
					padding=True,
				)
				if device != 'cpu':
					tokenized = {k: v.to(device) for k, v in tokenized.items()}

				gen_kwargs = dict(
					input_ids=tokenized.get("input_ids"),
					attention_mask=tokenized["attention_mask"],
					max_new_tokens=max_generated_tks,
					do_sample=TEMPERATURE > 0.0,
					temperature=TEMPERATURE,
					top_p=TOP_P,
					pad_token_id=tokenizer.pad_token_id,
					eos_token_id=tokenizer.eos_token_id,
				)
				# Generate response
				with torch.no_grad():
					with torch.amp.autocast(
						device_type=device.type,
						enabled=torch.cuda.is_available(),
						dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
					):
						outputs = model.generate(**gen_kwargs)
				decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
				
				# Parallel parsing per batch
				parsed_dict = _parse_batch_parallel(
					decoded_batch=decoded,
					batch_indices=batch_indices,
					batch_prompts=batch_prompts,
					model_id_=model_id,
					max_kws_=max_kws,
					verbose_=verbose,
				)
				# Assign results back to unique_results
				for idx, parsed in parsed_dict.items():
					unique_results[idx] = parsed
				# Successful batch => break out of retry loop
				break
			except Exception as e:
				print(f"❌ Batch {batch_num + 1} attempt {attempt + 1} failed:\n{e}")
				if attempt < max_retries:
					sleep_time = EXP_BACKOFF ** attempt
					print(f"⏳ Waiting {sleep_time}s before retry...")
					time.sleep(sleep_time)
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				else:
					print(f"💥 Batch {batch_num + 1} failed after {max_retries + 1} attempts")
					for idx in batch_indices:
						unique_results[idx] = None
		
		# Clean up batch tensors immediately after use
		try:
			del tokenized
		except NameError:
			pass
		try:
			del outputs
		except NameError:
			pass
		try:
			del decoded
		except NameError:
			pass
		
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
					f"[MEM] BATCH {batch_num} (GPU {device_idx}): {mem_usage_pct:.2f}% usage: "
					f"{mem_allocated:.2f}GB alloc / {mem_reserved:.2f}GB reserved (Total: {mem_total:.1f}GB)"
				)
			cleanup_threshold = 90
			if mem_usage_pct > cleanup_threshold: 
				need_cleanup = True
				memory_consumed_percent += mem_usage_pct

		if need_cleanup:
			print(f"[WARN] High memory usage ({memory_consumed_percent:.1f}%). Clearing cache...")
			torch.cuda.empty_cache() # clears all GPUs
			gc.collect()

	# HYBRID FALLBACK: Retry failed items individually with query_local_llm
	failed_indices = [
		i
		for i, result in enumerate(unique_results)
		if result is None and unique_inputs[i] is not None
	]

	if failed_indices and verbose:
		print(f"🔄 Retrying {len(failed_indices)} failed items individually using query_local_llm [sequential processing]...")
	
	for idx in failed_indices:
		desc = unique_inputs[idx]
		if verbose:
			print(f"Retrying individual item {idx}:\n{desc}\n")
		try:
			individual_result = query_local_llm(
				model=model,
				tokenizer=tokenizer,
				text=desc,
				device=device,
				max_generated_tks=max_generated_tks,
				max_kws=min(max_kws, len(desc.split())),
				verbose=verbose,
			)
			unique_results[idx] = individual_result
			if verbose and individual_result:
				print(f"✅ Individual retry successful: {individual_result}")
			elif verbose:
				print(f"❌ Individual retry failed for item {idx}:\n{desc}\n")
		except Exception as e:
			if verbose:
				print(f"💥 Individual retry error for item {idx}: {e}")
			unique_results[idx] = None

	# Map unique_results back to original order
	results: List[Optional[List[str]]] = []
	for orig_i, uniq_idx in tqdm(enumerate(original_to_unique_idx), desc="Mapping results", ncols=150,):
		results.append(unique_results[uniq_idx])
	
	# Stats
	if verbose:
		stats_start = time.time()
		n_ok = 0
		n_null = 0
		for inp, res in zip(inputs, results):
			if res is not None:
				n_ok += 1
			if inp is None or str(inp).strip() in ("", "nan", "None"):
				n_null += 1
		total_results = len(results)
		valid_inputs_count = total_results - n_null
		n_failed = valid_inputs_count - n_ok
		success_rate = (n_ok / valid_inputs_count) * 100 if valid_inputs_count > 0 else 0
		print(
			f"[STATS] {n_ok}/{valid_inputs_count} successful ({success_rate:.1f}%) "
			f"{n_null} null inputs {n_failed} failed "
			f"Elapsed_t: {time.time() - stats_start:.2f}s"
		)
	
	# Cleanup model and tokenizer
	if verbose:
		print(f"Cleaning up model and tokenizer...")
	del model, tokenizer
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	
	
	# Save results
	if csv_file:
		output_csv = csv_file.replace(".csv", "_llm_keywords.csv")
		if verbose:
			print(f"Saving results to {output_csv}...")
		df['llm_keywords'] = results
		df.to_csv(output_csv, index=False)
		try:
			df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")
		if verbose:
			print(f"Saved {len(results)} keywords to {output_csv} {df.shape}\n{list(df.columns)}")
	if verbose:
		print(f"Total LLM-based keyword extraction time: {time.time() - st_t:.1f} sec")
	return results

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="LLM-instruct-based keyword annotation for Historical Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, help="Path to the metadata CSV file")
	parser.add_argument("--model_id", '-llm', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
	parser.add_argument("--device", '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--description", '-desc', type=str, help="Description")
	parser.add_argument("--num_workers", '-nw', type=int, default=12, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=32, help="Batch size for processing (adjust based on GPU memory)")
	parser.add_argument("--max_generated_tks", '-mgt', type=int, default=256, help="Max number of generated tokens")
	parser.add_argument("--max_keywords", '-mkw', type=int, default=5, help="Max number of keywords to extract")
	parser.add_argument("--use_quantization", '-q', action='store_true', help="Use quantization")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	parser.add_argument("--debug", '-d', action='store_true', help="Debug mode")
	args = parser.parse_args()

	set_seeds(seed=42, debug=args.debug)
	args.device = torch.device(args.device)
	args.num_workers = min(args.num_workers, os.cpu_count())

	print(args)

	if args.debug or args.description:
		keywords = get_llm_based_labels_debug(
			model_id=args.model_id, 
			device=args.device,
			max_generated_tks=args.max_generated_tks,
			max_kws=args.max_keywords,
			csv_file=args.csv_file,
			description=args.description,
			use_quantization=args.use_quantization,
			verbose=args.verbose,
		)
	elif args.csv_file:
		keywords = get_llm_based_labels(
			model_id=args.model_id,
			device=args.device,
			batch_size=args.batch_size,
			max_generated_tks=args.max_generated_tks,
			max_kws=args.max_keywords,
			csv_file=args.csv_file,
			num_workers=args.num_workers,
			use_quantization=args.use_quantization,
			verbose=args.verbose,
		)
	else:
		print("Either --csv_file or --description must be provided")
		return

	if args.verbose and keywords:
		print(f"{len(keywords)} Extracted keywords")
		for i, kw in enumerate(keywords):
			print(f"{i:03d} {kw}")

if __name__ == "__main__":
	main()