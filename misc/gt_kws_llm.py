from utils import *
from nlp_utils import get_enriched_description

# how to run [local]:
# python gt_kws_llm.py -csv /home/farid/datasets/WW_DATASETs/HISTORY_X4/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -qb 8 -v -bs 2

# with description:
# small model for local testing:
# python gt_kws_llm.py -desc "Exhausted Marine weeping atop of Hill 200" -llm "Qwen/Qwen3.5-4B" -qb 4 -v

# large model:
# python gt_kws_llm.py -desc "Miltiano flag marching towards the Aragon front with the first columns of fighters, in Barcelona. A young militia officer with an abadera amongst the first columns of Republican fighters on his way to the front of Zaragoza, Barcelona." -llm "Qwen/Qwen3.5-122B-A10B" -v

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

# Hyperparameter defaults from module scope (falls back if undefined)
TEMPERATURE = 1e-8
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
RETRY_BATCH_SIZE = 8       # Safe micro-batch size for retries
RETRY_MAX_LENGTH = 4096    # Full prompt length preserved to avoid cutting off captions

# STOPWORDS = set(nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())) # all languages
STOPWORDS = set(nltk.corpus.stopwords.words('english')) # english only
# custom_stopwords_list = requests.get("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/refs/heads/master/stopwords-en.txt").content
# stopwords = set(custom_stopwords_list.decode().splitlines())
with open('meaningless_words.txt', 'r') as file_:
	stopwords = set([line.strip().lower() for line in file_])
STOPWORDS.update(stopwords)

with open('geographic_references.txt', 'r') as file_:
	geographic_references = set([line.strip().lower() for line in file_ if line.strip()])
STOPWORDS.update(geographic_references)

PROMPT_TEMPLATE = """Extract no more than {k} keywords.
Keywords must be semantically atomic, visually grounded, and broad with absolute maximum degree of breadth.
Return your response as a Python list of double-quoted strings containing keywords derived strictly from the caption without any reasoning, thinking, or explanation.
Opt for fewer keywords if the caption is short or lacks sufficient information.
Returning fewer keywords — or an empty list [] — is always better than returning one excluded term.

EXCLUDE:
  - Generic war terms ('World War I', 'Vietnam War', 'post war era', 'Post-war', 'aftermath of World War II', 'War', 'battle').
  - Quantities, counts, measurements, or numeric expressions (1 1/2 ton truck, 1 kilovolt, 7.3mm, 3 Dodge trucks).
  - Equipment identifiers, serial numbers, brands, or models.
  - Dates, times, years, decades, or any temporal references.
  - Names of locations, places, buildings, or structures (Plaza de Santiago, St. Louis Cathedral).
  - Individual people's names or honorifics (A. A. Robinson, A. Philip Randolph, Barbara Briggs, Allan M. Hardy, Josef Dietrich, Mrs. Howard Russell). 
  - Family relationship terms (mother, father, son, uncle).
  - Generic human category nouns (man, men, woman, person, people, children).
  - Geographical names such as continents, countries, states, provinces, cities, towns, islands, regions, roads, or landmarks.
  - Ordinal numeral keywords (fourth, 1st, 115th).
  - Roman numerals (I, II, IV, VIII).
  - Nationalities, ethnicities, or religions.
  - Misspelled keywords or non-standard spellings.
  - Acronyms, phrasal verbs, possessive constructions, or descriptive clauses.
  - Underscores, snake_case, camelCase, kebab-case, slashes, or punctuation to join words.

Color handling:
  - Remove color only if it is purely descriptive (white truck, blue sky).
  - Preserve color terms when they are part of a standardized or semantic label (Red Cross, Blue Cross gas shell, Green Berets).

Caption: {caption}"""

def _load_llm_(
	model_id: str,
	quantization_bits: Optional[int] = None,
	force_multi_gpu: bool = False,
	verbose: bool = False,
):
	if verbose:
		print(f"\n{'='*110}")
		print(f"[LOADING] {model_id} on cache_dir: {cache_directory.get(USER)}")
	
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
			print(f"[LOGIN INFO] HuggingFace Hub...")
		huggingface_hub.login(token=hf_tk)
	except Exception as e:
		print(f"<!> Failed to login to HuggingFace Hub:\n{e}")
		raise e
	
	# ========== Load config ==========
	config = tfs.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
	if verbose:
		print(f"[INFO] {model_id} Config summary")
		print(f"   • model_type        : {config.model_type}")
		print(f"   • architectures     : {config.architectures}")
		print(f"   • dtype (if set)    : {config.dtype}")
	
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
			use_auto_model = True
			if verbose:
				print(f"[INFO] Custom architecture detected: {cls_name}")
				print(f"[INFO] Will use AutoModelForCausalLM with trust_remote_code=True\n")
	else:
		use_auto_model = True
		if verbose:
			print(f"[INFO] No architecture specified in config")
			print(f"[INFO] Will use AutoModelForCausalLM\n")
	
	dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

	if verbose:
		print(f"[INFO] {model_id} Dtype selection: {dtype}")

	def _optimal_attn_impl() -> str:
		if not torch.cuda.is_available():
			return "eager"
		
		# Bypass flash-attn for gpt-oss to prevent transformers hub_kernels version-locking crashes
		if getattr(config, "model_type", None) == "gpt_oss" or "gpt-oss" in model_id.lower():
			if verbose:
				print("[INFO] gpt_oss detected: defaulting to 'sdpa' to bypass hub `kernels` requirement")
			return "eager"

		# Custom/non-standard architectures often don't support sdpa/flash
		if use_auto_model or (config.architectures and config.architectures[0] not in dir(tfs)):
			if verbose:
				print(f"[INFO] Custom architecture — defaulting to 'eager' attention")
			return "eager"
		
		# model config for FlashAttention dimension limits
		max_head_dim = getattr(config, "head_dim", 0)
		if hasattr(config, "text_config"):
			max_head_dim = max(max_head_dim, getattr(config.text_config, "head_dim", 0))
			max_head_dim = max(max_head_dim, getattr(config.text_config, "global_head_dim", 0))
		
		major, minor = torch.cuda.get_device_capability()
		compute_cap = major + minor / 10
		if compute_cap >= 8.0:
			if max_head_dim <= 256:
				try:
					import flash_attn
					if verbose: print(f"[INFO] Flash Attention 2 available (compute {compute_cap})")
					return "flash_attention_2"
				except ImportError:
					if verbose: print(f"[WARN] Flash Attention 2 not installed")
			else:
				if verbose: print(f"[INFO] Bypassing Flash Attention 2: max head_dim ({max_head_dim}) > 256")
		
		# ── SDPA: probe whether this architecture actually supports it ──
		if compute_cap >= 7.0 and torch.__version__ >= "2.0.0":
			sdpa_supported = getattr(model_cls, "_supports_sdpa", False)
			if sdpa_supported:
				if verbose: print(f"[INFO] Using SDPA attention (compute {compute_cap}, PyTorch {torch.__version__})")
				return "sdpa"
			else:
				if verbose: print(f"[INFO] {config.architectures[0]} does not declare _supports_sdpa — falling back to 'eager'")

		return "eager"

	attn_impl = _optimal_attn_impl()

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
		if verbose: print(f"[WARN] AutoTokenizer failed: {exc}. Trying fallbacks...")
		fallback_exc = None
		candidate_tokenizer_classes = [
			getattr(tfs, "MistralTokenizer", None), 
			getattr(tfs, "MistralTokenizerFast", None),
			getattr(tfs, "LlamaTokenizer", None), 
			getattr(tfs, "LlamaTokenizerFast", None),
		]

		for TokCls in [cls for cls in candidate_tokenizer_classes if cls is not None]:
			try:
				tokenizer = TokCls.from_pretrained(
					model_id, 
					trust_remote_code=True, 
					cache_dir=cache_directory[USER]
				)
				break
			except Exception as e: fallback_exc = e

		if tokenizer is None:
			try:
				tokenizer = tfs.AutoTokenizer.from_pretrained(
					model_id, 
					use_fast=False, 
					trust_remote_code=True, 
					cache_dir=cache_directory[USER]
				)
			except Exception as final_exc:
				raise RuntimeError(f"Failed to load tokenizer for '{model_id}'. Final error: {final_exc}") from final_exc
	
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id
	
	if hasattr(tokenizer, "padding_side") and tokenizer.padding_side is not None:
		tokenizer.padding_side = "left"
	
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
		print(f"\n[INFO] {model_id} Estimated size: {estimated_size_gb:.2f} GB (fp16)")
	
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
				
		# ADAPTIVE BUFFER
		if gpu_vram[0] < 10: vram_buffer_gb = 0.7
		elif gpu_vram[0] < 20: vram_buffer_gb = 2.0
		else: vram_buffer_gb = 3.5

		# Reduce buffer if quantization is used
		if quantization_bits is not None:
			vram_buffer_gb = max(0.5, vram_buffer_gb * 0.5)

		# Adjust estimated size for quantization
		adjusted_size = estimated_size_gb
		if quantization_bits == 8:
			adjusted_size = estimated_size_gb * 0.5
		elif quantization_bits == 4:
			adjusted_size = estimated_size_gb * 0.25
		
		# ========== PRE-FLIGHT VRAM VALIDATION ==========
		INFERENCE_OVERHEAD_MULTIPLIER = 1.2
		required_vram = adjusted_size * INFERENCE_OVERHEAD_MULTIPLIER
		usable_vram = total_vram_available - (n_gpus * vram_buffer_gb)

		if verbose:
			print(f"\n[VRAM CHECK] Pre-flight validation:")
			print(f"\t• Estimated Model size (fp16): {adjusted_size:.2f} GB (with {INFERENCE_OVERHEAD_MULTIPLIER}x overhead): {required_vram:.2f} GB")
			print(f"\t• Available VRAM (total):      {total_vram_available:.1f} GB")
			print(f"\t• Available VRAM (usable):     {usable_vram:.1f} GB ({n_gpus}x GPU(s), {vram_buffer_gb:.1f} GB buffer per GPU)")

		if required_vram > usable_vram:
			print("\n" + "="*80 + "\n❌ INSUFFICIENT VRAM ERROR\n" + "="*80)
			print(f"Model: {model_id}\nRequired: {required_vram:.1f} GB | Available: {usable_vram:.1f} GB")
			if quantization_bits is None:
				print("\nSOLUTIONS: Try setting quantization_bits=8 or quantization_bits=4")
			elif quantization_bits == 8:
				print("\nSOLUTIONS: Try setting quantization_bits=4")
			else:
				print("\nSOLUTIONS: Use a larger GPU or a smaller model.")
			
			raise RuntimeError(f"Insufficient VRAM. Required {required_vram:.2f} GB, found {usable_vram:.2f} GB.")
		
		# Decision: Single GPU vs Multi GPU
		single_gpu_capacity = gpu_vram[0] - vram_buffer_gb
		is_large_model = adjusted_size >= 20
		if verbose:
			print(f"\t• Single GPU capacity: {single_gpu_capacity:.1f} GB (GPU VRAM: {gpu_vram[0]:.1f} GB - {vram_buffer_gb:.1f} GB buffer)")
			print(f"\t• is {model_id} Large? ({adjusted_size:.1f} > 20GB) : {is_large_model}")

		use_single_gpu = (
			not force_multi_gpu and 
			not is_large_model and 
			adjusted_size < single_gpu_capacity * 0.8 and 
			(n_gpus == 1 or adjusted_size < 20)
		)
		
		max_memory = {}
		if use_single_gpu:
			max_memory[0] = f"{max(1, single_gpu_capacity):.0f}GB"
			strategy_desc = f"Single GPU (GPU 0, limit: {max_memory[0]})"
		else:
			for i in range(n_gpus):
				buffer = vram_buffer_gb if i == 0 else (0.5 if gpu_vram[i] < 10 else 2.0)
				if quantization_bits is not None: buffer *= 0.5
				max_memory[i] = f"{max(1, gpu_vram[i] - buffer):.0f}GB"
			strategy_desc = f"Multi-GPU [Model Parallelism] ({n_gpus} GPUs)"
	else:
		strategy_desc = "CPU (no GPUs)"
	
	if verbose: print(f"\n[INFO] Strategy: {strategy_desc}")

	# ========== Model loading kwargs ==========
	model_kwargs: Dict[str, Any] = {
		"low_cpu_mem_usage": True,
		# "low_cpu_mem_usage": False, # avoid illegal memory access during weight materialization
		"trust_remote_code": True,
		"cache_dir": cache_directory[USER],
		"attn_implementation": attn_impl,
		"dtype": dtype,
	}
	
	if quantization_config:
		model_kwargs["quantization_config"] = quantization_config
	
	if n_gpus > 0:
		model_kwargs["device_map"] = "auto"
		model_kwargs["max_memory"] = max_memory
	
		if torch.cuda.is_available():
			cur = torch.cuda.current_device()
			print("[DEBUG] CUDA memory BEFORE model load")
			print(f"   • allocated : {torch.cuda.memory_allocated(cur)//(1024**2)} MiB")
			print(f"   • reserved  : {torch.cuda.memory_reserved(cur)//(1024**2)} MiB\n")

	# ========== Load Model ==========
	if verbose:
		print("-"*70)
		print(f"[LOADING] {model_id}")
		model_loader_name = "AutoModelForCausalLM" if use_auto_model else model_cls.__name__
		print(f"[KWARGS] {model_loader_name}")
		pprint.pprint(model_kwargs)
		print("-"*70)

	loader = tfs.AutoModelForCausalLM if use_auto_model else model_cls
	try:
		model = loader.from_pretrained(model_id, **model_kwargs)
	except (ImportError, ValueError) as e:
		# if verbose: print(f"[ERROR] loading model {model_id}\n{e}")
		err_text = str(e).lower()
		if any(term in err_text for term in ["kernel", "flash", "attn"]) and model_kwargs.get("attn_implementation") != "eager":
			fallback = "sdpa" if model_kwargs["attn_implementation"] != "sdpa" else "eager"
			if verbose:
				print(f"\n[WARN] Failed with attn_implementation='{model_kwargs['attn_implementation']}'. Retrying with '{fallback}'...")
			model_kwargs["attn_implementation"] = fallback
			model = loader.from_pretrained(model_id, **model_kwargs)
		else:
			if verbose: print(f"[ERROR] loading model {model_id}\n{e}")
			raise e


	model.eval()
	
	# ========== Model Info & Verification ==========
	if verbose:
		print(f"\n[MODEL] {model_id} {model.__class__.__name__}")
		if hasattr(model, "hf_device_map"):
			dm = model.hf_device_map
			disk_layers = [k for k, v in dm.items() if v == "disk"]
			if disk_layers:
				print(f"\n{'='*70}\n❌ CRITICAL WARNING: {len(disk_layers)} layers on DISK!\n{'='*70}")
			elif not any(v == "cpu" for v in dm.values()):
				print(f"\n[OK] All layers on GPU - optimal performance!")

		print(f"{'='*75}")
		print(model.config)
		print(f"{'='*75}")

	return tokenizer, model

def get_prompt(
	tokenizer: tfs.PreTrainedTokenizer, 
	description: str, 
	max_kws: int,
	verbose: bool = False,
):
	if verbose:
		print(f"Generating prompt for text (length: {len(description.split()):7d}) max_kws: {max_kws}")

	messages = [
		{"role": "system", "content": "You are an archivist whose expertise lies in the 20th century."},
		{"role": "user", "content": PROMPT_TEMPLATE.format(k=max_kws, caption=description.strip())},
	]
	try:
		text = tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
			enable_thinking=False,
		)
	except Exception as e:
		print(f"[ERROR] {e}")
		# Fallback: manual formatting
		system_msg = messages[0]["content"]
		user_msg = messages[1]["content"]
		text = f"{system_msg}\n{user_msg}"

	return text

def parse_llm_response(
	model_id: str,
	llm_response: str,
	max_kws: int,
	verbose: bool = False
):
	if verbose:
		# print(f"\nRaw Caption:\n{caption}")
		print(f"\n[LLM: {model_id} RESPONSE]")
		print(llm_response)

	# Step 1: Find the assistant's response
	# Try multiple patterns for different model families
	response_content = None
	
	# Pattern 1: Mistral/Llama style with [/INST]
	inst_end_match = re.search(r'\[/INST\]\s*', llm_response)
	if inst_end_match:
		response_content = llm_response[inst_end_match.end():].strip()
		if verbose:
			print(f"[STEP 1] Found [/INST] tag at position {inst_end_match.end()}")
	
	# Pattern 2: ChatML style (Qwen, Yi, etc.) - look for "assistant\n"
	if response_content is None:
		assistant_match = re.search(r'\nassistant\s*\n', llm_response, re.IGNORECASE)
		if assistant_match:
			response_content = llm_response[assistant_match.end():].strip()
			if verbose:
				print(f"[STEP 1] Found assistant tag at position {assistant_match.end()}")
	
	# Pattern 3: Direct response - look for the last occurrence of a list pattern
	if response_content is None:
		# Just use the entire response
		response_content = llm_response.strip()
		if verbose:
			print(f"[STEP 1] No specific tag found, using entire response")

	if verbose:
		print(f"[STEP 1] Content to parse:\n{response_content}\n")
	
	# Step 2: Extract the Python list
	start_bracket = response_content.find('[')
	if start_bracket == -1:
		if verbose:
			print("[ERROR] No opening bracket '[' found in the response => skipping...")

		return None
	
	# Find matching closing bracket
	bracket_count = 0
	end_bracket = -1
	for i in range(start_bracket, len(response_content)):
		if response_content[i] == '[':
			bracket_count += 1
		elif response_content[i] == ']':
			bracket_count -= 1
			if bracket_count == 0:
				end_bracket = i
				break
	
	if end_bracket == -1:
		if verbose:
			print("[ERROR] No matching closing bracket ']' found")
		return None
	
	list_str = response_content[start_bracket:end_bracket + 1]
	
	if verbose:
		print(f"[STEP 2] Extracted list string: {list_str}\n")
	
	# Step 3: Parse with multiple strategies
	keywords_list, parsing_method = None, None
	
	# Strategy 1: Try ast.literal_eval directly
	try:
		keywords_list = ast.literal_eval(list_str)
		if isinstance(keywords_list, list):
			parsing_method = "ast.literal_eval (direct)"
	except Exception as e:
		if verbose:
			print(f"[FAILURE] ast.literal_eval {e}")
	
	# Strategy 2: Convert single quotes to double quotes for JSON
	if keywords_list is None:
		try:
			# Simple replacement works when there are no apostrophes inside strings
			json_str = list_str.replace("'", '"')
			keywords_list = json.loads(json_str)
			if isinstance(keywords_list, list):
				parsing_method = "json.loads (quote replacement)"
		except Exception as e:
			if verbose:
				print(f"[FAILURE] JSON parsing {e}")
	
	# Strategy 3: Smart quote conversion - handle apostrophes properly
	if keywords_list is None:
		try:
			if verbose:
				print(f"[STEP 3.3] Attempting smart quote conversion...")
			
			# Use a state machine approach
			result = []
			i = 0
			in_string = False
			
			while i < len(list_str):
				char = list_str[i]
				
				if char == '[':
					result.append(char)
					i += 1
					# Skip whitespace
					while i < len(list_str) and list_str[i].isspace():
						result.append(list_str[i])
						i += 1
					# Expect a quote to start a string
					if i < len(list_str) and list_str[i] in ('"', "'"):
						in_string = True
						result.append('"')  # Use double quote
						i += 1
					continue
				elif char == ']':
					if in_string:
						result.append('"')
						in_string = False
					result.append(char)
					i += 1
					continue
				elif char == ',' and not in_string:
					result.append(char)
					i += 1
					# Skip whitespace
					while i < len(list_str) and list_str[i].isspace():
						result.append(list_str[i])
						i += 1
					# Expect a quote to start next string
					if i < len(list_str) and list_str[i] in ('"', "'"):
						in_string = True
						result.append('"')  # Use double quote
						i += 1
					continue
				elif in_string:
					# Check if this is the closing quote
					if char in ('"', "'"):
						# Look ahead to see if this could be a closing quote
						j = i + 1
						while j < len(list_str) and list_str[j].isspace():
							j += 1
						
						if j < len(list_str) and list_str[j] in (',', ']'):
							# This is a closing quote
							result.append('"')  # Use double quote
							in_string = False
							i += 1
						else:
							# This is an apostrophe or quote inside the string
							result.append(char)
							i += 1
					else:
						result.append(char)
						i += 1
				else:
					result.append(char)
					i += 1
			
			normalized_list_str = ''.join(result)
			
			if verbose:
				print(f"[STEP 3.3] Converted to: {normalized_list_str}")
			
			keywords_list = ast.literal_eval(normalized_list_str)
			if isinstance(keywords_list, list):
				parsing_method = "smart quote conversion"
		except Exception as e:
			if verbose:
				print(f"[FAILURE] Smart conversion {e}")
	
	# Strategy 4: Regex extraction (most robust fallback)
	if keywords_list is None:
		try:
			if verbose:
				print(f"[STEP 3.4] Attempting regex extraction...")
			
			# Extract all quoted strings (handles both single and double quotes)
			pattern = r'''['"]([^'"\\]*(?:\\.[^'"\\]*)*)['"]'''
			matches = re.findall(pattern, list_str)
			
			if matches:
				keywords_list = matches
				parsing_method = "regex extraction"
		except Exception as e:
			if verbose:
				print(f"[FAILURE] Regex extraction {e}")
	
	# Validation
	if keywords_list is None or not isinstance(keywords_list, list):
		if verbose:
			print(f"[ERROR] All parsing strategies failed")
			print(f"[ERROR] Problematic string: {list_str}")
		return None

	if not keywords_list: # len() == 0
		if verbose:
			print(f"[WARNING] Empty list extracted: {keywords_list} => skipping...")
		return None

	if verbose:
		print(f"\n[STEP 3] parsing method: {parsing_method}")
		print(f"[SUCCESS] parsed {len(keywords_list)} items: {keywords_list}")
	
	# Step 4: Post-process keywords
	if verbose:
		print(f"\n[POST-PROCESSING] {keywords_list} (max allowed: {max_kws})")
	
	processed = []
	seen = set()
	for idx, kw in enumerate(keywords_list, 1):
		if verbose:
			print(f"\t[{idx}/{len(keywords_list)}]: {repr(kw)}")
		
		# Check if empty
		if not kw or not str(kw).strip():
			if verbose:
				print(f"    ✗ Skipped: empty/whitespace")
			continue
		
		# Normalize whitespace
		cleaned = re.sub(r'\s+', ' ', str(kw).strip())

		# Unescape any escaped characters
		cleaned = cleaned.replace("\\'", "'").replace('\\"', '"')
		
		if verbose:
			print(f"\t=> Cleaned: {repr(cleaned)}")

		# Check length
		if len(cleaned) < 3:
			if verbose:
				print(f"    ✗ Skipped: too short (len={len(cleaned)})")
			continue

		# Check for duplicates (case-insensitive)
		normalized = cleaned.lower()
		if normalized in seen:
			if verbose:
				print(f"    ✗ Skipped: {normalized} is a duplicate")
			continue
		
		seen.add(normalized)
		processed.append(cleaned)
	
	if verbose:
		print(f"[RESULT] Processed keywords (total: {len(processed)}): {processed}")
	
	return processed if processed else None

def query_local_llm(
	model: tfs.PreTrainedModel,
	tokenizer: tfs.PreTrainedTokenizer, 
	text: str,
	device: str,
	max_generated_tks: int,
	max_kws: int,
	verbose: bool=False,
) -> List[str]:
	
	if not isinstance(text, str) or not text.strip():
		return None

	keywords: Optional[List[str]] = None
	prompt = get_prompt(
		tokenizer=tokenizer, 
		description=text, 
		max_kws=max_kws, 
		verbose=verbose
	)

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

		if verbose:
			print(type(inputs), len(inputs), list(inputs.keys()))

			token_ids = inputs.get("input_ids", None) 
			print(type(token_ids), token_ids.shape, token_ids.dtype, token_ids.device)

		if verbose:
			print(f"[ELAPSED_t] Tokenization: {time.time() - tokenization_start:.4f} sec")
			print(f"Generating {max_generated_tks} tokens [model.generate(..)]")

		t0 = time.time()
		with torch.no_grad():
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported() else torch.float16,
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
		if verbose:
			print(f"[ELAPSED_t] model.generate(..): {time.time() - t0:.4f} sec")
		raw_llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)	
	except Exception as e:
		print(f"[ERROR] {e}")
		return None
	
	parsing_start = time.time()
	keywords = parse_llm_response(
		model_id=model_id,
		llm_response=raw_llm_response,
		max_kws=max_kws,
		verbose=verbose,
	)
	if verbose: 
		output_tokens = get_conversation_token_breakdown(raw_llm_response, model_id)
		print(f"[INFO] Token breakdown: {output_tokens}")
		print(f"[ELAPSED_TIME] Response parsing: {time.time() - parsing_start:.4f} sec")
	
	return keywords

def get_llm_based_labels_debug(
	model_id: str, 
	device: str, 
	max_generated_tks: int,
	max_kws: int,
	csv_file: str=None,
	description: str=None,
	quantization_bits: Optional[int]=None,
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
		quantization_bits=quantization_bits,
		verbose=verbose
	)

	all_keywords = list()
	for i, desc in enumerate(descriptions):
		if verbose: 
			print(f"Processing description {i+1}/{len(descriptions)}: {repr(desc)}")
		
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

		# try:
		# 	df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
		# except Exception as e:
		# 	print(f"Failed to write Excel file: {e}")

		if verbose:
			print(f"Saved {len(all_keywords)} keywords to {output_csv}")
			print(f"Done! dataframe: {df.shape} {list(df.columns)}")

	if verbose and description:
		print(f"Keywords: {all_keywords}")

	return all_keywords

def _split_batch(
	indices: List[int],
	prompts: List[str],
) -> Tuple[List[int], List[str], List[int], List[str]]:
	"""Split a batch into two roughly equal halves."""
	mid = len(indices) // 2
	return (
		indices[:mid], prompts[:mid],
		indices[mid:], prompts[mid:],
	)

def _generate_one_batch(
	tokenizer,
	model,
	device: torch.device,
	prompts: List[str],
	max_generated_tks: int,
	max_length: int,
	temperature: float,
	top_p: float = TOP_P,
) -> List[str]:
	"""
	Executes a single tokenize -> generate -> decode pass.
	
	Guarantees:
		- Strips token_type_ids to prevent crashes on causal decoder models.
		- Slices off prompt tokens (outputs[:, prompt_len:]) so callers receive 
			ONLY the model completion, preventing prompt leakage into regex parsers.
		- Propagates RuntimeError (including CUDA OOM) to the caller for handling.
	"""
	tokenized = tokenizer(
		prompts,
		return_tensors="pt",
		truncation=True,
		max_length=max_length,
		padding=True,
	)

	# Critical fix: Fast tokenizers often add token_type_ids, causing causal models to crash
	tokenized.pop("token_type_ids", None)
	prompt_len = tokenized["input_ids"].shape[1]
	if device.type != "cpu":
		tokenized = {k: v.to(device) for k, v in tokenized.items()}
	sampling = temperature > 1e-4
	gen_kwargs: Dict[str, Any] = {
		**tokenized,
		"max_new_tokens": max_generated_tks,
		"do_sample": sampling,
		"pad_token_id": tokenizer.pad_token_id,
		"eos_token_id": tokenizer.eos_token_id,
		"use_cache": True,
	}
	if sampling:
		gen_kwargs["temperature"] = temperature
		gen_kwargs["top_p"] = top_p

	use_amp = torch.cuda.is_available()
	amp_dtype = (
		torch.bfloat16
		if (use_amp and torch.cuda.is_bf16_supported())
		else torch.float16
	)
	with torch.no_grad():
		with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
			outputs = model.generate(**gen_kwargs)

	# Critical fix: Slice off prompt tokens to decode ONLY newly generated text
	generated_tokens = outputs[:, prompt_len:]
	decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

	del tokenized, outputs, generated_tokens

	return decoded

def parse_lenient(text: Optional[str], max_kws: int) -> Optional[List[str]]:
		"""
		Zero-GPU fallback parser for Cohort A:
		Recovers valid keywords when completions fail strict JSON or bracket parsing 
		(e.g., Markdown bullets, comma lists, unbracketed quotes).
		"""
		if not text or not isinstance(text, str):
				return None
		cleaned = text.strip()

		# Strip header/intro if model labeled the output
		m = re.search(r"keywords?\s*[:\-]\s*(.+)", cleaned, flags=re.IGNORECASE | re.DOTALL)
		if m:
				cleaned = m.group(1)

		# Clean markdown tokens and leading bullets/numbers
		cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
		cleaned = re.sub(r"^\s*(?:[#>*\-•]+|\d+[.)])\s*", "", cleaned, flags=re.MULTILINE)

		# Strategy 1: Quoted strings anywhere in the completion
		quotes = re.findall(r'["\']([^"\'\n\\]{2,50})["\']', cleaned)
		candidates = quotes if quotes else re.split(r"[,;\n|]+", cleaned)

		kws: List[str] = []
		seen = set()
		for item in candidates:
				token = item.strip().strip("\"'().").strip()
				if not token:
						continue
				token_lower = token.lower()
				if (
						1 <= len(token.split()) <= 5
						and token_lower not in ("keywords", "keyword", "none", "n/a", "na", "[]", "null")
						and token_lower not in seen
				):
						seen.add(token_lower)
						kws.append(token)

		return kws[:max_kws] if kws else None

def _lenient_salvage(
		indices: List[int],
		results: List[Optional[List[str]]],
		raw_responses: List[Optional[str]],
		max_kws: int,
) -> List[int]:
		"""
		Cohort A pass: Free CPU re-parse of retained raw generations.
		Returns the list of indices that remain unresolved.
		"""
		for idx in indices:
				if results[idx] is None and raw_responses[idx] is not None:
						salvaged = parse_lenient(raw_responses[idx], max_kws=max_kws)
						if salvaged:
								results[idx] = salvaged

		return [i for i in indices if results[i] is None]

def _run_generation_pass(
	*,
	tokenizer,
	model,
	device: torch.device,
	indices: List[int],
	prompts: List[Optional[str]],
	results: List[Optional[List[str]]],      # Mutated in place
	raw_responses: List[Optional[str]],      # Mutated in place
	batch_size: int,
	max_generated_tks: int,
	max_length: int,
	max_kws: int,
	model_id: str,
	max_retries: int,
	temperature: float,
	pass_name: str,
	checkpoint_fn: Optional[Callable[[], None]] = None,
	checkpoint_every: int = 50,
	verbose: bool = False,
) -> List[int]:
	"""
	Unified split-on-OOM generation loop used by both main pass and retry pass.
	Singletons emerge naturally when sub-batches are halved down to size 1.
	"""
	todo = [i for i in indices if results[i] is None]
	if not todo:
		return []

	# Length-homogeneous batching minimizes padding tokens and memory spikes
	todo.sort(key=lambda i: len(prompts[i] or ""))
	queue: deque = deque(
		(todo[i:i + batch_size], [prompts[j] for j in todo[i:i + batch_size]])
		for i in range(0, len(todo), batch_size)
	)

	if verbose:
		print(
			f"[{pass_name.upper()}] {len(todo)} prompts | batch_size={batch_size} | "
			f"temp={temperature} | max_length={max_length}"
		)

	pbar = tqdm(total=len(todo), desc=pass_name, ncols=100)
	n_pops = 0
	while queue:
		cur_idx, cur_prompts = queue.popleft()
		size = len(cur_idx)
		if size == 0:
			continue
		n_pops += 1
		for attempt in range(max_retries + 1):
			decoded: Optional[List[str]] = None
			try:
				decoded = _generate_one_batch(
					tokenizer=tokenizer,
					model=model,
					device=device,
					prompts=cur_prompts,
					max_generated_tks=max_generated_tks,
					max_length=max_length,
					temperature=temperature,
				)

				# Store completions immediately before parsing so failures can be salvaged
				for local_i, idx in enumerate(cur_idx):
					raw_text = decoded[local_i]
					raw_responses[idx] = raw_text
					try:
						results[idx] = parse_llm_response(
							model_id=model_id,
							llm_response=raw_text,
							max_kws=max_kws,
							verbose=verbose,
						)
					except Exception:
						results[idx] = None

				pbar.update(size)
				break  # Batch processed successfully
			except RuntimeError as e:
				err_msg = str(e).lower()
				is_oom = "out of memory" in err_msg or "cuda out of memory" in err_msg
				if not is_oom:
					raise  # Surface non-OOM errors

				if verbose:
					print(f"  ❌ [{pass_name}] OOM size={size}, attempt {attempt + 1}/{max_retries + 1}")

				decoded = None
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
					torch.cuda.synchronize()

				if attempt < max_retries:
					time.sleep(EXP_BACKOFF ** attempt)
					continue

				# Halve the batch size recursively
				if size > 1:
					left_idx, left_p, right_idx, right_p = _split_batch(cur_idx, cur_prompts)
					queue.appendleft((right_idx, right_p))
					queue.appendleft((left_idx, left_p))
				else:
					# Single sample truly failed with OOM
					results[cur_idx[0]] = None
					pbar.update(1)
				break

		# Periodic cache cleanup & checkpointing
		if n_pops % checkpoint_every == 0:
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			if checkpoint_fn is not None:
				checkpoint_fn()

	pbar.close()
	if checkpoint_fn is not None:
		checkpoint_fn()

	return [i for i in todo if results[i] is None]

def get_llm_based_labels_old(
	model_id: str,
	device: str,
	batch_size: int,
	max_generated_tks: int,
	max_kws: int,
	csv_file: str,
	num_workers: int,
	do_dedup: bool = True,
	max_retries: int = 2,
	quantization_bits: Optional[int]=None,
	verbose: bool = False,
) -> List[Optional[List[str]]]:

	output_csv = csv_file.replace(".csv", "_llm_keywords.csv")

	try:
		df = pd.read_csv(
			filepath_or_buffer=output_csv,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
			usecols = ['llm_keywords'],
		)
		return df['llm_keywords'].tolist()
	except Exception as e:
		print(f"{e} Generating from scratch...")
	
	# num_workers = min(os.cpu_count(), num_workers)
	if verbose:
		print(f"[INIT] Starting OPTIMIZED batch LLM processing with {num_workers} workers")

	st_t = time.time()

	# ========== Load data ==========
	if verbose:
		print(f"[PREP] Loading data (col: enriched_document_description) from {csv_file}...")
	wanted_cols = {
		'doc_url',
		'title',
		'description',
		'keywords', # SMU dataset
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
		print(df.head())

	# regenerate enriched_document_description
	df = get_enriched_description(
		df=df,
		eng_confidence_th=1e-2,
		verbose=verbose
	)
	
	if verbose:
		print(f"[READY] {type(df)} {df.shape} {list(df.columns)} ({time.time() - st_t:.2f}s)")

	descriptions = df['enriched_document_description'].tolist()
	inputs = descriptions
	if len(inputs) == 0:
		return None

	# Load tokenizer and model
	tokenizer, model = _load_llm_(
		model_id=model_id,
		quantization_bits=quantization_bits,
		verbose=verbose,
	)
	if verbose:
		valid_count = sum(
			1 for x in inputs
			if x is not None and str(x).strip() not in ("", "nan", "None")
		)
		null_count = len(inputs) - valid_count
		print(f"Input stats: {type(inputs)} {len(inputs)} total, {valid_count} valid, {null_count} null")
	
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
			prompt = get_prompt(
				tokenizer=tokenizer,
				description=s,
				max_kws=min(max_kws, len(s.split())),
				verbose=verbose,
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
					llm_response=decoded_batch[local_i],
					max_kws=max_kws_,
					verbose=verbose_,
				)
				return idx, parsed
			except Exception as e:
				if verbose_:
					print(f"[FAILED] Parsing batch index {idx}: {e}")
				return idx, None
		
		with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
			futures = {executor.submit(_parse_one, i): i for i in range(len(decoded_batch))}
			for future in concurrent.futures.as_completed(futures):
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

	# resilient batch processing loop
	for bn, (batch_indices, batch_prompts) in enumerate(tqdm(batches, desc="Processing (textual) batches", ncols=120)):
		# Use deque for O(1) popleft / appendleft
		queue = deque([(batch_indices, batch_prompts)])

		while queue:
			current_indices, current_prompts = queue.popleft()
			current_size = len(current_indices)
			if current_size == 0:
				continue

			tokenized = outputs = decoded = None
			success = False
			for attempt in range(max_retries + 1):
					if attempt > 0 and verbose:
							print(f"  🔄 Retry {attempt}/{max_retries} for sub-batch (size={current_size})")
					try:
							tokenized = tokenizer(
									current_prompts,
									return_tensors="pt",
									truncation=True,
									max_length=4096,
									padding=True,
							)
							if device.type != "cpu":
									tokenized = {k: v.to(device) for k, v in tokenized.items()}
							gen_kwargs = dict(
									**tokenized,
									max_new_tokens=max_generated_tks,
									do_sample=TEMPERATURE > 0.0,
									temperature=TEMPERATURE,
									top_p=TOP_P,
									pad_token_id=tokenizer.pad_token_id,
									eos_token_id=tokenizer.eos_token_id,
									use_cache=True,
							)
							with torch.no_grad():
									with torch.amp.autocast(
											device_type=device.type,
											enabled=torch.cuda.is_available(),
											dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
									):
											outputs = model.generate(**gen_kwargs)
							decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
							parsed_dict = _parse_batch_parallel(
									decoded_batch=decoded,
									batch_indices=current_indices,
									batch_prompts=current_prompts,
									model_id_=model_id,
									max_kws_=max_kws,
									verbose_=verbose,
							)
							for idx, parsed in parsed_dict.items():
									unique_results[idx] = parsed
							success = True
							break  # Success
					except RuntimeError as e:
							err_msg = str(e).lower()
							is_oom = "out of memory" in err_msg or "cuda out of memory" in err_msg
							if not is_oom:
								raise  # Re-raise non-OOM errors
							if verbose:
								print(f"  ❌ OOM on sub-batch size={current_size}, attempt {attempt+1}")
							# Heavy cleanup only on OOM
							if tokenized is not None:
								del tokenized
							if outputs is not None:
								del outputs
							if decoded is not None:
								del decoded
							tokenized = outputs = decoded = None
							gc.collect()
							if torch.cuda.is_available():
								torch.cuda.empty_cache()
								torch.cuda.synchronize()
							if attempt < max_retries:
									sleep_time = EXP_BACKOFF ** attempt
									time.sleep(sleep_time)
							else:
								# Retries exhausted → split
								if current_size > 1:
									left_idx, left_prompt, right_idx, right_prompt = _split_batch(current_indices, current_prompts)
									queue.appendleft((right_idx, right_prompt))
									queue.appendleft((left_idx, left_prompt))
									if verbose:
										print(f"Splitting size {current_size} → {len(left_idx)} + {len(right_idx)}")
								else:
										# Truly failed single sample
										idx = current_indices[0]
										unique_results[idx] = None
										if verbose:
											print(f"  💥 Single sample {idx} failed with OOM")
								break
					finally:
						# Lightweight guaranteed cleanup (no heavy GC here)
						if tokenized is not None:
							del tokenized
						if outputs is not None:
							del outputs
						if decoded is not None:
							del decoded
						tokenized = outputs = decoded = None

		# Periodic heavier cleanup (every 25 batches) to limit fragmentation
		if torch.cuda.is_available() and (bn + 1) % 25 == 0:
			gc.collect()
			torch.cuda.empty_cache()
			if verbose:
				for device_idx in range(torch.cuda.device_count()):
					mem_reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
					mem_total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
					print(f"  [MEM] GPU [{device_idx}] reserved: {mem_reserved:.1f}/{mem_total:.1f} GB")

	# HYBRID FALLBACK: Retry failed items individually with query_local_llm
	failed_indices = [
		i
		for i, result in enumerate(unique_results)
		if result is None and unique_inputs[i] is not None
	]

	if failed_indices and verbose:
		print(
			f"Retrying {len(failed_indices)} failed {type(failed_indices)} item(s) "
			f"=> query_local_llm [sequential processing]..."
		)
	
	for idx in failed_indices:
		desc = unique_inputs[idx]
		# if verbose:
		# 	print(f"Retrying item {idx}/{len(unique_inputs)}:\n{desc}\n")
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
				print(f"[SUCCES] Individual retry: {individual_result}")
			elif verbose:
				print(f"[FAILED] item {idx}:\n{desc}\n")
				print("-"*100)
		except Exception as e:
			if verbose:
				print(f"[FAILED] Individual retry for item {idx}: {e}")
			unique_results[idx] = None

	# Cleanup model and tokenizer
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	del model, tokenizer
	# gc.collect()

	# Map unique_results back to original order
	if verbose:
		print(f"Mapping {len(original_to_unique_idx)} original_to_unique_idx unique results back to original order: {len(unique_results)}")
	results: List[Optional[List[str]]] = []
	for _, uniq_idx in enumerate(original_to_unique_idx):
		results.append(unique_results[uniq_idx])

	# Save results
	if csv_file:
		output_csv = csv_file.replace(".csv", "_llm_keywords.csv")
		df['llm_keywords'] = results
		df.to_csv(output_csv, index=False)
		if verbose:
			print(f"Saved {len(results)} keywords to {output_csv} {df.shape}\n{list(df.columns)}")
			print(df.info(verbose=verbose, memory_usage="deep"))

	if verbose:
		n_ok, n_null = 0, 0

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
			f"[STATS] {n_ok}/{valid_inputs_count} successful ({success_rate:.2f}%) "
			f"{n_null} null, {n_failed} failed")

		print(f"Total Extracted LLM-based keywords: {len(results)} {type(results)}")
		print(f"[ELAPSED_TIME] {time.time() - st_t:.1f} sec")
		print("="*100)

	return results

def get_llm_based_labels(
	model_id: str,
	device: Any,
	batch_size: int,
	max_generated_tks: int,
	max_kws: int,
	csv_file: str,
	num_workers: int = 12,              # Retained for signature compatibility
	do_dedup: bool = True,
	max_retries: int = 2,
	quantization_bits: Optional[int] = None,
	verbose: bool = False,
) -> List[Optional[List[str]]]:
	"""
	Robust batch keyword extraction with:
		1. Sliced completion decoding (zero prompt leakage).
		2. Cohort A: Free CPU lenient recovery.
		3. Cohort B: Length-sorted micro-batched GPU retry with recursive OOM halving.
		4. Atomic .pkl checkpointing to prevent progress loss.
	"""
	output_csv = csv_file.replace(".csv", "_llm_keywords.csv")
	ckpt_path = csv_file.replace(".csv", "_llm_keywords_ckpt.pkl") if csv_file else None

	# 1. Return cached results if file already exists
	try:
		df_cached = pd.read_csv(
			filepath_or_buffer=output_csv,
			on_bad_lines="skip",
			dtype=globals().get("dtypes", None),
			low_memory=False,
			usecols=["llm_keywords"],
		)
		if verbose:
			print(f"[CACHE] Found completed results in {output_csv}")
		return df_cached["llm_keywords"].tolist()
	except Exception:
		if verbose:
			print(f"[INIT] Output CSV not found. Generating keywords from scratch...")

	# 2. Load and prepare source data
	st_t = time.time()
	wanted_cols = {
		"doc_url", 
		"title", 
		"description", 
		"keywords",
		"enriched_document_description",
	}
	try:
		df = pd.read_csv(
			filepath_or_buffer=csv_file,
			on_bad_lines="skip",
			dtype=globals().get("dtypes", None),
			low_memory=False,
			usecols=lambda c: c in wanted_cols,
		)
	except Exception as e:
		raise ValueError(f"Error loading CSV file {csv_file}: {e}")

	df = get_enriched_description(df=df, eng_confidence_th=1e-2, verbose=verbose)
	inputs = df["enriched_document_description"].tolist()
	if len(inputs) == 0:
		return None

	# 3. Model & Tokenizer loading
	tokenizer, model = _load_llm_(
		model_id=model_id,
		quantization_bits=quantization_bits,
		verbose=verbose,
	)

	# Ensure left-padding for causal/decoder LLM batch inference
	tokenizer.padding_side = "left"
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id

	# 4. Input Deduplication
	if do_dedup:
		unique_map: Dict[str, int] = {}
		unique_inputs: List[Optional[str]] = []
		original_to_unique_idx: List[int] = []
		for s in inputs:
			key = "__NULL__" if (s is None or str(s).strip() in ("", "nan", "None")) else str(s).strip()
			if key in unique_map:
				original_to_unique_idx.append(unique_map[key])
			else:
				idx = len(unique_inputs)
				unique_map[key] = idx
				unique_inputs.append(None if key == "__NULL__" else key)
				original_to_unique_idx.append(idx)
	else:
		unique_inputs = [
			None if (s is None or str(s).strip() in ("", "nan", "None")) else str(s).strip()
			for s in inputs
		]
		original_to_unique_idx = list(range(len(unique_inputs)))

	# 5. Build prompts
	unique_prompts: List[Optional[str]] = []
	for s in unique_inputs:
		if s is None:
			unique_prompts.append(None)
		else:
			unique_prompt = get_prompt(
				tokenizer=tokenizer,
				description=s,
				max_kws=min(max_kws, len(s.split())),
				verbose=verbose,
			)
			unique_prompts.append(unique_prompt)

	# Pre-allocate tracking buffers
	unique_results: List[Optional[List[str]]] = [None] * len(unique_prompts)
	raw_responses: List[Optional[str]] = [None] * len(unique_prompts)

	# 6. Atomic checkpointing helpers
	def _save_checkpoint() -> None:
			if not ckpt_path:
					return
			tmp_path = f"{ckpt_path}.tmp"
			try:
					with open(tmp_path, "wb") as f:
							pickle.dump(
									{"results": unique_results, "raw": raw_responses},
									f, protocol=pickle.HIGHEST_PROTOCOL,
							)
					os.replace(tmp_path, ckpt_path)
			except Exception as err:
					if verbose:
							print(f"[CHECKPOINT WARN] Failed to save checkpoint: {err}")

	# Resume from checkpoint if present
	if ckpt_path and os.path.exists(ckpt_path):
			try:
					with open(ckpt_path, "rb") as f:
							saved = pickle.load(f)
					saved_res = saved.get("results", [])
					saved_raw = saved.get("raw", [])
					if len(saved_res) == len(unique_results):
							for i, r in enumerate(saved_res):
									if r is not None and unique_results[i] is None:
											unique_results[i] = r
							for i, r in enumerate(saved_raw):
									if i < len(raw_responses) and r is not None and raw_responses[i] is None:
											raw_responses[i] = r
							if verbose:
									recovered_cnt = sum(r is not None for r in unique_results)
									print(f"[RESUME] Restored {recovered_cnt} item(s) from {ckpt_path}")
					elif verbose:
							print("[RESUME] Checkpoint size mismatch with current dataset. Starting fresh.")
			except Exception as e:
					if verbose:
							print(f"[RESUME WARN] Checkpoint unreadable ({e}). Starting fresh.")
	valid_indices = [i for i, p in enumerate(unique_prompts) if p is not None]
	if not valid_indices:
			return None

	# =========================================================================
	# PASS 1: Main Batched Generation
	# =========================================================================
	_run_generation_pass(
		tokenizer=tokenizer,
		model=model,
		device=device,
		indices=valid_indices,
		prompts=unique_prompts,
		results=unique_results,
		raw_responses=raw_responses,
		batch_size=batch_size,
		max_generated_tks=max_generated_tks,
		max_length=4096,
		max_kws=max_kws,
		model_id=model_id,
		max_retries=max_retries,
		temperature=TEMPERATURE,
		pass_name="main",
		checkpoint_fn=_save_checkpoint,
		checkpoint_every=50,
		verbose=verbose,
	)

	# =========================================================================
	# OPTIMIZED TWO-STAGE FALLBACK LADDER
	# =========================================================================
	failed = [
		i for i, r in enumerate(unique_results)
		if r is None and unique_inputs[i] is not None
	]

	# --- Cohort A: Free CPU Salvage on Retained Generations ---
	if failed:
		n_before = len(failed)
		failed = _lenient_salvage(failed, unique_results, raw_responses, max_kws)
		if verbose:
			print(f"[FALLBACK-A] Lenient extraction recovered {n_before - len(failed)}/{n_before} items (0 GPU compute)")

	# --- Cohort B: Micro-Batch GPU Retry with Flipped Decoding Mode ---
	if failed:
		# Flipped decoding: if primary was deterministic (temp near 0), sample mildly to escape loops
		is_greedy = TEMPERATURE < 1e-4
		retry_temperature = 0.7 if is_greedy else 0.0
		if verbose:
			print(
				f"[FALLBACK-B] {len(failed)} item(s) sent to GPU micro-batch retry "
				f"(bs={RETRY_BATCH_SIZE}, temp={retry_temperature}, max_len={RETRY_MAX_LENGTH})"
			)
		failed = _run_generation_pass(
			tokenizer=tokenizer,
			model=model,
			device=device,
			indices=failed,
			prompts=unique_prompts,
			results=unique_results,
			raw_responses=raw_responses,
			batch_size=RETRY_BATCH_SIZE,
			max_generated_tks=max_generated_tks,
			max_length=RETRY_MAX_LENGTH,
			max_kws=max_kws,
			model_id=model_id,
			max_retries=max_retries,
			temperature=retry_temperature,
			pass_name="retry",
			checkpoint_fn=_save_checkpoint,
			checkpoint_every=25,
			verbose=verbose,
		)
		# Final sweep: Salvage whatever was generated during the retry pass
		if failed:
			n_before_retry_salvage = len(failed)
			failed = _lenient_salvage(failed, unique_results, raw_responses, max_kws)
			if verbose:
				print(f"[FALLBACK-B] Final lenient sweep recovered {n_before_retry_salvage - len(failed)} items")

	if failed and verbose:
		print(f"[UNRESOLVED] {len(failed)} item(s) could not be parsed and are set to None")

	# 7. Cleanup GPU resources
	del model, tokenizer
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	gc.collect()

	# 8. Reconstruct original order and save results
	results: List[Optional[List[str]]] = [
		unique_results[uniq_idx] 
		for uniq_idx in original_to_unique_idx
	]

	if csv_file:
		df["llm_keywords"] = results
		df.to_csv(output_csv, index=False)
		# Remove checkpoint file on successful completion
		if ckpt_path and os.path.exists(ckpt_path):
			try:
				os.remove(ckpt_path)
			except OSError:
				pass
		if verbose:
			print(f"Saved {len(results)} keyword lists to {output_csv}")

	# 9. Summary stats
	if verbose:
		n_null = sum(1 for inp in inputs if inp is None or str(inp).strip() in ("", "nan", "None"))
		n_ok = sum(1 for r in results if r is not None)
		valid_count = len(results) - n_null
		rate = (n_ok / valid_count) * 100 if valid_count else 0.0
		print(f"\n[SUMMARY] {n_ok}/{valid_count} successful ({rate:.2f}%) | {n_null} null inputs | {valid_count - n_ok} failed")
		print(f"[ELAPSED TIME] {time.time() - st_t:.1f} sec\n{'=' * 100}")

	return results

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="LLM-instruct-based keyword annotation for Historical Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, help="Path to the metadata CSV file")
	parser.add_argument("--model_id", '-llm', type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="HuggingFace model ID")
	parser.add_argument("--device", '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--description", '-desc', type=str, help="Description")
	parser.add_argument("--num_workers", '-nw', type=int, default=12, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=32, help="Batch size for processing (adjust based on GPU memory)")
	parser.add_argument("--max_keywords", '-mkw', type=int, default=3, help="Max number of keywords to extract")
	parser.add_argument("--max_generated_tks", '-mgt', type=int, default=128, help="Max number of generated tokens")
	parser.add_argument("--quantization_bits", '-qb', type=int, default=None, help="Quantization bits")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	parser.add_argument("--debug", '-d', action='store_true', help="Debug mode")

	args = parser.parse_args()

	set_seeds(seed=42, debug=args.debug)
	args.device = torch.device(args.device)
	args.num_workers = min(args.num_workers, os.cpu_count())

	if args.verbose:
		print_args_table(args=args, parser=parser)
		print(args)

	if args.debug or args.description:
		keywords = get_llm_based_labels_debug(
			model_id=args.model_id, 
			device=args.device,
			max_generated_tks=args.max_generated_tks,
			max_kws=args.max_keywords,
			csv_file=args.csv_file,
			description=args.description,
			quantization_bits=args.quantization_bits,
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
			quantization_bits=args.quantization_bits,
			verbose=args.verbose,
		)
	else:
		print("Either --csv_file or --description must be provided")
		return

	if args.verbose and keywords:
		print(f"{len(keywords)} {type(keywords)} Extracted keywords")

if __name__ == "__main__":
	main()
