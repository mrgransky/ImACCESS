from utils import *
from nlp_utils import get_enriched_description

# better models:
# model_id = "Qwen/Qwen3-4B-Instruct-2507"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# model_id = "NousResearch/Hermes-2-Pro-Mistral-7B"
# model_id = "allenai/Olmo-3-7B-Instruct"
# model_id = "google/flan-t5-xxl"

# Qwen/Qwen3-30B-A3B-Instruct-2507 # multi-gpu required
# Qwen/Qwen3-Next-80B-A3B-Instruct # multi-gpu required

# how to run [local]:
# python gt_kws_llm.py -csv /home/farid/datasets/WW_DATASETs/HISTORY_X4/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -q -v -bs 2

# with description:
# small model for local testing:
# python gt_kws_llm.py -desc "Exhausted Marine weeping atop of Hill 200" -llm "Qwen/Qwen3-4B-Instruct-2507" -v

# large model:
# python gt_kws_llm.py -desc "A. A. Robinson and infantry." -llm "google/gemma-4-31B-it" -v

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
	stopwords = set([line.strip().lower() for line in file_])
STOPWORDS.update(stopwords)

with open('geographic_references.txt', 'r') as file_:
	geographic_references = set([line.strip().lower() for line in file_ if line.strip()])
STOPWORDS.update(geographic_references)

LLM_INSTRUCTION_TEMPLATE = """<s>[INST]
Given the caption below, extract **at most {k}** prominent, reusable, and semantically meaningful keywords.
Return **ONLY** a standardized, valid, and parsable **Python LIST** of keywords, without any explanatory text.
Opt for fewer keywords if the caption is short or lacks sufficient information.

Keywords must be:
	* **Semantically atomic** which represents a core concept.
	* **Visually grounded**: such as tangible objects, agents, scene elements, or observable actions.
	* **ABSOLUTE MAXIMUM GENERALITY**:
		- "nurse" instead of "nurse checking blood pressure"
		- "pilot" instead of "pilot Charles Matheson"
		- "Oberleutnant" instead of "Oberleutnant Bruno Kikillus"
		- "squadron" instead of "No. 10 Squadron RAAF"
		- "Corporal" instead of "Corporal Genevieve Wade"
		- "seaplane" instead of "seaplane on the water in the background"
		- "airplane" instead of "airplane in flight"
		- "airport" instead of "airport in the background"
		- "manufacturing loom" instead of "manufacturing looms for the government"
		- "mountain" instead of "Eastern Mountains"
		- "aerial view" instead of "aerial view of Osaka, Japan"
		- "Minister of War" instead of "Italian Minister of War Cipriano Facchinetti"	
		- "Army Hospital" instead of "United States Army General Hospital"
		- "Marine Corps" instead of "U.S. Marine Corps"
		- "Red Cross headquarter" instead of "American Red Cross headquarters in Rome, Italy"
		- "animal" instead of "man riding a camel in the desert"
		- "reservoir" instead of "Fort Loudoun Reservoir"
		- "Ballon Gun" instead of "6-pounder Ballon Gun"
		- "submarine" instead of "German submarine"

STRICTLY EXCLUDE:
	- Quantities, counts, measurements, or numeric expressions (e.g., 1 1/2 ton truck, 7.3mm, 3 Dodge trucks).
	- Equipment identifiers, serial numbers, or models.
	- Dates, times, years, decades, or any temporal references.
	- Names of individuals or Honorifics (e.g., A. A. Robinson, Barbara Briggs, Allan M. Hardy, Josef Dietrich, Mrs. Howard Russell). 
	- family relationship terms (e.g., mother, father, son, uncle).
	- Geographical names such as continents, countries, states, provinces, cities, towns, islands, regions, roads, or landmarks.
	- Ordinal numeral keywords (e.g., fourth, 1st, 115th).
	- Roman numerals (e.g., I, II, IV, VIII).
	- Nationalities, ethnicities, or religions.
	- Abbreviations, acronyms, phrasal verbs, possessive constructions, or descriptive clauses.

Color handling:
	Remove color only if it is purely descriptive (e.g., white truck, blue sky).
	Preserve color terms when they are part of a standardized or semantic label (e.g., Red Cross, Blue Cross gas shell, Green Berets).

ANTI-HALLUCINATION RULE:
	Only use the exact information given in the caption for keyword extraction, without making assumptions based on implied meanings.

{caption}	
[/INST]"""

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

	# Attention implementation selection
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
		print(f"\n[LOADING] {model_id}")
		model_loader_name = "AutoModelForCausalLM" if use_auto_model else model_cls.__name__
		print(f"\n[INFO] Model loading kwargs for {model_loader_name}:")
		pprint.pprint(model_kwargs)

	try:
		if use_auto_model:
			model = tfs.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
		else:
			model = model_cls.from_pretrained(model_id, **model_kwargs)
	except Exception as e:
		if verbose: print(f"[ERROR] Error loading model:\n{e}")
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

		print(f"{'='*110}\n")
		print(model.config)
		print(f"{'='*110}\n")

	return tokenizer, model


def get_prompt(
	tokenizer: tfs.PreTrainedTokenizer, 
	description: str, 
	max_kws: int
):
	messages = [
		{"role": "system", "content": "You are an expert image tagger and function as a historical archivist whose expertise lies in the 20th century."},
		{"role": "user", "content": LLM_INSTRUCTION_TEMPLATE.format(k=max_kws, caption=description.strip())},
	]
	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
		enable_thinking=False,
	)
	return text

def parse_llm_response(
	model_id: str, 
	input_prompt: str, 
	raw_llm_response: str, 
	max_kws: int,
	caption: str,
	verbose: bool = False
):
	if verbose:
		print(f"[DEBUG] Raw Caption:\n{caption}\n")
		print(f"[LLM: {model_id} RESPONSE]\n{raw_llm_response}\n")

	llm_response: Optional[str] = None
	llm_response = _qwen_llm_response(
		raw_llm_response, 
		max_kws, 
		verbose
	)

	return llm_response

def _qwen_llm_response(
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
	
	# Step 2: Extract the Python list
	start_bracket = response_content.find('[')
	if start_bracket == -1:
		if verbose:
			print("[ERROR] No opening bracket '[' found")
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
		print(f"\n[STEP 2] Extracted list string: {list_str}\n")
	
	# Step 3: Parse with multiple strategies
	keywords_list = None
	parsing_method = None
	
	# Strategy 1: Try ast.literal_eval directly
	try:
		keywords_list = ast.literal_eval(list_str)
		if isinstance(keywords_list, list):
			parsing_method = "ast.literal_eval (direct)"
			if verbose:
				print(f"[STEP 3.1] Success with ast.literal_eval")
	except Exception as e:
		if verbose:
			print(f"[STEP 3.1] ast.literal_eval failed: {e}")
	
	# Strategy 2: Convert single quotes to double quotes for JSON
	if keywords_list is None:
			try:
					# Simple replacement works when there are no apostrophes inside strings
					json_str = list_str.replace("'", '"')
					keywords_list = json.loads(json_str)
					if isinstance(keywords_list, list):
							parsing_method = "json.loads (quote replacement)"
							if verbose:
									print(f"[STEP 3.2] Success with JSON parsing")
			except Exception as e:
					if verbose:
							print(f"[STEP 3.2] JSON parsing failed: {e}")
	
	# Strategy 3: Smart quote conversion - handle apostrophes properly
	if keywords_list is None:
			try:
					if verbose:
							print(f"[STEP 3.3] Attempting smart quote conversion...")
					
					# Find all string boundaries by looking for quotes after [ or ,
					# This regex finds strings that are list items
					pattern = r'''(?:^|\[|,)\s*'([^']*(?:''[^']*)*)'(?=\s*(?:,|]))'''
					
					# Alternative: Use a state machine approach
					result = []
					i = 0
					in_string = False
					current_string = []
					
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
											current_string = []
									continue
							
							elif char == ']':
									if in_string:
											# Shouldn't happen, but handle it
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
											current_string = []
									continue
							
							elif in_string:
									# Check if this is the closing quote
									# It's a closing quote if:
									# 1. It's a quote character matching the list format (single or double)
									# 2. The next non-whitespace char is , or ]
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
							if verbose:
									print(f"[STEP 3.3] Success with smart conversion")
			except Exception as e:
					if verbose:
							print(f"[STEP 3.3] Smart conversion failed: {e}")
	
	# Strategy 4: Regex extraction (most robust fallback)
	if keywords_list is None:
			try:
					if verbose:
							print(f"[STEP 3.4] Attempting regex extraction...")
					
					# Extract all quoted strings (handles both single and double quotes)
					# This pattern captures content between quotes, including escaped quotes
					pattern = r'''['"]([^'"\\]*(?:\\.[^'"\\]*)*)['"]'''
					matches = re.findall(pattern, list_str)
					
					if matches:
							keywords_list = matches
							parsing_method = "regex extraction"
							if verbose:
									print(f"[STEP 3.4] Success with regex extraction")
			except Exception as e:
					if verbose:
							print(f"[STEP 3.4] Regex extraction failed: {e}")
	
	# Validation
	if keywords_list is None or not isinstance(keywords_list, list):
			if verbose:
					print(f"[ERROR] All parsing strategies failed")
					print(f"[ERROR] Problematic string: {list_str}")
			return None
	
	if verbose:
			print(f"\n[STEP 3] Parsing method: {parsing_method}")
			print(f"[STEP 3] Successfully parsed list with {len(keywords_list)} items:")
			for i, kw in enumerate(keywords_list, 1):
				print(f"  [{i}] {kw}")
	
	# Step 4: Post-process keywords
	if verbose:
		print(f"\n[POST-PROCESSING] {keywords_list} (max allowed: {max_kws})...")
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

		# if cleaned.lower() in STOPWORDS:
		# 	if verbose:
		# 		print(f"    ✗ Skipped: {kw} is a stopword!")
		# 	continue
				
		# # Check if cleaned is a number # 1940
		# if cleaned.isdigit():
		# 	if verbose:
		# 		print(f"    ✗ Skipped: number detected! {cleaned}")
		# 	continue

		# Check for duplicates (case-insensitive)
		normalized = cleaned.lower()
		if normalized in seen:
			if verbose:
				print(f"    ✗ Skipped: {normalized} is a duplicate")
			continue
		
		seen.add(normalized)
		processed.append(cleaned)
		
		
		# if len(processed) >= max_kws:
		# 	if verbose:
		# 		print(f"\n  [LIMIT] Reached max_kws={max_kws}, stopping")
		# 	break
	
	if verbose:
		print(f"[RESULT] Processed keywords (total: {len(processed)}): {processed}")

	# # Step 5: Return results
	# if verbose:
	# 	print(f"\n[RESULT] Final keywords ({len(processed)}/{len(keywords_list)} kept):")
	# 	for i, kw in enumerate(processed, 1):
	# 		print(f"  [{i}] {kw}")
	
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

		if verbose:
			print(type(inputs), len(inputs), list(inputs.keys()))

			token_ids = inputs.get("input_ids", None) 
			print(type(token_ids), token_ids.shape, token_ids.dtype, token_ids.device)

		tokenization_time = time.time() - tokenization_start
		if verbose: 
			print(f"Tokenization: {tokenization_time:.5f}s")

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

		raw_llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)	
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
		caption=text,
		max_kws=max_kws,
		verbose=verbose,
	)
	if verbose: 
		print(f"Response parsing elapsed time: {time.time() - parsing_start:.5f}s")
	
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
	for i, desc in tqdm(enumerate(descriptions), total=len(descriptions), desc="Processing descriptions"):
		if verbose: print(f"Processing description {i+1}/{len(descriptions)}: {repr(desc)}")
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

	if verbose and description:
		print(f"Keywords: {all_keywords}")

	return all_keywords

def get_llm_based_labels(
	model_id: str,
	device: str,
	batch_size: int,
	max_generated_tks: int,
	max_kws: int,
	csv_file: str,
	num_workers: int,
	mem_cleanup_th: int=95,
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
		print(f"<!> {e} Generating from scratch...")
	
	num_workers = min(os.cpu_count(), num_workers)
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

	# regenerate enriched_document_description
	df = get_enriched_description(df=df, check_english=True, verbose=verbose)
	
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
			# if verbose:
			# 	print(f"Generating prompt for text with len={len(s.split()):<10}max_kws={min(max_kws, len(s.split()))}")
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
					caption=unique_inputs[idx],
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
					**tokenized,
					max_new_tokens=max_generated_tks,
					do_sample=TEMPERATURE > 0.0,
					temperature=TEMPERATURE,
					top_p=TOP_P,
					pad_token_id=tokenizer.pad_token_id,
					eos_token_id=tokenizer.eos_token_id,
					use_cache=True,
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
			if mem_usage_pct > mem_cleanup_th: 
				need_cleanup = True
				memory_consumed_percent += mem_usage_pct

		if need_cleanup:
			print(f"[WARN] High memory usage ({memory_consumed_percent:.1f}% > {mem_cleanup_th}%) => Clearing cache...")
			torch.cuda.empty_cache() # clears all GPUs
			gc.collect()

	# HYBRID FALLBACK: Retry failed items individually with query_local_llm
	failed_indices = [
		i
		for i, result in enumerate(unique_results)
		if result is None and unique_inputs[i] is not None
	]

	if failed_indices and verbose:
		print(f"Retrying {len(failed_indices)} failed {type(failed_indices)} items individually using query_local_llm [sequential processing]...")
	
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
				print(f"OK: Individual retry successful: {individual_result}")
			elif verbose:
				print(f"<!> Individual retry FAILED for item {idx}:\n{desc}\n")
		except Exception as e:
			if verbose:
				print(f"Individual retry error for item {idx}: {e}")
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
	gc.collect()
	
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
		print(f"Total Extracted LLM-based keywords: {len(results)} {type(results)} | Elapsed time: {time.time() - st_t:.1f} sec")
		print("="*100)

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
