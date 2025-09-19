from utils import *
log = logging.getLogger(__name__)

# model_id = "meta-llama/Llama-3.2-1B-Instruct" # default for local
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "Qwen/Qwen3-4B-Instruct-2507"
# model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output

# not useful for instruction tuning:
# model_id = "microsoft/DialoGPT-large"  # Fallback if you can't run Hermes
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

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
MAX_KEYWORDS = 3

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

# PROMPT_TEMPLATE = """<s>[INST]
# As an expert historical archivist, analyze this historical description carefully and extract a list of concrete, factual and relevant keywords with their corresponding concise rationales.
# Duplicate and identical keywords are not allowed. Avoid keywords that contain numbers, temporal context, or time-related information.
# Description: {description}

# Your entire output MUST be ONLY a single JSON object with two keys: "keywords" and "rationales". The value of each key is a list of strings. Do not include any other text, explanations, or markdown formatting (e.g., ```json```) in your response.
# [/INST]
# """

PROMPT_TEMPLATE = """<s>[INST]
You are a meticulous historical archivist.  
Given the description below, **extract exactly {k}** concrete, factual, and *non‑numeric* keywords and give a short, one‑sentence rationale for each.

**Rules**
- Do NOT repeat or synonym‑duplicate keywords.
- Do NOT include any numbers, dates, years, or temporal expressions.
- Output MUST be **ONLY** a single JSON object with two keys: "keywords" and "rationales". 
- The value of each key is a list of strings. 
- Do not include any other text, explanations, or markdown formatting (e.g., ```json```) in the response.

**Description**: {description}
[/INST]
"""

class JsonStopCriteria(tfs.StoppingCriteria):
	def __init__(self, tokenizer):
		super().__init__()
		self.tokenizer = tokenizer
		self.brace_balance = 0
		self.seen_open = False
	def __call__(self, input_ids, scores, **kwargs):
		# decode only the *new* token(s) added in the latest step
		new_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
		# Update a simple brace counter
		for ch in new_text[-1:]:               # look at the last generated char only
			if ch == "{":
				self.seen_open = True
				self.brace_balance += 1
			elif ch == "}":
				if self.seen_open:
					self.brace_balance -= 1
					if self.brace_balance <= 0:   # we have closed everything we opened
						return True
		return False

def extract_json_from_text(text: str) -> Optional[dict]:
	try:
		json_match = re.search(r'\{[\s\S]*\}', text)
		if json_match:
			json_string = json_match.group(0)
			return json.loads(json_string)
	except (json.JSONDecodeError, AttributeError, TypeError) as e:
		pass
	return None

def extract_json_from_text_new(text: str) -> Optional[dict]:
	try:
		# Find the JSON part by looking for the curly braces
		json_match = re.search(r'\{.*\}', text, re.DOTALL)
		if json_match:
			json_string = json_match.group(0)
			return json.loads(json_string)
	except (json.JSONDecodeError, AttributeError) as e:
		print(f"Error decoding JSON: {e}")
		return None

def extract_json_gold_standard(text: str, *, first: bool = True) -> Optional[dict]:
	"""
	Extract *valid* JSON from an arbitrary string.
	Parameters
	----------
	text : str
			The raw output coming from an LLM or any other source.
	first : bool, optional
			If True (default) return only the first JSON object found.
			If False, return a list of all parsed objects (or None if none parse).
	Returns
	-------
	dict | None
			Parsed JSON dict, or ``None`` when no parsable JSON was found.
	"""
	JSON_RE = re.compile(r'\{[\s\S]*?\}', re.MULTILINE)   # non‑greedy, matches first JSON block
	if not isinstance(text, str):
		log.debug("extract_json received a non‑string: %r", text)
		return None
	# Find *all* candidate JSON substrings (non‑greedy)
	candidates = JSON_RE.findall(text)
	if not candidates:
		log.debug("No JSON bracket pattern found in text.")
		return None
	parsed = []
	for cand in candidates:
		try:
			parsed.append(json.loads(cand))
		except json.JSONDecodeError as exc:
			# Fine‑grained debug; not noisy in production
			log.debug("Failed to decode candidate JSON: %s | error=%s", cand, exc)
	if not parsed:
		log.debug("No valid JSON found in text.")
		return None
	return parsed[0] if first else parsed

def query_local_llm(model, tokenizer, text: str, device) -> Tuple[List[str], List[str]]:
	if not isinstance(text, str) or not text.strip():
		return None, None		
	keywords: Optional[List[str]] = None
	rationales: Optional[List[str]] = None
	prompt = PROMPT_TEMPLATE.format(k=MAX_KEYWORDS, description=text.strip())
	stop_criteria = tfs.StoppingCriteriaList([JsonStopCriteria(tokenizer)])

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
		outputs = model.generate(
			**inputs, 
			max_new_tokens=MAX_NEW_TOKENS,
			temperature=TEMPERATURE,
			top_p=TOP_P,
			do_sample=TEMPERATURE > 0.0,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
			stopping_criteria=stop_criteria
		)
		raw_llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
	except Exception as e:
		print(f"<!> Error {e}")
		return None, None

	print(f"=== Raw Output from LLM ===")
	print(f"{raw_llm_response}")
	print("="*150)

	print(f"\n=== Extracted (Raw) JSON Data ===")
	# Extract the JSON data from the response string
	json_data = extract_json_from_text(raw_llm_response)
	print(json_data)

	print(f"\n=== Extracted Listed results from JSON Data ===")
	if json_data:
		keywords = json_data.get("keywords", None)
		rationales = json_data.get("rationales", None)
		print(f"Extracted {len(keywords)} Keywords({type(keywords)}): {keywords}")
		print(f"Extracted {len(rationales)} Rationales({type(rationales)}): {rationales}")
	else:
		print("Could not extract JSON data from the response.")

	print(f"\n=== Extracted (Raw) JSON Data (NEW)===")
	json_data_new = extract_json_from_text_new(raw_llm_response)
	print(json_data_new)

	print(f"\n=== Extracted Listed results from JSON Data (NEW) ===")
	if json_data_new:
		keywords_new = json_data_new.get("keywords", None)
		rationales_new = json_data_new.get("rationales", None)
		print(f"Extracted {len(keywords_new)} Keywords({type(keywords_new)}): {keywords_new}")
		print(f"Extracted {len(rationales_new)} Rationales({type(rationales_new)}): {rationales_new}")
	else:
		print("Could not extract JSON data from the response.")

	print(f"\n=== Extracted JSON Data (Gold Standard) ===")
	json_payload = extract_json_gold_standard(raw_llm_response)
	print(json_payload)

	print(f"\n=== Extracted Listed results from JSON Data (Gold Standard) ===")
	if json_payload:
		keywords_gold = json_payload.get("keywords", None)
		rationales_gold = json_payload.get("rationales", None)
		print(f"Extracted {len(keywords_gold)} Keywords({type(keywords_gold)}): {keywords_gold}")
		print(f"Extracted {len(rationales_gold)} Rationales({type(rationales_gold)}): {rationales_gold}")
	else:
		print("Could not extract JSON data from the response.")

	return keywords, rationales

def get_labels(model_id: str, device: str) -> None:
	tokenizer = tfs.AutoTokenizer.from_pretrained(
		model_id, 
		use_fast=True, 
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id

	model = tfs.AutoModelForCausalLM.from_pretrained(
		model_id,
		device_map=device,
		torch_dtype=torch.float16,
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	).eval()

	debug_llm_info(model, tokenizer, device)

	test_description = "Bus trip from Algiers to Bou Saada, deep in the Sahara According to Shaffer: ''[This is] the bus from Algiers to Bou Saada and Biskra. Kay Shelby, the Red Cross girl to the left, enjoyed her job as hostess to soldiers on leave and operated this little tour for their benefit.''"

	query_local_llm(
		model=model, 
		tokenizer=tokenizer, 
		text=test_description, 
		device= device
	)

def main():
	parser = argparse.ArgumentParser(description="Textual-label annotation for Historical Archives Dataset using instruction-tuned LLMs")
	parser.add_argument("--model_id", '-m', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	args = parser.parse_args()
	print(args)
	get_labels(model_id=args.model_id, device=args.device)

if __name__ == "__main__":
	main()