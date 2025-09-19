from utils import *

# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.2-1B-Instruct" # default for local
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "Qwen/Qwen3-4B-Instruct-2507"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output

# not useful for instruction tuning:
# model_id = "microsoft/DialoGPT-large"  # Fallback if you can't run Hermes
# model_id = "gpt2-xl"

# {{"keywords": ["keyword1", "keyword2", "keyword3"], "rationales": ["rationale1", "rationale2", "rationale3"]}}

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
TEMPERATURE = 1e-8
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
MAX_KEYWORDS = 3
JSON_OUTPUT_TEMPLATE = {"keywords": ["keyword1", "keyword2", "keyword3"], "rationales": ["rationale1", "rationale2", "rationale3"]}
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
- Do not include any other text or explanations in the response.

**Description**: {description}

Adhere to the following example output format (no extra spaces, no extra characters, no extra lines):
```json
{desired_json_output}
```
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

def extract_json_from_llm_response(text: str) -> Optional[dict]:
		"""
		Extracts a JSON object from a complex LLM response string.
		
		This function handles:
		- JSON wrapped in Markdown fences (```json...```)
		- Lone JSON objects with extraneous text before or after
		- Multiple JSON objects in a single response, by taking the last one
		- Extra characters/tokens before or after the JSON object
		"""
		all_json_objects = []
		
		# First, look for JSON objects wrapped in a markdown fence.
		# This is a very common and preferred pattern.
		markdown_matches = re.findall(r'```json\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
		for json_string in markdown_matches:
				try:
						data = json.loads(json_string)
						all_json_objects.append(data)
				except json.JSONDecodeError:
						continue

		# If no markdown-wrapped JSON found, look for all standalone JSON objects.
		if not all_json_objects:
				# A non-greedy regex to find all bracket-enclosed objects
				raw_json_matches = re.findall(r'\{[\s\S]*?\}', text, re.DOTALL)
				for json_string in raw_json_matches:
						try:
								data = json.loads(json_string.strip())
								# Add to list only if it's a valid JSON with "keywords" and "rationales"
								if isinstance(data, dict) and "keywords" in data and "rationales" in data:
										all_json_objects.append(data)
						except json.JSONDecodeError:
								continue

		# Return the last valid JSON object found, as it's most likely the final answer.
		if all_json_objects:
				return all_json_objects[-1]
				
		return None

def query_local_llm(model, tokenizer, text: str, device) -> Tuple[List[str], List[str]]:
	if not isinstance(text, str) or not text.strip():
		return None, None		
	keywords: Optional[List[str]] = None
	rationales: Optional[List[str]] = None
	prompt = PROMPT_TEMPLATE.format(k=MAX_KEYWORDS, description=text.strip(), desired_json_output=JSON_OUTPUT_TEMPLATE)
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

	print(f"\n=== Extracted Listed results from JSON Data ===")
	json_data_improved = extract_json_from_llm_response(raw_llm_response)
	if json_data_improved:
		keywords_improved = json_data_improved.get("keywords", None)
		rationales_improved = json_data_improved.get("rationales", None)
		print(f"Extracted {len(keywords_improved)} Keywords({type(keywords_improved)}): {keywords_improved}")
		print(f"Extracted {len(rationales_improved)} Rationales({type(rationales_improved)}): {rationales_improved}")
	else:
		print("Could not extract JSON data from the response.")
	
	return keywords, rationales

def get_labels(model_id: str, device: str, test_description: str) -> None:
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
	parser.add_argument("--test_description", '-td', type=str, required=True, help="Test description")
	args = parser.parse_args()
	print(args)
	get_labels(model_id=args.model_id, device=args.device, test_description=args.test_description)

if __name__ == "__main__":
	main()