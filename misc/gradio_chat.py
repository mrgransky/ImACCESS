from utils import *

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

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
TOP_K = 3
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

if USER == "farid":
	model_id = "meta-llama/Llama-3.2-1B-Instruct" # for local
else:
	# model_id = "Qwen/Qwen3-4B-Instruct-2507"
	model_id = "microsoft/Phi-4-mini-instruct"
	# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
	# model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"

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

print("\n=== Model Attributes ===")
print(dir(model))
print("="*100)
print("\n=== Tokenizer Attributes ===")
print(dir(tokenizer))
print("="*100)

prompt = """<s>[INST]
As an expert historical archivist, analyze this historical description carefully and extract a maximum of three concrete, factual and relevant keywords with concise rationales.
Duplicate keywords are not allowed. Avoid keywords that contain numbers, temporal context, or time-related information.
Description: [Close-up Aerial View of John Gillin's House, Designed by Frank Lloyd Wright, Dallas, TX] aerials; construction sites Oilman John A. Gillin commissioned Frank Lloyd Wright in 1950 to design a work of art that would also be suitable for living and entertaining. Completed in 1954, it's the only Frank Lloyd Wright residence in Dallas. The result is a house that has three wings built off a central hexagon. The roof is made of copper, and the outer walls are of understated sandstone. It is a magnificent example of the type of work Wright was doing later in his career. Source: USA Today http://usatoday30.usatoday.com/travel/destinations/cityguides/dallas/2003-10-20-spotlight-bighouses_x.htm

Your entire output MUST be ONLY a single JSON object with two keys: "keywords" and "rationales". The value of each key is a list of strings. Do not include any other text, explanations, or markdown formatting (e.g., ```json```) in your response.
[/INST]
"""

inputs = tokenizer(
	prompt,
	return_tensors="pt", 
	truncation=True, 
	max_length=512, 
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
)
llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(llm_response)

def extract_json_from_text(text: str) -> dict:
	try:
		# Find the JSON part by looking for the curly braces
		json_match = re.search(r'\{.*\}', text, re.DOTALL)
		if json_match:
			json_string = json_match.group(0)
			return json.loads(json_string)
	except (json.JSONDecodeError, AttributeError) as e:
		print(f"Error decoding JSON: {e}")
		return None

# Extract the JSON data from the response string
json_data = extract_json_from_text(llm_response)
print(json_data)

if json_data:
	keywords = json_data.get("keywords", [])
	rationales = json_data.get("rationales", [])
	print(f"Extracted {len(keywords)} Keywords({type(keywords)}): {keywords}")
	print(f"Extracted {len(rationales)} Rationales({type(rationales)}): {rationales}")
else:
	print("Could not extract JSON data from the response.")

