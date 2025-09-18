from utils import *

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
TOP_K = 3

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
huggingface_hub.login(token=hf_tk)

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
	trust_remote_code=True,
	cache_dir=cache_directory[USER],
).eval()

prompt = """<s>[INST] 
<s>[INST]
As an expert historical archivist, analyze this historical description carefully and extract maximum of three (not more) concrete, factual and relevant keywords with concise rationales.
Duplicate keywords are not allowed. Keywords with numbers, temporal context and time-related information are strongly discouraged.
Description: American soldier tourists, Venice, Summer 1945 Museum and Medical Arts Service; MAMAS According to Shaffer: ''The Venetians were pleased to have American soldiers as visitors to their city, even this ad hoc group. They had been denied the benefits of tourism for several years. Many of the traditional Venetian craftsmen had worked during the war years and were well stocked with glassware and jewelry for sale to the soldiers.''

Your entire output must be a single JSON object with two keys: "keywords" and "rationales". The value of each key is a list of strings. Do not include any other text, explanations, or markdown formatting (e.g., ```json```) in your response.
[/INST]
"""

# Generate a response
inputs = tokenizer(
	prompt,
	return_tensors="pt", 
	truncation=True, 
	max_length=512, 
	padding=True
)

if device != 'cpu':
	inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model.generate(
	**inputs, 
	max_new_tokens=MAX_NEW_TOKENS,
	temperature=TEMPERATURE,
	top_p=TOP_P,
	do_sample=TEMPERATURE > 0.0,
	pad_token_id=tokenizer.pad_token_id,
	eos_token_id=tokenizer.eos_token_id,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)

def extract_json_from_text(text):
    """
    Extracts the JSON object from a string that may contain other text.
    """
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
json_data = extract_json_from_text(response)

# Extract the lists if the JSON data was successfully parsed
if json_data:
    keywords = json_data.get("keywords", [])
    rationales = json_data.get("rationales", [])

    print(f"Extracted {len(keywords)} Keywords({type(keywords)}): {keywords}")
    print(f"Extracted {len(rationales)} Rationales({type(rationales)}): {rationales}")
else:
    print("Could not extract JSON data from the response.")
