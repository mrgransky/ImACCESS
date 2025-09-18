from utils import *
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
TOP_K = 3

# Load the model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
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

debug_llm_info(model, tokenizer, device)


# prompt = """<s>[INST] You are a helpful assistant. Answer the following question concisely.
# Question: Hello!! What is your background in Computer Science?
# Answer: [/INST]
# """
prompt = """<s>[INST] 
As an expert historical archivist, analyze this historical description carefully and extract maximum of three (not more) concrete, factual and relevant keywords with concise rationales.
Duplicate keywords are not allowed. Keywords with numbers, temporal context and time-related information are strongly discouraged.
Description: American soldier tourists, Venice, Summer 1945 Museum and Medical Arts Service; MAMAS According to Shaffer: ''The Venetians were pleased to have American soldiers as visitors to their city, even this ad hoc group. They had been denied the benefits of tourism for several years. Many of the traditional Venetian craftsmen had worked during the war years and were well stocked with glassware and jewelry for sale to the soldiers.''

Respond in the JSON format containing two keys: "keywords" and "rationales". The value of each key is a list of strings.
Example: {"keywords": ["medicine", "salesman", "Algiers"], "rationales": ["medicine is mentioned in the description", "salesman is mentioned in the description", "Algiers is mentioned in the description"]}
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