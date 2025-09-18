from utils import *
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
TOP_K = 3

# Load the model and tokenizer
model_id = "meta-llama/Llama-3.2-3B-Instruct"
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

Description: Medicine salesman, Algiers, 1943 According to Shaffer: ''[This is] a medicine man in the Medina, Algiers, November 1943. His collection included antler, animal skin, snakeskin, ostrich egg and ostrich egg shell, dried lizard (a favorite among medicine men through the Muslim world at that time), roots, some spices and goat eyeballs. A made-to-order system prevailed: a patient would present the apothecary with a compliant; he would then pick bits and pieces from his stockpile, grind them in a mortar, throw in a little goat fat and hand it over. I have no clinical data to demonstrate the value of his services, but people on the desert lived for centuries (and in some areas still do) with this level of medical skill at their disposal.''
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