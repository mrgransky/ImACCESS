from numpy import dtype
from torch import device
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Set the environment variable as suggested by the error message
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
device=torch.device("cuda:0")
# --- 1. Configure 4-bit Quantization ---
bnb_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.bfloat16,
	bnb_4bit_use_double_quant=True,
)

# --- 2. Load the tokenizer and the model ---
tokenizer = AutoTokenizer.from_pretrained(
	model_name,
	trust_remote_code=True,
	cache_dir=cache_directory[USER],
)
model = AutoModelForCausalLM.from_pretrained(
	model_name,
	quantization_config=bnb_config,
	device_map=device,
	# attn_implementation="flash_attention_2", # Use Flash Attention for efficiency
	trust_remote_code=True,
  dtype=torch.bfloat16,
	cache_dir=cache_directory[USER],
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
		{"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
		**model_inputs,
		max_new_tokens=2048 # Start with a reasonable number
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)