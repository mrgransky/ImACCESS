import os
import huggingface_hub
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

# Set environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Login to Hugging Face
USER = os.getenv('USER')
hf_tk = os.getenv("HUGGINGFACE_TOKEN")
print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True  # Enable double quantization for extra memory savings
)

# Custom device map to offload more to CPU
device_map = {
    "model.embed_tokens": "cuda:0",
    "model.layers": "cpu",  # Offload all layers to CPU
    "model.norm": "cpu",
    "lm_head": "cpu"
}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map=device_map,
    quantization_config=quantization_config
)
print(model)

# Test inference
prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")  # Keep inputs on GPU
generated_ids = model.generate(**inputs, max_new_tokens=50)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(outputs)