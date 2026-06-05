import os
from typing import Dict, Any
from transformers import AutoProcessor, AutoModelForMultimodalLM
HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER

cache_directory = {
	"farid": "/home/farid/datasets/models",
	"alijanif": "/scratch/project_2004072/models",
	"ubuntu": "/media/volume/models",
}

MODEL_ID = "google/gemma-4-E2B-it"
base_model_kwargs: Dict[str, Any] = {
	"low_cpu_mem_usage": True,
	"trust_remote_code": True,
	"cache_dir": cache_directory[USER],
}

# Load model
processor = AutoProcessor.from_pretrained(
  MODEL_ID,
	trust_remote_code=True,
	cache_dir=cache_directory[USER],  
)
model = AutoModelForMultimodalLM.from_pretrained(
	MODEL_ID,
  **base_model_kwargs,
	dtype="auto",
	device_map="auto",
)
# Prompt - add image before text
messages = [
 	{
		"role": "user", "content": [
			{"type": "image", "image": "/home/farid/Bilder/GoldenGate.png"},
			{"type": "text", "text": "What is shown in this image?"}
		]
	}
]

# Process input
inputs = processor.apply_chat_template(
	messages,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
	add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]
print(type(inputs["input_ids"]), inputs["input_ids"].shape, input_len)

# Generate output
outputs = model.generate(**inputs, max_new_tokens=128)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
res = processor.parse_response(response)
print(res)