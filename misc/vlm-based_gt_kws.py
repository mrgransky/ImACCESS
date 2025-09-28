from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
import requests
import torch

# model_id = "tiiuae/falcon-11B-vlm"
# model_id = "utter-project/EuroVLM-1.7B-Preview"
model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "llava-hf/llava-1.5-13b-hf"
# model_id = "OpenGVLab/InternVL-Chat-V1-2"

# Load and preprocess image
url = "https://digitalcollections.smu.edu/digital/api/singleitem/image/stn/989/default.jpg"
img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
print(f"IMG: {type(img)} {img.size} {img.mode}")

# Load processor and model
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to('cuda:0')

# Prepare text prompt
instruction = 'Describe the image in three words.'
prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{instruction} ASSISTANT:"

# Process inputs
inputs = processor(
    text=prompt,
    images=img,
    return_tensors="pt"
).to('cuda:0')

# Generate output
output = model.generate(**inputs, max_new_tokens=128)

# Decode output
results = processor.decode(output[0], skip_special_tokens=True).strip()
print("Generated output:")
print(results)