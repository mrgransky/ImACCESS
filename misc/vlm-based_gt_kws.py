from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
import requests
import torch

model_id = "tiiuae/falcon-11B-vlm"
# model_id = "utter-project/EuroVLM-1.7B-Preview"

processor = LlavaNextProcessor.from_pretrained(model_id, tokenizer_class='PreTrainedTokenizerFast', use_fast=True)
model = LlavaNextForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16)

url = "https://digitalcollections.smu.edu/digital/api/singleitem/image/stn/989/default.jpg"
cats_image = Image.open(requests.get(url, stream=True).raw)
print(cats_image.size)
instruction = 'Describe the image in three words.'

prompt = f"User: <image>\n{instruction} Assistant:"
inputs = processor(prompt, images=cats_image, return_tensors="pt", padding=True).to('cuda:0')

model.to('cuda:0')
output = model.generate(**inputs, max_new_tokens=256)


prompt_length = inputs['input_ids'].shape[1]
results = processor.decode(output[0], skip_special_tokens=True).strip()

print(results)