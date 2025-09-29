from utils import *

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

# Load and preprocess image
url = "https://digitalcollections.smu.edu/digital/api/singleitem/image/stn/989/default.jpg"
img = Image.open(requests.get(url, stream=True).raw)#.convert('RGB')
print(f"IMG: {type(img)} {img.size} {img.mode}")

processor = tfs.LlavaNextProcessor.from_pretrained(model_id, use_fast=True, trust_remote_code=True, cache_dir=cache_directory[USER],)
model = tfs.LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=cache_directory[USER],)
model.to("cuda:0")

instruction = 'Act as a meticulous historical archivist specializing in 20th century documentation. Describe the image in three keywords.'
conversation = [
	{
		"role": "user",
		"content": [
			{"type": "text", "text": instruction},
			{"type": "image"},
		],
	},
]
txt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=img, text=txt, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=256)
print("Generated output:")
print(processor.decode(output[0], skip_special_tokens=True))
print("="*100)
