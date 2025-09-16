# import transformers as tfs
# import requests
# from PIL import Image
# import io
# import torch

# # model_id = "Salesforce/blip-image-captioning-large"
# model_id = "microsoft/git-large-coco"
# # model_id = "openai/clip-vit-base-patch32"
# # model_id = "google/vit-base-patch16-224"

# # Load config
# config = tfs.AutoConfig.from_pretrained(model_id)
# print(f"Model type for {model_id}: {config.model_type}")
# print(f"Architectures: {config.architectures}")

# # Always safe to load processor this way
# processor = tfs.AutoProcessor.from_pretrained(model_id, use_fast=True)

# # Dynamically pick the first architecture class (if available)
# if config.architectures:
# 		cls_name = config.architectures[0]
# 		model_cls = getattr(tfs, cls_name)
# 		model = model_cls.from_pretrained(model_id, config=config, device_map="auto", dtype="auto")
# else:
# 		# Fallback: generic AutoModel (works for base encoders like ViT, BERT, etc.)
# 		try:
# 				model = tfs.AutoModel.from_pretrained(model_id, config=config, device_map="auto", dtype="auto")
# 		except Exception:
# 				# Some cases need task-specific auto-classes
# 				model = tfs.AutoModelForImageClassification.from_pretrained(model_id, config=config, device_map="auto", dtype="auto")

# print(f"Loaded model class: {model.__class__.__name__}")

# model = model.eval()
# device = next(model.parameters()).device  # Get the device of the model
# print(f"Model loaded on {device}")

# # 2. Get the image from the URL
# url = "https://truck-encyclopedia.com/ww2/us/photos/Mack_NO.JPG"

# # Add a User-Agent header to mimic a browser
# headers = {
# 		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
# }

# # Download the image
# response = requests.get(url, headers=headers)
# response.raise_for_status()  # This will check for download errors (like 404, 403, etc.)
# image = Image.open(io.BytesIO(response.content))

# # 3. Process the image and generate a caption
# inputs = processor(images=image, return_tensors="pt").to(device)

# # Generate caption
# generated_ids = model.generate(**inputs, max_length=50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_caption)


import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

prompt = "<OD>"

url = "https://cas-bridge.xethub.hf.co/xet-bridge-us/621ffdd236468d709f1835cf/dcf539b14bbf0a3b2dae3f6e94112b0762acf325d7edc17ff640a3390719e8ef?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20250916%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250916T084420Z&X-Amz-Expires=3600&X-Amz-Signature=36701c363d678f2d171f890ef50f722d3080937ef545cc08b0ecc064205eebb4&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=64425ec7afba527ed36637ba&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27car.jpg%3B+filename%3D%22car.jpg%22%3B&response-content-type=image%2Fjpeg&x-id=GetObject&Expires=1758015860&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1ODAxNTg2MH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82MjFmZmRkMjM2NDY4ZDcwOWYxODM1Y2YvZGNmNTM5YjE0YmJmMGEzYjJkYWUzZjZlOTQxMTJiMDc2MmFjZjMyNWQ3ZWRjMTdmZjY0MGEzMzkwNzE5ZThlZioifV19&Signature=QfONBHcFhn3TI6oCLEC1rmHeG%7EtVFdoCMFtYT7eW-amo%7E3rDuNd1ifdPdM6uO8PNczIi5Rbln%7EU4PrI7NWjgJ4V8VcEIJPnq3h1uTy%7EeOGxFjNFLZAnAmZhOebL4h0nkoZykJA69ywZn6NU6mkq1Wl5-fajc9sppeD0LkmZRYRJ8NjYnh-SwpkwjWuHxNDZV99whSTGuM37CTtf8V1buseTLKSUcBza7GbPG3g0oDWOB9Rer7proAE5MKzR0rIbo0scUZGCjU3ysYsNZTfNJAUD9iE8wdnOJVw1F%7E%7ELOINmK7EMpImIHrijoaDltApaFSK4toa5J5XZAt1hjDVF7pw__&Key-Pair-Id=K2L8F4GPSG1IFC"
image = Image.open(requests.get(url, stream=True).raw)
print(f"Image size: {image.size}")
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=4096,
    num_beams=3,
    do_sample=False
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)
