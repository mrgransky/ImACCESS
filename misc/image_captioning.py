import transformers as tfs
import requests
from PIL import Image
import io
import torch

# model_id = "Salesforce/blip-image-captioning-large"
model_id = "microsoft/git-large-coco"
# model_id = "openai/clip-vit-base-patch32"
# model_id = "google/vit-base-patch16-224"

# Load config
config = tfs.AutoConfig.from_pretrained(model_id)
print(f"Model type for {model_id}: {config.model_type}")
print(f"Architectures: {config.architectures}")

# Always safe to load processor this way
processor = tfs.AutoProcessor.from_pretrained(model_id, use_fast=True)

# Dynamically pick the first architecture class (if available)
if config.architectures:
		cls_name = config.architectures[0]
		model_cls = getattr(tfs, cls_name)
		model = model_cls.from_pretrained(model_id, config=config, device_map="auto", dtype="auto")
else:
		# Fallback: generic AutoModel (works for base encoders like ViT, BERT, etc.)
		try:
				model = tfs.AutoModel.from_pretrained(model_id, config=config, device_map="auto", dtype="auto")
		except Exception:
				# Some cases need task-specific auto-classes
				model = tfs.AutoModelForImageClassification.from_pretrained(model_id, config=config, device_map="auto", dtype="auto")

print(f"Loaded model class: {model.__class__.__name__}")

model = model.eval()
device = next(model.parameters()).device  # Get the device of the model
print(f"Model loaded on {device}")

# 2. Get the image from the URL
url = "https://truck-encyclopedia.com/ww2/us/photos/Mack_NO.JPG"

# Add a User-Agent header to mimic a browser
headers = {
		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Download the image
response = requests.get(url, headers=headers)
response.raise_for_status()  # This will check for download errors (like 404, 403, etc.)
image = Image.open(io.BytesIO(response.content))

# 3. Process the image and generate a caption
inputs = processor(images=image, return_tensors="pt").to(device)

# Generate caption
generated_ids = model.generate(**inputs, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)