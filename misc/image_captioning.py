from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image
import io

# 1. Load the model and processor
processor = AutoProcessor.from_pretrained("microsoft/git-large-coco", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

# 2. Get the image from the URL
# url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Tatra_T81.jpg/640px-Tatra_T81.jpg"
url = "https://truck-encyclopedia.com/ww2/us/photos/Mack_NO.JPG"
# Add a User-Agent header to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# --- This is the ONLY block you need to get the image ---
# It uses the headers and includes error checking.
response = requests.get(url, headers=headers)
response.raise_for_status()  # This will check for download errors (like 404, 403, etc.)
image = Image.open(io.BytesIO(response.content))
# --- End of image loading ---

# 3. Process the image and generate a caption
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_caption)