from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

model_id = "facebook/dinov2-giant"
url = 'https://i.pinimg.com/564x/e3/06/2a/e3062ab542537be8403d2b84ae120ad7.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
model = AutoModel.from_pretrained(model_id)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state


print(type(last_hidden_states), last_hidden_states.shape, last_hidden_states.dtype)