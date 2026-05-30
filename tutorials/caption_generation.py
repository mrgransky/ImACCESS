# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# import torch

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")

# image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(images=image, text="Question: What is the weather in this image? Answer:", return_tensors="pt").to("cuda", torch.float16)

# outputs = model.generate(**inputs, max_new_tokens=100)
# answer = processor.decode(outputs[0], skip_special_tokens=True)

# print(answer)

from transformers import pipeline

generator = pipeline('text-generation', model="facebook/opt-2.7b")
response = generator("What are we having for dinner?")
print(response)