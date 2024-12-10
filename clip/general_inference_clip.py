import os
import torch
import clip
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100

device = "cuda" if torch.cuda.is_available() else "cpu"
print(clip.available_models())

model, preprocess = clip.load("ViT-B/32", device=device)
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

img = Image.open("/home/farid/WS_Farid/ImACCESS/TEST_IMGs/dog.jpeg") # <class 'PIL.JpegImagePlugin.JpegImageFile'> (863, 625)
image = preprocess(img).unsqueeze(0).to(device) # <class 'torch.Tensor'> torch.Size([1, 3, 224, 224])

cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
cifar10 = CIFAR10(os.path.expanduser("~/.cache"), transform=preprocess, download=True)

# LABELS = [f"{label}" for label in cifar100.classes]
LABELS = [f"{label}" for label in cifar10.classes] # list
print(len(LABELS))
print(LABELS)

tokenized_labels_tensor = clip.tokenize(texts=LABELS).to(device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
with torch.no_grad():
	image_features = model.encode_image(image)
	text_features = model.encode_text(tokenized_labels_tensor)

	logits_per_image, logits_per_text = model(image, tokenized_labels_tensor)
	probs = logits_per_image.softmax(dim=-1).cpu().numpy()

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"{len(LABELS)} Label probs:", probs)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

print(top_probs, top_labels)
print("#"*100)
print([LABELS[i] for i in top_labels.numpy().flatten()])