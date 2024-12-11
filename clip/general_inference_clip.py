import os
import torch
import clip
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.metrics import precision_score, recall_score
from typing import List
device = "cuda" if torch.cuda.is_available() else "cpu"

# $ nohup python -u general_inference_clip.py > /media/volume/trash/ImACCESS/prec_at_K.out &

def load_model():
	model, preprocess = clip.load("ViT-B/32", device=device)
	input_resolution = model.visual.input_resolution
	context_length = model.context_length
	vocab_size = model.vocab_size
	print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
	print("Input resolution:", input_resolution)
	print("Context length:", context_length)
	print("Vocab size:", vocab_size)
	return model, preprocess

def load_dataset(transform=None):
	dataset = CIFAR10(
		root=os.path.expanduser("~/.cache"), 
		transform=transform, 
		download=True,
		# train=False,
	)
	# dataset = CIFAR100(os.path.expanduser("~/.cache"), transform=transform, download=True)
	return dataset

def tokenize_(labels:List[str]=["dog", "cat"]):
	tokenized_labels = clip.tokenize(texts=labels) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	return tokenized_labels # <class 'torch.Tensor'> torch.Size([10, 77])

def compute_similarities(model, image, tokenized_labels):
	image_features = model.encode_image(image)
	tokenized_labels_features = model.encode_text(tokenized_labels)
	similarities = (100.0 * image_features @ tokenized_labels_features.T).softmax(dim=-1)
	return similarities

def zero_shot(img_path:str="/home/farid/WS_Farid/ImACCESS/TEST_IMGs/dog.jpeg"):
	model, preprocess = load_model()
	
	dataset = load_dataset(preprocess) # cifar10 or cifar 100
	labels = dataset.classes

	tokenized_labels_tensor = tokenize_(labels=labels).to(device) # <class 'torch.Tensor'> torch.Size([10, 77])
	
	img = Image.open(img_path)
	image_tensor = preprocess(img).unsqueeze(0).to(device) # <class 'torch.Tensor'> torch.Size([1, 3, 224, 224])

	topk = 5
	# Compute the similarities
	similarities = compute_similarities(model, image_tensor, tokenized_labels_tensor)
	topk_probs, topk_labels_idx = similarities.topk(topk, dim=-1)
	print(topk_probs)
	print(topk_labels_idx)
	print("#"*100)
	print(f"Top-{topk} predicted labels: {[dataset.classes[i] for i in topk_labels_idx.cpu().numpy().flatten()]}")

def get_prec_at_(K:int=5):
	# Precision at K measures how many items with the top K positions are relevant. 
	model, preprocess = clip.load("ViT-B/32", device=device)
	dataset = CIFAR100(
		root=os.path.expanduser("~/.cache"), 
		transform=None,
		download=True,
		train=False,
	) # <class 'torchvision.datasets.cifar.CIFAR10'>
	print(dataset)

	labels = dataset.classes # <class 'list'> ['airplane', 'automobile', ...]
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77

	predicted_labels = []
	true_labels = []

	for i, (img_raw, gt_lbl) in enumerate(dataset): #img: <class 'PIL.Image.Image'>
		img_tensor = preprocess(img_raw).unsqueeze(0).to(device)
		image_features = model.encode_image(img_tensor)
		labels_features = model.encode_text(tokenized_labels_tensor)
		similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
		_, topk_labels_idx = similarities.topk(K, dim=-1)
		predicted_labels.append(topk_labels_idx.cpu().numpy().flatten())
		true_labels.append(gt_lbl)
	print(len(predicted_labels), len(true_labels))
	print(type(predicted_labels[0]), predicted_labels[0].shape,)
	print("#"*100)
	print(predicted_labels[:10])
	print(true_labels[:10])
	prec_at_k = 0
	for i, v in enumerate(true_labels):
		preds = predicted_labels[i] # <class 'numpy.ndarray'>
		if v in preds:
			prec_at_k += 1
	avg_prec_at_k = prec_at_k/len(true_labels)
	print(f"[OWN] top-{K}: {prec_at_k} | {avg_prec_at_k}")

	# Calculate Precision at K
	prec_at_k = sum(1 for i, v in enumerate(true_labels) if v in predicted_labels[i])
	avg_prec_at_k = prec_at_k / len(true_labels)
	
	# Calculate Recall at K
	recall_at_k = sum(1 / K for i, v in enumerate(true_labels) if v in predicted_labels[i])
	avg_recall_at_k = recall_at_k / len(true_labels)
	
	print(f"top-{K} Precision: {prec_at_k} | {avg_prec_at_k}")
	print(f"top-{K} Recall: {recall_at_k} | {avg_recall_at_k}")

def main():
	print(clip.available_models())
	# zero_shot() # only for a given image
	get_prec_at_(K=5)

if __name__ == "__main__":
		main()