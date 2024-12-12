import os
import torch
import clip
import time
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
from typing import List
import matplotlib.pyplot as plt

# $ nohup python -u inference.py > /media/volume/ImACCESS/trash/prec_at_K.out &

USER = os.getenv('USER')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"USER: {USER} device: {device}")

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

def load_dataset():
	dataset = CIFAR10(
		root=os.path.expanduser("~/.cache"), 
		transform=None,
		download=True,
		train=False, # split Test
	)
	print(dataset)
	return dataset

def tokenize_(labels:List[str]=["dog", "cat"]):
	tokenized_labels = clip.tokenize(texts=labels) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	return tokenized_labels # <class 'torch.Tensor'> torch.Size([10, 77])

def compute_similarities(model, image, tokenized_labels):
	image_features = model.encode_image(image)
	tokenized_labels_features = model.encode_text(tokenized_labels)
	similarities = (100.0 * image_features @ tokenized_labels_features.T).softmax(dim=-1)
	return similarities

def get_zero_shot(img_path:str="/home/farid/WS_Farid/ImACCESS/TEST_IMGs/dog.jpeg"):
	model, preprocess = load_model()
	
	dataset = load_dataset() # cifar10 or cifar 100
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

def get_zero_shot_precision_at_(K:int=5):
	# Precision at K measures how many items with the top K positions are relevant. 
	model, preprocess = clip.load("ViT-B/32", device=device)
	dataset = CIFAR100(
		root=os.path.expanduser("~/.cache"), 
		transform=None,
		download=True,
		# train=False,
	) # <class 'torchvision.datasets.cifar.CIFAR10'>
	print(dataset)

	labels = dataset.classes # <class 'list'> ['airplane', 'automobile', ...]
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	labels_features = model.encode_text(tokenized_labels_tensor)
	
	predicted_labels = []
	true_labels = []
	floop_st = time.time()
	for i, (img_raw, gt_lbl) in enumerate(dataset): #img: <class 'PIL.Image.Image'>
		img_tensor = preprocess(img_raw).unsqueeze(0).to(device)
		image_features = model.encode_image(img_tensor)
		similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
		_, topk_labels_idx = similarities.topk(K, dim=-1)
		predicted_labels.append(topk_labels_idx.cpu().numpy().flatten())
		true_labels.append(gt_lbl)
	print(f"Total (for loop): {time.time()-floop_st:.3f} sec")
	print(len(predicted_labels), len(true_labels))
	print(type(predicted_labels[0]), predicted_labels[0].shape,)
	print("#"*100)
	print(predicted_labels[:10])
	print(true_labels[:10])

	##################################################################################################
	pred_st = time.time()
	prec_at_k = 0
	for i, v in enumerate(true_labels):
		preds = predicted_labels[i] # <class 'numpy.ndarray'>
		if v in preds:
			prec_at_k += 1
	avg_prec_at_k = prec_at_k/len(true_labels)
	print(f"[OWN] top-{K}: {prec_at_k} | {avg_prec_at_k} Elapsed_t: {time.time()-pred_st:.2f} sec")
	##################################################################################################

	pred_st = time.time()
	# Calculate Precision at K
	prec_at_k = sum(1 for i, v in enumerate(true_labels) if v in predicted_labels[i])
	avg_prec_at_k = prec_at_k / len(true_labels)
	
	# Calculate Recall at K
	recall_at_k = sum(1 / K for i, v in enumerate(true_labels) if v in predicted_labels[i])
	avg_recall_at_k = recall_at_k / len(true_labels)
	
	print(
		f"top-{K} Precision: {prec_at_k} | {avg_prec_at_k} "
		f"Recall: {recall_at_k} | {avg_recall_at_k} "
		f"Elapsed_t: {time.time()-pred_st:.2f} sec"
	)

def get_image_retrieval(query:str="cat", topk:int=5, batch_size:int=1024):
	print(f"Image Retrieval {device} CLIP".center(100, "-"))
	print(f"Top-{topk} image(s) for Query: « {query} »".center(100, " "))
	model, preprocess = load_model()
	dataset = load_dataset()
	labels = dataset.classes
	tokenized_query_tensor = clip.tokenize(texts=query).to(device)#<class 'torch.Tensor'> torch.Size([1, 77])
	query_features = model.encode_text(tokenized_query_tensor) # <class 'torch.Tensor'> torch.Size([1, 512])
	
	# Encode all the images
	all_image_features = []
	for i in range(0, len(dataset), batch_size):
		batch_images = [dataset[j][0] for j in range(i, min(i + batch_size, len(dataset)))]
		batch_tensors = torch.stack([preprocess(img).to(device) for img in batch_images])
		with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
			image_features = model.encode_image(batch_tensors)
		all_image_features.append(image_features)
		torch.cuda.empty_cache()  # Clear CUDA cache

	all_image_features = torch.cat(all_image_features, dim=0)

	# Compute similarities between query and all images
	similarities = (100.0 * query_features @ all_image_features.T).softmax(dim=-1)

	# Get the top-k most similar images
	topk_probs, topk_indices = similarities.topk(topk, dim=-1)
	
	# Retrieve the top-k images
	topk_pred_images = [dataset[idx][0] for idx in topk_indices.squeeze().cpu().numpy()] # [<PIL.Image.Image image mode=RGB size=32x32 at 0x7C16A47D8F40>, ...]
	topk_pred_labels = [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()] # [3, 3, 8, 8, 3]
	topk_probs = topk_probs.squeeze().cpu().detach().numpy()
	print(topk_pred_images)
	print(topk_pred_labels, dataset.classes)
	print(topk_probs)

	# Save the top-k images in a single file
	fig, axes = plt.subplots(1, topk, figsize=(12, 5))
	fig.suptitle(f"Top-{topk} Query: {query}", fontsize=11)
	for i, img in enumerate(topk_pred_images):
		axes[i].imshow(img)
		axes[i].axis('off')
		axes[i].set_title(f"Top-{i+1}\nprob: {topk_probs[i]:.8f}\nGT: {labels[topk_pred_labels[i]]}", fontsize=9)
		
	# plt.savefig(os.path.join("/media/volume/ImACCESS/results/", f"top{topk}_IMGs_query_{query}.png"))
	plt.tight_layout()
	plt.savefig(f"top{topk}_IMGs_query_{query}.png")

def get_image_retrieval_precision_recall_at_(K:int=5, batch_size:int=1024):
	print(f"Image Retrieval {device} CLIP [performance metrics: Precision@{K}]".center(100, "-"))
	model, preprocess = load_model()
	dataset = load_dataset()
	labels = dataset.classes
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)#<class 'torch.Tensor'> torch.Size([num_lbls, 77])
	tokenized_labels_features = model.encode_text(tokenized_labels_tensor) # <class 'torch.Tensor'> torch.Size([num_lbls, 512])
	
	# Encode all the images
	all_image_features = []
	for i in range(0, len(dataset), batch_size):
		batch_images = [dataset[j][0] for j in range(i, min(i + batch_size, len(dataset)))]
		batch_tensors = torch.stack([preprocess(img).to(device) for img in batch_images])
		with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
			image_features = model.encode_image(batch_tensors)
		all_image_features.append(image_features)
		torch.cuda.empty_cache()  # Clear CUDA cache

	all_image_features = torch.cat(all_image_features, dim=0)

	prec_at_k = 0
	recall_at_k = []
	for i, label_features in enumerate(tokenized_labels_features):
		sim = (100.0 * label_features @ all_image_features.T).softmax(dim=-1) # similarities between query and all images
		topk_probs, topk_indices = sim.topk(K, dim=-1)
		topk_pred_labels = [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()] # [3, 3, 8, 8, 3]
		recall_at_k.append(topk_pred_labels.count(i)/K)
		if i in topk_pred_labels:
			prec_at_k += 1
	avg_prec_at_k = prec_at_k / len(tokenized_labels_features)
	avg_recall_at_k = sum(recall_at_k) / len(labels)
	print(f"Precision@{K}: {prec_at_k} {avg_prec_at_k}")
	print(f"Recall@{K}: {recall_at_k} {avg_recall_at_k} {np.mean(recall_at_k)}")
	print(labels)

	precision_at_k = 0
	recall_at_k_values = []
	for i, label_feature in enumerate(tokenized_labels_features):
		similarities = (100.0 * label_feature @ all_image_features.T).softmax(dim=-1)
		_, topk_indices = similarities.topk(K, dim=-1)
		topk_pred_labels = [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()]
		relevant_images = [idx for idx, (_, label) in enumerate(dataset) if label == i]
		print(len(relevant_images))
		relevant_images_in_topk = [label for label in topk_pred_labels if label == i]
		print(len(relevant_images_in_topk), relevant_images_in_topk)
		precision_at_k += len(relevant_images_in_topk) / K
		recall_at_k_values.append(len(relevant_images_in_topk) / len(relevant_images))
	avg_precision_at_k = precision_at_k / len(tokenized_labels_features)
	avg_recall_at_k = sum(recall_at_k_values) / len(labels)
	print(f"Average Precision@{K}: {avg_precision_at_k:.4f}")
	print(f"Average Recall@{K}: {avg_recall_at_k:.4f}")

def main():
	print(clip.available_models())
	# get_zero_shot() # only for a given image
	# get_zero_shot_precision_at_(K=5)
	# get_image_retrieval(query=q)
	# for q in ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']:
	# 	get_image_retrieval(query=q)
	get_image_retrieval_precision_recall_at_(K=5)

if __name__ == "__main__":
	main()