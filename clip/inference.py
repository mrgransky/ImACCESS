import os
import torch
import clip
import time
import re
import argparse
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

parser = argparse.ArgumentParser(description="Evaluate CLIP for CIFAR10x")
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--query_image', '-qi', type=str, default="/home/farid/WS_Farid/ImACCESS/TEST_IMGs/dog.jpeg", help='image path for zero shot classification')
parser.add_argument('--query_label', '-ql', type=str, default="airplane", help='image path for zero shot classification')
parser.add_argument('--topK', '-k', type=int, default=5, help='TopK results')
parser.add_argument('--batch_size', '-ba', type=int, default=1024, help='TopK results')
parser.add_argument('--dataset', '-d', type=str, choices=['CIFAR10', 'CIFAR100'], default='CIFAR10', help='Choose dataset (CIFAR10/CIFAR100)')

args, unknown = parser.parse_known_args()
print(args)

args.device = torch.device(args.device)

# $ nohup python -u inference.py > /media/volume/ImACCESS/trash/prec_at_K.out &

USER = os.getenv('USER')
print(f"USER: {USER} device: {args.device}")

def load_model():
	model, preprocess = clip.load("ViT-B/32", device=args.device)
	input_resolution = model.visual.input_resolution
	context_length = model.context_length
	vocab_size = model.vocab_size
	print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
	print("Input resolution:", input_resolution)
	print("Context length:", context_length)
	print("Vocab size:", vocab_size)
	return model, preprocess

def get_dataset(dname:str="CIFAR10"):
	if dname == 'CIFAR10':
			dataset = CIFAR10(
					root=os.path.expanduser("~/.cache"), 
					transform=None,
					download=True,
					train=False,  # split Test
			)
	elif dname == 'CIFAR100':
			dataset = CIFAR100(
					root=os.path.expanduser("~/.cache"), 
					transform=None,
					download=True,
					train=False,  # split Test
			)
	else:
			raise ValueError(f"Invalid dataset name: {dname}. Supported datasets are 'CIFAR10' and 'CIFAR100'.")
	print(dataset)
	return dataset

def get_zero_shot(dataset, model, preprocess, img_path, topk:int=5):
	print(f"Zero-Shot Image Classification: {img_path}".center(160, " "))
	labels = dataset.classes
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(args.device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	img = Image.open(img_path)
	image_tensor = preprocess(img).unsqueeze(0).to(args.device) # <class 'torch.Tensor'> torch.Size([1, 3, 224, 224])

	with torch.no_grad():
		image_features = model.encode_image(image_tensor)
		labels_features = model.encode_text(tokenized_labels_tensor)

	image_features /= image_features.norm(dim=-1, keepdim=True)
	labels_features /= labels_features.norm(dim=-1, keepdim=True)

	similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
	topk_pred_probs, topk_pred_labels_idx = similarities.topk(topk, dim=-1)
	print(topk_pred_probs)
	print(topk_pred_labels_idx)
	print(f"Top-{topk} predicted labels: {[labels[i] for i in topk_pred_labels_idx.cpu().numpy().flatten()]}")
	print("-"*160)

def get_zero_shot_precision_at_(dataset, model, preprocess, K:int=5):
	print(f"Zero-Shot Image Classification {args.device} CLIP [performance metrics: Precision@{K}]".center(160, " "))
	
	labels = dataset.classes # <class 'list'> ['airplane', 'automobile', ...]
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(args.device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	labels_features = model.encode_text(tokenized_labels_tensor)
	labels_features /= labels_features.norm(dim=-1, keepdim=True)

	predicted_labels = []
	true_labels = []
	floop_st = time.time()

	with torch.no_grad():
		for i, (img_raw, gt_lbl) in enumerate(dataset): #img: <class 'PIL.Image.Image'>
			img_tensor = preprocess(img_raw).unsqueeze(0).to(args.device)
			image_features = model.encode_image(img_tensor)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
			_, topk_labels_idx = similarities.topk(K, dim=-1)
			predicted_labels.append(topk_labels_idx.cpu().numpy().flatten())
			true_labels.append(gt_lbl)
	print(f"Total (for loop): {time.time()-floop_st:.3f} sec")
	print(len(predicted_labels), len(true_labels))
	print(type(predicted_labels[0]), predicted_labels[0].shape,)
	print(predicted_labels[:10])
	print(true_labels[:10])

	##################################################################################################
	pred_st = time.time()
	prec_at_k = 0
	for ilbl, vlbl in enumerate(true_labels):
		preds = predicted_labels[ilbl] # <class 'numpy.ndarray'>
		if vlbl in preds:
			prec_at_k += 1
	avg_prec_at_k = prec_at_k/len(true_labels)
	print(f"[OWN] Precision@{K}: {prec_at_k} | {avg_prec_at_k} Elapsed_t: {time.time()-pred_st:.2f} sec")
	##################################################################################################

	# pred_st = time.time()
	# # Calculate Precision at K
	# prec_at_k = sum(1 for i, v in enumerate(true_labels) if v in predicted_labels[i])
	# avg_prec_at_k = prec_at_k / len(true_labels)
	
	# # Calculate Recall at K
	# recall_at_k = sum(1 / K for i, v in enumerate(true_labels) if v in predicted_labels[i])
	# avg_recall_at_k = recall_at_k / len(true_labels)
	
	# print(
	# 	f"top-{K} Precision: {prec_at_k} | {avg_prec_at_k} "
	# 	f"Recall: {recall_at_k} | {avg_recall_at_k} "
	# 	f"Elapsed_t: {time.time()-pred_st:.2f} sec"
	# )
	print("-"*160)

def get_image_retrieval(dataset, model, preprocess, query:str="cat", topk:int=5, batch_size:int=1024):
	print(f"Top-{topk} Image Retrieval {args.device} CLIP Query: « {query} »".center(160, " "))
	labels = dataset.classes
	tokenized_query_tensor = clip.tokenize(texts=query).to(args.device) #<class 'torch.Tensor'> torch.Size([1, 77])
	query_features = model.encode_text(tokenized_query_tensor) # <class 'torch.Tensor'> torch.Size([1, 512])
	query_features /= query_features.norm(dim=-1, keepdim=True)
	# Encode all the images
	all_image_features = []
	for i in range(0, len(dataset), batch_size):
		batch_images = [dataset[j][0] for j in range(i, min(i + batch_size, len(dataset)))]
		batch_tensors = torch.stack([preprocess(img).to(args.device) for img in batch_images])
		with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
			image_features = model.encode_image(batch_tensors)
			image_features /= image_features.norm(dim=-1, keepdim=True)
		all_image_features.append(image_features)
		torch.cuda.empty_cache() # Clear CUDA cache

	all_image_features = torch.cat(all_image_features, dim=0)

	# Compute similarities between query and all images
	similarities = (100.0 * query_features @ all_image_features.T).softmax(dim=-1)

	# Get the top-k most similar images
	topk_probs, topk_indices = similarities.topk(topk, dim=-1)
	
	# Retrieve the top-k images
	topk_pred_images = [dataset[topk_indices.squeeze().item()][0]] if topk==1 else [dataset[idx][0] for idx in topk_indices.squeeze().cpu().numpy()] # [<PIL.Image.Image image mode=RGB size=32x32 at 0x7C16A47D8F40>, ...]
	topk_pred_labels = [dataset[topk_indices.squeeze().item()][1]] if topk==1 else [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()] # [3, 3, 8, 8, 3]

	topk_probs = topk_probs.squeeze().cpu().detach().numpy()
	print(topk_pred_images)
	print(topk_pred_labels, labels)
	print(topk_probs)

	# Save the top-k images in a single file
	fig, axes = plt.subplots(1, topk, figsize=(16, 8))
	if topk == 1:
		axes = [axes]  # Convert to list of axes
	fig.suptitle(f"Top-{topk} Query: {query}", fontsize=11)
	for i, (img, ax) in enumerate(zip(topk_pred_images, axes)):
		ax.imshow(img)
		ax.axis('off')
		# ax.set_title(f"Top-{i+1}\nprob: {topk_probs[i]:.8f}\nGT: {labels[topk_pred_labels[i]]}", fontsize=9)
		if topk == 1:
			ax.set_title(f"Top-1\nprob: {topk_probs:.8f}\nGT: {labels[topk_pred_labels[0]]}", fontsize=9)
		else:
			ax.set_title(f"Top-{i+1}\nprob: {topk_probs[i]:.8f}\nGT: {labels[topk_pred_labels[i]]}", fontsize=9)
				
	plt.tight_layout()
	plt.savefig(f"top{topk}_IMGs_query_{re.sub(' ', '_', query)}.png")
	print("-"*160)

def get_image_retrieval_precision_recall_at_(dataset, model, preprocess, K:int=5, batch_size:int=1024):
	torch.cuda.empty_cache()  # Clear CUDA cache
	print(f"Image Retrieval {args.device} CLIP [performance metrics: Precision@{K}]".center(160, " "))
	labels = dataset.classes
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(args.device)#<class 'torch.Tensor'> torch.Size([num_lbls, 77])
	tokenized_labels_features = model.encode_text(tokenized_labels_tensor) # <class 'torch.Tensor'> torch.Size([num_lbls, 512])
	tokenized_labels_features /= tokenized_labels_features.norm(dim=-1, keepdim=True)
	# Encode all the images
	all_image_features = []
	for i in range(0, len(dataset), batch_size):
		batch_images = [dataset[j][0] for j in range(i, min(i + batch_size, len(dataset)))]
		batch_tensors = torch.stack([preprocess(img).to(args.device) for img in batch_images])
		with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
			image_features = model.encode_image(batch_tensors)
			image_features /= image_features.norm(dim=-1, keepdim=True)
		all_image_features.append(image_features)
		torch.cuda.empty_cache()  # Clear CUDA cache

	all_image_features = torch.cat(all_image_features, dim=0)

	###################################### Wrong approach for P@K and R@K ######################################
	# it is rather accuracy or hit rate!
	# prec_at_k = 0
	# recall_at_k = []
	# for i, label_features in enumerate(tokenized_labels_features):
	# 	sim = (100.0 * label_features @ all_image_features.T).softmax(dim=-1) # similarities between query and all images
	# 	topk_probs, topk_indices = sim.topk(K, dim=-1)
	# 	topk_pred_labels = [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()] # [3, 3, 8, 8, 3]
	# 	recall_at_k.append(topk_pred_labels.count(i)/K)
	# 	if i in topk_pred_labels: # just checking if the label is present
	# 		prec_at_k += 1
	# avg_prec_at_k = prec_at_k / len(tokenized_labels_features)
	# avg_recall_at_k = sum(recall_at_k) / len(labels)
	# print(f"Precision@{K}: {prec_at_k} {avg_prec_at_k}")
	# print(f"Recall@{K}: {recall_at_k} {avg_recall_at_k} {np.mean(recall_at_k)}")
	# print(labels)
	###################################### Wrong approach for P@K and R@K ######################################

	prec_at_k = []
	recall_at_k = []
	for i, label_features in enumerate(tokenized_labels_features):
		sim = (100.0 * label_features @ all_image_features.T).softmax(dim=-1) # similarities between query and all images
		topk_probs, topk_indices = sim.topk(K, dim=-1)
		# topk_pred_labels = [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()] # [3, 3, 8, 8, 3] # only K@(>1)
		topk_pred_labels = [dataset[topk_indices.squeeze().item()][1]] if K==1 else [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()]# K@1, 5, ...
		relevant_retrieved_images_for_label_i = topk_pred_labels.count(i)  # counting relevant images in top-K retrieved images
		prec_at_k.append(relevant_retrieved_images_for_label_i/K)
		all_images_with_label_i = [idx for idx, (img, lbl) in enumerate(dataset) if lbl == i]
		num_all_images_with_label_i = len(all_images_with_label_i)
		recall_at_k.append(relevant_retrieved_images_for_label_i/num_all_images_with_label_i)

	avg_prec_at_k = sum(prec_at_k)/len(labels)
	avg_recall_at_k = sum(recall_at_k) / len(labels)
	# print(f"Precision@{K}: {prec_at_k} {avg_prec_at_k} {np.mean(prec_at_k)}")
	# print(f"Recall@{K}: {recall_at_k} {avg_recall_at_k} {np.mean(recall_at_k)}")
	print(f"Precision@{K}: {avg_prec_at_k} {np.mean(prec_at_k)}")
	print(f"Recall@{K}: {avg_recall_at_k} {np.mean(recall_at_k)}")
	print(labels)

	fpr_values = []
	tpr_values = []
	precision_values = []
	recall_values = []
	for i, label_features in enumerate(tokenized_labels_features):
		sim = (100.0 * label_features @ all_image_features.T).softmax(dim=-1)
		sim = sim.squeeze().cpu().detach().numpy()
		predicted_labels = np.argsort(-sim)
		true_labels = [1 if dataset[j][1] == i else 0 for j in range(len(dataset))]
		prec, rec, thresh = precision_recall_curve(true_labels, sim)
		fpr, tpr, _ = roc_curve(true_labels, sim)
		precision_values.append(prec)
		recall_values.append(rec)
		fpr_values.append(fpr)
		tpr_values.append(tpr)

	fig, ax = plt.subplots(1, 2, figsize=(22,11))
	for i in range(len(tokenized_labels_features)):
		ax[0].plot(recall_values[i], precision_values[i], label=f'{labels[i]}')
		ax[0].set_xlabel('Recall')
		ax[0].set_ylabel('Precision')
		ax[0].set_title('Precision-Recall')
		ax[1].plot(fpr_values[i], tpr_values[i])
		ax[1].set_xlabel('False Positive')
		ax[1].set_ylabel('True Positive')
		ax[1].set_title('ROC')

	fig.legend(bbox_to_anchor=(0.5, 0.99), fontsize=7, ncol=len(labels), frameon=False)
	# fig.tight_layout()
	plt.savefig(f"{args.dataset}_PR_ROC_x{len(labels)}_labels_prec_at_{K}.png")

	print("-"*160)

def main():
	print(clip.available_models())
	model, preprocess = load_model()
	dataset = get_dataset(dname=args.dataset)

	if USER == "farid":
		get_zero_shot(
			dataset=dataset,
			model=model,
			preprocess=preprocess,
			img_path=args.query_image,
			topk=args.topK,
		)

	get_zero_shot_precision_at_(
		dataset=dataset,
		model=model,
		preprocess=preprocess,
		K=args.topK,
	)

	if USER == "farid":
		get_image_retrieval(
			dataset=dataset,
			model=model,
			preprocess=preprocess,
			query=args.query_label,
			topk=args.topK,
			batch_size=args.batch_size,
		)

	if USER == "farid":
		for q in ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']:
			get_image_retrieval(
				dataset=dataset,
				model=model,
				preprocess=preprocess,
				query=q,
				topk=args.topK,
				batch_size=args.batch_size,
			)

	get_image_retrieval_precision_recall_at_(
		dataset=dataset,
		model=model,
		preprocess=preprocess,
		K=args.topK,
		batch_size=args.batch_size,
	)

if __name__ == "__main__":
	main()