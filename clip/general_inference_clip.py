import os
import torch
import clip
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.metrics import precision_score, recall_score

device = "cuda" if torch.cuda.is_available() else "cpu"

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
		train=False,
	)
	# dataset = CIFAR100(os.path.expanduser("~/.cache"), transform=transform, download=True)
	return dataset

def tokenize_labels(dataset):
	labels = [f"{label}" for label in dataset.classes] # list: ["dog", "cat", ...]
	tokenized_labels = clip.tokenize(texts=labels) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	return tokenized_labels

def compute_similarities(model, image, tokenized_labels):
	image_features = model.encode_image(image)
	text_features = model.encode_text(tokenized_labels)
	similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
	return similarities

def zero_shot():
	model, preprocess = load_model()
	# Load the test image
	img_path = "/home/farid/WS_Farid/ImACCESS/TEST_IMGs/dog.jpeg"
	img = Image.open(img_path)
	image = preprocess(img).unsqueeze(0).to(device)
	
	dataset = load_dataset(preprocess) # cifar10 or cifar 100
	tokenized_labels = tokenize_labels(dataset).to(device)
	
	topk = 5
	# Compute the similarities
	similarities = compute_similarities(model, image, tokenized_labels)
	topk_probs, topk_labels_idx = similarities.topk(topk, dim=-1)
	print(topk_probs)
	print(topk_labels_idx)
	print("#"*100)
	print(f"Top-{topk} predicted labels: {[dataset.classes[i] for i in topk_labels_idx.cpu().numpy().flatten()]}")

def get_prec_at_k():
	batch_size = 16
	model, preprocess = load_model()
	dataset = load_dataset() # <class 'torchvision.datasets.cifar.CIFAR10'>
	labels = dataset.classes # <class 'list'> ['airplane', 'automobile', ...]
	images = []
	for i, (img, label) in enumerate(dataset): #img: <class 'PIL.Image.Image'>
		images.append(preprocess(img).unsqueeze(0).to(device))

	# we create image embeddings and text embeddings
	with torch.no_grad():
		image_embeddings = model.encode_images(images, batch_size=batch_size)
		text_embeddings = model.encode_text(labels, batch_size=batch_size)

	# we normalize the embeddings to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
	image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
	text_embeddings = text_embeddings/np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)

	precision = 0
	# we could batch this operation to make it faster
	for index, t in enumerate(text_embeddings):
			arr = t.dot(image_embeddings.T)

			best = arr.argsort()[-5:][::-1]

			if index in best:
					precision +=1

	round(precision/len(text_embeddings), 2)

def calculate_prec_at_K(true_labels, predicted_labels, K):
		"""
		Calculate precision at K.
		
		Args:
				true_labels: The true labels.
				predicted_labels: The predicted labels.
				K: The number of top results to consider.
		
		Returns:
				prec_at_K: The precision at K.
		"""
		true_labels = true_labels.tolist()
		predicted_labels = predicted_labels[:, :K].tolist()
		prec_at_K = []
		for i in range(len(true_labels)):
				predicted = predicted_labels[i]
				true = true_labels[i]
				prec_at_K.append(precision_score([true], [1 if x==true else 0 for x in predicted], zero_division=0, average='binary'))
		return sum(prec_at_K) / len(prec_at_K)

def calculate_r_at_K(true_labels, predicted_labels, K):
		"""
		Calculate recall at K.
		
		Args:
				true_labels: The true labels.
				predicted_labels: The predicted labels.
				K: The number of top results to consider.
		
		Returns:
				r_at_K: The recall at K.
		"""
		true_labels = true_labels.tolist()
		predicted_labels = predicted_labels[:, :K].tolist()
		r_at_K = []
		for i in range(len(true_labels)):
				predicted = predicted_labels[i]
				true = true_labels[i]
				r_at_K.append(recall_score([true], [1 if x==true else 0 for x in predicted], zero_division=0, average='binary'))
		return sum(r_at_K) / len(r_at_K)

def evaluate_retrieval(model, dataset, metadata_df, K):
		"""
		Evaluate the retrieval performance of the model on a given dataset.
		
		Args:
				model: The CLIP model.
				dataset: The dataset to evaluate on.
				metadata_df: The metadata dataframe containing the image and text pairs.
				K: The number of top results to consider.
		
		Returns:
				prec_at_K: The precision at K.
				r_at_K: The recall at K.
		"""
		device = "cuda" if torch.cuda.is_available() else "cpu"
		preprocess = clip.load("ViT-B/32", device=device)[1]
		tokenized_labels = tokenize_labels(dataset).to(device)
		
		predicted_labels = []
		true_labels = []
		for index, row in metadata_df.iterrows():
				img_path = row['image_path']
				img = Image.open(img_path)
				image = preprocess(img).unsqueeze(0).to(device)
				similarities = compute_similarities(model, image, tokenized_labels)
				_, topk_labels_idx = similarities.topk(K, dim=-1)
				predicted_labels.append(topk_labels_idx.cpu().numpy().flatten())
				true_labels.append(dataset.classes.index(row['label']))
		
		prec_at_K = calculate_prec_at_K(torch.tensor(true_labels), torch.tensor(predicted_labels), K)
		r_at_K = calculate_r_at_K(torch.tensor(true_labels), torch.tensor(predicted_labels), K)
		return prec_at_K, r_at_K

def main():
	print(clip.available_models())
	# zero_shot() # only for a given image
	get_prec_at_k()

	

if __name__ == "__main__":
		main()