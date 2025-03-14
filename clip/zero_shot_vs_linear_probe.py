import os
import clip
import torch
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
model = model.float()
# Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)
print(train)
print("-"*25)
print(test)

# Get the class names
class_names = test.classes

def get_features(dataset, batch_size:int=1024):
	all_features = []
	all_labels = []
	with torch.no_grad():
		for images, labels in tqdm(DataLoader(dataset, batch_size=batch_size)):
			# print(type(images), type(labels)) # <class 'torch.Tensor'> <class 'torch.Tensor'>
			features = model.encode_image(images.to(device))
			all_features.append(features)
			all_labels.append(labels)
	return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
print(f"Getting training features", end="\t")
t0 = time.time()
train_features, train_labels = get_features(train)
print(f"Elapsed_t: {time.time()-t0:.2f} sec")

print(f"Getting test features", end="\t")
t0 = time.time()
test_features, test_labels = get_features(test)
print(f"Elapsed_t: {time.time()-t0:.2f} sec")

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Linear Probe Accuracy = {accuracy:.1f}")

##############################################################################################
# Encode the text descriptions of the classes
text_descriptions = [f"a photo of a {label}" for label in class_names]
text_inputs = torch.cat([clip.tokenize(desc) for desc in text_descriptions]).to(device)
with torch.no_grad():
	text_features = model.encode_text(text_inputs)

# Normalize the features
test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Convert test_features to a PyTorch tensor
test_features = torch.from_numpy(test_features).to(device)

# Calculate the similarity scores
similarity_scores = (100.0 * test_features @ text_features.T).softmax(dim=-1)

# Get the predicted class indices
predicted_class_indices = np.argmax(similarity_scores.cpu().numpy(), axis=1)

# Calculate the accuracy
accuracy = np.mean((test_labels == predicted_class_indices).astype(float)) * 100.
print(f"Zero-shot Accuracy = {accuracy:.1f}")

##############################################################################################
def zero_shot_classifier(class_names, model, device):
	"""
	Creates text embeddings from class names.

	Args:
			class_names: A list of class names.
			model: The CLIP model.
			device: Device for computation.

	Returns:
			Text embeddings.
	"""
	text_descriptions = [f"This is a photo of a {label}" for label in class_names]
	text_tokens = clip.tokenize(text_descriptions).to(device)
	with torch.no_grad():
			text_features = model.encode_text(text_tokens)
			text_features /= text_features.norm(dim=-1, keepdim=True)
	return text_features

def get_image_features(dataset, model, device):
		"""
		Extract image features from the dataset using the CLIP model.

		Args:
			dataset: The dataset to extract features from.
			model: The CLIP model.
			device: Device for computation.

		Returns:
			Image features and their corresponding labels.
		"""

		all_features = []
		all_labels = []

		with torch.no_grad():
				for images, labels in tqdm(DataLoader(dataset, batch_size=1024)):
					features = model.encode_image(images.to(device))
					all_features.append(features)
					all_labels.append(labels)

		return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def evaluate_zero_shot(image_features, image_labels, text_features, model, device):
	"""
	Evaluates zero-shot classification performance using image and text features.

	Args:
			image_features: Image features.
			image_labels: Image labelsl.
			text_features: Text features (text descriptions).
			model: The CLIP model.
			device: Device for computation.

	Returns:
			Accuracy in percentage.
	"""

	with torch.no_grad():
			image_features_tensor = torch.tensor(image_features).float().to(device)
			# Calculate similarity between all images and all text features
			similarities = (100.0 * image_features_tensor @ text_features.T).softmax(dim=-1)
			_, predictions = similarities.cpu().topk(1, dim=-1) # get top predictions

	accuracy = np.mean((image_labels == predictions.flatten().numpy()).astype(float)) * 100.
	return accuracy

# Get image features from the test set
test_features, test_labels = get_image_features(test, model, device)

# Get the class names
class_names = test.classes

# Generate text embeddings for the classes
text_features = zero_shot_classifier(class_names, model, device)

# Evaluate using zero-shot classification
accuracy = evaluate_zero_shot(test_features, test_labels, text_features, model, device)
print(f"[Google AI studio] Zero-Shot Accuracy = {accuracy:.3f}")