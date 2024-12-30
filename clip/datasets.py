import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from utils import *

class CUSTOMIZEDDATASET(Dataset):
	def __init__(self, dataset, transformer=None,):
		self.dataset = dataset
		# self.images = [img for idx, (img,lbl) in enumerate(self.dataset)]
		self.labels = clip.tokenize(texts=[dataset.classes[lbl_idx] for i, (img, lbl_idx) in enumerate(self.dataset)])
		if transformer:
			self.transform = transformer
		else:
			self.transform = T.Compose(
				[
					T.ToTensor(),
					T.Normalize(
						(0.491, 0.482, 0.446), 
						(0.247, 0.243, 0.261)
					)
				]
			)

	def __getitem__(self, index):
		# image = self.images[index]
		img = self.dataset[index][0]
		lbl = self.labels[index]
		return self.transform(img), lbl

	def __len__(self):
		return len(self.dataset)

class ImageNet(Dataset):
		def __init__(self, root, train=False, transform=None):
				self.root = root
				self.train = train
				self.transform = transform
				self.classes = []
				self.synset_to_index = {}
				self.data = []
				self.labels = []
				self.synset_ids = []

				# Read the synset to index mapping
				synset_file = os.path.join(self.root, 'LOC_synset_mapping.txt')
				with open(synset_file, 'r') as f:
						for idx, line in enumerate(f):
								synset_id, class_name = line.strip().split(maxsplit=1)
								self.synset_to_index[synset_id] = idx
								self.classes.append(class_name)

				if train:
						self.data, self.labels, self.synset_ids = self._load_train_data(os.path.join(root, 'ILSVRC', 'Data', 'CLS-LOC', 'train'))
				else:
						self.data, self.labels, self.synset_ids = self._load_val_data(
								os.path.join(root, 'ILSVRC', 'Data', 'CLS-LOC', 'val'),
								os.path.join(root, 'LOC_val_solution.csv')
						)

		def _load_train_data(self, directory):
				data = []
				labels = []
				ids = []
				for synset_id in os.listdir(directory):
						if synset_id in self.synset_to_index:
								class_dir = os.path.join(directory, synset_id)
								label = self.synset_to_index[synset_id]
								for file_name in os.listdir(class_dir):
										file_path = os.path.join(class_dir, file_name)
										data.append(file_path)
										labels.append(label)
										ids.append(synset_id)
				return data, labels, ids

		def _load_val_data(self, directory, val_label_file):
				data = []
				labels = []
				ids = []
				# Read the validation solution file
				df = pd.read_csv(val_label_file)
				
				# Create a mapping from ImageId to label
				image_id_to_label = {}
				for _, row in df.iterrows():
						image_id = row['ImageId']
						# The CSV has multiple predictions, we take the first one
						first_prediction = row['PredictionString'].split()[0]
						label = self.synset_to_index.get(first_prediction)
						if label is not None:
								image_id_to_label[image_id] = label
				
				# Construct the file paths and labels
				for idx in sorted(image_id_to_label.keys(), key=lambda x: int(x.split('_')[-1])):  # Sort by the numeric part
						file_name = f'{idx}.JPEG'
						file_path = os.path.join(directory, file_name)
						label = image_id_to_label[idx]
						synset_id = [k for k, v in self.synset_to_index.items() if v == label][0]
						data.append(file_path)
						labels.append(label)
						ids.append(synset_id)
				
				return data, labels, ids

		def __len__(self):
				return len(self.data)

		def __getitem__(self, index):
				file_path, label = self.data[index], self.labels[index]
				image = Image.open(file_path).convert('RGB')
				if self.transform is not None:
						image = self.transform(image)
				return image, label

		def __repr__(self):
				split = 'Train' if self.train else 'Validation'
				return (
						f'Dataset IMAGENET\n' \
						f'    Number of datapoints: {len(self)}\n' \
						f'    Root location: {self.root}\n' \
						f'    Split: {split}'
				)

class CINIC10(Dataset):
	classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	def __init__(self, root, train=True, download=False, transform=None):
		self.root = root
		self.train = train
		self.transform = transform
		if train:
			self.data = self._load_data(os.path.join(root, 'train'))
		else:
			self.data = self._load_data(os.path.join(root, 'valid'))

	def _load_data(self, directory):
		data = []
		labels = []
		for idx, class_name in enumerate(self.classes):
			# print(f"Loading {idx} {class_name} images...")
			class_dir = os.path.join(directory, class_name)
			for file_name in os.listdir(class_dir):
				file_path = os.path.join(class_dir, file_name)
				data.append(file_path)
				labels.append(idx)
		return list(zip(data, labels))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		file_path, label = self.data[index]
		image = Image.open(file_path)
		if self.transform is not None:
			image = self.transform(image)
		return image, label

	def __repr__(self):
		split = 'Train' if self.train else 'Test'
		return (
			f'Dataset CINIC10\n' \
			f'    Number of datapoints: {len(self)}\n' \
			f'    Root location: {self.root}\n' \
			f'    Split: {split}'
		)