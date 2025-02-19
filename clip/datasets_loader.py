from utils import *
from torchvision.datasets import CIFAR10, CIFAR100

def _convert_image_to_rgb(image):
	return image.convert("RGB")

def get_dataset_transform(dname:str="CIFAR10"):
	dname = dname.upper()
	mean_std_dict = {
		'CIFAR10': ((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
		'CIFAR100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
		'IMAGENET': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		'CINIC10': ((0.128, 0.109, 0.075), (0.202, 0.185, 0.167)),
	}
	if dname in mean_std_dict.keys():
		mean = mean_std_dict[dname][0]
		std = mean_std_dict[dname][1]
		return T.Compose([
			T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
			T.CenterCrop(224),
			_convert_image_to_rgb,
			T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		])
	else:
		raise ValueError(f"Invalid dataset name: {dname}. Available: [CIFAR10, CIFAR100, IMAGENET, CINIC10]")

def get_dataset(
	dname:str="CIFAR10",
	transform=None,
	USER:str="USER",
	):
	if transform is None:
		transform = get_dataset_transform(dname=dname)
	dname = dname.upper()
	ddir = {
		"farid": f'/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/{dname}',
		"ubuntu": f'/media/volume/ImACCESS/WW_DATASETs/{dname}',
		"alijanif": f'/scratch/project_2004072/ImACCESS/WW_DATASETs/{dname}',
	}
	if dname == 'CIFAR100':
		train_dataset = CIFAR100(
			root=os.path.expanduser("~/.cache"), 
			train=True,
			download=True,
			transform=transform,
		)
		validation_dataset = CIFAR100(
			root=os.path.expanduser("~/.cache"), 
			train=False,
			download=True,
			transform=transform,
		)
	elif dname == 'CIFAR10':
		train_dataset = CIFAR10(
			root=os.path.expanduser("~/.cache"), 
			train=True,
			download=True,
			transform=transform,
		)
		validation_dataset = CIFAR10(
			root=os.path.expanduser("~/.cache"), 
			train=False,
			download=True,
			transform=transform,
		)
	elif dname == 'IMAGENET':
		train_dataset = ImageNet_1K(
			root=ddir.get(USER),
			train=True,
			transform=transform,
		)
		validation_dataset = ImageNet_1K(
			root=ddir.get(USER),
			train=False,
			transform=transform,
	)	
	elif dname == 'CINIC10':
		train_dataset = CINIC10(
			root=ddir.get(USER),
			train=True,
			download=True,
			transform=transform,
		)
		validation_dataset = CINIC10(
			root=ddir.get(USER),
			train=False,
			download=True,
			transform=transform,
		)
	else:
		raise ValueError(f"Invalid dataset name: {dname}. Available: [CIFAR10, cifar100, IMAGENET, CINIC10]")
	print(train_dataset)
	print(validation_dataset)
	return train_dataset, validation_dataset

def get_dataloaders(
	dataset_name: str='CIFAR10',
	batch_size: int=32,
	nw: int=10,
	USER: str="farid",
	):

	train_dataset, validation_dataset = get_dataset(
		dname=dataset_name,
		USER=USER,
	)

	trainset = IMAGE_TEXT_DATASET(dataset=train_dataset,)
	validset = IMAGE_TEXT_DATASET(dataset=validation_dataset,)
	
	train_loader = DataLoader(
		dataset=trainset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=nw,
		pin_memory=True, # Move data to GPU faster if using CUDA
		persistent_workers=(nw > 1),  # Keep workers alive if memory allows
	)
	validation_loader = DataLoader(
		dataset=validset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=nw,
		pin_memory=True, # when using CUDA
	)
	return train_loader, validation_loader

class IMAGE_TEXT_DATASET(Dataset):
	def __init__(self, dataset):
		self.dataset = dataset
		self.label_names = dataset.classes  # Class names like 'airplane', 'automobile', etc.
	
	def __getitem__(self, index):
		img, lbl_idx = self.dataset[index]
		label = self.label_names[lbl_idx]  # Use label name as text prompt
		lbl_tokenized = clip.tokenize(texts=[label]).squeeze(0)  # Tokenize the class name
		return img, lbl_tokenized, lbl_idx
	
	def __len__(self):
		return len(self.dataset)

class ImageNet_1K_LT(Dataset):
	"""
	ImageNet-1K Long Tail (LT) contains a subset of the original 1,281,167 training images, 
	with 1,000 classes, and a long tail distribution.
	"""
	def __init__(self, root, train=False, transform=None, imbalance_factor=0.01, min_size=10):
			self.root = root
			self.train = train
			self.transform = transform
			self.imbalance_factor = imbalance_factor
			self.min_size = min_size
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
			self._apply_imbalance_factor()
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
	def _apply_imbalance_factor(self):
			class_sizes = {}
			for label in set(self.labels):
					class_sizes[label] = self.labels.count(label)
			sorted_classes = sorted(class_sizes.items(), key=lambda x: x[1], reverse=True)
			new_data = []
			new_labels = []
			for label, size in sorted_classes:
					num_images_to_load = max(int(size * (self.imbalance_factor ** (999 - label))), self.min_size)
					indices = [i for i, x in enumerate(self.labels) if x == label]
					for index in indices[:num_images_to_load]:
							new_data.append(self.data[index])
							new_labels.append(self.labels[index])
			self.data = new_data
			self.labels = new_labels
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
			transform_str = f"StandardTransform\nTransform: {self.transform}\n" if self.transform else ""
			return (
					f'Dataset ImageNet-1K Long Tail\n' \
					f'    Number of datapoints: {len(self)}\n' \
					f'    Root location: {self.root}\n' \
					f'    Split: {split}\n' \
					f'{transform_str}'
			)

class ImageNet_1K(Dataset):
	"""
	ImageNet-1K contains 1,281,167 training images, 50,000 validation images and 100,000 test images.
	The images are labeled with 1,000 classes.
	"""
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
		transform_str = f"StandardTransform\nTransform: {self.transform}\n" if self.transform else ""
		return (
			f'IMAGENET-1K\n' \
			f'    Number of datapoints: {len(self)}\n' \
			f'    Root location: {self.root}\n' \
			f'    Split: {split}\n' \
			f'{transform_str}'
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
			f'CINIC10\n' \
			f'    Number of datapoints: {len(self)}\n' \
			f'    Root location: {self.root}\n' \
			f'    Split: {split}'
		)