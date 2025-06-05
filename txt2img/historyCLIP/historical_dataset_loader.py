from utils import *

dtypes = {
	'doc_id': str,
	'id': str,
	'label': str,
	'title': str,
	'description': str,
	'img_url': str,
	'label_title_description': str,
	'raw_doc_date': str,
	'doc_year': float,
	'doc_url': str,
	'img_path': str,
	'doc_date': str,
	'dataset': str,
	'date': str,
	'country': str,
}

def _convert_image_to_rgb(image: Image) -> Image:
	return image.convert("RGB")

def get_preprocess(dataset_dir: str, input_resolution: int) -> T.Compose:
	"""
	Create a preprocessing transformation pipeline for image data.
	
	Args:
			dataset_dir: Directory containing the dataset and possibly mean/std statistics
			input_resolution: Target resolution for the images
			
	Returns:
			A torchvision.transforms.Compose object containing the preprocessing pipeline
	"""
	try:
		mean = load_pickle(fpath=os.path.join(dataset_dir, "img_rgb_mean.gz"))
		std = load_pickle(fpath=os.path.join(dataset_dir, "img_rgb_std.gz"))
		print(f"{os.path.basename(dataset_dir)} mean: {mean} std: {std}")
	except Exception as e:
		mean = [0.52, 0.50, 0.48]
		std = [0.27, 0.27, 0.26]
		print(f"Could not load mean and std from {dataset_dir}. Using default values: mean={mean} std={std}")
	
	preprocess = T.Compose(
		[
			T.Resize(size=input_resolution, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
			T.CenterCrop(size=input_resolution),
			_convert_image_to_rgb,
			T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		]
	)
	
	return preprocess

def get_single_label_datasets(ddir: str, seed:int=42,):
	metadata_fpth = os.path.join(ddir, "metadata.csv")
	print(f"Loading dataset: {metadata_fpth}")
	############################################################################
	# debugging types of columns
	# df = pd.read_csv(filepath_or_buffer=metadata_fpth, on_bad_lines='skip')
	# for col in df.columns:
	# 	print(f"Column: {col}")
	# 	print(df[col].apply(type).value_counts())
	# 	print("-" * 50)
	############################################################################
	df = pd.read_csv(
		filepath_or_buffer=metadata_fpth, 
		on_bad_lines='skip',
		dtype=dtypes, 
		low_memory=False,
	)
	print(f"FULL Dataset {type(df)} {df.shape}")
	metadata_train_fpth = os.path.join(ddir, "metadata_train.csv")
	metadata_val_fpth = os.path.join(ddir, "metadata_val.csv")
	print(f"Loading training dataset: {metadata_train_fpth}")
	df_train = pd.read_csv(
		filepath_or_buffer=metadata_train_fpth, 
		on_bad_lines='skip',
		dtype=dtypes, 
		low_memory=True,
	)
	print(f"Loading validation dataset: {metadata_val_fpth}")
	df_val = pd.read_csv(
		filepath_or_buffer=metadata_val_fpth,
		on_bad_lines='skip',
		dtype=dtypes, 
		low_memory=True,
	)
	# # ######################################################################################
	# Create deterministic label mapping from all data
	all_labels = sorted(set(df_train["label"].unique()) | set(df_val["label"].unique()))
	label_dict = {label: idx for idx, label in enumerate(all_labels)}
	# print(json.dumps(label_dict, indent=2, ensure_ascii=False))
	# Map labels to integers
	df_train['label_int'] = df_train['label'].map(label_dict)
	df_val['label_int'] = df_val['label'].map(label_dict)
	# Validate that all validation labels exist in training
	val_labels = set(df_val["label"].unique())
	train_labels = set(df_train["label"].unique())
	unknown_labels = val_labels - train_labels
	if unknown_labels:
		print(f"WARNING: Validation set contains labels not in training: {unknown_labels}")
	# # ######################################################################################
	return df_train, df_val

def get_single_label_dataloaders(
		dataset_dir: str,
		batch_size: int,
		num_workers: int,
		input_resolution: int,
		memory_threshold_gib: float = 500.0,  # Minimum available memory (GiB) to preload images
	)-> Tuple[DataLoader, DataLoader]:
	dataset_name = os.path.basename(dataset_dir)

	print(f"Creating single-label dataloaders for {dataset_name}...")
	train_dataset, val_dataset = get_single_label_datasets(ddir=dataset_dir)

	preprocess = get_preprocess(dataset_dir=dataset_dir, input_resolution=input_resolution)
	
	train_dataset = HistoricalArchivesSingleLabelDataset(
		dataset_name=dataset_name,
		train=True,
		data_frame=train_dataset.sort_values(by="img_path").reset_index(drop=True),
		transform=preprocess,
		memory_threshold_gib=memory_threshold_gib,
	)

	print(train_dataset)
	# print(f"image paths:\n{train_dataset.images[:10]}")
	# print("labels:", train_dataset.labels[:10])
	# print("label_int:", train_dataset.labels_int[:10])

	train_loader = DataLoader(
		dataset=train_dataset,
		batch_size=batch_size,
		shuffle=True,
		pin_memory=torch.cuda.is_available(), # Move data to GPU faster if CUDA available
		persistent_workers=(num_workers > 0),  # Keep workers alive if memory allows
		num_workers=num_workers,
		prefetch_factor=2, # 
	)
	train_loader.name = f"{dataset_name.lower()}_train".upper()

	validation_dataset = HistoricalArchivesSingleLabelDataset(
		dataset_name=dataset_name,
		train=False,
		data_frame=val_dataset.sort_values(by="img_path").reset_index(drop=True),
		transform=preprocess,
		memory_threshold_gib=memory_threshold_gib,
	)
	
	print(validation_dataset)
	# print(f"image paths:\n{validation_dataset.images[:10]}")
	# print("labels:", validation_dataset.labels[:10])
	# print("label_int:", validation_dataset.labels_int[:10])

	val_loader = DataLoader(
		dataset=validation_dataset,
		batch_size=batch_size,
		shuffle=False,
		pin_memory=torch.cuda.is_available(), # Move data to GPU faster if CUDA available
		num_workers=num_workers,
		prefetch_factor=2, # Number of batches loaded in advance by each worker
		persistent_workers=(num_workers > 0 and os.environ.get('USER') != "farid"),  # Keep workers alive if memory allows
	)
	val_loader.name = f"{dataset_name.lower()}_validation".upper()

	return train_loader, val_loader

class HistoricalArchivesSingleLabelDataset(Dataset):
	def __init__(
		self,
		dataset_name: str,
		train: bool,
		data_frame: pd.DataFrame,
		transform,
		memory_threshold_gib: float = 500.0,  # Minimum available memory (GiB) to preload images
	):
		self.dataset_name = dataset_name
		self.train = train
		self.data_frame = data_frame
		self.images = self.data_frame["img_path"].values
		self.labels = self.data_frame["label"].values
		self.labels_int = self.data_frame["label_int"].values
		# Sort unique labels to ensure deterministic ordering across runs (fixes non-reproducible results):
		self.unique_labels = sorted(list(set(self.labels)))  # Sort the unique labels
		self._num_classes = len(np.unique(self.labels_int))
		self.transform = transform
		# Preload images into memory if available memory exceeds threshold
		available_memory_gib = psutil.virtual_memory().available / (1024 ** 3)  # Convert bytes to GiB
		if available_memory_gib >= memory_threshold_gib:
			print(f"Available memory ({available_memory_gib:.2f} GiB) exceeds threshold ({memory_threshold_gib} GiB). Preloading images...")
			self.image_cache = self._preload_images()
		else:
			print(f"Available memory ({available_memory_gib:.2f} GiB) below threshold ({memory_threshold_gib} GiB). Skipping preloading.")
			self.image_cache = None
	
	def _preload_images(self):
		print(f"Preloading images into memory for {self.dataset_name} ({'train' if self.train else 'validation'})...")
		cache = []
		for img_path in tqdm(self.images, desc="Loading images"):
			try:
				img = Image.open(img_path).convert("RGB")
				cache.append(img)
			except Exception as e:
				print(f"ERROR: {img_path}\t{e}")
				cache.append(None)
		print(f"Preloaded {sum(1 for img in cache if img is not None)}/{len(cache)} images successfully")
		return cache
	
	def __len__(self):
		return len(self.data_frame)
	
	def __repr__(self):
		transform_str = f"StandardTransform\nTransform: {self.transform}\n" if self.transform else ""
		split = 'Train' if self.train else 'Validation'
		cache_status = "Preloaded" if self.image_cache is not None else "Not preloaded"
		return (
			f"{self.dataset_name}\n"
			f"\tSplit: {split} {self.data_frame.shape}\n"
			f"\t{list(self.data_frame.columns)}\n"
			f"\tlabels={self.labels.shape}\n"
			f"\tCache: {cache_status}\n"
			f"{transform_str}"
		)
	
	def __getitem__(self, idx):
		doc_label = self.labels[idx]
		doc_label_int = self.labels_int[idx]
		
		# Use cached image if available, otherwise load from disk
		if self.image_cache is not None:
			image = self.image_cache[idx]
			if image is None:
				raise ValueError(f"Failed to load image at index {idx} (cached as None)")
		else:
			doc_image_path = self.images[idx]
			try:
				image = Image.open(doc_image_path).convert("RGB")
			except Exception as e:
				print(f"ERROR: {doc_image_path}\t{e}")
				raise

		image_tensor = self.transform(image)
		tokenized_label_tensor = clip.tokenize(texts=doc_label).squeeze(0)
		return image_tensor, tokenized_label_tensor, doc_label_int

def get_multi_label_datasets(ddir: str, seed: int = 42):
	metadata_fpth = os.path.join(ddir, "metadata_multimodal.csv")
	print(f"Loading multi-label dataset: {metadata_fpth}")
	
	dtypes = {
			'img_url': str,
			'id': str,
			'title': str,
			'description': str,
			'user_query': str,
			'enriched_document_description': str,
			'raw_doc_date': str,
			'doc_url': str,
			'img_path': str,
			'doc_date': str,
			'label': str,
			'textual_based_labels': str,
			'visual_based_labels': str,
			'multimodal_labels': str,
	}
	
	df = pd.read_csv(
			filepath_or_buffer=metadata_fpth, 
			on_bad_lines='skip',
			dtype=dtypes, 
			low_memory=False,
	)
	print(f"FULL Multi-label Dataset {type(df)} {df.shape}")
	
	# Split into train and validation
	metadata_train_fpth = os.path.join(ddir, "metadata_multimodal_train.csv")
	metadata_val_fpth = os.path.join(ddir, "metadata_multimodal_val.csv")
	
	print(f"Loading multi-label training dataset: {metadata_train_fpth}")
	df_train = pd.read_csv(
			filepath_or_buffer=metadata_train_fpth, 
			on_bad_lines='skip',
			dtype=dtypes, 
			low_memory=True,
	)
	
	print(f"Loading multi-label validation dataset: {metadata_val_fpth}")
	df_val = pd.read_csv(
			filepath_or_buffer=metadata_val_fpth,
			on_bad_lines='skip',
			dtype=dtypes, 
			low_memory=True,
	)
	
	# Create label mapping from all unique labels in the dataset
	all_labels = set()
	for labels_str in df['multimodal_labels']:
			try:
					labels = ast.literal_eval(labels_str)
					all_labels.update(labels)
			except (ValueError, SyntaxError):
					continue
	
	# Convert to sorted list for deterministic ordering
	all_labels = sorted(all_labels)
	label_dict = {label: idx for idx, label in enumerate(all_labels)}
	
	# Add label vectors to dataframes
	for df_split in [df_train, df_val]:
			label_vectors = []
			for labels_str in df_split['multimodal_labels']:
					try:
							labels = ast.literal_eval(labels_str)
							vector = np.zeros(len(all_labels), dtype=np.float32)
							for label in labels:
									if label in label_dict:
											vector[label_dict[label]] = 1.0
							label_vectors.append(vector)
					except (ValueError, SyntaxError):
							label_vectors.append(np.zeros(len(all_labels), dtype=np.float32))
			
			df_split['label_vector'] = label_vectors
	
	return df_train, df_val, label_dict

def get_multi_label_dataloaders(
		dataset_dir: str,
		batch_size: int,
		num_workers: int,
		input_resolution: int,
		memory_threshold_gib: float = 500.0,
	) -> Tuple[DataLoader, DataLoader]:
	dataset_name = os.path.basename(dataset_dir)
	print(f"Creating multi-label dataloaders for {dataset_name}...")
	
	train_dataset, val_dataset, label_dict = get_multi_label_datasets(ddir=dataset_dir)
	preprocess = get_preprocess(dataset_dir=dataset_dir, input_resolution=input_resolution)
	
	train_dataset = HistoricalArchivesMultiLabelDataset(
			dataset_name=dataset_name,
			train=True,
			data_frame=train_dataset.sort_values(by="img_path").reset_index(drop=True),
			transform=preprocess,
			memory_threshold_gib=memory_threshold_gib,
			label_dict=label_dict,
	)
	
	print(train_dataset)
	
	train_loader = DataLoader(
			dataset=train_dataset,
			batch_size=batch_size,
			shuffle=True,
			pin_memory=torch.cuda.is_available(),
			persistent_workers=(num_workers > 0),
			num_workers=num_workers,
			prefetch_factor=2,
	)
	train_loader.name = f"{dataset_name.lower()}_multilabel_train".upper()
	
	validation_dataset = HistoricalArchivesMultiLabelDataset(
			dataset_name=dataset_name,
			train=False,
			data_frame=val_dataset.sort_values(by="img_path").reset_index(drop=True),
			transform=preprocess,
			memory_threshold_gib=memory_threshold_gib,
			label_dict=label_dict,
	)
	
	print(validation_dataset)
	
	val_loader = DataLoader(
			dataset=validation_dataset,
			batch_size=batch_size,
			shuffle=False,
			pin_memory=torch.cuda.is_available(),
			num_workers=num_workers,
			prefetch_factor=2,
			persistent_workers=(num_workers > 0 and os.environ.get('USER') != "farid"),
	)
	val_loader.name = f"{dataset_name.lower()}_multilabel_validation".upper()
	
	return train_loader, val_loader

class HistoricalArchivesMultiLabelDataset(Dataset):
	def __init__(
		self,
		dataset_name: str,
		train: bool,
		data_frame: pd.DataFrame,
		transform,
		memory_threshold_gib: float = 500.0,
		label_dict: dict = None,
		text_augmentation: bool = True
	):
		self.dataset_name = dataset_name
		self.train = train
		self.data_frame = data_frame
		self.images = self.data_frame["img_path"].values
		self.labels = self.data_frame["multimodal_labels"].values
		self.label_dict = label_dict
		self._num_classes = len(label_dict) if label_dict else 0
		self.transform = transform
		self.text_augmentation = text_augmentation
		
		# Initialize caches
		self.image_cache = None
		self.text_cache = [None] * len(self.data_frame)
		
		# Preload if memory allows
		available_memory_gib = psutil.virtual_memory().available / (1024 ** 3)
		if available_memory_gib >= memory_threshold_gib:
			print(f"Available memory ({available_memory_gib:.2f} GiB) exceeds threshold. Preloading...")
			self.image_cache = self._preload_images()
			self._preload_texts()  # Cache tokenized texts
	
	@property
	def unique_labels(self):
		"""Return sorted list of all possible class names"""
		return sorted(self.label_dict.keys()) if self.label_dict else []
	
	def _preload_images(self):
		print(f"Preloading images for {self.dataset_name}...")
		cache = []
		for img_path in tqdm(self.images, desc="Loading images"):
			try:
				img = Image.open(img_path).convert("RGB")
				cache.append(img)
			except Exception as e:
				print(f"ERROR: {img_path}\t{e}")
				cache.append(None)
		print(f"Preloaded {sum(1 for img in cache if img is not None)}/{len(cache)} images")
		return cache
	
	def _preload_texts(self):
		print(f"Preprocessing texts for {self.dataset_name}...")
		for idx in tqdm(range(len(self.labels)), desc="Tokenizing texts"):
			self.text_cache[idx] = self._tokenize_labels(self.labels[idx])
	
	def _tokenize_labels(self, labels_str):
		try:
			labels = ast.literal_eval(labels_str)
			text_desc = self._create_text_description(labels)
			return clip.tokenize(text_desc).squeeze(0)
		except (ValueError, SyntaxError):
			return clip.tokenize("").squeeze(0)
	
	def _create_text_description(self, labels):
		"""Convert list of labels to natural language string"""
		if not labels:
			return ""
				
		if not self.text_augmentation:
			return " ".join(labels)
				
		if len(labels) == 1:
			return labels[0]
		elif len(labels) == 2:
			return f"{labels[0]} and {labels[1]}"
		else:
			return ", ".join(labels[:-1]) + f", and {labels[-1]}"
	
	def _get_label_vector(self, labels_str):
		"""Convert label string to multi-hot vector"""
		try:
			labels = ast.literal_eval(labels_str)
			vector = torch.zeros(self._num_classes, dtype=torch.float32)
			for label in labels:
				if label in self.label_dict:
					vector[self.label_dict[label]] = 1.0
			return vector
		except (ValueError, SyntaxError):
			return torch.zeros(self._num_classes, dtype=torch.float32)
	
	def __len__(self):
		return len(self.data_frame)
	
	def __repr__(self):
		transform_str = f"Transform: {self.transform}\n" if self.transform else ""
		split = 'Train' if self.train else 'Validation'
		cache_status = []
		if self.image_cache: cache_status.append("Images")
		if any(self.text_cache): cache_status.append("Texts")
		cache_str = "Preloaded: " + ", ".join(cache_status) if cache_status else "Not preloaded"
		
		return (
			f"{self.dataset_name}\n"
			f"\tSplit: {split} {self.data_frame.shape}\n"
			f"\tColumns: {list(self.data_frame.columns)}\n"
			f"\tNum classes: {self._num_classes}\n"
			f"\tCache: {cache_str}\n"
			f"{transform_str}"
		)

	def __getitem__(self, idx):
		if self.image_cache is not None:
			image = self.image_cache[idx]
			if image is None:
				raise ValueError(f"Failed to load image at index {idx}")
		else:
			try:
				image = Image.open(self.images[idx]).convert("RGB")
			except Exception as e:
				print(f"ERROR: {self.images[idx]}\t{e}")
				raise
		if self.text_cache[idx] is None:
			self.text_cache[idx] = self._tokenize_labels(self.labels[idx])
		tokenized_text = self.text_cache[idx]
		label_vector = self._get_label_vector(self.labels[idx])
		image_tensor = self.transform(image)
		return image_tensor, tokenized_text, label_vector