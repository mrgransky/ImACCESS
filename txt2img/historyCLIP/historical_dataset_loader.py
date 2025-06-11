from utils import *
from functools import lru_cache
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

dtypes={
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
	'textual_based_labels': str,
	'visual_based_labels': str,
	'multimodal_labels': str,
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
	df = pd.read_csv(
		filepath_or_buffer=metadata_fpth, 
		on_bad_lines='skip',
		dtype=dtypes, 
		low_memory=False,
	)
	print(f"FULL Multi-label Dataset {type(df)} {df.shape}")
	
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

def detect_environment() -> dict:
		"""Detect system environment for cache optimization."""
		memory = psutil.virtual_memory()
		total_gb = memory.total / (1024**3)
		available_gb = memory.available / (1024**3)
		
		# Environment detection
		is_hpc = any(env in os.environ for env in ['SLURM_JOB_ID', 'PBS_JOBID', 'SGE_TASK_ID'])
		is_laptop = total_gb <= 32 and not is_hpc
		is_workstation = total_gb > 32 and not is_hpc
		
		# Memory pressure
		memory_pressure = "high" if memory.percent > 80 else "medium" if memory.percent > 60 else "low"
		
		return {
				"total_gb": total_gb,
				"available_gb": available_gb,
				"is_hpc": is_hpc,
				"is_laptop": is_laptop,
				"is_workstation": is_workstation,
				"memory_pressure": memory_pressure
		}

def get_cache_strategy(env: dict) -> dict:
	"""Determine cache strategy based on environment."""
	if env["is_hpc"]:
		return {"memory_ratio": 0.4, "target_coverage": 0.8}
	elif env["is_workstation"]:
		return {"memory_ratio": 0.25, "target_coverage": 0.6}
	elif env["is_laptop"]:
		return {"memory_ratio": 0.15, "target_coverage": 0.4}
	else:
		return {"memory_ratio": 0.2, "target_coverage": 0.5}

def get_cache_size(
		dataset_size: int,
		image_estimate_mb: float = 7.0,
		safety_factor: float = 0.8,
		verbose: bool = True
	) -> int:
	"""
	Calculate optimal cache size based on system environment and dataset.
	
	Args:
			dataset_size: Number of images in dataset
			image_estimate_mb: Estimated size per image in MB
			safety_factor: Safety margin (0.8 = use 80% of calculated safe memory)
			verbose: Print analysis
			
	Returns:
			Optimal cache size (number of images)
	"""
	env = detect_environment()
	strategy = get_cache_strategy(env)
	
	# Adjust available memory for pressure
	available_gb = env["available_gb"]
	if env["memory_pressure"] == "high":
			available_gb *= 0.5
	elif env["memory_pressure"] == "medium":
			available_gb *= 0.7
	
	# Calculate memory-based limit
	usable_memory_gb = available_gb * strategy["memory_ratio"] * safety_factor
	memory_based_size = int((usable_memory_gb * 1024) / image_estimate_mb)
	
	# Calculate coverage-based limit
	target_coverage = strategy["target_coverage"]
	
	# Adjust coverage for dataset size
	if dataset_size < 1000:
		target_coverage = min(1.0, target_coverage * 1.5)
	elif dataset_size > 100000:
		target_coverage = max(0.1, target_coverage * 0.5)
	
	coverage_based_size = int(dataset_size * target_coverage)
	
	# Take minimum of both constraints
	cache_size = min(memory_based_size, coverage_based_size)
	
	# Apply bounds
	cache_size = max(100, min(cache_size, dataset_size))
	
	if verbose:
		actual_coverage = cache_size / dataset_size * 100
		actual_memory = (cache_size * image_estimate_mb) / 1024
		
		env_type = "HPC" if env["is_hpc"] else "Workstation" if env["is_workstation"] else "Laptop"
		
		print(f"\nOptimized Cache Analysis:")
		print(f"\tEnvironment: {env_type}")
		print(f"\tAvailable memory: {env['available_gb']:.1f}GB")
		print(f"\tDataset size: {dataset_size:,} images")
		print(f"\tCache size: {cache_size:,} images ({actual_coverage:.1f}% coverage)")
		print(f"\tMemory usage: {actual_memory:.1f}GB")
		
		# Expected speedup estimate
		if actual_coverage >= 70:
			speedup = "60-80% faster"
		elif actual_coverage >= 50:
			speedup = "40-60% faster"
		elif actual_coverage >= 30:
			speedup = "20-40% faster"
		else:
			speedup = "10-20% faster"
		
		print(f"\tExpected speedup: {speedup} (after epoch 1)")
	
	return cache_size

def get_multi_label_dataloaders_old(
		dataset_dir: str,
		batch_size: int,
		num_workers: int,
		input_resolution: int,
		cache_size: int = None,  # Auto-detect if None
	) -> Tuple[DataLoader, DataLoader]:
	dataset_name = os.path.basename(dataset_dir)
	print(f"Creating multi-label dataloaders for {dataset_name}...")
	train_dataset, val_dataset, label_dict = get_multi_label_datasets(ddir=dataset_dir)
	preprocess = get_preprocess(dataset_dir=dataset_dir, input_resolution=input_resolution)
	total_samples = len(train_dataset) + len(val_dataset)
	
	if cache_size is None:
		cache_size = get_cache_size(dataset_size=total_samples, verbose=True)
		print(f"Auto-detected LRU cache size: {cache_size}")

	
	cache_multiplier = 0.6 if total_samples > 150000 else 0.8

	train_dataset = HistoricalArchivesMultiLabelDataset(
		dataset_name=dataset_name,
		train=True,
		data_frame=train_dataset.sort_values(by="img_path").reset_index(drop=True),
		transform=preprocess,
		label_dict=label_dict,
		cache_size=int(cache_size * cache_multiplier),
	)
	
	print(train_dataset)
	
	train_loader = DataLoader(
			dataset=train_dataset,
			batch_size=batch_size,
			shuffle=True,
			pin_memory=torch.cuda.is_available(),
			num_workers=num_workers,
			prefetch_factor=1,
			persistent_workers=(num_workers > 0 and os.environ.get('USER') != "farid"),
			collate_fn=custom_collate_fn,
	)
	train_loader.name = f"{dataset_name.lower()}_multilabel_train".upper()
	
	validation_dataset = HistoricalArchivesMultiLabelDataset(
			dataset_name=dataset_name,
			train=False,
			data_frame=val_dataset.sort_values(by="img_path").reset_index(drop=True),
			transform=preprocess,
			label_dict=label_dict,
			cache_size=int(cache_size * (1.0 - cache_multiplier)), # Allocate less cache to validation set,
	)
	
	print(validation_dataset)
	
	val_loader = DataLoader(
			dataset=validation_dataset,
			batch_size=batch_size,
			shuffle=False,
			pin_memory=torch.cuda.is_available(),
			num_workers=num_workers,
			prefetch_factor=1,
			persistent_workers=(num_workers > 0 and os.environ.get('USER') != "farid"),
			collate_fn=custom_collate_fn,
	)
	val_loader.name = f"{dataset_name.lower()}_multilabel_validation".upper()
	
	return train_loader, val_loader

class HistoricalArchivesMultiLabelDatasetWithCaching(Dataset):
	def __init__(
			self,
			dataset_name: str,
			train: bool,
			data_frame: pd.DataFrame,
			transform,
			label_dict: dict,
			text_augmentation: bool = True,
			cache_size: int = 1000,
		):
		self.dataset_name = dataset_name
		self.train = train
		self.data_frame = data_frame
		self.images = self.data_frame["img_path"].values
		self.labels = self.data_frame["multimodal_labels"].values
		self.label_dict = label_dict
		self._num_classes = len(label_dict)
		self.transform = transform
		self.text_augmentation = text_augmentation
		self.split = 'Train' if self.train else 'Validation'
		enable_cache = self._should_enable_cache(cache_size, verbose=True)		
		self.cache_enabled = enable_cache		
		if self.cache_enabled:
			self._load_image = self._get_cached_loader(cache_size)
			print(f"🟢 LRU caching ENABLED for {self.dataset_name}_{self.split} with cache_size: {cache_size}")
		else:
			self._load_image = self._load_image_no_cache
			print(f"🔴 LRU caching DISABLED for {self.dataset_name}_{self.split} (insufficient memory)")
		
		self.text_cache = [None] * len(self.data_frame)
		self._preload_texts()
		self._monitor_memory()

	def _should_enable_cache(self, cache_size: int, verbose: bool = False) -> bool:
		try:
			memory = psutil.virtual_memory()
			available_gb = memory.available / (1024**3)
			
			# Estimate cache memory usage (conservative: 10MB per image)
			estimated_cache_gb = (cache_size * 10) / 1024
			
			# Only enable cache if it uses less than 25% of available memory
			safe_threshold = available_gb * 0.25
			
			if verbose:
				print(
					f"Memory check: Available={available_gb:.1f}GB, "
					f"Estimated cache={estimated_cache_gb:.1f}GB, "
					f"Threshold={safe_threshold:.1f}GB"
				)
			return estimated_cache_gb < safe_threshold
		except Exception:
			return False

	def _get_cached_loader(self, cache_size: int):
		@lru_cache(maxsize=cache_size)
		def load_and_transform_image(img_path: str) -> torch.Tensor:
			try:
				with Image.open(img_path) as image:
					image = image.convert("RGB")
					if self.transform:
						tensor = self.transform(image)
					else:
						tensor = T.ToTensor()(image)
					return tensor
			except Exception as e:
				print(f"Error loading {img_path}: {e}")
				return torch.zeros(3, 224, 224, dtype=torch.float32)
		return load_and_transform_image

	def _load_image_no_cache(self, img_path: str) -> torch.Tensor:
		try:
			with Image.open(img_path) as image:
				image = image.convert("RGB")
				if self.transform:
					return self.transform(image)
				else:
					return T.ToTensor()(image)
		except Exception as e:
			print(f"Error loading {img_path}: {e}")
			return torch.zeros(3, 224, 224, dtype=torch.float32)

	def _monitor_memory(self):
		try:
			memory = psutil.virtual_memory()
			if memory.percent > 80:
				print(f"⚠️  WARNING: High memory usage ({memory.percent:.1f}%)")
				if self.cache_enabled:
					print("   Consider disabling cache or reducing cache_size")
		except:
			pass

	@property
	def unique_labels(self):
		return sorted(self.label_dict.keys()) if self.label_dict else []
	def get_cache_info(self):
		if self.cache_enabled and hasattr(self._load_image, 'cache_info'):
			return self._load_image.cache_info()
		else:
			return None

	def clear_cache(self):
		if self.cache_enabled and hasattr(self._load_image, 'cache_clear'):
			self._load_image.cache_clear()
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()		
		print(f"Cache cleared for {self.dataset_name}_{self.split}")

	def _preload_texts(self):
		print(f"Preprocessing texts for {self.dataset_name}...")
		for idx in tqdm(range(len(self.labels)), desc=f"Text Tokenization for {self.dataset_name}"):
			self.text_cache[idx] = self._tokenize_labels(self.labels[idx])

	def _tokenize_labels(self, labels_str):
		try:
			labels = ast.literal_eval(labels_str)
			text_desc = self._create_text_description(labels)
			return clip.tokenize(text_desc).squeeze(0)
		except (ValueError, SyntaxError):
			return clip.tokenize("").squeeze(0)

	def _create_text_description(self, labels: list) -> str:
		if not labels:
			return ""
		if not self.train or not self.text_augmentation:
			return " ".join(labels)
		if len(labels) == 1:
			return labels[0]
		if len(labels) == 2:
			return f"{labels[0]} and {labels[1]}"
		np.random.shuffle(labels)
		return ", ".join(labels[:-1]) + f", and {labels[-1]}"

	def _get_label_vector(self, labels_str: str) -> torch.Tensor:
		vector = torch.zeros(self._num_classes, dtype=torch.float32)
		try:
			labels = ast.literal_eval(labels_str)
			for label in labels:
				if label in self.label_dict:
					vector[self.label_dict[label]] = 1.0
		except (ValueError, SyntaxError):
			pass
		return vector

	def __len__(self):
		return len(self.data_frame)

	def __repr__(self):
		transform_str = f"Transform: {self.transform}\n" if self.transform else ""
		
		if self.cache_enabled:
			try:
				cache_info = self.get_cache_info()
				if cache_info:
					cache_str = (
						f"Image Cache (LRU): ENABLED - "
						f"Size={cache_info.currsize}/{cache_info.maxsize}, "
						f"Hits={cache_info.hits}, Misses={cache_info.misses}, "
						f"Hit Rate={cache_info.hits/(cache_info.hits + cache_info.misses)*100:.1f}%" 
						if (cache_info.hits + cache_info.misses) > 0 else "Hit Rate=0%"
					)
				else:
					cache_str = "Image Cache (LRU): ENABLED but not yet used"
			except Exception:
				cache_str = "Image Cache (LRU): ENABLED"
		else:
			cache_str = "Image Cache (LRU): DISABLED (streaming mode)"
		return (
			f"{self.dataset_name}\n"
			f"\tSplit: {self.split} ({self.data_frame.shape[0]} samples)\n"
			f"\tNum classes: {self._num_classes}\n"
			f"\t{cache_str}\n"
			f"{transform_str}"
		)

	def __getitem__(self, idx: int):
		try:
			image_path = self.images[idx]
			
			image_tensor = self._load_image(image_path)
			
			tokenized_text = self.text_cache[idx]
			label_vector = self._get_label_vector(self.labels[idx])
			
			return image_tensor, tokenized_text, label_vector
		except Exception as e:
			print(f"WARNING: Skipping sample {idx} due to error: {e} | Path: {self.images[idx]}")
			return None

	def __del__(self):
		try:
			self.clear_cache()
		except:
			pass



def custom_collate_fn(batch):
	valid_samples = [item for item in batch if item is not None]
	if not valid_samples:
		return torch.empty(0), torch.empty(0), torch.empty(0)
	
	# Use manual collation (recommended for simple, fixed-size tuples)
	try:
		images, texts, labels = zip(*valid_samples)
		return (
			torch.stack(images),
			torch.stack(texts),
			torch.stack(labels)
		)
	except ValueError:
		# Fallback: if the structure is not unpackable, use default_collate
		return torch.utils.data.dataloader.default_collate(valid_samples)

class ImageCache:
	def __init__(self, image_paths, cache_size, num_workers=4):
		self.cache = {}
		self.cache_size = cache_size
		self.image_paths = image_paths
		self.lock = threading.Lock()
		
		if cache_size > 0:
			self._preload_images(num_workers)
	
	def _preload_images(self, num_workers):
		print(f"\nPreloading {self.cache_size} images using {num_workers} workers...")
		
		# For training, load random subset; for validation, load first N images
		indices_to_load = np.random.choice(
			len(self.image_paths), 
			size=min(self.cache_size, len(self.image_paths)), 
			replace=False
		)
		
		def load_image(idx):
			try:
				img_path = self.image_paths[idx]
				with Image.open(img_path) as img:
					# Store as numpy array (more compact than PIL Image)
					img_array = np.array(img.convert("RGB"), dtype=np.uint8)
				return idx, img_array
			except Exception as e:
				print(f"Failed to load image at index {idx}: {e}")
				return idx, None
		
		# Load images in parallel
		with ThreadPoolExecutor(max_workers=num_workers) as executor:
			futures = [executor.submit(load_image, idx) for idx in indices_to_load]
			
			with tqdm(total=len(indices_to_load), desc="Loading images") as pbar:
				for future in as_completed(futures):
					idx, img_array = future.result()
					if img_array is not None:
						with self.lock:
							self.cache[idx] = img_array
					pbar.update(1)
		
		print(f"Successfully cached {len(self.cache)} images")
	
	def get(self, idx):
		"""Get image from cache, returns None if not cached."""
		with self.lock:
				img_array = self.cache.get(idx)
				if img_array is not None:
						# Convert back to PIL Image
						return Image.fromarray(img_array)
				return None
	
	def __len__(self):
		return len(self.cache)

class HistoricalArchivesMultiLabelDataset(Dataset):
		def __init__(
				self,
				dataset_name: str,
				train: bool,
				data_frame: pd.DataFrame,
				transform,
				label_dict: dict,
				text_augmentation: bool = True,
				cache_size: int = 0,
				cache_workers: int = 4,
			):
			self.dataset_name = dataset_name
			self.train = train
			self.data_frame = data_frame
			self.images = self.data_frame["img_path"].values
			self.labels = self.data_frame["multimodal_labels"].values
			self.label_dict = label_dict
			self._num_classes = len(label_dict)
			self.transform = transform
			self.text_augmentation = text_augmentation
			self.split = 'Train' if self.train else 'Validation'
			self.cache_size = cache_size

			# Initialize cache
			if self.cache_size > 0:
				self.image_cache = ImageCache(
					self.images,
					self.cache_size,
					num_workers=cache_workers
				)
				self.cache_hits = 0
				self.cache_misses = 0
			else:
				self.image_cache = None

			# Precompute text tokens
			self.text_cache = [None] * len(self.data_frame)
			self._preload_texts()
				
		def _preload_texts(self):
				"""Precompute all text tokens."""
				print(f"Preprocessing texts for {self.dataset_name}...")
				for idx in tqdm(range(len(self.labels)), desc=f"Tokenizing texts for {self.dataset_name}"):
						self.text_cache[idx] = self._tokenize_labels(self.labels[idx])
		
		def _tokenize_labels(self, labels_str):
				"""Tokenize label string."""
				try:
						labels = ast.literal_eval(labels_str)
						text_desc = self._create_text_description(labels)
						return clip.tokenize(text_desc).squeeze(0)
				except (ValueError, SyntaxError):
						return clip.tokenize("").squeeze(0)
		
		def _create_text_description(self, labels: list) -> str:
				"""Create text description from labels."""
				if not labels:
						return ""
				if not self.train or not self.text_augmentation:
						return " ".join(labels)
				if len(labels) == 1:
						return labels[0]
				if len(labels) == 2:
						return f"{labels[0]} and {labels[1]}"
				np.random.shuffle(labels)
				return ", ".join(labels[:-1]) + f", and {labels[-1]}"
		
		def _get_label_vector(self, labels_str: str) -> torch.Tensor:
				"""Convert label string to binary vector."""
				vector = torch.zeros(self._num_classes, dtype=torch.float32)
				try:
						labels = ast.literal_eval(labels_str)
						for label in labels:
								if label in self.label_dict:
										vector[self.label_dict[label]] = 1.0
				except (ValueError, SyntaxError):
						pass
				return vector
		
		def _load_image(self, idx: int) -> Image.Image:
				"""Load image with caching."""
				# Try cache first
				if self.image_cache is not None:
						cached_img = self.image_cache.get(idx)
						if cached_img is not None:
								self.cache_hits += 1
								return cached_img
						else:
								self.cache_misses += 1
				
				# Load from disk
				img_path = self.images[idx]
				try:
						with Image.open(img_path) as img:
								return img.convert("RGB")
				except Exception as e:
						print(f"Error loading {img_path}: {e}")
						# Return a blank image instead of None
						return Image.new("RGB", (224, 224), color='white')
		
		@property
		def unique_labels(self):
				return sorted(self.label_dict.keys()) if self.label_dict else []
		
		def get_cache_stats(self):
			"""Get cache statistics."""
			if self.image_cache is not None:
				total_requests = self.cache_hits + self.cache_misses
				hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
				return {
					"cache_size": len(self.image_cache),
					"hits": self.cache_hits,
					"misses": self.cache_misses,
					"hit_rate": hit_rate
				}
			return None
		
		def __len__(self):
				return len(self.data_frame)
		
		def __repr__(self):
				transform_str = f"Transform: {self.transform}\n" if self.transform else ""
				
				if self.image_cache is not None:
						stats = self.get_cache_stats()
						cache_str = (f"Image Cache: Size={len(self.image_cache)}/{len(self.images)}, "
												f"Hit Rate={stats['hit_rate']:.1f}%")
				else:
						cache_str = "Image Cache: DISABLED (streaming mode)"
				
				return (
						f"{self.dataset_name}\n"
						f"\tSplit: {self.split} ({self.data_frame.shape[0]} samples)\n"
						f"\tNum classes: {self._num_classes}\n"
						f"\t{cache_str}\n"
						f"{transform_str}"
				)
		
		def __getitem__(self, idx: int):
			try:
				# Load and transform image
				image = self._load_image(idx)
				image_tensor = self.transform(image)
				
				# Get precomputed text
				tokenized_text = self.text_cache[idx]
				
				# Get label vector
				label_vector = self._get_label_vector(self.labels[idx])
				
				return image_tensor, tokenized_text, label_vector
			except Exception as e:
				print(f"Error processing sample {idx}: {e}")
				# Return valid tensors with zeros instead of None
				return (
					torch.zeros(3, 224, 224),  # Blank image
					torch.zeros(77, dtype=torch.long),  # Empty text
					torch.zeros(self._num_classes)  # No labels
				)

def get_estimated_image_size_mb(
		image_paths: Union[List[str], pd.Series],
		sample_size: int = 100,
	)-> float:

	if not image_paths.any() if isinstance(image_paths, pd.Series) else not image_paths:
		print("Warning: No image paths provided for estimation. Returning default estimate.")
		return 7.0 # Default estimate of 7MB per image

	actual_sample_size = min(sample_size, len(image_paths))

	if actual_sample_size == 0:
		print("Warning: image_paths list is empty. Returning default estimate.")
		return 7.0 # Default estimate of 7MB per image

	print(f"Estimating average image RAM size from {actual_sample_size} samples...")

	if isinstance(image_paths, pd.Series):
		indices_to_sample_from = range(len(image_paths))
	else:
		indices_to_sample_from = range(len(image_paths))

	sampled_indices = random.sample(indices_to_sample_from, actual_sample_size)
	total_bytes = 0
	successfully_loaded = 0

	if isinstance(image_paths, pd.Series):
		get_path_fn = lambda integer_pos: image_paths.iloc[integer_pos]
	else:
		get_path_fn = lambda integer_pos: image_paths[integer_pos]

	for i in tqdm(sampled_indices, desc="Sampling images for size estimation"):
		img_path = get_path_fn(i)
		try:
			with Image.open(img_path) as img:
				img_array = np.array(img.convert("RGB"), dtype=np.uint8)
			total_bytes += img_array.nbytes
			successfully_loaded += 1
		except FileNotFoundError:
			print(f"Warning: Image not found at {img_path}, skipping for estimation.")
		except Exception as e:
			print(f"Warning: Could not load or process image {img_path} for estimation: {e}, skipping.")

	if successfully_loaded == 0:
		print("Warning: Failed to load any images from the sample. Returning default estimate.")
		return 7.0  # Fallback default if no images could be loaded

	average_bytes = total_bytes / successfully_loaded
	average_mb = average_bytes / (1024**2)

	print(f"Successfully loaded {successfully_loaded}/{actual_sample_size} sampled images.")
	print(f"Estimated average raw image RAM size: {average_mb:.2f} MB")
	
	return average_mb

def get_cache_size_v2(
		dataset_size: int,
		available_memory_gb: float,
		average_image_size_mb: float,
		is_hpc: bool = False,
		min_coverage: float = 0.15,  # Minimum 15% coverage to be worthwhile
		max_memory_fraction: float = 0.15,  # Use max 15% of available memory
	) -> int:
	
	# Calculate minimum cache size for effectiveness
	min_cache_size = int(dataset_size * min_coverage)

	# Calculate maximum cache size from memory
	max_cache_from_memory = int((available_memory_gb * max_memory_fraction * 1024) / average_image_size_mb)
	
	# If we can't achieve minimum coverage, disable cache
	if max_cache_from_memory < min_cache_size:
		print(f"Cannot achieve minimum {min_coverage*100:.0f}% coverage with available memory.")
		print(f"Would need {min_cache_size} images but can only fit {max_cache_from_memory}")
		return 0
	
	# Target coverage based on environment
	if is_hpc:
		target_coverage = 0.3  # 30% on HPC
	else:
		target_coverage = 0.4  # 40% on workstation
	
	target_cache_size = int(dataset_size * target_coverage)
	
	# Final cache size
	cache_size = min(target_cache_size, max_cache_from_memory)
	
	actual_coverage = cache_size / dataset_size
	actual_memory_gb = (cache_size * average_image_size_mb) / 1024
	
	print(f"\nCache Analysis:")
	print(f"\tDetected Environment: {'HPC' if is_hpc else 'Workstation'} (target coverage: {target_coverage*100:.1f}%)")
	print(f"\tAvailable RAM memory: {available_memory_gb:.1f}GB => {max_memory_fraction*100:.0f}% => {available_memory_gb*max_memory_fraction:.1f}GB for cache")
	print(f"\tMinimum cache size for effectiveness: {min_cache_size:,} images ({min_coverage*100:.0f}% minimum coverage)")
	print(f"\tMaximum nominal cache from memory: {max_cache_from_memory:,} images (considering image size: {average_image_size_mb:.2f}MB)")
	print(f"\tDataset size: {dataset_size:,} images")
	print(f"\tCache size: {cache_size:,} images ({actual_coverage*100:.1f}% actual coverage)")
	print(f"\tMemory usage: {actual_memory_gb:.1f}GB")
	print(f"\tExpected speedup: {int(actual_coverage * 100)}% faster (after warmup)")
	
	return cache_size

def get_multi_label_dataloaders(
		dataset_dir: str,
		batch_size: int,
		num_workers: int,
		input_resolution: int,
		cache_size: int = None,
	) -> Tuple[DataLoader, DataLoader]:
	"""Create multi-label dataloaders with improved caching."""
	dataset_name = os.path.basename(dataset_dir)
	print(f"Creating multi-label dataloaders for {dataset_name}...")
	
	train_dataset, val_dataset, label_dict = get_multi_label_datasets(ddir=dataset_dir)
	preprocess = get_preprocess(dataset_dir=dataset_dir, input_resolution=input_resolution)
	
	# Estimate memory per image
	average_image_size_mb = get_estimated_image_size_mb(
		image_paths=train_dataset["img_path"].values.tolist()+val_dataset["img_path"].values.tolist(),
		sample_size=5000,
	)

	# Determine cache size if not specified
	if cache_size is None:
		print(">> No cache size specified. Detecting environment for cache optimization...")
		memory = psutil.virtual_memory()
		available_gb = memory.available / (1024**3)
		is_hpc = any(env in os.environ for env in ['SLURM_JOB_ID', 'PBS_JOBID'])
		
		total_samples = len(train_dataset) + len(val_dataset)
		cache_size = get_cache_size_v2(
			dataset_size=total_samples,
			available_memory_gb=available_gb,
			average_image_size_mb=average_image_size_mb,
			is_hpc=is_hpc,
		)
	
	# Allocate cache between train/val
	train_pct = 0.7
	val_pct = 1.0 - train_pct

	if cache_size > 0:
		train_cache_size = int(cache_size * train_pct)
		val_cache_size = cache_size - train_cache_size
	else:
		train_cache_size = val_cache_size = 0
	print(f">> Total cache size: {cache_size:,} Distributed (train[{train_pct*100:.0f}%]: {train_cache_size:,}, validation[{val_pct*100:.0f}%]: {val_cache_size:,})")
	
	train_dataset = HistoricalArchivesMultiLabelDataset(
		dataset_name=dataset_name,
		train=True,
		data_frame=train_dataset.sort_values(by="img_path").reset_index(drop=True),
		transform=preprocess,
		label_dict=label_dict,
		cache_size=train_cache_size,
		cache_workers=min(4, num_workers),
	)
	
	print(train_dataset)
	
	val_dataset = HistoricalArchivesMultiLabelDataset(
		dataset_name=dataset_name,
		train=False,
		data_frame=val_dataset.sort_values(by="img_path").reset_index(drop=True),
		transform=preprocess,
		label_dict=label_dict,
		cache_size=val_cache_size,
		cache_workers=min(4, num_workers),
	)
	
	print(val_dataset)
	
	# Create dataloaders
	train_loader = DataLoader(
			dataset=train_dataset,
			batch_size=batch_size,
			shuffle=True,
			pin_memory=torch.cuda.is_available(),
			num_workers=num_workers,
			prefetch_factor=2 if num_workers > 0 else None,
			persistent_workers=(num_workers > 0),
			drop_last=False,
	)
	train_loader.name = f"{dataset_name.lower()}_multilabel_train".upper()
	
	val_loader = DataLoader(
			dataset=val_dataset,
			batch_size=batch_size,
			shuffle=False,
			pin_memory=torch.cuda.is_available(),
			num_workers=num_workers,
			prefetch_factor=2 if num_workers > 0 else None,
			persistent_workers=(num_workers > 0),
			drop_last=False,
	)
	val_loader.name = f"{dataset_name.lower()}_multilabel_validation".upper()
	
	return train_loader, val_loader