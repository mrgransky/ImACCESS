from utils import *

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
	metadata_fpth = os.path.join(ddir, "metadata_multi_label_multimodal.csv")
	print(f"Loading multi-label dataset: {metadata_fpth}")
	df = pd.read_csv(
		filepath_or_buffer=metadata_fpth, 
		on_bad_lines='skip',
		dtype=dtypes, 
		low_memory=False,
	)
	print(f"FULL Multi-label Dataset {type(df)} {df.shape}")
	
	# metadata_train_fpth = os.path.join(ddir, "metadata_multimodal_train.csv")
	# metadata_val_fpth = os.path.join(ddir, "metadata_multimodal_val.csv")

	metadata_train_fpth = os.path.join(ddir, metadata_fpth.replace('.csv', '_train.csv'))
	metadata_val_fpth = os.path.join(ddir, metadata_fpth.replace('.csv', '_val.csv'))

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

def estimate_image_size_mb(path: str) -> float:
	try:
		with Image.open(path) as img:
			w, h = img.size
			# Assume RGB (3 channels), 1 byte/channel
			return (w * h * 3) / (1024 ** 2)
	except:
		return None

def get_estimated_image_size_mb(
		image_paths: Union[List[str], pd.Series],
		sample_size: int = 100,
		num_workers: int = 8
	) -> float:

	if isinstance(image_paths, pd.Series):
		image_paths = image_paths.dropna().tolist()

	if not image_paths:
		print("Warning: No image paths provided for estimation. Returning default estimate.")
		return 7.0

	actual_sample_size = min(sample_size, len(image_paths))
	print(f"Estimating average image RAM size from {actual_sample_size} samples...")

	sample_paths = random.sample(image_paths, actual_sample_size)

	t0 = time.time()
	sizes = []

	with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
		for size in tqdm(executor.map(estimate_image_size_mb, sample_paths), total=actual_sample_size):
			if size is not None:
				sizes.append(size)

	if not sizes:
		print("Warning: Failed to estimate any image sizes. Using fallback estimate.")
		return 7.0

	avg_mb = sum(sizes) / len(sizes)
	print(f"Successfully estimated {len(sizes)}/{actual_sample_size} images.")
	print(f"Estimated avg image RAM size: {avg_mb:.2f} MB | Elapsed: {time.time() - t0:.2f} sec")
	return avg_mb

def get_cache_size(
		dataset_size: int,
		available_memory_gb: float,
		average_image_size_mb: float,
		is_hpc: bool = False,
		min_desired_converage: float = 0.15,
		max_memory_fraction: float = 0.40,
	) -> int:
	detected_platform = "HPC" if is_hpc else f"{platform.system()} Workstation (Laptop/VM)"
	# Calculate minimum desired cache size for effectiveness
	min_desired_cache_size = int(dataset_size * min_desired_converage)

	# Calculate maximum allowed cache size from memory
	max_allowed_cache_size = int((available_memory_gb * max_memory_fraction * 1024) / average_image_size_mb)
	
	# If we can't achieve minimum coverage, disable cache
	if max_allowed_cache_size < min_desired_cache_size:
		print(f"<!> Cannot achieve {min_desired_converage*100:.0f}% minimum coverage with available memory: {available_memory_gb:.1f}GB for {average_image_size_mb:.2f}MB image average size.")
		print(f"\tComputed minimum desired cache size: {min_desired_cache_size:,} images! but {detected_platform} can only fit {max_allowed_cache_size:,} images.")
		rounded_up_cache_size = round_up(num=max_allowed_cache_size)
		print(f"\t=> rounded up to {rounded_up_cache_size:,} images.")
		return rounded_up_cache_size
		# print(f"\t=> using cache size: {max_allowed_cache_size:,}")
		# return max_allowed_cache_size
	
	# Target coverage based on environment
	if is_hpc:
		target_coverage = 0.40
	else:
		target_coverage = 0.50
	
	target_cache_size = int(dataset_size * target_coverage)
	
	# Final cache size
	cache_size = min(target_cache_size, max_allowed_cache_size)
	
	actual_coverage = cache_size / dataset_size
	actual_memory_gb = (cache_size * average_image_size_mb) / 1024
	
	print(f"\nCache Analysis:")
	print(f"\tDetected Environment: {detected_platform} (target coverage: {target_coverage*100:.1f}%)")
	print(f"\tAvailable RAM memory: {available_memory_gb:.1f}GB => {max_memory_fraction*100:.0f}% Allocated => {available_memory_gb*max_memory_fraction:.1f}GB for cache")
	print(f"\tMinimum desired cache size for effectiveness: {min_desired_cache_size:,} images ({min_desired_converage*100:.0f}% minimum desired coverage)")
	print(f"\tMaximum allowed cache size [given available RAM memory]: {max_allowed_cache_size:,} images (considering image size: {average_image_size_mb:.2f}MB)")
	print(f"\tDataset size: {dataset_size:,} images")
	print(f"\tCache size: {cache_size:,} images ({actual_coverage*100:.1f}% actual coverage)")
	print(f"\tMemory usage: {actual_memory_gb:.1f}GB")
	print(f"\tExpected speedup: {int(actual_coverage * 100)}% faster (after warmup)")
	
	return cache_size

class HistoricalArchivesMultiLabelDatasetWithCaching(Dataset):
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
				cache_str = (
					f"Image Cache: Size={len(self.image_cache)}/{len(self.images)}, "
					f"Hit Rate={stats['hit_rate']:.1f}%"
				)
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

def get_multi_label_dataloaders_with_caching(
		dataset_dir: str,
		batch_size: int,
		num_workers: int,
		input_resolution: int,
		cache_size: int = None,
	) -> Tuple[DataLoader, DataLoader]:
	dataset_name = os.path.basename(dataset_dir)
	print(f"Creating multi-label dataloaders for {dataset_name}...")
	
	train_dataset, val_dataset, label_dict = get_multi_label_datasets(ddir=dataset_dir)
	preprocess = get_preprocess(dataset_dir=dataset_dir, input_resolution=input_resolution)
	total_samples = len(train_dataset) + len(val_dataset)

	# Estimate memory per image
	average_image_size_mb = get_estimated_image_size_mb(
		image_paths=train_dataset["img_path"].values.tolist()+val_dataset["img_path"].values.tolist(),
		sample_size=int(total_samples*0.1) if total_samples > int(1e5) else 5000,
	)
	
	if cache_size is None:
		memory = psutil.virtual_memory()
		available_gb = memory.available / (1024**3)
		total_gb = memory.total / (1024**3)
		is_hpc = any(env in os.environ for env in ['SLURM_JOB_ID', 'PBS_JOBID'])
		print(
			f">> Obtaining optimal cache size for multi-label dataloader. "
			f"Total RAM memory: {total_gb:.2f}GB | "
			f"Available RAM memory: {available_gb:.2f}GB => {average_image_size_mb:.2f}MB/image"
		)
		
		cache_size = get_cache_size(
			dataset_size=total_samples,
			available_memory_gb=available_gb,
			average_image_size_mb=average_image_size_mb,
			is_hpc=is_hpc,
		)
	
	# Allocate cache between train/val
	train_pct = 0.6
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

class HistoricalArchivesMultiLabelDataset(Dataset):
		"""
		A robust and high-performance Dataset class for multi-label archives.
		This implementation combines LRU caching for performance with graceful
		error handling to prevent crashes from corrupted data.
		"""
		def __init__(
				self,
				dataset_name: str,
				train: bool,
				data_frame: pd.DataFrame,
				transform,
				label_dict: dict,
				text_augmentation: bool = True,
				cache_size: int = 10000
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

				# This is the core of the performance improvement.
				# It creates a new, separate cache for each instance (train, val).
				self._load_image = lru_cache(maxsize=cache_size)(self._load_image_base)
				print(f"LRU cache enabled for image loading with maxsize={cache_size} for {self.dataset_name}")

				# Pre-caching tokenized text is still a good, lightweight optimization.
				self.text_cache = [None] * len(self.data_frame)
				self._preload_texts()

		@property
		def unique_labels(self):
			return sorted(self.label_dict.keys()) if self.label_dict else []

		@staticmethod
		def _load_image_base(img_path: str) -> Image.Image:
				"""
				Base function for loading an image. It's wrapped by the lru_cache.
				It should be simple: it either succeeds or raises an exception.
				"""
				return Image.open(img_path).convert("RGB")

		def __getitem__(self, idx: int):
				"""
				Attempts to load a single sample. Returns None on any loading error.
				This method is now simple and clean.
				"""
				try:
						image_path = self.images[idx]
						# This call is cached. It's fast after the first load.
						image = self._load_image(image_path)
						
						# Get other data (text is pre-cached, label vector is computed)
						tokenized_text = self.text_cache[idx]
						label_vector = self._get_label_vector(self.labels[idx])
						
						# Apply transformations
						image_tensor = self.transform(image)
						
						return image_tensor, tokenized_text, label_vector

				except Exception as e:
						# If any part of the process fails (especially image loading),
						# we print a non-fatal warning and return None.
						# The custom_collate_fn will handle this gracefully.
						# print(f"WARNING: Skipping sample {idx} due to error: {e} | Path: {self.images[idx]}")
						return None
						
		# --- The following helper methods are preserved from your original class ---
		
		def _preload_texts(self):
				print(f"Preprocessing texts for {self.dataset_name}...")
				for idx in tqdm(range(len(self.labels)), desc=f"Tokenizing texts for {self.dataset_name}"):
						self.text_cache[idx] = self._tokenize_labels(self.labels[idx])
		
		def _tokenize_labels(self, labels_str):
				try:
						labels = ast.literal_eval(labels_str)
						text_desc = self._create_text_description(labels)
						return clip.tokenize(text_desc).squeeze(0)
				except (ValueError, SyntaxError):
						return clip.tokenize("").squeeze(0)
		
		def _create_text_description(self, labels: list) -> str:
				if not labels: return ""
				if not self.train or not self.text_augmentation: return " ".join(labels)
				if len(labels) == 1: return labels[0]
				if len(labels) == 2: return f"{labels[0]} and {labels[1]}"
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
				split = 'Train' if self.train else 'Validation'
				# The cache info is now managed per-worker, so a central view isn't possible.
				# Stating the configuration is the correct approach.
				cache_str = f"Image Cache (LRU): Maxsize={self._load_image.cache_info().maxsize}"
				
				return (
						f"{self.dataset_name}\n"
						f"\tSplit: {split} ({self.data_frame.shape[0]} samples)\n"
						f"\tNum classes: {self._num_classes}\n"
						f"\t{cache_str}\n"
						f"{transform_str}"
				)

def get_multi_label_dataloaders(
		dataset_dir: str,
		batch_size: int,
		num_workers: int,
		input_resolution: int,
		cache_size: int = int(1e4),
	):
	dataset_name = os.path.basename(dataset_dir)
	print(f"Creating multi-label dataloaders for {dataset_name}...")

	# --- 1. Load data from CSVs (this part remains the same) ---
	train_df, val_df, label_dict = get_multi_label_datasets(ddir=dataset_dir)
	preprocess = get_preprocess(dataset_dir=dataset_dir, input_resolution=input_resolution)
	total_samples = len(train_df) + len(val_df)
	cache_size = min(cache_size, int(total_samples*0.70))
	train_pct = 0.70
	val_pct = 1.0 - train_pct
	train_cache_size = int(cache_size * train_pct)
	val_cache_size = int(cache_size * val_pct)
	print(
		f">> Total Samples: {total_samples:,} | Total cache size: {cache_size:,} "
		f"Distributed (train[{train_pct*100:.0f}%]: {train_cache_size:,}, validation[{val_pct*100:.0f}%]: {val_cache_size:,})"
	)

	# --- 2. Create the full dataset instances with LRU caching ---
	train_dataset = HistoricalArchivesMultiLabelDataset(
			dataset_name=f"{dataset_name}_TRAIN",
			train=True,
			data_frame=train_df,
			transform=preprocess,
			label_dict=label_dict,
			cache_size=train_cache_size,
	)
	print(train_dataset)
	
	val_dataset = HistoricalArchivesMultiLabelDataset(
			dataset_name=f"{dataset_name}_VALIDATION",
			train=False,
			data_frame=val_df,
			transform=preprocess,
			label_dict=label_dict,
			cache_size=val_cache_size,
	)
	print(val_dataset)
	
	# # --- 3. Create a small, fixed subset for QUICK validation checks ---
	# # This is critical for solving the slow validation loop bottleneck.
	# quick_val_indices = np.random.choice(len(val_dataset), size=min(2048, len(val_dataset)), replace=False)
	# quick_val_subset = torch.utils.data.Subset(val_dataset, quick_val_indices)
	# print(f"Created a quick validation subset with {len(quick_val_subset)} samples.")
	
	# --- 4. Create the DataLoader instances using the custom collate function ---
	common_loader_args = {
		'batch_size': batch_size,
		'num_workers': min(num_workers, 4),
		'pin_memory': torch.cuda.is_available(),
		'persistent_workers': (num_workers > 0),
		'collate_fn': custom_collate_fn # Use the robust collate function
	}
	train_loader = DataLoader(dataset=train_dataset, shuffle=True, **common_loader_args)
	train_loader.name = f"{dataset_name.upper()}_MULTILABEL_TRAIN"
	
	# Loader for the FULL validation set (for infrequent, detailed metrics)
	full_val_loader = DataLoader(dataset=val_dataset, shuffle=False, **common_loader_args)
	full_val_loader.name = f"{dataset_name.upper()}_MULTILABEL_VALIDATION"

	# # Loader for the QUICK validation subset (for fast, per-epoch loss checks)
	# quick_val_loader = DataLoader(dataset=quick_val_subset, shuffle=False, **common_loader_args)
	# quick_val_loader.name = f"{dataset_name.upper()}_MULTILABEL_QUICK_VAL"

	return train_loader, full_val_loader	
	# return (
	# 	train_loader, 
	# 	full_val_loader, 
	# 	quick_val_loader
	# )