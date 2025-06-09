from utils import *
from functools import lru_cache
import platform
import subprocess

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

def is_virtual_machine(verbose: bool = False) -> bool:
    vm_keywords = ['kvm', 'virtualbox', 'vmware', 'hyper-v', 'qemu', 'xen', 'bhyve', 'parallels', 'bochs', 'google', 'amazon', 'azure', 'digitalocean']

    def check_file(path):
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    content = f.read().lower()
                    return any(keyword in content for keyword in vm_keywords)
        except Exception:
            pass
        return False

    def check_systemd_detect_virt():
        try:
            result = subprocess.run(
                ["systemd-detect-virt"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return result.stdout.strip().lower() not in ("", "none")
        except Exception:
            return False

    # Check SLURM (HPC cluster)
    if "SLURM_JOB_ID" in os.environ or "SLURM_NODELIST" in os.environ:
        if verbose:
            print("[VM DETECTED] SLURM HPC environment.")
        return True

    # Check typical Linux virtualization indicators
    if platform.system() == "Linux":
        dmi_hits = {
            "product_name": check_file("/sys/devices/virtual/dmi/id/product_name"),
            "sys_vendor": check_file("/sys/devices/virtual/dmi/id/sys_vendor"),
            "cpuinfo": check_file("/proc/cpuinfo"),
        }
        systemd_result = check_systemd_detect_virt()

        if verbose:
            print(f"[VM Check] DMI hits: {dmi_hits}, systemd-detect-virt: {systemd_result}")

        return any(dmi_hits.values()) or systemd_result

    # Check for known VM indicators on Windows
    if platform.system() == "Windows":
        info = platform.uname()
        return any(keyword in info.node.lower() or keyword in info.system.lower() for keyword in vm_keywords)

    return False

def get_cache_size(
		image_estimate_mb: float = 7.0,
		max_cap_gb: float = 6.0,
		min_cache_items: int = 64,
		verbose: bool = True
	) -> int:
	if "SLURM_JOB_ID" in os.environ or "SLURM_NODELIST" in os.environ:
		mode = "high"
	elif platform.system() == "Linux" and "WSL" in platform.release():
		mode = "low"  # WSL may share memory with host
	elif is_virtual_machine(verbose=True):
		mode = "medium"
	else:
		mode = "auto"

	available_gb = psutil.virtual_memory().available / 1024**3
	total_gb = psutil.virtual_memory().total / 1024**3
	if verbose:
		print(f">> {platform.system()} {platform.uname().node} [mode: {mode}] Available RAM: {available_gb:.2f} GiB | Total: {total_gb:.2f} GiB")
	
	if mode == "low":
		usage_ratio = 0.02
	elif mode == "medium":
		usage_ratio = 0.06
	elif mode == "high":
		usage_ratio = 0.07
	elif mode == "auto":
		if total_gb <= 8:
			usage_ratio = 0.02  # Laptop
		elif total_gb <= 32:
			usage_ratio = 0.05  # VM or dev machine
		else:
			usage_ratio = 0.08  # HPC or SLURM
	else:
		raise ValueError(f"Unknown mode '{mode}'. Use auto, low, medium, or high.")

	print(f">> Usage ratio: {usage_ratio:.2f} => available_gb * usage_ratio: {available_gb * usage_ratio:.2f} GiB | max_cap_gb: {max_cap_gb:.2f} GiB")
	cache_budget_gb = min(available_gb * usage_ratio, max_cap_gb)
	print(f">> cache_budget_gb: {cache_budget_gb:.2f} GiB")
	cache_items = int((cache_budget_gb * 1024) / image_estimate_mb)
	cache_items = max(min_cache_items, cache_items)
	if verbose:
		print(f">> Cache budget: {cache_budget_gb:.2f} GiB → cache contains {cache_items} images (est. {image_estimate_mb:.1f} MB/item)")
	
	return cache_items

def get_multi_label_dataloaders(
		dataset_dir: str,
		batch_size: int,
		num_workers: int,
		input_resolution: int,
		memory_threshold_gib: float = 500.0,
		cache_size: int = None,  # Auto-detect if None
	) -> Tuple[DataLoader, DataLoader]:
	dataset_name = os.path.basename(dataset_dir)
	print(f"Creating multi-label dataloaders for {dataset_name}...")
	
	if cache_size is None:
		cache_size = get_cache_size()
		print(f"Auto-detected LRU cache size: {cache_size}")
	# return
	train_dataset, val_dataset, label_dict = get_multi_label_datasets(ddir=dataset_dir)
	preprocess = get_preprocess(dataset_dir=dataset_dir, input_resolution=input_resolution)
	
	train_dataset = HistoricalArchivesMultiLabelDataset(
		dataset_name=dataset_name,
		train=True,
		data_frame=train_dataset.sort_values(by="img_path").reset_index(drop=True),
		transform=preprocess,
		# memory_threshold_gib=memory_threshold_gib,
		label_dict=label_dict,
		cache_size=cache_size,
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
			collate_fn=custom_collate_fn,
	)
	train_loader.name = f"{dataset_name.lower()}_multilabel_train".upper()
	
	validation_dataset = HistoricalArchivesMultiLabelDataset(
			dataset_name=dataset_name,
			train=False,
			data_frame=val_dataset.sort_values(by="img_path").reset_index(drop=True),
			transform=preprocess,
			# memory_threshold_gib=memory_threshold_gib,
			label_dict=label_dict,
			cache_size=cache_size,
	)
	
	print(validation_dataset)
	
	val_loader = DataLoader(
			dataset=validation_dataset,
			batch_size=batch_size,
			shuffle=False,
			pin_memory=torch.cuda.is_available(),
			num_workers=num_workers,
			prefetch_factor=2,
			collate_fn=custom_collate_fn,
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
		self._load_image = lru_cache(maxsize=cache_size)(self._load_image_base)
		print(f"LRU cache enabled for image loading with maxsize={cache_size}")
		self.text_cache = [None] * len(self.data_frame)
		self._preload_texts()

	@property
	def unique_labels(self):
		"""Return sorted list of all possible class names"""
		return sorted(self.label_dict.keys()) if self.label_dict else []

	@staticmethod
	def _load_image_base(img_path: str) -> Image.Image:
		return Image.open(img_path).convert("RGB")

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
		split = 'Train' if self.train else 'Validation'
		try:
			cache_info = self._load_image.cache_info()
			cache_str = (
				f"Image Cache (LRU): Maxsize={cache_info.maxsize}, "
				f"CurrentSize={cache_info.currsize}, Hits={cache_info.hits}, Misses={cache_info.misses}"
			)
		except AttributeError:
			cache_str = "Image Cache (LRU): Not yet used"
		return (
			f"{self.dataset_name}\n"
			f"\tSplit: {split} ({self.data_frame.shape[0]} samples)\n"
			f"\tNum classes: {self._num_classes}\n"
			f"\t{cache_str}\n"
			f"{transform_str}"
		)

	def __getitem__(self, idx: int):
		"""
		- Uses LRU cached image loading
		- Returns None on any loading error (graceful handling)
		"""
		try:
			image_path = self.images[idx]
			image = self._load_image(image_path)
			
			tokenized_text = self.text_cache[idx]
			label_vector = self._get_label_vector(self.labels[idx])
			image_tensor = self.transform(image)
			
			return image_tensor, tokenized_text, label_vector
		except Exception as e:
			print(f"WARNING: Skipping sample {idx} due to error: {e} | Path: {self.images[idx]}")
			return None

def custom_collate_fn(batch):
	"""
	Returns tuple format compatible with existing training code
	Handles batches where some samples may be None due to loading errors
	"""
	# Filter out None samples
	valid_samples = [item for item in batch if item is not None]
	
	if not valid_samples:
		# Return empty tensors with correct structure if entire batch fails
		return torch.empty(0), torch.empty(0), torch.empty(0)
	
	# Standard collate for valid samples - returns tuple format
	images, texts, labels = zip(*valid_samples)
	
	return (
		torch.stack(images),    # Shape: [batch_size, channels, height, width]
		torch.stack(texts),     # Shape: [batch_size, sequence_length] 
		torch.stack(labels)     # Shape: [batch_size, num_classes]
	)