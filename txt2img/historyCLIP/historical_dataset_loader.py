from utils import *

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

def get_datasets(
		ddir: str, # Dataset directory
		sampling: str, # "stratified_random" or "kfold_stratified"
		kfolds:int=None,  # Number of folds for K-Fold
		force_regenerate:bool=False, # Force regenerate K-Fold splits
		seed:int=42, # Seed for random sampling
	):
	valid_sampling_methods = ["stratified_random", "kfold_stratified"]
	if sampling not in valid_sampling_methods:
		raise ValueError(f"Invalid sampling. Choose from: {', '.join(valid_sampling_methods)}")

	if sampling == "kfold_stratified":
		if kfolds is None:
			raise ValueError("kfolds must be specified for K-Fold stratified sampling.")
		if kfolds < 2:
			raise ValueError("kfolds must be at least 2.")

	metadata_fpth = os.path.join(ddir, "metadata.csv")
	############################################################################
	# debugging types of columns
	# df = pd.read_csv(filepath_or_buffer=metadata_fpth, on_bad_lines='skip')
	# for col in df.columns:
	# 	print(f"Column: {col}")
	# 	print(df[col].apply(type).value_counts())
	# 	print("-" * 50)
	############################################################################
	dtypes = {
		'doc_id': str,
		'id': str,
		'label': str,
		'title': str,
		'description': str,
		'img_url': str,
		'label_title_description': str,
		'raw_doc_date': str,  # Adjust based on actual data
		'doc_year': float,    # Adjust based on actual data
		'doc_url': str,
		'img_path': str,
		'doc_date': str,      # Adjust based on actual data
		'dataset': str,
		'date': str,          # Adjust based on actual data
	}
	df = pd.read_csv(
		filepath_or_buffer=metadata_fpth, 
		on_bad_lines='skip',
		dtype=dtypes, 
		low_memory=False, # Set to False to avoid memory issues
	)
	print(f"FULL Dataset {type(df)} {df.shape}")
	if sampling == "stratified_random":
		print(f">> Using stratified random sampling...")
		metadata_train_fpth = os.path.join(ddir, "metadata_train.csv")
		metadata_val_fpth = os.path.join(ddir, "metadata_val.csv")

		# Load training and validation datasets
		df_train = pd.read_csv(
			filepath_or_buffer=metadata_train_fpth, 
			on_bad_lines='skip',
			dtype=dtypes, 
			low_memory=False, # Set to False to avoid memory issues
		)

		df_val = pd.read_csv(
			filepath_or_buffer=metadata_val_fpth,
			on_bad_lines='skip',
			dtype=dtypes, 
			low_memory=False, # Set to False to avoid memory issues
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
	elif sampling == "kfold_stratified":
		fold_dir = os.path.join(ddir, sampling)
		if os.path.exists(fold_dir) and not force_regenerate:
			print(f"K-Fold splits already exist in {fold_dir}. Loading existing splits...")
			folds = []
			for fold in range(1, kfolds + 1):
				train_fpth = os.path.join(fold_dir, f"fold_{fold}", "metadata_train.csv")
				val_fpth = os.path.join(fold_dir, f"fold_{fold}", "metadata_val.csv")
				df_train = pd.read_csv(
					filepath_or_buffer=train_fpth,
					on_bad_lines='skip',
					dtype=dtypes, 
					low_memory=False, # Set to False to avoid memory issues
				)
				df_val = pd.read_csv(
					filepath_or_buffer=val_fpth,
					on_bad_lines='skip',
					dtype=dtypes, 
					low_memory=False, # Set to False to avoid memory issues
				)
				folds.append((df_train, df_val))
			return folds

		print(f"K-Fold Stratified sampling with K={kfolds} folds...")

		if "label" not in df.columns:
			raise ValueError("The dataset must have a 'label' column for stratified sampling.")

		# Exclude labels that occur only once
		label_counts = df["label"].value_counts()
		labels_to_drop = label_counts[label_counts == 1].index # 
		if len(labels_to_drop)>0:
			print(f"Removing {len(labels_to_drop)} labels that occur only once")
			df = df[~df["label"].isin(labels_to_drop)]

		# Check if we still have data after dropping rare labels
		if df.empty:
			raise ValueError("No valid labels for stratified sampling (after removing labels with one occurrence).")

		all_labels = list(set(df["label"].tolist())) # Get unique labels
		all_labels = sorted(all_labels) # Get sorted unique labels
		label_dict = {lbl: idx for idx, lbl in enumerate(all_labels)}
		df["label_int"] = df["label"].map(label_dict)

		# Create stratified K-Fold splits
		folding_method = StratifiedKFold(
			n_splits=kfolds,
			shuffle=True,
			random_state=seed,
		)

		# Create each fold
		folds = []
		for fold, (train_idx, val_idx) in enumerate(folding_method.split(df, df["label"])):
			fold_dir = os.path.join(ddir, sampling, f"fold_{fold + 1}")
			os.makedirs(fold_dir, exist_ok=True)
			
			train_fpth = os.path.join(fold_dir, "metadata_train.csv")
			val_fpth = os.path.join(fold_dir, "metadata_val.csv")

			df_train = df.iloc[train_idx].copy()
			df_val = df.iloc[val_idx].copy()
			
			# Map labels to integers for both train and validation
			df_train["label_int"] = df_train["label"].map(label_dict)
			df_val["label_int"] = df_val["label"].map(label_dict)
			
			df_train.to_csv(train_fpth, index=False)
			df_val.to_csv(val_fpth, index=False)
			
			folds.append((df_train, df_val))
		print(f"K(={kfolds})-Fold splits saved successfully in {ddir}")
		print("*"*100)
		return folds
	else:
		raise ValueError("Invalid sampling. Use 'stratified_random' or 'kfold_stratified'.")

def get_dataloaders(
		dataset_dir: str,
		sampling: str,
		batch_size: int,
		num_workers: int,
		input_resolution: int,
	)-> Tuple[DataLoader, DataLoader]:
	dataset_name = os.path.basename(dataset_dir)
	print(f"Loading dataset: {dataset_name} using {sampling} strategy...")
	train_dataset, val_dataset = get_datasets(ddir=dataset_dir, sampling=sampling)
	preprocess = get_preprocess(dataset_dir=dataset_dir, input_resolution=input_resolution)
	
	train_dataset = HistoricalArchivesDataset(
		dataset_name=dataset_name,
		train=True,
		data_frame=train_dataset.sort_values(by="img_path").reset_index(drop=True),
		transform=preprocess,
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

	validation_dataset = HistoricalArchivesDataset(
		dataset_name=dataset_name,
		train=False,
		data_frame=val_dataset.sort_values(by="img_path").reset_index(drop=True),
		transform=preprocess,
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

from utils import *
from tqdm import tqdm
import psutil

class HistoricalArchivesDataset(Dataset):
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

class HistoryDataset(Dataset):
	def __init__(
			self,
			data_frame,
			dataset_directory:str="path/2/images",
			mean:List[float]=[0.52, 0.50, 0.48],
			std:List[float]=[0.27, 0.27, 0.26],
			transform=None,
		):
		self.data_frame = data_frame
		self.dataset_directory = dataset_directory
		self.tokenized_doc_descriptions = clip.tokenize(texts=self.data_frame["label"])
		if transform:
			self.transform = transform
		else:
			self.transform = T.Compose(
				[
					T.Resize((224, 224)),
					T.ToTensor(),
					T.Normalize(mean=mean, std=std),
					# T.Normalize(
					# 	(0.48145466, 0.4578275, 0.40821073), 
					# 	(0.26862954, 0.26130258, 0.27577711)
					# )
				]
			)

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		sample = self.data_frame.iloc[idx] # Retrieve sample DataFrame row

		doc_image_path = os.path.join(self.dataset_directory, f"{sample['id']}.jpg")

		if not os.path.exists(doc_image_path): # Try to load the image and handle errors gracefully
			print(f"{doc_image_path} Not found!")
			# raise FileNotFoundError(f"Image not found at path: {img_path}") debugging purpose
			return None

		try:
			Image.open(doc_image_path).verify() # # Validate the image
			image = Image.open(doc_image_path).convert("RGB")
		except (FileNotFoundError, IOError, Exception) as e:
			# raise IOError(f"Error: Could not load image: {img_path} {e}") # debugging
			print(f"ERROR: {doc_image_path}\t{e}")
			return None

		image_tensor = self.transform(image) # <class 'torch.Tensor'> torch.Size([3, 224, 224])
		tokenized_description_tensor = self.tokenized_doc_descriptions[idx] # torch.Size([num_lbls, context_length]) [10 x 77]
		
		return image_tensor, tokenized_description_tensor 

class ResizeWithPad:
	def __init__(self, target_size, pad_color=(128, 128, 128)):
		self.target_size = target_size
		self.pad_color = pad_color

	def __call__(self, img):
		img_np = np.array(img)
		scale = min(self.target_size[0] / img_np.shape[0], self.target_size[1] / img_np.shape[1])
		new_size = tuple(int(dim * scale) for dim in img_np.shape[:2])
		resized = Image.fromarray(img_np).resize(new_size[::-1], Image.LANCZOS)
		avg_colors = tuple(np.mean(img_np, axis=(0, 1)).astype(int))
		new_img = Image.new(
			mode="RGB", 
			size=self.target_size, 
			color=tuple(avg_colors), # self.pad_color
		) 
		# Paste resized image onto padded image
		new_img.paste(
			resized, 
			((self.target_size[1] - new_size[1]) // 2, (self.target_size[0] - new_size[0]) // 2)
		)
		return new_img

class ContrastEnhanceAndDenoise:
	def __init__(self, contrast_cutoff=2, blur_radius=0.1):
		self.contrast_cutoff = contrast_cutoff
		self.blur_radius = blur_radius

	def __call__(self, img):
		img = ImageOps.autocontrast(img, cutoff=self.contrast_cutoff)
		img = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))				
		return img

	def __repr__(self):
		return self.__class__.__name__ + f'(contrast_cutoff={self.contrast_cutoff}, blur_radius={self.blur_radius})'

class HistoricalDataset(Dataset):
	def __init__(
			self, 
			data_frame, 
			img_sz:int=28, 
			txt_category:str="label", 
			dataset_directory:str="path/2/images",
			max_seq_length:int=128,
			mean:List[float]=[0.5644510984420776, 0.5516530275344849, 0.5138059854507446],
			std:List[float]=[0.2334197610616684, 0.22689250111579895, 0.2246231734752655],
			augment_data:bool=False,
		):
		self.data_frame = data_frame
		self.img_sz = img_sz  # Desired size for the square image
		self.transform = T.Compose(
			[
				# ContrastEnhanceAndDenoise(contrast_cutoff=1, blur_radius=0.1),  # Mild enhancement
				# ResizeWithPad((img_sz, img_sz)),
				T.RandomResizedCrop(size=img_sz, scale=(0.85, 1.0), ratio=(1.0, 1.0)),
				# T.RandomApply(
				# 	[
				# 		T.RandomAffine(
				# 				degrees=15,
				# 				translate=(0.1, 0.1),
				# 				scale=(0.9, 1.1),
				# 				shear=5,
				# 				fill=0
				# 		)
				# 	], 
				# 	p=0.5,
				# ),
				# T.RandomHorizontalFlip(p=0.5),
				# T.RandomApply(
				# 	[
				# 		T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
				# 	],
				# 	p=0.3,
				# ),
				# T.RandomApply(
				# 	[
				# 		T.RandomPosterize(bits=3)
				# 	],
				# 	p=0.1,
				# ),
				# # #############################################################################
				# T.RandomApply(
				# 	[
				# 		T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
				# 	],
				# 	p=0.3,
				# ),
				# T.RandomApply(
				# 	[
				# 		T.RandomRotation(degrees=15)
				# 	],
				# 	p=0.3,
				# ),
				# # T.RandomApply(
				# # 	[
				# # 		T.RandomCrop(size=img_sz, padding=4)
				# # 	],
				# # 	p=0.3,
				# # ),
				# # #############################################################################
				T.ToTensor(),
				T.Normalize(mean=mean, std=std),
			]
		)
		self.txt_category = txt_category
		self.dataset_directory = dataset_directory
		self.max_seq_length = max_seq_length

	def __len__(self) -> int:
		return len(self.data_frame)

	def __getitem__(self, idx):
		sample = self.data_frame.iloc[idx] # Retrieve sample from DataFrame (row)
		img_path = os.path.join(self.dataset_directory, f"{sample['id']}.jpg")
		if not os.path.exists(img_path): # Try to load the image and handle errors gracefully
			print(f"{img_path} Not found!")
			# raise FileNotFoundError(f"Image not found at path: {img_path}") debugging purpose
			return None
		try:
			Image.open(img_path).verify() # # Validate the image
			image = Image.open(img_path).convert("RGB")
		except (FileNotFoundError, IOError, Exception) as e:
			# raise IOError(f"Error: Could not load image: {img_path} {e}") # debugging
			print(f"ERROR: {img_path}\t{e}")
			return None
		# image = self.contrast_enhance_denoise(image)
		# image = self.resize_and_pad(image, self.img_sz)
		# print(type(image), image.size, image.mode)
		image = self.transform(image)
		label = sample[self.txt_category].lower()
		cap, mask = tokenizer(
			text=label,
			encode=True, 
			max_seq_length=self.max_seq_length,
		)
		mask = mask.repeat(len(mask), 1) # 1D tensor => (max_seq_length x max_seq_length)
		# print(f"img: {type(image)} {image.shape} ")
		# print(f"cap: {type(cap)} {cap.shape} ")
		# print(f"mask: {type(mask)} {mask.shape} ") # 1D tensor of size max_seq_length
		# print("#"*100)
		return {
			"image": image, #  <class 'torch.Tensor'> torch.Size([C, img_sz, img_sz])
			"caption": cap, # <class 'torch.Tensor'> torch.Size([max_seq_length])
			"mask": mask, # <class 'torch.Tensor'> torch.Size([max_seq_length, max_seq_length])
			"image_filepath": img_path,
		}

	def contrast_enhance_denoise(self, image, contrast_cutoff=2, blur_radius=0.1):
		image = ImageOps.autocontrast(image, cutoff=contrast_cutoff)
		image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
		return image

	def resize_and_pad(self, image, img_sz):
		original_width, original_height = image.size
		aspect_ratio = original_width / original_height
		if aspect_ratio > 1:
			new_width = img_sz
			new_height = int(img_sz / aspect_ratio)
		else:
			new_height = img_sz
			new_width = int(img_sz * aspect_ratio)
		image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
		# Compute padding to center the image
		pad_width = (img_sz - new_width) // 2
		pad_height = (img_sz - new_height) // 2
		# Apply padding to ensure the image is square
		padding = (
			pad_width, 
			pad_height, 
			img_sz - new_width - pad_width, 
			img_sz - new_height - pad_height
		)
		image = ImageOps.expand(
			image=image, 
			border=padding, 
			fill=0,
		)
		return image