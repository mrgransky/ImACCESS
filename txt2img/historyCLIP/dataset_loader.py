from utils import *

class AddFilmGrain(object):
	def __init__(self, grain_intensity: float = 0.1, blend_factor: float = 0.2):
		self.grain_intensity = grain_intensity
		self.blend_factor = blend_factor
	def __call__(self, image: Image.Image) -> Image.Image:
				# Convert image to numpy array
				img_np = np.array(image).astype(np.float32)
				
				# Generate subtle random noise
				noise = np.random.normal(0, 1, img_np.shape) * self.grain_intensity * 255
				
				# Blend noise with original image
				img_with_grain = img_np * (1 - self.blend_factor) + noise * self.blend_factor
				
				# Ensure values stay in valid range
				img_with_grain = np.clip(img_with_grain, 0, 255).astype(np.uint8)
				
				# Convert back to PIL Image
				return Image.fromarray(img_with_grain)

class ResizeWithPad:
		def __init__(self, target_size):
			self.target_size = target_size
		def __call__(self, img):
			img_np = np.array(img)
			# Calculate scaling factor
			scale = min(self.target_size[0] / img_np.shape[0], self.target_size[1] / img_np.shape[1])
			new_size = tuple(int(dim * scale) for dim in img_np.shape[:2])
			resized = Image.fromarray(img_np).resize(new_size[::-1], Image.LANCZOS)
			new_img = Image.new("RGB", self.target_size, color=0) # Create new image with padding
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
			captions,
			img_sz:int=28, 
			txt_category:str="query", 
			dataset_directory:str="path/2/images",
			max_seq_length:int=128,
			mean:List[float]=[0.5644510984420776, 0.5516530275344849, 0.5138059854507446],
			std:List[float]=[0.2334197610616684, 0.22689250111579895, 0.2246231734752655],
		):
		self.data_frame = data_frame
		self.img_sz = img_sz  # Desired size for the square image
		# self.transform = T.Compose([
		# 	ContrastEnhanceAndDenoise(contrast_cutoff=1, blur_radius=0.1),  # Mild enhancement
		# 	# T.Lambda(lambda img: ImageOps.equalize(img)),  # Adaptive Histogram Equalization
		# 	ResizeWithPad((img_sz, img_sz)),  # Custom resize and pad
		# 	# AddFilmGrain(grain_intensity=0.1, blend_factor=0.2),
		# 	T.RandomHorizontalFlip(p=0.5),
		# 	T.RandomAffine(
		# 		degrees=10, 
		# 		translate=(0.05, 0.05), 
		# 		scale=(0.95, 1.05),
		# 		shear=5,
		# 		fill=0,
		# 	),
		# 	T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05)], p=0.2),
		# 	# T.RandomApply([T.Lambda(lambda img: ImageOps.invert(img))], p=0.05),  # Occasional invert
		# 	T.ToTensor(),  # Convert image to tensor
		# 	T.Normalize(mean=[mean], std=[std]),  # Normalize using calculated mean and std
		# ])
		self.transform = T.Compose(
			[
				ContrastEnhanceAndDenoise(contrast_cutoff=1, blur_radius=0.1),  # Mild enhancement
				ResizeWithPad((img_sz, img_sz)),
				T.RandomApply(
					[
						T.RandomAffine(
								degrees=15,
								translate=(0.1, 0.1),
								scale=(0.9, 1.1),
								shear=5,
								fill=0
						)
					], 
					p=0.5,
				),
				T.RandomHorizontalFlip(p=0.5),
				T.RandomApply(
					[
						T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
					],
					p=0.3,
				),
				T.RandomApply(
					[
						T.RandomPosterize(bits=3)
					],
					p=0.1,
				),
				T.ToTensor(),
				T.Normalize(mean=mean, std=std),
			]
		)
		self.captions = captions
		self.txt_category = txt_category
		self.dataset_directory = dataset_directory
		self.max_seq_length = max_seq_length

	def __len__(self) -> int:
		return len(self.data_frame)

	def __getitem__(self, idx):
		sample = self.data_frame.iloc[idx] # Retrieve the sample from the DataFrame
		img_path = os.path.join(self.dataset_directory, f"{sample['id']}.jpg")
		if not os.path.exists(img_path): # Try to load the image and handle errors gracefully
			print(f"{img_path} Not found!")
			# raise FileNotFoundError(f"Image not found at path: {img_path}") debugging purpose
			return None
		try:
			Image.open(img_path).verify() # # Validate the image
			image = Image.open(img_path).convert("RGB")
		except (FileNotFoundError, IOError, Exception) as e:
			# raise IOError(f"Could not load image: {img_path}, Error: {str(e)}") # debugging
			print(f"{img_path} ERROR: {e}")
			return None
		# image = self.contrast_enhance_denoise(image)
		# image = self.resize_and_pad(image, self.img_sz)
		# print(type(image), image.size, image.mode)
		image = self.transform(image)
		label = sample[self.txt_category].lower()
		if label not in self.captions.values():
			# raise KeyError(f"Label '{label}' not found in captions dictionary")
			return None
		label_idx = next(idx for idx, class_name in self.captions.items() if class_name == label)
		cap, mask = tokenizer(
			text=self.captions[label_idx], 
			encode=True, 
			max_seq_length=self.max_seq_length,
		)
		mask = torch.tensor(mask)
		if len(mask.size()) == 1:
			mask = mask.unsqueeze(0)
		return {
			"image": image,
			"caption": cap,
			"mask": mask,
			"image_filepath": img_path
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