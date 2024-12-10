from utils import *

# Define a custom dataset class
class HistoryDataset(Dataset):
	def __init__(
			self,
			data_frame,
			dataset_directory:str="path/2/images",
			mean:List[float]=[0.52, 0.50, 0.48],
			std:List[float]=[0.27, 0.27, 0.26],
			transformer=None,
		):
		self.data_frame = data_frame
		self.dataset_directory = dataset_directory
		self.tokenized_doc_descriptions = clip.tokenize(texts=self.data_frame["label"])
		if transformer:
			self.transform = transformer
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