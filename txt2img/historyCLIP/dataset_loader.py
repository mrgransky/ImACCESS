from utils import *

class HistoricalDataset(Dataset):
	def __init__(self, data_frame, captions, img_sz=28, txt_category="query", dataset_directory="path/2/images"):
		self.data_frame = data_frame
		self.img_sz = img_sz  # Desired size for the square image
		self.transform = T.Compose([
			T.RandomResizedCrop(img_sz),
			T.RandomHorizontalFlip(p=0.5),
			T.ColorJitter(brightness=0.3, contrast=0.3),  # Adjust brightness/contrast
			T.ToTensor(),  # Convert image to tensor
		])
		self.captions = captions
		self.txt_category = txt_category
		self.dataset_directory = dataset_directory

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		# Retrieve the sample from the DataFrame
		sample = self.data_frame.iloc[idx]
		# Construct the image path
		img_path = os.path.join(self.dataset_directory, f"{sample['id']}.jpg")
		# Try to load the image and handle errors gracefully
		if not os.path.exists(img_path):
			print(f"{img_path} Not found!")
			# raise FileNotFoundError(f"Image not found at path: {img_path}") debugging purpose
			return None
		try:
			# print(img_path)
			image = Image.open(img_path).convert('RGB')
			# image = Image.open(img_path).convert('L')
		except (FileNotFoundError, IOError, Exception) as e:
			# raise IOError(f"Could not load image: {img_path}, Error: {str(e)}") # debugging
			print(f"{img_path} ERROR: {e}")
			return None
		# Resize the image to maintain aspect ratio and apply transformations
		image = self.resize_and_pad(image, self.img_sz)
		image = self.transform(image)
		# Retrieve label and its corresponding caption
		label = sample[self.txt_category].lower()
		# Check if label exists in captions dictionary
		if label not in self.captions.values():
			# raise KeyError(f"Label '{label}' not found in captions dictionary")
			return None
		label_idx = next(idx for idx, class_name in self.captions.items() if class_name == label)
		# Tokenize the caption using the tokenizer function
		cap, mask = tokenizer(self.captions[label_idx])
		# Ensure the mask is a tensor and correct the shape if necessary
		mask = torch.tensor(mask)
		if len(mask.size()) == 1:
			mask = mask.unsqueeze(0)
		return {
			"image": image,
			"caption": cap,
			"mask": mask,
			"image_filepath": img_path
		}

	def resize_and_pad(self, image, img_sz):
		"""Resize the image to maintain aspect ratio and pad it to the target size."""
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
		image = ImageOps.expand(image, padding, fill=(0, 0, 0))
		# print(image.size) # must be square => e.g., (250 x 250)
		return image