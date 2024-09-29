# import torch
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# from PIL import Image
# import clip
# import os
# from tqdm import tqdm
# import numpy as np

# # Step 1: Set up the environment
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# # Step 2: Prepare the dataset
# def load_dataset(image_folder):
# 		return ImageFolder(
# 				root=image_folder,
# 				transform=preprocess
# 		)

# # Step 3: Encode images
# def encode_images(dataset, batch_size=32):
# 		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# 		all_features = []
# 		all_paths = []

# 		with torch.no_grad():
# 				for images, _ in tqdm(dataloader):
# 						features = model.encode_image(images.to(device))
# 						all_features.append(features)
# 						all_paths.extend([dataset.imgs[i][0] for i in range(len(features))])

# 		return torch.cat(all_features).cpu().numpy(), all_paths

# # Step 4: Perform text-to-image retrieval
# def retrieve_images(text_query, image_features, image_paths, top_k=5):
# 		with torch.no_grad():
# 				text_features = model.encode_text(clip.tokenize(text_query).to(device))
# 				text_features /= text_features.norm(dim=-1, keepdim=True)

# 		image_features = torch.from_numpy(image_features).to(device)
# 		image_features /= image_features.norm(dim=-1, keepdim=True)

# 		similarity = (100.0 * image_features @ text_features.T).softmax(dim=0)
# 		values, indices = similarity.topk(top_k)

# 		return [(image_paths[i], values[i].item()) for i in indices]

# # Step 5: Main function
# def main():
# 		# Replace with your dataset path
# 		image_folder = "path/to/your/historical_wartime_images"
		
# 		print("Loading dataset...")
# 		dataset = load_dataset(image_folder)
		
# 		print("Encoding images...")
# 		image_features, image_paths = encode_images(dataset)
		
# 		while True:
# 				query = input("Enter a text query (or 'quit' to exit): ")
# 				if query.lower() == 'quit':
# 						break
				
# 				results = retrieve_images(query, image_features, image_paths)
				
# 				print("\nTop 5 matching images:")
# 				for path, score in results:
# 						print(f"Image: {path}, Score: {score:.2f}")
# 				print()

# if __name__ == "__main__":
# 		main()

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image, UnidentifiedImageError
import clip
import os
from tqdm import tqdm
import numpy as np
import requests
from io import BytesIO

# Step 1: Set up the environment
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Step 2: Prepare a sample dataset
def download_image(url, save_path):
		try:
				response = requests.get(url, timeout=10)
				response.raise_for_status()
				img = Image.open(BytesIO(response.content))
				img.save(save_path)
				return True
		except (requests.RequestException, UnidentifiedImageError, OSError) as e:
				print(f"Error downloading {url}: {str(e)}")
				return False

def create_sample_dataset():
		sample_images = [
				("https://picsum.photos/800/600?random=1", "category1"),
				("https://picsum.photos/800/600?random=2", "category2"),
				("https://picsum.photos/800/600?random=3", "category3"),
				("https://picsum.photos/800/600?random=4", "category4"),
				("https://picsum.photos/800/600?random=5", "category5"),
		]
		
		base_dir = "sample_images"
		os.makedirs(base_dir, exist_ok=True)
		
		successful_downloads = 0
		for url, category in sample_images:
				category_dir = os.path.join(base_dir, category)
				os.makedirs(category_dir, exist_ok=True)
				file_name = f"image_{category}.jpg"
				save_path = os.path.join(category_dir, file_name)
				if download_image(url, save_path):
						successful_downloads += 1
		
		if successful_downloads == 0:
				raise RuntimeError("Failed to download any images. Please check your internet connection and try again.")
		
		print(f"Successfully downloaded {successful_downloads} out of {len(sample_images)} images.")
		return base_dir

# Step 3: Load the dataset
def load_dataset(image_folder):
		return ImageFolder(
				root=image_folder,
				transform=preprocess
		)

# Step 4: Encode images
def encode_images(dataset, batch_size=32):
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
		all_features = []
		all_paths = []

		with torch.no_grad():
				for images, _ in tqdm(dataloader):
						features = model.encode_image(images.to(device))
						all_features.append(features)
						all_paths.extend([dataset.imgs[i][0] for i in range(len(features))])

		return torch.cat(all_features).cpu().numpy(), all_paths

# Step 5: Perform text-to-image retrieval
def retrieve_images(text_query, image_features, image_paths, top_k=5):
		with torch.no_grad():
				text_features = model.encode_text(clip.tokenize(text_query).to(device))
				text_features /= text_features.norm(dim=-1, keepdim=True)

		image_features = torch.from_numpy(image_features).to(device)
		image_features /= image_features.norm(dim=-1, keepdim=True)

		similarity = (100.0 * image_features @ text_features.T).softmax(dim=0)
		values, indices = similarity.topk(top_k)

		return [(image_paths[i], values[i].item()) for i in indices]

# Step 6: Main function
def main():
		try:
				print("Creating sample dataset...")
				image_folder = create_sample_dataset()
				
				print("Loading dataset...")
				dataset = load_dataset(image_folder)
				
				print("Encoding images...")
				image_features, image_paths = encode_images(dataset)
				
				while True:
						query = input("Enter a text query (or 'quit' to exit): ")
						if query.lower() == 'quit':
								break
						
						results = retrieve_images(query, image_features, image_paths)
						
						print("\nTop matching images:")
						for path, score in results:
								print(f"Image: {path}, Score: {score:.2f}")
						print()
		except Exception as e:
				print(f"An error occurred: {str(e)}")
				print("Please check your internet connection and try running the script again.")

if __name__ == "__main__":
		main()