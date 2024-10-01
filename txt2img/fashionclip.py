import os
import sys
import re
from tqdm import tqdm
import random
from collections import Counter
import json
import warnings
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from PIL import Image, ImageDraw, ImageOps
from typing import List, Set, Dict, Tuple, Union
warnings.filterwarnings('ignore')

# how to run [Local]:
# $ python fashionclip.py --query tie
# $ python fashionclip.py --query tie --dataset_dir myntradataset --num_epochs 7 # original functioning!
# $ nohup python -u fashionclip.py --num_epochs 100 > $HOME/datasets/trash/logs/fashionclip.out & 

# how to run [Pouta]:
# $ python fashionclip.py --dataset_dir /media/volume/ImACCESS/myntradataset --num_epochs 13
# $ nohup python -u --dataset_dir /media/volume/ImACCESS/myntradataset --num_epochs 3 --query "topwear" > /media/volume/ImACCESS/trash/logs/fashionclip.out & 

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--query', type=str, default="bags", help='Query')
parser.add_argument('--topk', type=int, default=5, help='Top-K images')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--validation_dataset_share', type=float, default=0.20, help='share of Validation set')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--validate', type=bool, default=True, help='Model Validation upon request')
parser.add_argument('--product_description_col', type=str, default="subCategory", help='caption col ["articleType", "subCategory", "customized_caption"]')
args = parser.parse_args()
print(args)

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
visualize: bool = False

if USER=="farid": # local laptop
	WDIR = os.path.join(HOME, "datasets")
	models_dir = os.path.join(WDIR, "trash", "models")
	visualize = True
elif USER=="alijanif": # Puhti
	WDIR = "/scratch/project_2004072/ImACCESS"
	models_dir = os.path.join(WDIR, "trash", "models")
else: # Pouta
	WDIR = "/media/volume/ImACCESS"
	models_dir = os.path.join(WDIR, "models")

os.makedirs(os.path.join(args.dataset_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(args.dataset_dir, "outputs"), exist_ok=True)

# Vision
emb_dim = 128
vit_d_model = 32 # vit_heads * vit_layers = vit_d_model
img_size = (80,80)
patch_size = (5,5) 
n_channels = 3
vit_layers = 8
vit_heads = 4 

# Text
vocab_size = 256
text_d_model = 64 #  -->  text_heads * text_layers = text_d_model
max_seq_length = 128
text_heads = 8
text_layers = 8
wd = 1e-4 # L2 Regularization
nw:int = multiprocessing.cpu_count() # def: 8
mdl_fpth:str = os.path.join(
	args.dataset_dir, 
	"models",
	f"fashionclip_nEpochs_{args.num_epochs}_"
	f"batchSZ_{args.batch_size}_"
	f"lr_{args.learning_rate}_"
	f"val_{args.validation_dataset_share}_"
	f"descriptions_{args.product_description_col}.pt",
)
outputs_dir:str = os.path.join(
	args.dataset_dir, 
	"outputs",
)

def get_dframe(fpth: str="path/2/file.csv", img_dir: str="path/2/images"):
	print(f"Laoding style (csv): {fpth}")
	replacement_dict = {
		"lips": "lipstick",
		"eyes": "eyelash",
		"nails": "nail polish"
	}
	styles_df = pd.read_csv(
		filepath_or_buffer=fpth,
		usecols=[
			"id",
			"gender",
			"masterCategory",
			"subCategory",
			"articleType",
			"baseColour",
			"season",
			"year",
			"usage",
			"productDisplayName",
		], 
		on_bad_lines='skip',
	)
	# Convert all text columns to lowercase
	styles_df[styles_df.select_dtypes(include=['object']).columns] = styles_df.select_dtypes(include=['object']).apply(lambda x: x.str.lower())
	styles_df['subCategory'] = styles_df['subCategory'].replace(replacement_dict)
	# Create a new column 'customized_caption'
	styles_df['customized_caption'] = styles_df.apply(
		# lambda row: f"{row['subCategory']} {row['articleType']}" if row['subCategory'] != row['articleType'] else row['subCategory'],
		lambda row: row['articleType'] if row['subCategory'] in row['articleType'] else f"{row['subCategory']} {row['articleType']}",
		axis=1,
	)


	# Check for existence of images and filter DataFrame
	styles_df['image_exists'] = styles_df['id'].apply(lambda x: os.path.exists(os.path.join(img_dir, f"{x}.jpg")))
	# Drop rows where the image does not exist
	filtered_df = styles_df[styles_df['image_exists']].drop(columns=['image_exists'])

	# df = styles_df.copy() # without checking image dir
	df = filtered_df.copy()

	# print(f"df: {df.shape}")
	# print(df.head(10))
	# print(df['subCategory'].value_counts())
	# print("#"*100)

	return df

def get_img_name_without_suffix(fpth):
	# Get the basename of the file path (removes directory)
	basename = os.path.basename(fpth)
	# Split the basename into filename and extension
	filename, extension = os.path.splitext(basename)
	return int(filename)

def plot_loss(losses, num_epochs, save_path):
	"""
	Plots the loss with respect to epoch and saves the plot.
	Parameters:
	losses (list): List of loss values for each epoch.
	num_epochs (int): Number of epochs.
	save_path (str): Path to save the plot.
	"""
	epochs = range(1, num_epochs + 1)		
	plt.figure(figsize=(10, 5))
	plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(f'Loss vs. Epoch (LR: {args.learning_rate})')
	plt.grid(True)
	plt.savefig(save_path)
	
def set_seeds():
	# fix random seeds
	SEED_VALUE = 42
	random.seed(SEED_VALUE)
	np.random.seed(SEED_VALUE)
	torch.manual_seed(SEED_VALUE)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(SEED_VALUE)
		torch.cuda.manual_seed_all(SEED_VALUE)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True

def tokenizer(text, encode=True, mask=None, max_seq_length=32):
	if encode:
		# Adding SOT and EOT tokens
		out = chr(2) + text + chr(3)
		# Truncate if length exceeds max_seq_length
		if len(out) > max_seq_length:
			out = out[:max_seq_length]
		# Add padding if needed
		out = out + "".join([chr(0) for _ in range(max_seq_length - len(out))])
		# Encode the text
		out = torch.IntTensor(list(out.encode("utf-8")))
		# Create the mask
		mask = torch.ones(len(out.nonzero()))
		# Pad the mask to max_seq_length
		if len(mask) < max_seq_length:
			mask = torch.cat((mask, torch.zeros(max_seq_length - len(mask)))).type(torch.IntTensor)
		else:
			mask = mask.type(torch.IntTensor)
	else:
		# Decode the text
		out = [chr(x) for x in text[1:len(mask.nonzero()) - 1]]
		out = "".join(out)
		mask = None
	return out, mask

def get_info(dataloader):
	tot_samples = len(dataloader.dataset)
	n_chunks = len(dataloader) # ceil(tot_samples / dataloader.batch_size)
	print(
		# f"dataloader organization has main {len(next(iter(dataloader)))} element(s)\n"
		f"Total samples:: {tot_samples} "
		f"divided into {n_chunks} chunk(s) "
		f"using batch size: {dataloader.batch_size} "
		f"in {dataloader.num_workers} CPU(s)"
	)
	for i, data in enumerate(dataloader):
		print(
			f'[{i+1}/{n_chunks}] '
			f'{len(data["image_filepath"])} image_filepath {type(data["image_filepath"])} '
			f'{data["caption"].shape} caption {type(data["image_filepath"])}')
		print(f"Batch {i+1}: {len([img for img in data['image_filepath']])} {[img for img in data['image_filepath']]}")
		c = Counter(data["image_filepath"])
		# print(f"{json.dumps(c, indent=2, ensure_ascii=False)}")
		print("#"*100)
		# print()
		# if i == 0:  # Just show the first batch as an example
		# 	print(f"For Sample batch {i}:")
		# 	# for key in data.keys():
		# 	# 	print(f"{key}")  # Print shape of each item in the batch
		# 	# 	# print(data["caption"])
		# 	print(f'caption: {data["caption"].shape} {type(data["caption"])}')
		# 	print(f'image: {data["image"].shape} {type(data["image"])}')
		# 	print(f'image_filepath: {len(data["image_filepath"])} {data["image_filepath"][:5]} {type(data["image_filepath"])}')
		# 	break  # Exit after printing the first batch

class TransformerEncoder(nn.Module):
	def __init__(self, d_model, n_heads, mlp_ratio =4):
		super().__init__()
		self.d_model = d_model
		self.n_heads = n_heads
		self.ln1 = nn.LayerNorm(d_model)
		self.mha = MultiheadAttention(d_model, n_heads)
		self.ln2 = nn.LayerNorm(d_model)
		self.mlp = nn.Sequential(
			nn.Linear(d_model, d_model*mlp_ratio),
			nn.GELU(),
			nn.Linear(d_model * mlp_ratio, d_model)
		)
	#For clip even though its a encoder model it requires mask ->to account for padded for max seq_length
	def forward(self, x, mask = None):
		x_n = self.mha(self.ln1(x), mask = mask)
		x = x + self.mlp(self.ln2(x_n))
		return x  # x.shape -->  [B,max_seq_len,d_model]

class MultiheadAttention(nn.Module):
	def __init__(self, d_model, n_heads):
		super().__init__()
		# d_model --> embed dimension 
		# n_heads --> number of heads 
		self.qkv_dim = d_model //  n_heads #or self.head_size
		self.W_o = nn.Linear(d_model,d_model) #Dense layer
		self.multi_head = nn.ModuleList([AttentionHead(d_model, self.qkv_dim) for _ in range(n_heads)])
	def forward(self,x,mask = None):
		 #x.shape --> [B,max_seq_len,d_model]
		#Concatenates the outputs from all attention heads along the last dimension (dim=-1)
		out = torch.cat([head(x, mask=mask) for head in self.multi_head], dim = -1) #  [B,max_seq_len,d_model]
		# Apply the linear transformation
		out = self.W_o(out)   #---> (Concat --> Dense)  -- [B,max_seq_len,d_model]
		return out

class AttentionHead(nn.Module):
	def __init__(self, d_model, qkv_dim):
		super().__init__()
		self.qkv_dim = qkv_dim
		self.query = nn.Linear(d_model, qkv_dim)
		self.key = nn.Linear(d_model, qkv_dim)
		self.value = nn.Linear(d_model, qkv_dim)
	def forward(self, x, mask = None):
		# x.shape -->  [B,max_seq_len,d_model]
		Q = self.query(x) #[B,max_seq_len,vit_heads]
		K = self.key(x)
		V = self.value(x)
		attention = Q @ K.transpose(-2,-1) #eg: -2 -second last dim and -1 last dim -->  [B,max_seq_len,max_seq_len]
		#Scaling
		attention = attention / self.qkv_dim ** 0.5  #  [B,max_seq_len,max_seq_len]
		#Apply attention mask for padded sequence
		if mask is not None:
			mask = attention.masked_fill(mask == 0, float("-inf")) # torch.tensor.masked_fill
		# Apply softmax to obtain attention weights [Wij]
		attention  = torch.softmax(attention, dim = -1) #along last dim  # (softmax(Q_K^T)/sqrt(d_k)).V -->  [B,max_seq_len,max_seq_len]
		attention = attention @ V #  [B,max_seq_len,max_seq_len]
		return attention  #Y_i

class PositionalEmbedding(nn.Module):
	def __init__(self, d_model, max_seq_length):
		super().__init__()
		self.d_model = d_model
		self.max_seq_length = max_seq_length
		pe = torch.zeros(max_seq_length, d_model)
		position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe.unsqueeze(0))
	def forward(self, x):
		seq_len = x.size(1)
		return x + self.pe[:, :seq_len]

class VisionEncoder(nn.Module):
	def __init__(self, d_model,img_size,patch_size, n_channels, n_heads,n_layers, emb_dim):
		super().__init__()
		assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] ==0, "image dimensions should be divisible by patch dim"
		assert d_model % n_heads == 0, "d_model should be divisible by n_heads"
		self.num_patches = (img_size[0] * img_size[1] ) // (patch_size[0] * patch_size[1]) # max_seq_length
		self.max_seq_length = self.num_patches +1
		self.linear_proj = nn.Conv2d(in_channels = n_channels,out_channels = d_model, kernel_size = patch_size[0], stride = patch_size[0])
		self.cls_token = nn.Parameter(torch.randn(1,1,d_model), requires_grad = True)
		self.positional_embedding =  PositionalEmbedding(d_model, self.max_seq_length)
		self.transformer_encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])
		self.projection = nn.Parameter(torch.randn(d_model, emb_dim))
	def forward(self,x, mask = None):			 
		x  = self.linear_proj(x)  # (B, C, H, W) -> (B, d_model, Patch_col_d_model, Patch_row_height)  
		x = x.flatten(2).transpose(-2, -1)   # (B, d_model, Patch_col_d_model, Patch_row_height) --> Flatten (B, d_model, Patch) --> .transpose(-2,-1) (B, Patch, d_model)
		# The input to the transformer we need to pass a sequence of patches or tokens so we need num_patches to be before hidden dim
		x = torch.cat((self.cls_token.expand(x.shape[0], -1,-1), x), dim = 1) #add cls token at the beginning of patch_sequence   -->  [B,max_seq_len,d_model]
		x =  self.positional_embedding(x)  #  [B,max_seq_len,d_model]
		for encoder_layer in self.transformer_encoder:
			x = encoder_layer(x, mask)  #  [B, d_model]
		# Get learned class tokens
		x = x[:, 0, :]
		# Project to shared embedding space
		if self.projection is not None:
			x = x @ self.projection  #[B, emb_dim]
		x  = x / torch.norm(x , dim = -1 , keepdim = True) 
		return x

class TextEncoder(nn.Module):
	def __init__(self, vocab_size, d_model, max_seq_length, n_layers,n_heads, emb_dim):
		super().__init__()
		self.max_seq_length = max_seq_length
		self.embed = nn.Embedding(vocab_size, d_model)
		self.positional_embedding = PositionalEmbedding(d_model, max_seq_length)
		self.transformer_encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])
		self.projection = nn.Parameter(torch.randn(d_model, emb_dim))
	# For training
	def forward(self, text, mask = None):
		x = self.embed(text)
		x = self.positional_embedding(x)
		for encoder_layer in self.transformer_encoder:
			x = encoder_layer(x, mask=mask)
		#The output of the encoder layers is the text features. We are going to be using the features from the EOT embedding.
		x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:,0],dim=1),1)]
		if self.projection is not None:
			x = x @ self.projection
		x = x / torch.norm(x, dim=-1, keepdim = True)
		return x

class TextEncoder_Retrieval(nn.Module):
	def __init__(self, vocab_size, d_model, max_seq_length, n_layers,n_heads, emb_dim):
		super().__init__()
		self.max_seq_length = max_seq_length
		self.embed = nn.Embedding(vocab_size, d_model)
		self.positional_embedding = PositionalEmbedding(d_model, max_seq_length)
		self.transformer_encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])
		self.projection = nn.Parameter(torch.randn(d_model, emb_dim))
	# # For image retrieval
	def forward(self, text, mask=None):
		x = self.embed(text)
		x = self.positional_embedding(x)
		for encoder_layer in self.transformer_encoder:
			x = encoder_layer(x, mask=mask)
		if mask is not None:
			# Get the lengths of each sequence (i.e., find the last non-padded token)
			seq_lengths = mask.sum(dim=1) - 1  # Subtract 1 to get the index
			x = x[torch.arange(text.shape[0]), seq_lengths]
		else:
			x = x[:, -1]  # If no mask is provided, take the last token in the sequence.
		if self.projection is not None:
			x = x @ self.projection
		x = x / torch.norm(x, dim=-1, keepdim=True)		
		return x

class CLIP(nn.Module):
	def __init__(self, emb_dim, vit_layers, vit_d_model, img_size, patch_size, n_channels, vit_heads, vocab_size, max_seq_length, text_heads, text_layers, text_d_model, retrieval=False):
		super().__init__()
		self.vision_encoder = VisionEncoder(vit_d_model, img_size, patch_size, n_channels, vit_heads, vit_layers, emb_dim)
		if retrieval:
			self.text_encoder = TextEncoder_Retrieval(vocab_size, text_d_model, max_seq_length, text_layers, text_heads, emb_dim)
		else:
			self.text_encoder = TextEncoder(vocab_size, text_d_model, max_seq_length, text_layers, text_heads, emb_dim)
		self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	def CLIPLoss(self, logits, device = "cuda"):
		#Symmetric or Contrastive loss
		# arange generates a list between 0 and n-1
		labels = torch.arange(logits.shape[0]).to(device)  # For row 1 we want 1,1 to be max, and row n-1 we want (n-1,n-1) text pairs to be max --> time 15.43 umar
		loss_v = nn.functional.cross_entropy(logits.transpose(-2,-1), labels)
		loss_t = nn.functional.cross_entropy(logits, labels)
		loss = (loss_v + loss_t) / 2
		return loss
	def forward(self, image, text, mask=None):
		V_e = self.vision_encoder(image)  # Vision encoder output [B, emb_dim]
		T_e = self.text_encoder(text, mask)  # Text encoder output [B, emb_dim]
		# print(f"V_e shape: {V_e.shape}, T_e shape: {T_e.shape}")
		logits = (V_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)
		loss = self.CLIPLoss(logits, self.device)
		return loss

class MyntraDataset(Dataset):
		def __init__(self, data_frame, captions, img_sz=28, txt_category="subCategory", dataset_directory="path/2/images"):
				self.data_frame = data_frame
				self.img_sz = img_sz  # Desired size for the square image
				self.transform = T.Compose([
						T.ToTensor()  # Convert image to tensor
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
					# raise FileNotFoundError(f"Image not found at path: {img_path}") debugging purpose
					return None

				try:
					image = Image.open(img_path).convert('RGB')
				except (FileNotFoundError, IOError) as e:
					# raise IOError(f"Could not load image: {img_path}, Error: {str(e)}") # debugging
					return None

				# Resize the image to maintain aspect ratio and apply transformations
				image = self.resize_and_pad(image, self.img_sz)
				image = self.transform(image)

				# Retrieve the subCategory label and its corresponding caption
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

				image = image.resize((new_width, new_height))

				# Compute padding to center the image
				pad_width = (img_sz - new_width) // 2
				pad_height = (img_sz - new_height) // 2

				# Apply padding to ensure the image is square
				padding = (pad_width, pad_height, img_sz - new_width - pad_width, img_sz - new_height - pad_height)
				image = ImageOps.expand(image, padding, fill=(0, 0, 0))

				return image

def get_product_description(df, col:str="colmun_name"):
	class_names = list(df[col].unique())
	captions = {idx: class_name for idx, class_name in enumerate(class_names)}
	# print(f"{len(list(captions.keys()))} Captions:\n{json.dumps(captions, indent=2, ensure_ascii=False)}")
	return captions, class_names

def validate(model_fpth: str=f"path/to/models/clip.pt", TOP_K: int=10):
	print(f"Validation {model_fpth} using {device}".center(100, "-"))
	vdl_st = time.time()
	print(f"Creating Validation Dataloader for {len(val_df)} samples", end="\t")
	vdl_st = time.time()
	val_dataset = MyntraDataset(
		data_frame=val_df,
		captions=captions,
		img_sz=80,
		dataset_directory=os.path.join(args.dataset_dir, "images")
	)
	val_loader = DataLoader(
		dataset=val_dataset, 
		shuffle=False,
		batch_size=args.batch_size, 
		num_workers=nw,
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(val_loader.dataset)} Elapsed_t: {time.time()-vdl_st:.5f} sec")
	# get_info(dataloader=val_loader)

	# Loading Best Model
	model = CLIP(
		emb_dim, 
		vit_layers, 
		vit_d_model,
		img_size,
		patch_size,
		n_channels,
		vit_heads,
		vocab_size,
		max_seq_length,
		text_heads,
		text_layers,
		text_d_model,
		retrieval=False,
	).to(device)
	
	model.load_state_dict(torch.load(model_fpth, map_location=device))
	text = torch.stack([tokenizer(x)[0] for x in val_dataset.captions.values()]).to(device)
	mask = torch.stack([tokenizer(x)[1] for x in val_dataset.captions.values()])
	mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)
	correct, total = 0,0

	with torch.no_grad():
		for data in val_loader:				
			images, labels = data["image"].to(device), data["caption"].to(device)
			image_features = model.vision_encoder(images)
			text_features = model.text_encoder(text, mask=mask)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			text_features /= text_features.norm(dim=-1, keepdim=True)
			similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
			_, indices = torch.max(similarity,1)
			pred = torch.stack([tokenizer(val_dataset.captions[int(i)])[0] for i in indices]).to(device)
			correct += int(sum(torch.sum((pred==labels),dim=1)//len(pred[0])))
			total += len(labels)

	print(f'\nModel Accuracy: {100 * correct // total} %')
	text = torch.stack([tokenizer(x)[0] for x in class_names]).to(device)
	mask = torch.stack([tokenizer(x)[1] for x in class_names])
	mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)
	# idx = 19
	idx = random.randint(0, len(val_df))
	img = val_dataset[idx]["image"][None,:]

	if visualize:
		plt.imshow(img[0].permute(1, 2, 0)  ,cmap="gray")
		plt.title(tokenizer(val_dataset[idx]["caption"], encode=False, mask=val_dataset[idx]["mask"][0])[0])
		plt.show()

	img = img.to(device)
	with torch.no_grad():
		image_features = model.vision_encoder(img)
		text_features = model.text_encoder(text, mask=mask)

	image_features /= image_features.norm(dim=-1, keepdim=True)
	text_features /= text_features.norm(dim=-1, keepdim=True)
	similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
	values, indices = similarity[0].topk(TOP_K)

	# Print the result
	print(f'\nTop-{TOP_K} Prediction(s) for IMG: {idx} {tokenizer(val_dataset[idx]["caption"], encode=False, mask=val_dataset[idx]["mask"][0])}:\n')
	for value, index in zip(values, indices):
		print(f"index: {index}: {class_names[int(index)]:>30s}: {100 * value.item():.3f}%")
	print(f"Elapsed_t: {time.time()-vdl_st:.2f} sec")

def img_retrieval(query:str="bags", model_fpth: str=f"path/to/models/clip.pt", TOP_K: int=10):
	print(f"Top-{TOP_K} Image Retrieval for Query: {query}".center(100, "-"))
	print(f"val_df: {val_df.shape} | {val_df['subCategory'].value_counts().shape} / {df['subCategory'].value_counts().shape}")
	if query not in val_df['subCategory'].value_counts():
		print(f"Query: {query} Not Found! Search something else! from the list:")
		print(val_df['subCategory'].value_counts())
		return

	model = CLIP(
		emb_dim,
		vit_layers, 
		vit_d_model, 
		img_size,
		patch_size,
		n_channels,
		vit_heads,
		vocab_size,
		max_seq_length,
		text_heads,
		text_layers,
		text_d_model,
		retrieval=False,
	).to(device)

	retrieval_model = CLIP(
		emb_dim, 
		vit_layers, 
		vit_d_model, 
		img_size,patch_size,
		n_channels,
		vit_heads,
		vocab_size,
		max_seq_length,
		text_heads,
		text_layers,
		text_d_model,
		retrieval=True
	).to(device)
	
	retrieval_model.load_state_dict(torch.load(model_fpth, map_location=device))

	# Step 1: Encode the text query using your tokenizer and TextEncoder
	query_text, query_mask = tokenizer(query)
	print(type(query_text), type(query_mask))
	query_text = query_text.unsqueeze(0).to(device) # Add batch dimension
	query_mask = query_mask.unsqueeze(0).to(device)

	with torch.no_grad():
		query_features = retrieval_model.text_encoder(query_text, mask=query_mask)
		query_features /= query_features.norm(dim=-1, keepdim=True)

	# Step 2: Encode all images in the dataset and store features
	image_features_list = []
	val_images_paths = []
	val_images_descriptions = []

	print(f"Creating Validation Dataloader for {len(val_df)} images", end="\t")
	vdl_st = time.time()
	val_dataset = MyntraDataset(
		data_frame=val_df,
		captions=captions,
		img_sz=80,
		dataset_directory=os.path.join(args.dataset_dir, "images")
	)
	val_loader = DataLoader(
		dataset=val_dataset, 
		shuffle=False,
		batch_size=args.batch_size, #32, # double check!!!! 
		num_workers=nw,
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(val_loader.dataset)} Elapsed_t: {time.time()-vdl_st:.5f} sec")
	# get_info(dataloader=val_loader)
	# return

	with torch.no_grad():
		for batch in val_loader:
			# print(batch)
			images = batch["image"].to(device)
			features = retrieval_model.vision_encoder(images)
			features /= features.norm(dim=-1, keepdim=True)			
			image_features_list.append(features)
			# print(type(batch["image_filepath"]), type(batch.get("caption")))
			val_images_paths.extend(batch["image_filepath"])  # Assuming batch contains image paths or IDs
			val_images_descriptions.extend(batch.get("caption"))
	
	print(f"val_images_paths {type(val_images_paths)} {len(val_images_paths)}")
	print(f"val_images_descriptions: {type(val_images_descriptions)} {len(val_images_descriptions)}")
	# print(val_images_descriptions) # Tensor [0. 10. 11. 19, 2727. ...]

	# Concatenate all image features
	image_features = torch.cat(image_features_list, dim=0)

	# Step 3: Compute similarity using the CLIP model's logic
	# In your CLIP model, this is done using logits and temperature scaling
	similarities = (query_features @ image_features.T) * torch.exp(model.temperature)

	# Apply softmax to the similarities if needed
	similarities = similarities.softmax(dim=-1)
	print(type(similarities), similarities.shape)
	print(similarities)

	# Retrieve topK matches
	top_values, top_indices = similarities.topk(TOP_K)
	print(type(top_values), type(top_indices))
	print(top_values.shape, top_indices.shape, TOP_K)
	print(top_values)
	print(top_indices)

	# Step 4: Retrieve and display (or save) top N images:
	print(f"Top-{TOP_K} images from Validation: ({len(val_loader.dataset)}) | Query: '{query}':\n")
	fig, axes = plt.subplots(1, TOP_K, figsize=(18, 4))  # Adjust figsize as needed
	for ax, value, index in zip(axes, top_values[0], top_indices[0]):
		img_path = val_images_paths[index]
		img_fname = get_img_name_without_suffix(fpth=img_path)
		img_GT = df.loc[df['id'] == img_fname, 'subCategory'].values
		print(f"vidx: {index} | Similarity: {100 * value.item():.6f}% | {img_path} | GT: {img_GT}")
		img = Image.open(img_path).convert("RGB")
		img_title = f"vidx_{index}_sim_{100 * value.item():.2f}%\nGT: {img_GT}"
		ax.set_title(img_title, fontsize=9)
		ax.axis('off')
		ax.imshow(img)
	plt.tight_layout()
	plt.savefig(os.path.join(outputs_dir, f"Top_{TOP_K}_imgs_Q_{re.sub(' ', '-', query)}_{args.num_epochs}_epochs.png"))

df = get_dframe(
	fpth=os.path.join(args.dataset_dir, "styles.csv"), 
	img_dir=os.path.join(args.dataset_dir, "images"), 
)
# sys.exit()

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(
	df, 
	shuffle=True, 
	test_size=args.validation_dataset_share, # 0.05
	random_state=42,
)

# Print the sizes of the datasets
print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

captions, class_names = get_product_description(df=df, col=args.product_description_col)
# sys.exit()

def custom_collate_fn(batch):
	# Filter out the None values from the batch
	batch = [item for item in batch if item is not None]
	# Use default collate function on the filtered batch
	return default_collate(batch)

def fine_tune():
	print(f"Fine-tuning in {torch.cuda.get_device_name(device)} using {nw} CPU(s)".center(150, "-"))
	print(f"Creating Train Dataloader", end="\t")
	tdl_st = time.time()
	train_dataset = MyntraDataset(
		data_frame=train_df,
		captions=captions, 
		img_sz=80,
		dataset_directory=os.path.join(args.dataset_dir, "images")
	)
	train_loader = DataLoader(
		dataset=train_dataset,
		shuffle=True,
		batch_size=args.batch_size,
		num_workers=nw,
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(train_loader.dataset)} Elapsed_t: {time.time()-tdl_st:.5f} sec")
	# get_info(dataloader=train_loader)
	model = CLIP(
		emb_dim, 
		vit_layers,
		vit_d_model,
		img_size,
		patch_size,
		n_channels,
		vit_heads,
		vocab_size,
		max_seq_length,
		text_heads,
		text_layers,
		text_d_model,
		retrieval=False,
	).to(device)
	optimizer = optim.AdamW(
		params=model.parameters(), 
		lr=args.learning_rate, 
		weight_decay=wd, # weight decay (L2 regularization)
	)
	scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

	total_params = 0
	total_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
	print(f"Total trainable parameters: {total_params} ~ {total_params/int(1e+6):.2f} M")
	best_loss = np.inf
	print(f"Training {args.num_epochs} Epoch(s) in {device}".center(100, "-"))
	training_st = time.time()
	average_losses = list()
	for epoch in range(args.num_epochs):
		print(f"Epoch [{epoch+1}/{args.num_epochs}]")
		epoch_loss = 0.0  # To accumulate the loss over the epoch
		for batch_idx, data in enumerate(train_loader):
			img = data["image"].to(device) 
			cap = data["caption"].to(device)
			mask = data["mask"].to(device)
			optimizer.zero_grad()
			loss = model(img, cap, mask)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			if batch_idx % 100 == 0:
				print(f"\tBatch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.5f}")
			epoch_loss += loss.item()
		avg_loss = epoch_loss / len(train_loader)
		scheduler.step(avg_loss)
		print(f"Average Loss: {avg_loss:.5f} @ Epoch: {epoch+1}")
		average_losses.append(avg_loss)
		if avg_loss <= best_loss:
			best_loss = avg_loss
			torch.save(model.state_dict(), mdl_fpth)
			print(f"Saving model in {mdl_fpth} for best avg loss: {best_loss:.5f}")
	print(f"Elapsed_t: {time.time()-training_st:.5f} sec")
	plot_loss(
		losses=average_losses, 
		num_epochs=args.num_epochs, 
		save_path=os.path.join(outputs_dir, f'loss_{args.num_epochs}_nEpochs.png'),
	)

def main():
	set_seeds()
	if not os.path.exists(mdl_fpth):
		fine_tune()
	if args.validate:
		validate(
			model_fpth=mdl_fpth,
			TOP_K=args.topk,
		)
	img_retrieval(
		query=args.query, 
		model_fpth=mdl_fpth, 
		TOP_K=args.topk,
	)

if __name__ == "__main__":
	main()