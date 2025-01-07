import os
import torch
import clip
import datetime
import time
import json
import re
import argparse
import random
import numpy as np
from PIL import Image
from typing import Tuple, Union, List

def visualize_(dataloader, num_samples=5, ):
	for batch_idx, (batch_imgs, batch_lbls) in enumerate(dataloader):
		print(batch_idx, batch_imgs.shape, batch_lbls.shape, len(batch_imgs), len(batch_lbls)) # torch.Size([32, 3, 224, 224]) torch.Size([32])
		if batch_idx >= num_samples:
			break
		
		image = batch_imgs[batch_idx].permute(1, 2, 0).numpy() # Convert tensor to numpy array and permute dimensions
		caption_idx = batch_lbls[batch_idx]
		print(image.shape, caption_idx)
		print()
			
		# # Denormalize the image
		image = image * np.array([0.2268645167350769]) + np.array([0.6929051876068115])
		image = np.clip(image, 0, 1)  # Ensure pixel values are in [0, 1] range
		
		plt.figure(figsize=(10, 10))
		plt.imshow(image)
		plt.title(f"Caption {caption_idx.shape}\n{caption_idx}", fontsize=5)
		plt.axis('off')
		plt.show()

def set_seeds(seed:int=42, debug:bool=False):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		if debug: # slows down training but ensures reproducibility
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False