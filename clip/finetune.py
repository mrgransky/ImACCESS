import os
import torch
import clip
import time
import re
import argparse
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from typing import List
import matplotlib.pyplot as plt
import torchvision.transforms as T

parser = argparse.ArgumentParser(description="FineTune CLIP for CIFAR10x Dataset")
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--num_workers', '-nw', type=int, default=10, help='Number of CPUs [def: max cpus]')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=50, help='Batch size for training')
# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)

args.device = torch.device(args.device)

# $ nohup python -u finetune.py > /media/volume/ImACCESS/trash/prec_at_K.out &

USER = os.getenv('USER')
print(f"USER: {USER} device: {args.device}")

class CIFARDATASET(torch.utils.data.Dataset):
		def __init__(self, dataset, transformer=None,):
				self.dataset = dataset
				self.images = [img for idx, (img,lbl) in enumerate(dataset)]
				self.labels = clip.tokenize(texts=[dataset.classes[lbl_idx] for i, (img, lbl_idx) in enumerate(dataset)])
				if transformer:
					self.transform = transformer
				else:
					self.transform = T.Compose(
					[
						T.ToTensor(),
						T.Normalize(
							(0.48145466, 0.4578275, 0.40821073), 
							(0.26862954, 0.26130258, 0.27577711)
						)
					]
				)

		def __getitem__(self, index):
				image = self.images[index]
				text = self.labels[index]
				return self.transform(image), text

		def __len__(self):
				return len(self.dataset)

def get_dataloaders(train_dataset, test_dataset, preprocess, batch_size=32, num_workers=10):
		train_dataset = CIFARDATASET(train_dataset, transformer=preprocess)
		test_dataset = CIFARDATASET(test_dataset, transformer=preprocess)

		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
		return train_loader, test_loader

def load_model():
	model, preprocess = clip.load("ViT-B/32", device=args.device, jit=False) # training or finetuning => jit=False
	input_resolution = model.visual.input_resolution
	context_length = model.context_length
	vocab_size = model.vocab_size
	print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
	print("Input resolution:", input_resolution)
	print("Context length:", context_length)
	print("Vocab size:", vocab_size)
	return model, preprocess

def get_dataset(large_dataset:bool=False):
		if large_dataset:
				train_dataset = CIFAR100(
						root=os.path.expanduser("~/.cache"), 
						train=True,
						download=True,
						transform=None
				)
				test_dataset = CIFAR100(
						root=os.path.expanduser("~/.cache"), 
						train=False,
						download=True,
						transform=None
				)
		else: # cifar10 with 10K samples
				train_dataset = CIFAR10(
						root=os.path.expanduser("~/.cache"), 
						train=True,
						download=True,
						transform=None
				)
				test_dataset = CIFAR10(
						root=os.path.expanduser("~/.cache"), 
						train=False,
						download=True,
						transform=None
				)
		print(train_dataset)
		print(test_dataset)
		return train_dataset, test_dataset

def finetune(model, train_loader, test_loader, num_epochs=5):
		print(f"Fine-Tuning CLIP model {num_epochs} Epoch(s) device: {args.device} & {args.num_workers} CPU(s)".center(160, "-"))
		if torch.cuda.is_available():
				print(f"{torch.cuda.get_device_name(args.device)}".center(160, " "))

		# Unfreeze all layers
		for param in model.parameters():
			param.requires_grad = True
		
		# Define the optimizer and the loss function
		optimizer = optim.Adam(model.parameters(), lr=1e-5)
		criterion_img, criterion_txt = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()

		for epoch in range(num_epochs):
				model.train()
				total_loss = 0
				for batch in train_loader:
						images, labels = batch # torch.Size([b, 3, 224, 224]), torch.Size([b, 77])
						images, labels = images.to(args.device), labels.to(args.device)
						optimizer.zero_grad() # Zero the parameter gradients
						logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
						print(logits_per_image.shape, logits_per_text.shape)


				print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

				# Evaluate the model on the test set
				model.eval()
				total_correct = 0
				with torch.no_grad():
						for batch in test_loader:
								images, labels = batch
								images = images.to(args.device)
								labels = labels.to(args.device)
								outputs = model(images, labels)
								_, predicted = torch.max(outputs.logits, dim=1)
								total_correct += (predicted == labels).sum().item()
				accuracy = total_correct / len(test_loader.dataset)
				print(f'Test Accuracy: {accuracy:.4f}')

def main():
	print(clip.available_models())
	model, preprocess = load_model()
	train_dataset, test_dataset = get_dataset() # cifar10 or cifar 100
	train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, preprocess=preprocess, batch_size=args.batch_size, num_workers=args.num_workers)
	finetune(model, train_loader, test_loader, num_epochs=args.num_epochs)

if __name__ == "__main__":
	main()