import os
import torch
import clip
import time
import re
import argparse
import random
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
import warnings
import datetime
warnings.filterwarnings("ignore", category=UserWarning, module='torch.optim.lr_scheduler')

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

def plot_loss_accuracy(
		train_losses,
		val_losses,
		validation_accuracy_text_description_for_each_image_list,
		validation_accuracy_text_image_for_each_text_description_list,
		losses_file_path: str="losses.png",
		accuracy_file_path: str="accuracy.png",
	):
	num_epochs = len(train_losses)
	if num_epochs == 1:
		return
	epochs = range(1, num_epochs + 1)

	plt.figure()
	plt.plot(epochs, train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
	plt.plot(epochs, val_losses, marker='o', linestyle='-', color='r', label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.tight_layout()
	plt.legend()
	plt.savefig(losses_file_path)
	plt.close()

	plt.figure()
	plt.plot(epochs, validation_accuracy_text_description_for_each_image_list, marker='o', linestyle='-', color='b', label='Validation Accuracy [text description for each image]')
	plt.plot(epochs, validation_accuracy_text_image_for_each_text_description_list, marker='o', linestyle='-', color='r', label='Validation Accuracy [image for each text description]')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.tight_layout()
	plt.legend()
	plt.savefig(accuracy_file_path)
	plt.close()


def load_model(model_name:str="ViT-B/32", device:str="cuda", jit:bool=False):
	model, preprocess = clip.load(model_name, device=device, jit=jit) # training or finetuning => jit=False
	model = model.float() # Convert model parameters to FP32
	input_resolution = model.visual.input_resolution
	context_length = model.context_length
	vocab_size = model.vocab_size
	print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
	print("Input resolution:", input_resolution)
	print("Context length:", context_length)
	print("Vocab size:", vocab_size)
	return model, preprocess

def get_dataset(dname:str="CIFAR10"):
	if dname == 'CIFAR100':
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
	elif dname == 'CIFAR10':
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
	else:
		raise ValueError(f"Invalid dataset name: {dname}. Choose from CIFAR10 or CIFAR100")
	print(train_dataset)
	print(test_dataset)
	return train_dataset, test_dataset