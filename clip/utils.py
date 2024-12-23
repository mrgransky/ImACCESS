import os
import torch
import clip
import time
import re
import argparse
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

def plot_loss_accuracy(
		train_losses,
		val_losses,
		validation_accuracy_text_description_for_each_image_list,
		validation_accuracy_text_image_for_each_text_description_list,
		dataset_name:str="CIFAR10",
		learning_rate:float=1e-5,
		weight_decay:float=1e-3,
		batch_size:int=64,
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
	plt.savefig(f"{dataset_name}_train_val_loss_ep_{len(train_losses)}_lr_{learning_rate}_wd_{weight_decay}_{batch_size}_batch_size.png")
	plt.close()

	plt.figure()
	plt.plot(epochs, validation_accuracy_text_description_for_each_image_list, marker='o', linestyle='-', color='b', label='Validation Accuracy [text description for each image]')
	plt.plot(epochs, validation_accuracy_text_image_for_each_text_description_list, marker='o', linestyle='-', color='r', label='Validation Accuracy [image for each text description]')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.tight_layout()
	plt.legend()
	plt.savefig(f"{dataset_name}_validation_accuracy_ep_{len(train_losses)}_txt_img.png")
	plt.close()


def load_model(model_name:str="ViT-B/32", device:str="cuda", jit:bool=False):
	model, preprocess = clip.load(model_name, device=device, jit=jit) # training or finetuning => jit=False
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