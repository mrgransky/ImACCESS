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
from typing import List
import matplotlib.pyplot as plt
import torchvision.transforms as T

parser = argparse.ArgumentParser(description="FineTune CLIP for CIFAR10x Dataset")
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--num_workers', '-nw', type=int, default=18, help='Number of CPUs [def: max cpus]')
parser.add_argument('--num_epochs', '-ne', type=int, default=5, help='Number of epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=103, help='Batch size for training')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='small learning rate for better convergence [def: 1e-3]')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3, help='Weight decay [def: 5e-4]')
parser.add_argument('--print_every', type=int, default=150, help='Print loss')

args, unknown = parser.parse_known_args()
args.device = torch.device(args.device)
print(args)

# run in pouta:
# $ nohup python -u finetune.py -bs 256 -ne 25 -lr 1e-4 > /media/volume/ImACCESS/trash/finetune_cifar.out &

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

def evaluate(model, test_loader, criterion, args):
	model.eval()
	total_loss = 0
	total_correct_text_description_for_each_image = 0
	total_correct_image_for_each_text_description = 0
	with torch.no_grad():
		for batch_idx, batch in enumerate(test_loader):
			images, labels = batch
			images = images.to(args.device)
			labels = labels.to(args.device)
			logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
			_, predicted_idxs_imgs = torch.max(input=logits_per_image, dim=1, keepdim=True)
			_, predicted_idxs_txts = torch.max(input=logits_per_text, dim=1, keepdim=True)
			# Get the indices of the correct text descriptions for each image
			correct_text_description_idxs = torch.argmax(labels, dim=1)
			# Compare the predicted indexes with the correct indexes
			total_correct_text_description_for_each_image += (predicted_idxs_imgs == correct_text_description_idxs.unsqueeze(1)).sum().item()
			total_correct_image_for_each_text_description += (predicted_idxs_txts == correct_text_description_idxs.unsqueeze(1)).sum().item()
			# Compute validation loss
			ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=args.device)
			loss_img = criterion(logits_per_image, ground_truth) 
			loss_txt = criterion(logits_per_text, ground_truth)
			valid_loss = 0.5 * (loss_img + loss_txt)
			total_loss += valid_loss.item()
	avg_loss = total_loss / len(test_loader)
	accuracy_text_description_for_each_image = total_correct_text_description_for_each_image / len(test_loader.dataset)
	accuracy_text_image_for_each_text_description = total_correct_image_for_each_text_description / len(test_loader.dataset)
	return avg_loss, accuracy_text_description_for_each_image, accuracy_text_image_for_each_text_description

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

def plot_(train_losses, val_losses, validation_accuracy_text_description_for_each_image_list, validation_accuracy_text_image_for_each_text_description_list):
	num_epochs = len(train_losses)
	if num_epochs == 1:
		return
	epochs = range(1, num_epochs + 1)

	# # Move tensors to CPU and convert to NumPy arrays
	# train_losses = [loss.cpu().item() for loss in train_losses]
	# validation_accuracy_text_description_for_each_image_list = [acc.cpu().item() for acc in validation_accuracy_text_description_for_each_image_list]
	# validation_accuracy_text_image_for_each_text_description_list = [acc.cpu().item() for acc in validation_accuracy_text_image_for_each_text_description_list]

	plt.figure()
	plt.plot(epochs, train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
	plt.plot(epochs, val_losses, marker='o', linestyle='-', color='r', label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(f"train_vs_validation_loss_ep_{len(train_losses)}.png")
	plt.close()

	plt.figure()
	plt.plot(epochs, validation_accuracy_text_description_for_each_image_list, marker='o', linestyle='-', color='b', label='Validation Accuracy [text description for each image]')
	plt.plot(epochs, validation_accuracy_text_image_for_each_text_description_list, marker='o', linestyle='-', color='r', label='Validation Accuracy [image for each text description]')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(f"validation_accuracy_ep_{len(train_losses)}_txt_img.png")
	plt.close()

def finetune(model, train_loader, test_loader, num_epochs=5):
	print(f"Fine-Tuning CLIP model {num_epochs} Epoch(s) device: {args.device} & {args.num_workers} CPU(s)".center(160, "-"))
	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(args.device)}".center(160, " "))

	# # update all parameters in the model during fine-tuning:
	# for param in model.parameters():
	# 	param.requires_grad = True

	# Print the names of all modules
	print("Modules".center(100, "-"))
	for name, module in model.named_modules():
		print(name)
	print("Parameters".center(100, "-"))
	# Print the names of all parameters
	for name, param in model.named_parameters():
		print(name)

	# Freeze the early layers in the vision encoder
	for name, param in model.visual.named_parameters():
		if name in ['conv1.weight', 'conv1.bias', 'layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias', 'layer3.weight', 'layer3.bias']:
			param.requires_grad = False

	# Freeze the early layers in the text encoder
	for name, param in model.text.named_parameters():
		if name in ['embedding.weight', 'encoder.layers[0].weight', 'encoder.layers[0].bias', 'encoder.layers[1].weight', 'encoder.layers[1].bias', 'encoder.layers[2].weight', 'encoder.layers[2].bias', 'encoder.layers[3].weight', 'encoder.layers[3].bias']:
			param.requires_grad = False

	# Update the remaining layers
	for param in model.parameters():
		if param.requires_grad == False:
			continue
		param.requires_grad = True

	optimizer = optim.AdamW(
		params=model.parameters(),
		lr=args.learning_rate,
		betas=(0.9,0.98),
		eps=1e-6,
		weight_decay=args.weight_decay,
	)

	criterion = nn.CrossEntropyLoss()
	training_losses, validation_losses = [], []
	validation_accuracy_text_description_for_each_image_list = []
	validation_accuracy_text_image_for_each_text_description_list = []

	for epoch in range(num_epochs):
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		epoch_loss = 0.0  # To accumulate the loss over the epoch
		for batch_idx, batch in enumerate(train_loader):
			optimizer.zero_grad() # Clear gradients from previous batch
			images, labels = batch # torch.Size([b, 3, 224, 224]), torch.Size([b, 77])
			images, labels = images.to(args.device), labels.to(args.device)
			# logits_per_image: similarity between image embeddings and all text embeddings in batch
			# logits_per_text: similarity between text embeddings and all image embeddings in batch
			logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
			ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=args.device)
			loss_img = criterion(logits_per_image, ground_truth) 
			loss_txt = criterion(logits_per_text, ground_truth)
			total_loss = 0.5 * (loss_img + loss_txt)
			# print(loss_img.item(), loss_txt.item(), total_loss.item())
			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
			optimizer.step() # Update weights
			if batch_idx%args.print_every==0 or batch_idx+1==len(train_loader):
				print(
					f"\tBatch [{batch_idx+1}/{len(train_loader)}] "
					f"Loss: {total_loss.item():.7f}",
				)
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		print(f"Average Training Loss: {avg_training_loss:.5f} @ Epoch: {epoch+1}")
		training_losses.append(avg_training_loss)

		avg_valid_loss, accuracy_text_description_for_each_image, accuracy_text_image_for_each_text_description = evaluate(model, test_loader, criterion, args)
		validation_losses.append(avg_valid_loss)
		validation_accuracy_text_description_for_each_image_list.append(accuracy_text_description_for_each_image)
		validation_accuracy_text_image_for_each_text_description_list.append(accuracy_text_image_for_each_text_description)

		print(
			f'Training Loss: {avg_training_loss:.4f} '
			f'Validation Loss: {avg_valid_loss:.4f} '
			f'Validation Accuracy [text description for each image]: {accuracy_text_description_for_each_image:.4f} '
			f'[image for each text description]: {accuracy_text_image_for_each_text_description:.4f}'
		)

	plot_(
		train_losses=training_losses,
		val_losses=validation_losses,
		validation_accuracy_text_description_for_each_image_list=validation_accuracy_text_description_for_each_image_list,
		validation_accuracy_text_image_for_each_text_description_list=validation_accuracy_text_image_for_each_text_description_list,
	)

def main():
	print(clip.available_models())
	model, preprocess = load_model()
	train_dataset, test_dataset = get_dataset() # cifar10 or cifar 100
	train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, preprocess=preprocess, batch_size=args.batch_size, num_workers=args.num_workers)
	finetune(model, train_loader, test_loader, num_epochs=args.num_epochs)

if __name__ == "__main__":
	main()