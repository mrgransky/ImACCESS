from backend import run_backend
import datetime
import time
import os
import json
import random
import matplotlib.pyplot as plt
from datasets_loader import CINIC10, ImageNet_1K
import seaborn as sns
from collections import Counter
import math
import numpy as np
# from torchvision.datasets import ImageNet
# train_dataset = ImageNet(
# 	root=os.path.expanduser("~/.cache"),
# 	split='train',
# 	download=True,
# 	transform=None
# )
# test_dataset = ImageNet(
# 	root=os.path.expanduser("~/.cache"),
# 	split='val',
# 	download=True,
# 	transform=None
# )

# cinic10_dataset_train = CINIC10(root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/CINIC10', train=True)
# random_indices = random.sample(range(len(cinic10_dataset_train)), 10)
# print(len(random_indices), type(random_indices), random_indices)
# for i in random_indices:
# 	image, label = cinic10_dataset_train[i]
# 	print(f"Image path: {cinic10_dataset_train.data[i][0]}")
# 	print(f"Label: {label} ({cinic10_dataset_train.classes[label]})")
# 	print("-" * 50)
# fig, axs = plt.subplots(2, 5, figsize=(20, 8))
# for i, idx in enumerate(random_indices):
# 	image, label = cinic10_dataset_train[idx]
# 	axs[i // 5, i % 5].imshow(image)
# 	axs[i // 5, i % 5].set_title(cinic10_dataset_train.classes[label])
# 	axs[i // 5, i % 5].axis('off')
# plt.title(f"CINIC10 Dataset {len(random_indices)} Random Images")
# plt.tight_layout()
# plt.show()

imgnet = ImageNet_1K(
	root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/IMAGENET', 
	train=True,
)
print(imgnet)
random_indices = random.sample(range(len(imgnet)), 10)
print(random_indices)
print(len(random_indices), type(random_indices), random_indices)
for i in random_indices:
	image, label_idx = imgnet[i]
	# print(f"Image path: {imgnet.data[i][0]}")
	print(f"Image path: {imgnet.data[i]}")
	print(f"[Label] index: {label_idx} SynsetID: {imgnet.synset_ids[label_idx]} ({imgnet.classes[label_idx]})")
	print("-" * 50)
fig, axs = plt.subplots(2, 5, figsize=(22, 8))

for i, idx in enumerate(random_indices):
	image, label_idx = imgnet[idx]
	axs[i // 5, i % 5].imshow(image)
	axs[i // 5, i % 5].set_title(f"{imgnet.synset_ids[label_idx]} {imgnet.classes[label_idx]}", fontsize=8)
	axs[i // 5, i % 5].axis('off')
plt.suptitle(f"IMAGENET {len(random_indices)} Random Images")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10, CIFAR100

def plot_class_distribution(dataset, title):
	"""
	Plots the distribution of classes in a dataset.
	Args:
			dataset: The dataset to analyze.
			title: The title for the plot.
	"""
	labels = dataset.targets if hasattr(dataset, 'targets') else [label for _, label in dataset]
	class_counts = Counter(labels)
	classes = dataset.classes if hasattr(dataset, 'classes') else list(class_counts.keys())
	# Sort by count directly
	sorted_class_counts = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
	sorted_classes, sorted_counts = zip(*sorted_class_counts)

	# Determine optimal figure size
	num_classes = len(sorted_classes)
	aspect_ratio = max(1, num_classes / 20)  # Adjust aspect ratio based on number of classes
	plt.figure(figsize=(14, 9))

	sns.barplot(x=range(len(sorted_classes)), y=sorted_counts)
	plt.title(title, pad=20, fontsize=14)
	plt.xlabel('Labels', fontsize=12)
	plt.ylabel('Sample Count', fontsize=12)
	# Dynamically set x-ticks
	step = max(1, num_classes // 20)  # Show up to 50 ticks
	plt.xticks(
		ticks=np.arange(0, num_classes, step),
		labels=sorted_classes[::step],
		rotation=90,
		fontsize=8,
	)
	plt.tight_layout()
	plot_file = f"{title.replace(' ', '_')}.png"
	plt.savefig(plot_file)
	plt.close()
	print(f"Saved plot to {plot_file}")

# Load datasets
cifar10_train = CIFAR10(root='~/.cache', train=True, download=True)
cifar10_test = CIFAR10(root='~/.cache', train=False, download=True)
cifar100_train = CIFAR100(root='~/.cache', train=True, download=True)
cifar100_test = CIFAR100(root='~/.cache', train=False, download=True)

# Plot class distributions
plot_class_distribution(cifar10_train, '[Original] CIFAR10 Train Label Distribution')
plot_class_distribution(cifar10_test, '[Original] CIFAR10 Test Label Distribution')
plot_class_distribution(cifar100_train, '[Original] CIFAR100 Train Label Distribution')
plot_class_distribution(cifar100_test, '[Original] CIFAR100 Test Label Distribution')

imgnet_train = ImageNet_1K(
	root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/IMAGENET', 
	train=True,
)
print(imgnet_train)

imgnet_val = ImageNet_1K(
	root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/IMAGENET', 
	train=False,
)
print(imgnet_val)

# # Plot class distributions
plot_class_distribution(imgnet_val, '[Original] ImageNet-1K Val Label Distribution')
plot_class_distribution(imgnet_train, '[Original] ImageNet-1K Train Label Distribution')

# def main():
#   run_backend()
#   return

# if __name__ == "__main__":
# 	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
# 	START_EXECUTION_TIME = time.time()
# 	main()
# 	END_EXECUTION_TIME = time.time()
# 	print(
# 		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
# 		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
# 		.center(160, " ")
# 	)