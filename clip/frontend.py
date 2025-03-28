# from backend import run_backend
# import datetime
# import time
# import os
# import json
# import random
# from datasets_loader import CINIC10, ImageNet_1K
# import seaborn as sns
# from collections import Counter
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from torchvision.datasets import CIFAR10, CIFAR100
# from matplotlib.lines import Line2D

# # # from torchvision.datasets import ImageNet
# # # train_dataset = ImageNet(
# # # 	root=os.path.expanduser("~/.cache"),
# # # 	split='train',
# # # 	download=True,
# # # 	transform=None
# # # )
# # # test_dataset = ImageNet(
# # # 	root=os.path.expanduser("~/.cache"),
# # # 	split='val',
# # # 	download=True,
# # # 	transform=None
# # # )

# # # cinic10_dataset_train = CINIC10(root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/CINIC10', train=True)
# # # random_indices = random.sample(range(len(cinic10_dataset_train)), 10)
# # # print(len(random_indices), type(random_indices), random_indices)
# # # for i in random_indices:
# # # 	image, label = cinic10_dataset_train[i]
# # # 	print(f"Image path: {cinic10_dataset_train.data[i][0]}")
# # # 	print(f"Label: {label} ({cinic10_dataset_train.classes[label]})")
# # # 	print("-" * 50)
# # # fig, axs = plt.subplots(2, 5, figsize=(20, 8))
# # # for i, idx in enumerate(random_indices):
# # # 	image, label = cinic10_dataset_train[idx]
# # # 	axs[i // 5, i % 5].imshow(image)
# # # 	axs[i // 5, i % 5].set_title(cinic10_dataset_train.classes[label])
# # # 	axs[i // 5, i % 5].axis('off')
# # # plt.title(f"CINIC10 Dataset {len(random_indices)} Random Images")
# # # plt.tight_layout()
# # # plt.show()

# # imgnet = ImageNet_1K(
# # 	root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/IMAGENET', 
# # 	train=True,
# # )
# # print(imgnet)
# # random_indices = random.sample(range(len(imgnet)), 10)
# # print(random_indices)
# # print(len(random_indices), type(random_indices), random_indices)
# # for i in random_indices:
# # 	image, label_idx = imgnet[i]
# # 	# print(f"Image path: {imgnet.data[i][0]}")
# # 	print(f"Image path: {imgnet.data[i]}")
# # 	print(f"[Label] index: {label_idx} SynsetID: {imgnet.synset_ids[label_idx]} ({imgnet.classes[label_idx]})")
# # 	print("-" * 50)
# # fig, axs = plt.subplots(2, 5, figsize=(22, 8))

# # for i, idx in enumerate(random_indices):
# # 	image, label_idx = imgnet[idx]
# # 	axs[i // 5, i % 5].imshow(image)
# # 	axs[i // 5, i % 5].set_title(f"{imgnet.synset_ids[label_idx]} {imgnet.classes[label_idx]}", fontsize=8)
# # 	axs[i // 5, i % 5].axis('off')
# # plt.suptitle(f"IMAGENET {len(random_indices)} Random Images")
# # plt.tight_layout()
# # plt.show()

# # def plot_class_distribution(dataset, title):
# # 	"""
# # 	Plots the distribution of classes in a dataset.
# # 	Args:
# # 			dataset: The dataset to analyze.
# # 			title: The title for the plot.
# # 	"""
# # 	labels = dataset.targets if hasattr(dataset, 'targets') else [label for _, label in dataset]
# # 	class_counts = Counter(labels)
# # 	classes = dataset.classes if hasattr(dataset, 'classes') else list(class_counts.keys())
# # 	# Sort by count directly
# # 	sorted_class_counts = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
# # 	sorted_classes, sorted_counts = zip(*sorted_class_counts)

# # 	# Determine optimal figure size
# # 	num_classes = len(sorted_classes)
# # 	aspect_ratio = max(1, num_classes / 20)  # Adjust aspect ratio based on number of classes
# # 	plt.figure(figsize=(14, 9))

# # 	sns.barplot(x=range(len(sorted_classes)), y=sorted_counts)
# # 	plt.title(title, pad=20, fontsize=14)
# # 	plt.xlabel('Labels', fontsize=12)
# # 	plt.ylabel('Sample Count', fontsize=12)
# # 	# Dynamically set x-ticks
# # 	step = max(1, num_classes // 20)  # Show up to 50 ticks
# # 	plt.xticks(
# # 		ticks=np.arange(0, num_classes, step),
# # 		labels=sorted_classes[::step],
# # 		rotation=90,
# # 		fontsize=8,
# # 	)
# # 	plt.tight_layout()
# # 	plot_file = f"{title.replace(' ', '_')}.png"
# # 	plt.savefig(plot_file)
# # 	plt.close()
# # 	print(f"Saved plot to {plot_file}")

# # # Load datasets
# # cifar10_train = CIFAR10(root='~/.cache', train=True, download=True)
# # cifar10_test = CIFAR10(root='~/.cache', train=False, download=True)
# # cifar100_train = CIFAR100(root='~/.cache', train=True, download=True)
# # cifar100_test = CIFAR100(root='~/.cache', train=False, download=True)

# # # Plot class distributions
# # plot_class_distribution(cifar10_train, '[Original] CIFAR10 Train Label Distribution')
# # plot_class_distribution(cifar10_test, '[Original] CIFAR10 Test Label Distribution')
# # plot_class_distribution(cifar100_train, '[Original] CIFAR100 Train Label Distribution')
# # plot_class_distribution(cifar100_test, '[Original] CIFAR100 Test Label Distribution')

# # imgnet_train = ImageNet_1K(
# # 	root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/IMAGENET', 
# # 	train=True,
# # )
# # print(imgnet_train)

# # imgnet_val = ImageNet_1K(
# # 	root='/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/IMAGENET', 
# # 	train=False,
# # )
# # print(imgnet_val)

# # # # Plot class distributions
# # plot_class_distribution(imgnet_val, '[Original] ImageNet-1K Val Label Distribution')
# # plot_class_distribution(imgnet_train, '[Original] ImageNet-1K Train Label Distribution')

# # # def main():
# # #   run_backend()
# # #   return

# # # if __name__ == "__main__":
# # # 	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
# # # 	START_EXECUTION_TIME = time.time()
# # # 	main()
# # # 	END_EXECUTION_TIME = time.time()
# # # 	print(
# # # 		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
# # # 		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
# # # 		.center(160, " ")
# # # 	)

# # fig, ax = plt.subplots(figsize=(10, 8))
# # ax.set_title("CLIP Fine-Tuning Strategies for Retrieval Tasks", fontsize=14)
# # ax.set_xlabel("Similarity to Pre-trained CLIP Data (Different to Similar)")
# # ax.set_ylabel("Dataset Size (Small: <1000, Large: >10000)")
# # ax.axhline(y=0.5, color='k', linestyle='-', linewidth=1)
# # ax.axvline(x=0.5, color='k', linestyle='-', linewidth=1)

# # # Quadrant labels and strategies
# # quadrants = [
# # 		("Large, Different", (0.25, 0.75), {"Text": True, "Image": True}),
# # 		("Large, Similar", (0.75, 0.75), {"Text": True, "Image": False}),
# # 		("Small, Different", (0.25, 0.25), {"Text": False, "Image": True}),
# # 		("Small, Similar", (0.75, 0.25), {"Text": True, "Image": False}),
# # ]
# # for (label, pos, strategy) in quadrants:
# # 		ax.text(pos[0], pos[1], label, ha='center', va='center')
# # 		# Add encoder blocks (simplified)
# # 		if strategy["Text"]:
# # 				ax.add_patch(patches.Rectangle((pos[0]-0.2, pos[1]+0.1), 0.1, 0.2, fc='blue'))
# # 				ax.text(pos[0]-0.15, pos[1]+0.2, "Text Encoder", ha='left')
# # 		else:
# # 				ax.add_patch(patches.Rectangle((pos[0]-0.2, pos[1]+0.1), 0.1, 0.2, fc='gray'))
# # 				ax.text(pos[0]-0.15, pos[1]+0.2, "Text Encoder", ha='left')
# # 		if strategy["Image"]:
# # 				ax.add_patch(patches.Rectangle((pos[0]+0.1, pos[1]+0.1), 0.1, 0.2, fc='blue'))
# # 				ax.text(pos[0]+0.15, pos[1]+0.2, "Image Encoder", ha='right')
# # 		else:
# # 				ax.add_patch(patches.Rectangle((pos[0]+0.1, pos[1]+0.1), 0.1, 0.2, fc='gray'))
# # 				ax.text(pos[0]+0.15, pos[1]+0.2, "Image Encoder", ha='right')

# # # Legend
# # legend_elements = [
# # 	Line2D([0], [0], marker='s', color='w', label='Freeze', markerfacecolor='gray', markersize=15),
# # 	Line2D([0], [0], marker='s', color='w', label='Fine-tune', markerfacecolor='blue', markersize=15)
# # ]
# # ax.legend(handles=legend_elements, loc='best', fontsize=10)

# # plt.savefig("clip_finetuning_strategies.png", dpi=300, bbox_inches='tight')
# # plt.close()

# plt.rcParams['axes.linewidth'] = 1.2

# fig, ax = plt.subplots(figsize=(14, 10))
# fig.patch.set_facecolor('white')

# # Set axis limits and remove default ticks
# ax.set_xlim(-0.1, 1.1)
# ax.set_ylim(-0.1, 1.1)
# ax.set_xticks([])
# ax.set_yticks([])

# # Add quadrant dividers with better styling
# ax.axhline(y=0.5, color='#2C3E50', linestyle='--', linewidth=1.5)
# ax.axvline(x=0.5, color='#2C3E50', linestyle='--', linewidth=1.5)

# # Quadrant definitions with improved formatting
# quadrants = [
# 		("Large Dataset, Different Domain", (0.1, 0.75), {"Text": True, "Image": True}),
# 		("Large Dataset, Similar Domain", (0.75, 0.75), {"Text": True, "Image": False}),
# 		("Small Dataset, Different Domain", (0.1, 0.25), {"Text": False, "Image": True}),
# 		("Small Dataset, Similar Domain", (0.75, 0.25), {"Text": True, "Image": False}),
# ]

# # Improved drawing of components
# for label, (x, y), strategy in quadrants:
# 		# Quadrant label with bold styling
# 		ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', color='#2C3E50')
		
# 		# Positioning encoder blocks
# 		block_y = y - 0.18  # Space below label
		
# 		# Text Encoder block
# 		tx_x = x - 0.12
# 		ax.add_patch(
# 			patches.Rectangle(
# 				(tx_x, block_y), 0.22, 0.12,
# 				facecolor='#3498DB' if strategy["Text"] else '#BDC3C7',
# 				edgecolor='#2C3E50', linewidth=1,
# 			)
# 		)
# 		ax.text(tx_x + 0.11, block_y + 0.06, "Text Encoder", ha='center', va='center', fontsize=9, color='white')
		
# 		# Image Encoder block
# 		img_x = x + 0.12
# 		ax.add_patch(
# 			patches.Rectangle(
# 				(img_x, block_y), 0.22, 0.12,
# 				facecolor='#3498DB' if strategy["Image"] else '#BDC3C7',
# 				edgecolor='#2C3E50', linewidth=1,
# 			)
# 		)
# 		ax.text(img_x + 0.11, block_y + 0.06, "Image Encoder", ha='center', va='center', fontsize=9, color='white')

# # Improved axis labels and title
# ax.set_title("CLIP Fine-Tuning Strategies for Retrieval Tasks", fontsize=16, pad=20, fontweight='bold', color='#2C3E50')
# ax.set_xlabel("Similarity to Pre-trained Data", fontsize=12, labelpad=15, color='#2C3E50')
# ax.set_ylabel("Dataset Size", fontsize=12, labelpad=20, color='#2C3E50')

# # Custom legend with improved styling
# legend_elements = [
# 	patches.Patch(facecolor='#3498DB', edgecolor='#2C3E50', label='Fine-Tuned'),
# 	patches.Patch(facecolor='#BDC3C7', edgecolor='#2C3E50', label='Frozen'),
# ]
# ax.legend(
# 	handles=legend_elements, 
# 	loc='upper center', bbox_to_anchor=(0.5, -0.15),
# 	ncol=2, frameon=False, fontsize=10,
# )

# # Add axis annotations
# ax.annotate('Different', xy=(0, 0.5), xycoords='axes fraction',
# 					 xytext=(-100, 0), textcoords='offset points',
# 					 ha='right', va='center', rotation=90, fontsize=10,
# 					 color='#2C3E50')
# ax.annotate('Similar', xy=(1, 0.5), xycoords='axes fraction',
# 					 xytext=(100, 0), textcoords='offset points',
# 					 ha='left', va='center', rotation=270, fontsize=10,
# 					 color='#2C3E50')

# ax.annotate('Small', xy=(0.5, 0), xycoords='axes fraction',
# 					 xytext=(0, -50), textcoords='offset points',
# 					 ha='center', va='top', fontsize=10, color='#2C3E50')
# ax.annotate('Large', xy=(0.5, 1), xycoords='axes fraction',
# 					 xytext=(0, 40), textcoords='offset points',
# 					 ha='center', va='bottom', fontsize=10, color='#2C3E50')

# # Remove unnecessary spines
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

# plt.tight_layout()
# plt.savefig("clip_finetuning_strategies_v2.png", 
# 						dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
# plt.close()

import matplotlib.pyplot as plt 
import numpy as np
def plot_label_distribution_pie_chart(labels=None, counts=None):
	if labels is None or counts is None:
		# Default data for demonstration
		# Sample data: skewed distribution for a historical wartime dataset
		labels = ["Aircraft", "Infantry", "Tanks", "Ships", "Military Bases", "Artillery", "Border"]
		counts = [np.random.randint(1, int(5e2)) if i == 0 else np.random.randint(1, 10) for i in range(len(labels))]  # Highly imbalanced (long-tailed) 

	# colors = plt.cm.tab20c.colors
	# colors = plt.cm.tab20c.colors
	colors = plt.cm.cividis(np.linspace(0, 0.98, len(labels)))
	# Set up the figure and axis. Adjust the size for a research paper quality figure.
	fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True, facecolor='white', edgecolor='black')

	# Create the pie chart with enhanced aesthetics:
	wedges, texts, autotexts = ax.pie(
			counts,
			labels=labels,
			autopct='',
			startangle=0,              # Rotate so that the first slice starts at 90 degrees
			colors=colors,
			textprops={'fontsize': 12, 'color': 'black'},
			wedgeprops={'edgecolor': 'white', 'linewidth': 2}  # Crisp white borders between slices
	)

	# Add a title that suits a research paper style
	# ax.set_title("Long-tailed Label Distribution", fontsize=18, weight='bold', pad=20)
	ax.axis('equal')  # Ensure the pie chart is a circle

	# Improve layout and display the chart
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	plot_label_distribution_pie_chart()