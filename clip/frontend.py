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

# import matplotlib.pyplot as plt 
# import numpy as np
# def plot_label_distribution_pie_chart(labels=None, counts=None):
# 	if labels is None or counts is None:
# 		# Default data for demonstration
# 		# Sample data: skewed distribution for a historical wartime dataset
# 		labels = ["Aircraft", "Infantry", "Tanks", "Ships", "Military Bases", "Artillery", "Border"]
# 		counts = [np.random.randint(1, int(5e2)) if i == 0 else np.random.randint(1, 10) for i in range(len(labels))]  # Highly imbalanced (long-tailed) 

# 	# colors = plt.cm.tab20c.colors
# 	# colors = plt.cm.tab20c.colors
# 	colors = plt.cm.cividis(np.linspace(0, 0.98, len(labels)))
# 	# Set up the figure and axis. Adjust the size for a research paper quality figure.
# 	fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True, facecolor='white', edgecolor='black')

# 	# Create the pie chart with enhanced aesthetics:
# 	wedges, texts, autotexts = ax.pie(
# 			counts,
# 			labels=labels,
# 			autopct='',
# 			startangle=0,              # Rotate so that the first slice starts at 90 degrees
# 			colors=colors,
# 			textprops={'fontsize': 12, 'color': 'black'},
# 			wedgeprops={'edgecolor': 'white', 'linewidth': 2}  # Crisp white borders between slices
# 	)

# 	# Add a title that suits a research paper style
# 	# ax.set_title("Long-tailed Label Distribution", fontsize=18, weight='bold', pad=20)
# 	ax.axis('equal')  # Ensure the pie chart is a circle

# 	# Improve layout and display the chart
# 	plt.tight_layout()
# 	plt.show()


# if __name__ == "__main__":
# 	plot_label_distribution_pie_chart()

from visualize import *

# import numpy as np

# # Define K values
# topK_values = [1, 3, 5, 10, 15, 20]

# # Helper function to generate synthetic metric values
# def generate_metric_values(base_value, trend='decrease', noise_level=0.05, k_values=topK_values):
#     values = []
#     for k in k_values:
#         # Base trend: decrease, increase, or stable
#         if trend == 'decrease':
#             val = base_value * (1 - 0.1 * (k / max(k_values)))  # Decrease with K
#         elif trend == 'increase':
#             val = base_value * (1 + 0.1 * (k / max(k_values)))  # Increase with K
#         else:
#             val = base_value  # Stable
#         # Add some noise
#         noise = np.random.uniform(-noise_level, noise_level)
#         val = max(0, min(1, val + noise))  # Ensure value is between 0 and 1
#         values.append(val)
#     return {str(k): val for k, val in zip(k_values, values)}

# # Pre-trained CLIP ViT-B/32 (Image-to-Text)
# pretrained_img2txt_dict = {
#     'ViT-B/32': {
#         'mP': generate_metric_values(0.8, trend='decrease', noise_level=0.05),
#         'mAP': generate_metric_values(0.6, trend='stable', noise_level=0.03),
#         'Recall': generate_metric_values(0.4, trend='increase', noise_level=0.04),
#     }
# }

# # Pre-trained CLIP ViT-B/32 (Text-to-Image)
# pretrained_txt2img_dict = {
#     'ViT-B/32': {
#         'mP': generate_metric_values(0.7, trend='decrease', noise_level=0.05),
#         'mAP': generate_metric_values(0.5, trend='stable', noise_level=0.03),
#         'Recall': generate_metric_values(0.3, trend='increase', noise_level=0.04),
#     }
# }

# # Fine-tuned CLIP ViT-B/32 (Image-to-Text)
# finetuned_img2txt_dict = {
#     'ViT-B/32': {
#         'full': {
#             'mP': generate_metric_values(0.85, trend='decrease', noise_level=0.05),
#             'mAP': generate_metric_values(0.65, trend='stable', noise_level=0.03),
#             'Recall': generate_metric_values(0.45, trend='increase', noise_level=0.04),
#         },
#         'lora': {
#             'mP': generate_metric_values(0.75, trend='decrease', noise_level=0.05),
#             'mAP': generate_metric_values(0.55, trend='stable', noise_level=0.03),
#             'Recall': generate_metric_values(0.35, trend='increase', noise_level=0.04),
#         },
#         'progressive': {
#             'mP': generate_metric_values(0.9, trend='decrease', noise_level=0.05),
#             'mAP': generate_metric_values(0.7, trend='stable', noise_level=0.03),
#             'Recall': generate_metric_values(0.5, trend='increase', noise_level=0.04),
#         },
#     }
# }

# # Fine-tuned CLIP ViT-B/32 (Text-to-Image)
# finetuned_txt2img_dict = {
#     'ViT-B/32': {
#         'full': {
#             'mP': generate_metric_values(0.75, trend='decrease', noise_level=0.05),
#             'mAP': generate_metric_values(0.55, trend='stable', noise_level=0.03),
#             'Recall': generate_metric_values(0.35, trend='increase', noise_level=0.04),
#         },
#         'lora': {
#             'mP': generate_metric_values(0.65, trend='decrease', noise_level=0.05),
#             'mAP': generate_metric_values(0.45, trend='stable', noise_level=0.03),
#             'Recall': generate_metric_values(0.25, trend='increase', noise_level=0.04),
#         },
#         'progressive': {
#             'mP': generate_metric_values(0.8, trend='decrease', noise_level=0.05),
#             'mAP': generate_metric_values(0.6, trend='stable', noise_level=0.03),
#             'Recall': generate_metric_values(0.4, trend='increase', noise_level=0.04),
#         },
#     }
# }
# # Define parameters
# dataset_name = "DummyDataset"
# model_name = "ViT-B/32"
# finetune_strategies = ["full", "lora", "progressive"]
# results_dir = "./dummy_plots"
# os.makedirs(results_dir, exist_ok=True)

# # Call plot_comparison_metrics_split()
# plot_comparison_metrics_split(
#     dataset_name=dataset_name,
#     pretrained_img2txt_dict=pretrained_img2txt_dict,
#     pretrained_txt2img_dict=pretrained_txt2img_dict,
#     finetuned_img2txt_dict=finetuned_img2txt_dict,
#     finetuned_txt2img_dict=finetuned_txt2img_dict,
#     model_name=model_name,
#     finetune_strategies=finetune_strategies,
#     results_dir=results_dir,
#     topK_values=topK_values,
#     figure_size=(7, 7),
#     DPI=300,
# )

# # Call plot_comparison_metrics_merged()
# plot_comparison_metrics_merged(
#     dataset_name=dataset_name,
#     pretrained_img2txt_dict=pretrained_img2txt_dict,
#     pretrained_txt2img_dict=pretrained_txt2img_dict,
#     finetuned_img2txt_dict=finetuned_img2txt_dict,
#     finetuned_txt2img_dict=finetuned_txt2img_dict,
#     model_name=model_name,
#     finetune_strategies=finetune_strategies,
#     results_dir=results_dir,
#     topK_values=topK_values,
#     figure_size=(14, 5),
#     DPI=300,
# )
# from utils import select_qualitative_samples

# # --- Example Usage ---
# if __name__ == '__main__':
# 	DATASET_DIRECTORY = {
# 	"farid": "/home/farid/datasets/WW_DATASETs/HISTORY_X3",
# 	"alijanif": "/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4",
# 	"ubuntu": "/media/volume/ImACCESS/WW_DATASETs/HISTORY_X4",
# 	"alijani": "/lustre/sgn-data/ImACCESS/WW_DATASETs/HISTORY_X4",
# 	}

# 	# Replace with the actual paths to your metadata files
# 	# Assumes you are running this script from a directory where these paths are valid
# 	full_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata.csv")
# 	train_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata_train.csv")
# 	val_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata_val.csv")
# 	# Use a fixed seed for reproducible sample selection during development/analysis
# 	random.seed(42)
# 	np.random.seed(42)
# 	i2t_samples, t2i_samples = select_qualitative_samples(
# 			metadata_path=full_meta,
# 			metadata_train_path=train_meta, # Train path not strictly needed for this version but good practice
# 			metadata_val_path=val_meta,
# 			num_samples_per_segment=5 # Select 5 samples per segment
# 	)
# 	if i2t_samples and t2i_samples:
# 			print("\n--- Selected I2T Queries (Image Path, GT Label) ---")
# 			for i, sample in enumerate(i2t_samples):
# 					print(f"Sample {i+1} (GT: {sample['label']}): {sample['image_path']}")
# 					# Example of how to pass to your inference code:
# 					# !python history_clip_inference.py -qi "{sample['image_path']}" ... (other args)
# 			print("\n--- Selected T2I Queries (Label String) ---")
# 			for i, sample in enumerate(t2i_samples):
# 					print(f"Sample {i+1}: '{sample['label']}'")
# 					# Example of how to pass to your inference code:
# 					# !python history_clip_inference.py -ql "{sample['label']}" ... (other args)
# 			# Example of a hand-picked "bias" example if needed
# 			# print("\n--- Example 'Bias' T2I Query ---")
# 			# print("Query Label: 'political figure'") # Or the specific label for the nature photo GT
# 			# print("Expected Output: Visually relevant images of politicians")
# 			# print("Observed Output (Progressive FT, Top-1): Image of nature scene due to metadata.")
# 			# Add the image path and GT label of that specific nature photo here if you want to reference it.
# 			# bias_image_path = "/path/to/that/specific/nature/image.jpg"
# 			# bias_image_gt_label = "political figure" # Or whatever its actual label is
# 			# print(f"Specific image illustrating bias (GT: {bias_image_gt_label}): {bias_image_path}")

# from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
# from PIL import Image
# import requests

# # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# # image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open("/home/farid/datasets/TEST_IMGs/6002_107454.jpg")

# processor = ViTImageProcessor.from_pretrained('google/vit-large-patch32-384')
# model = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384')

# # processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
# # model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# print(type(logits), logits.shape, logits.dtype, logits.device)
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])

# # Convert logits to probabilities using softmax
# probs = torch.nn.functional.softmax(logits, dim=1)

# # Get top-5 predictions
# top_probs, top_indices = torch.topk(probs, 5)
# print(f"\nTop 5 Predictions (ImageNet Labels):\n{'-'*40}")

# for i in range(5):
#     idx = top_indices[0][i].item()
#     prob = top_probs[0][i].item()
#     label = model.config.id2label[idx]
#     print(f"{i+1}. {label} ({prob:.2%})")

# print("\nLogits info:")
# print(f"Type: {type(logits)}, Shape: {logits.shape}, Dtype: {logits.dtype}, Device: {logits.device}")


import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch >= 2.31.0, timm >= 1.0.15

model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-gopt-16-SigLIP2-384')
tokenizer = get_tokenizer('hf-hub:timm/ViT-gopt-16-SigLIP2-384')

image = Image.open(urlopen(
	# 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/8inchHowitzerTowedByScammellPioneer12Jan1940.jpg/632px-8inchHowitzerTowedByScammellPioneer12Jan1940.jpg'
	"https://media.cnn.com/api/v1/images/stellar/prod/180207010106-military-parades-us-new-york-1946.jpg"
))
image = preprocess(image).unsqueeze(0)


# Define category sets for different aspects of visual content
object_categories = [
	# Military vehicles
	"tank", "jeep", "armored car", "truck", "military aircraft", "helicopter",
	"submarine", "battleship", "aircraft carrier", "fighter jet", "bomber aircraft",
	
	# Military personnel
	"soldier", "officer", "military personnel", "pilot", "sailor", "cavalry",
	
	# Weapons
	"gun", "rifle", "machine gun", "artillery", "cannon", "missile", "bomb",
	
	# Other military objects
	"military base", "bunker", "trench", "fortification", "flag", "military uniform"
]

scene_categories = [
	# Terrain types
	"desert", "forest", "urban area", "beach", "mountain", "field", "ocean", "river",
	
	# Military scenes
	"battlefield", "military camp", "airfield", "naval base", "military parade",
	"military exercise", "war zone", "training ground", "military factory"
]

era_categories = [
	"World War I era", "World War II era", "Cold War era", "modern military",
	"1910s style", "1940s style", "1960s style", "1980s style", "2000s style"
]

activity_categories = [
	"driving", "flying", "marching", "fighting", "training", "maintenance",
	"loading equipment", "unloading equipment", "towing", "firing weapon",
	"military parade", "crossing terrain", "naval operation"
]

labels_list = object_categories + scene_categories + era_categories + activity_categories
text = tokenizer(labels_list, context_length=model.context_length)

with torch.no_grad(), with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
	image_features = model.encode_image(image, normalize=True)
	text_features = model.encode_text(text, normalize=True)
	text_probs = torch.sigmoid(image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias)

zipped_list = list(zip(labels_list, [100 * round(p.item(), 3) for p in text_probs[0]]))
print("Label probabilities: ", zipped_list)




def get_visual_based_annotation_ensemble(
				csv_file: str,
				confidence_threshold: float = 0.5,
				batch_size: int = 16,
				device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
				verbose: bool = True
		) -> List[List[str]]:
		"""
		Generate visual-based annotations using an ensemble of vision models with different architectures.
		This function uses object detection, scene recognition, fine-grained classification, and 
		human-centric models to create a comprehensive set of labels.
		
		Args:
				csv_file: Path to metadata CSV containing image paths
				confidence_threshold: Minimum confidence score to accept a prediction
				batch_size: Number of images to process at once
				device: Device to run models on ('cuda:0' or 'cpu')
				verbose: Whether to print progress information
				
		Returns:
				List of label lists, one per image
		"""
		print(f"Automatic label extraction from image data (Ensemble Approach)".center(160, "-"))
		start_time = time.time()
		
		# Load dataset
		if verbose:
				print(f"Loading metadata from {csv_file}...")
		df = pd.read_csv(csv_file, dtype={'img_path': str}, low_memory=False)
		image_paths = df['img_path'].tolist()
		if verbose:
				print(f"Found {len(image_paths)} images to process")
		
		# Define category mappings for each model (in production, these would be more extensive)
		COCO_CATEGORIES = {
				1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
				6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
				# Add more classes that are relevant to historical/military images
		}
		
		SCENE_CATEGORIES = [
				"battlefield", "military base", "bunker", "trench", "airfield",
				"desert", "forest", "urban area", "beach", "mountain", "river", "field",
				"destroyed building", "military parade", "military camp", "naval base",
				# Add more scene categories relevant to historical images
		]
		
		MILITARY_EQUIPMENT_CATEGORIES = [
				"tank", "artillery", "rifle", "machine gun", "fighter plane", "bomber",
				"jeep", "military truck", "battleship", "submarine", "armored vehicle",
				"missile launcher", "anti-aircraft gun", "military helicopter",
				# Add more fine-grained military equipment categories
		]
		
		UNIFORM_CATEGORIES = [
				"army uniform", "navy uniform", "air force uniform", "officer", "soldier",
				"pilot", "civilian", "prisoner", "medical personnel", "resistance fighter",
				# Add more uniform/human categories relevant to historical context
		]
		

		# Load models - each from a different architecture family
		if verbose:
				print("Loading ensemble of vision models...")
		
		# 1. Object Detection Model: Faster R-CNN
		object_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
		object_detector.to(device).eval()
		
		# 2. Scene Recognition: DenseNet trained on Places365
		# Note: If Places365 weights aren't available, use ImageNet weights and map to relevant categories
		scene_classifier = torchvision.models.densenet161(pretrained=True)
		num_ftrs = scene_classifier.classifier.in_features
		scene_classifier.classifier = torch.nn.Linear(num_ftrs, len(SCENE_CATEGORIES))
		# In production, you would load saved weights for Places365
		# scene_classifier.load_state_dict(torch.load('densenet161_places365.pth'))
		scene_classifier.to(device).eval()
		
		# 3. Fine-grained Recognition: EfficientNet for military equipment
		finegrained_classifier = torchvision.models.efficientnet_b3(pretrained=True)
		num_ftrs = finegrained_classifier.classifier[1].in_features
		finegrained_classifier.classifier[1] = torch.nn.Linear(num_ftrs, len(MILITARY_EQUIPMENT_CATEGORIES))
		# In production, you would load domain-specific weights
		# finegrained_classifier.load_state_dict(torch.load('efficientnet_military.pth'))
		finegrained_classifier.to(device).eval()
		
		# 4. Human Parsing: HRNet for uniforms and human attributes
		human_classifier = torchvision.models.resnet50(pretrained=True)
		num_ftrs = human_classifier.fc.in_features
		human_classifier.fc = torch.nn.Linear(num_ftrs, len(UNIFORM_CATEGORIES))
		# In production, you would load domain-specific weights
		# human_classifier.load_state_dict(torch.load('resnet_uniforms.pth'))
		human_classifier.to(device).eval()
		

		# Define preprocessing transforms
		object_transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
		])
		
		classifier_transform = torchvision.transforms.Compose([
				torchvision.transforms.Resize(256),
				torchvision.transforms.CenterCrop(224),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		
		# Define functions to process images with each model type
		def process_object_detection_batch(batch_paths):
				"""Process a batch of images with the object detection model"""
				valid_images = []
				valid_indices = []
				
				# Load images
				for i, path in enumerate(batch_paths):
						try:
								if os.path.exists(path):
										img = Image.open(path).convert('RGB')
										img_tensor = object_transform(img)
										valid_images.append(img_tensor)
										valid_indices.append(i)
						except Exception as e:
								if verbose:
										print(f"Error loading image {path}: {e}")
				
				if not valid_images:
						return [[] for _ in range(len(batch_paths))]
				
				batch_results = [[] for _ in range(len(batch_paths))]
				
				# Process with Faster R-CNN
				with torch.no_grad():
						for i, img_tensor in enumerate(valid_images):
								# Faster R-CNN expects a list of tensors
								predictions = object_detector([img_tensor.to(device)])
								
								for pred in predictions:
										boxes = pred['boxes']
										labels = pred['labels']
										scores = pred['scores']
										
										# Filter by confidence threshold
										high_conf_indices = scores > confidence_threshold
										filtered_labels = labels[high_conf_indices]
										
										# Convert label indices to category names
										batch_idx = valid_indices[i]
										for label_idx in filtered_labels:
												label_idx = label_idx.item()
												if label_idx in COCO_CATEGORIES:
														batch_results[batch_idx].append(COCO_CATEGORIES[label_idx])
				
				return batch_results
		
		def process_classifier_batch(batch_paths, model, categories, transform):
				"""Process a batch of images with a classifier model"""
				valid_images = []
				valid_indices = []
				
				# Load images
				for i, path in enumerate(batch_paths):
						try:
								if os.path.exists(path):
										img = Image.open(path).convert('RGB')
										img_tensor = transform(img)
										valid_images.append(img_tensor)
										valid_indices.append(i)
						except Exception as e:
								if verbose:
										print(f"Error loading image {path}: {e}")
				
				if not valid_images:
						return [[] for _ in range(len(batch_paths))]
				
				batch_results = [[] for _ in range(len(batch_paths))]
				
				# Stack tensors for batch processing
				batch_tensor = torch.stack(valid_images).to(device)
				
				# Process with classifier
				with torch.no_grad():
						outputs = model(batch_tensor)
						probabilities = torch.nn.functional.softmax(outputs, dim=1)
						
						# Get predictions for each image
						for i, probs in enumerate(probabilities):
								batch_idx = valid_indices[i]
								
								# Get high confidence predictions
								high_conf_indices = (probs > confidence_threshold).nonzero(as_tuple=True)[0]
								for idx in high_conf_indices:
										category = categories[idx.item()]
										batch_results[batch_idx].append(category)
				
				return batch_results
		
		# Process images in batches with each model
		all_object_labels = []
		all_scene_labels = []
		all_equipment_labels = []
		all_uniform_labels = []
		
		# 1. Object Detection
		if verbose:
				print("Performing object detection...")
		for i in tqdm(range(0, len(image_paths), batch_size), desc="Object Detection"):
				batch_paths = image_paths[i:i+batch_size]
				batch_results = process_object_detection_batch(batch_paths)
				all_object_labels.extend(batch_results)
		
		# 2. Scene Recognition
		if verbose:
				print("Performing scene classification...")
		for i in tqdm(range(0, len(image_paths), batch_size), desc="Scene Classification"):
				batch_paths = image_paths[i:i+batch_size]
				batch_results = process_classifier_batch(
						batch_paths, scene_classifier, SCENE_CATEGORIES, classifier_transform
				)
				all_scene_labels.extend(batch_results)
		
		# 3. Fine-grained Equipment Recognition
		if verbose:
				print("Identifying military equipment...")
		for i in tqdm(range(0, len(image_paths), batch_size), desc="Equipment Classification"):
				batch_paths = image_paths[i:i+batch_size]
				batch_results = process_classifier_batch(
						batch_paths, finegrained_classifier, MILITARY_EQUIPMENT_CATEGORIES, classifier_transform
				)
				all_equipment_labels.extend(batch_results)
		
		# 4. Human/Uniform Classification
		if verbose:
				print("Analyzing uniforms and personnel...")
		for i in tqdm(range(0, len(image_paths), batch_size), desc="Uniform Classification"):
				batch_paths = image_paths[i:i+batch_size]
				batch_results = process_classifier_batch(
						batch_paths, human_classifier, UNIFORM_CATEGORIES, classifier_transform
				)
				all_uniform_labels.extend(batch_results)
		
		# Ensemble the results with weighted voting
		combined_labels = []
		for i in range(len(image_paths)):
				# Initialize empty set for this image
				image_labels = set()
				
				# Add results from each model
				if i < len(all_object_labels):
						image_labels.update(all_object_labels[i])
				
				if i < len(all_scene_labels):
						image_labels.update(all_scene_labels[i])
				
				if i < len(all_equipment_labels):
						image_labels.update(all_equipment_labels[i])
				
				if i < len(all_uniform_labels):
						image_labels.update(all_uniform_labels[i])
				
				# Handle duplicates and near-duplicates
				final_labels = deduplicate_labels(list(image_labels))
				
				# Apply semantic categorization and balance
				categorized = assign_semantic_categories(final_labels)
				final_labels = sorted(set(final_labels + categorized))
				
				# Add to results
				combined_labels.append(final_labels)
		
		# Save results
		df['visual_based_labels'] = combined_labels
		output_path = os.path.join(os.path.dirname(csv_file), "metadata_visual_based_labels_ensemble.csv")
		df.to_csv(output_path, index=False)
		
		if verbose:
				total_labels = sum(len(labels) for labels in combined_labels)
				print(f"Vision-based ensemble annotation completed in {time.time() - start_time:.2f} seconds")
				print(f"Generated {total_labels} labels for {len(image_paths)} images")
				print(f"Average labels per image: {total_labels/len(image_paths):.2f}")
				
				# Print distribution of label sources
				obj_count = sum(len(labels) for labels in all_object_labels)
				scene_count = sum(len(labels) for labels in all_scene_labels)
				equip_count = sum(len(labels) for labels in all_equipment_labels)
				uniform_count = sum(len(labels) for labels in all_uniform_labels)
				
				print("\nLabel source distribution:")
				print(f"Object detection: {obj_count} labels ({obj_count/total_labels*100:.1f}%)")
				print(f"Scene classification: {scene_count} labels ({scene_count/total_labels*100:.1f}%)")
				print(f"Equipment recognition: {equip_count} labels ({equip_count/total_labels*100:.1f}%)")
				print(f"Uniform/personnel: {uniform_count} labels ({uniform_count/total_labels*100:.1f}%)")
		
		print(f"Vision-based ensemble annotation elapsed time: {time.time() - start_time:.2f} sec".center(160, " "))
		return combined_labels

