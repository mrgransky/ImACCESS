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
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch >= 2.31.0, timm >= 1.0.15

model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-gopt-16-SigLIP2-384')
tokenizer = get_tokenizer('hf-hub:timm/ViT-gopt-16-SigLIP2-384')

# url = "https://media.cnn.com/api/v1/images/stellar/prod/180207010106-military-parades-us-new-york-1946.jpg"
# url = "https://truck-encyclopedia.com/ww1/img/photos/German_WWI_armoured_car_destroyed.jpg"
url = "https://truck-encyclopedia.com/ww1/img/photos/Dart-CC4-production.jpg"
image = Image.open(urlopen(url))
image = preprocess(image).unsqueeze(0)

object_categories = [
	# Military Vehicles
	"tank", "tank destroyer", "armored car", "utility vehicle"
	"half-track", "armored personnel carrier", "armored train", "Kubelwagen",
	"jeep", "military truck", "supply truck", "artillery tractor", "amphibious vehicle",
	# Aircraft
	"military aircraft", "fighter aircraft", "bomber aircraft", "reconnaissance aircraft",
	"seaplane", "biplane", "monoplane",
	# Naval Vessels
	"submarine", "destroyer", "cruiser", "battleship", "aircraft carrier",
	"corvette", "minesweeper", "landing craft", "hospital ship", "troop transport", "tugboat",
	# Military Personnel
	"soldier", "infantryman", "officer", "pilot", "sailor", "marine", "medic",
	"Wehrmacht soldier", "Red Army soldier", "nurse",
	# Weapons
	"rifle", "machine gun", "pistol", "bayonet", "mortar", "artillery piece",
	"cannon", "anti-tank gun", "grenade", "landmine", "torpedo",
	# Military Infrastructure
	"bunker", "gun emplacement", "observation post", "barbed wire", "trenches",
	"fortification", "coastal defense", "minefield",
	# Military Symbols
	"military flag", "Union Jack", "military insignia",
	"medal", "military helmet", "gas mask",
	# Military Equipment
	"military uniform", "backpack", "entrenching tool", "binoculars", "field telephone",
	"radio equipment", "parachute", "jerry can", "stretcher",
	# Infrastructure
	"construction crane", "tunnel slab", "gasometer", "railway track", "locomotive wheel",
	"train car", "ferry slip", "locomotive engine", "railway signal", "construction site",
	# Civilian Objects
	"wheelbarrow", "leather apron", "signature document", "blood pressure monitor",
	"medical chart",
	# Cultural
	"cannon carriage", "museum", "pavilion structure",
	"market stall", "religious monument", "basilica statue", "palace garden", "historical plaque",
	# Animals
	"donkey", "camel", "workhorse", "mule", "ox", "reindeer",
	# Damaged Equipment and Infrastructure
	"wreck", "debris", "damaged vehicle", "damaged aircraft", "damaged ship",
	"ruined building", "collapsed bridge"
]

scene_categories = [
	# Military Theaters
	"Western Front", "Eastern Front", "coastline", "city ruins", "battlefield", "military camp",
	# Terrain
	"desert", "forest", "winter forest", "urban area", "mountain", "mountain valley",
	"field", "rural farmland", "snow-covered field", "ocean", "coastal waters",
	"river", "river crossing", "bridge", "lake shore", "coastal landscape",
	# Military Settings
	"military hospital", "field hospital", "military cemetery", "aircraft factory",
	"military depot", "military port", "dry dock",
	# Military Infrastructure
	"airfield", "naval base", "army barracks", "military headquarters",
	"coastal defense",
	# Civilian & Cultural
	"urban construction", "industrial site", "railroad station", "ferry dock",
	"harbor port", "bathing area", "palace courtyard",
	"historical plaza", "library hall", "urban plaza", "cathedral",
	"palace grounds",
	# Damaged Scenes
	"wreckage site", "battle damage", "disaster area", "ruined building", "shipwreck", "plane wreck"
]

era_categories = [
	"world war one", "world war two",
]

activity_categories = [
	# Combat
	"fighting", "tank battle", "infantry assault", "naval engagement", "barrage",
	"firing weapon", "bombing",
	# Movement
	"driving", "troop transport", "marching", "reconnaissance flight", "river crossing",
	# Military Operations
	"digging trenches", "fortification", "laying mines", "setting up artillery",
	"establishing field hospital",
	# Logistics
	"loading equipment", "loading ammunition", "evacuating casualties", "aircraft maintenance",
	# Military Life
	"training", "receiving briefing", "standing guard", "medical treatment",
	"distributing supplies",
	# Ceremonial
	"military parade", "flag raising", "ceremonial speech", "ceremony",
	"officer briefing", "civilian interaction", "hand shaking",
	# Civilian & Cultural
	"tunnel digging", "restoration", "railway maintenance", "wood stacking",
	"bathing", "portrait", "museum curation",
	"signature documentation",
	"farming harvest", "exhibition",
	# Transport
	"ferry operation", "train operation", "freight loading",
	# Damage-Related Activities
	"marine salvage", "demolition", "repair"
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Context length: {model.context_length}")
labels_list = object_categories + scene_categories + era_categories + activity_categories
print(f"labels_list: {len(labels_list)}")
text = tokenizer(labels_list, context_length=model.context_length)
print(f"text: {type(text)} {text.shape}")
with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
	image_features = model.encode_image(image, normalize=True)
	text_features = model.encode_text(text, normalize=True)
	text_probs = torch.sigmoid(image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias)

zipped_list = list(zip(labels_list, [100 * round(p.item(), 3) for p in text_probs[0]]))
# Sort by probability in descending order (highest probability first)
sorted_list = sorted(zipped_list, key=lambda x: x[1], reverse=True)
print(f"")
print("Label probabilities (sorted by confidence):")
print("-" * 50)
for label, prob in sorted_list:
	print(f"{label:<30}: {prob:>6.1f}%")

topk = 25
print("\n" + "="*50)
print(f"Top-{topk} predictions:")
print("="*50)
for i, (label, prob) in enumerate(sorted_list[:topk], 1):
	print(f"{i:2d}. {label:<25}: {prob:>6.1f}%")