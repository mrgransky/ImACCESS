from utils import *

import matplotlib.pyplot as plt
from typing import List, Dict

def plot_retrieval_metrics_best_model(
	image_to_text_metrics: Dict[str, Dict[str, float]],
	text_to_image_metrics: Dict[str, Dict[str, float]],
	topK_values: List[int],
	fname: str ="Retrieval_Performance_Metrics_best_model.png",
	):
	# Ensure the metrics are in the correct format
	metrics = list(image_to_text_metrics.keys())  # ['mP', 'mAP', 'Recall']
	suptitle_text = f"Retrieval Performance Metrics [Best Model]: "
	# suptitle_text = r"Retrieval Performance Metrics [\textcolor{green}{Best Model}]: "
	# suptitle_text = "Retrieval Performance Metrics [<span style='color:green'>Best Model</span>]: "
	for metric in metrics:
		suptitle_text += f"{metric}@K | " 
	suptitle_text = suptitle_text[:-3]  # Remove trailing " | "
	modes = ['Image-to-Text', 'Text-to-Image']
	
	fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
	fig.suptitle(suptitle_text, fontsize=15, fontweight='bold')
	
	# Store legend handles and labels
	legend_handles = []
	legend_labels = []
	
	for i, metric in enumerate(metrics):
		ax = axes[i]
		
		# Plotting for Image-to-Text
		it_values = [image_to_text_metrics[metric].get(str(K), 0) for K in topK_values]
		line, = ax.plot(topK_values, it_values, marker='o', label=modes[0], color='blue')
		if modes[0] not in legend_labels:
			legend_handles.append(line)
			legend_labels.append(modes[0])
		
		# Plotting for Text-to-Image
		ti_values = [text_to_image_metrics[metric].get(str(K), 0) for K in topK_values]
		line, = ax.plot(topK_values, ti_values, marker='s', label=modes[1], color='red')
		if modes[1] not in legend_labels:
			legend_handles.append(line)
			legend_labels.append(modes[1])
		
		ax.set_xlabel('K', fontsize=12)
		ax.set_ylabel(f'{metric}@K', fontsize=12)
		ax.set_title(f'{metric}@K', fontsize=14)
		ax.grid(True, linestyle='--', alpha=0.7)
		
		# Set the x-axis to only show integer values
		ax.set_xticks(topK_values)
		
		# Adjust y-axis to start from 0 for better visualization
		ax.set_ylim(bottom=0.0, top=1.05)
	
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.legend(
		legend_handles,
		legend_labels,
		fontsize=12,
		loc='upper center',
		ncol=len(modes),
		bbox_to_anchor=(0.5, 0.96),
		bbox_transform=fig.transFigure,
		frameon=True,
		shadow=True,
		fancybox=True,
		edgecolor='black',
		facecolor='white',
	)
	plt.savefig(fname, dpi=300, bbox_inches='tight')
	plt.close()

def plot_retrieval_metrics_grok3(
		image_to_text_metrics_list: List[Dict[str, Dict[str, float]]],
		text_to_image_metrics_list: List[Dict[str, Dict[str, float]]],
		topK_values: List[int],
		fname: str = "Retrieval_Performance_Metrics_Grok3.png",
):
		num_epochs = len(image_to_text_metrics_list)
		if num_epochs < 2:
				return

		epochs = range(1, num_epochs + 1)
		metrics = list(image_to_text_metrics_list[0].keys())  # ['mP', 'mAP', 'Recall']
		
		# Use a colormap with distinct colors
		cmap = plt.get_cmap("tab10")
		colors = [cmap(i) for i in range(len(topK_values))]
		
		# Define different line styles and markers for better differentiation
		line_styles = ['-', '--', '-.']
		markers = ['o', 's', 'D']
		
		# Create a larger figure for better visibility
		fig, axs = plt.subplots(2, 3, figsize=(22, 14), constrained_layout=True)
		fig.suptitle("Retrieval Performance Metrics: mP@K | mAP@K | Recall@K", fontsize=18)
		
		# Store legend handles and labels
		legend_handles = []
		legend_labels = []

		for i, task_metrics_list in enumerate([image_to_text_metrics_list, text_to_image_metrics_list]):
				for j, metric in enumerate(metrics):
						ax = axs[i, j]
						for idx, (K, color, linestyle, marker) in enumerate(zip(topK_values, colors, line_styles, markers)):
								values = []
								for metrics_dict in task_metrics_list:
										if metric in metrics_dict and str(K) in metrics_dict[metric]:
												values.append(metrics_dict[metric][str(K)])
										else:
												values.append(0)
								
								# Plot with distinct line styles, markers, and transparency
								line, = ax.plot(
										epochs, values, 
										marker=marker, 
										linestyle=linestyle, 
										label=f'K={K}', 
										color=color, 
										alpha=0.8, 
										markersize=8, 
										linewidth=2
								)
								
								# Collect handles and labels for the legend
								if f'K={K}' not in legend_labels:
										legend_handles.append(line)
										legend_labels.append(f'K={K}')
						
						# Set labels and title with larger fonts
						ax.set_xlabel('Epoch', fontsize=14)
						ax.set_ylabel(f'{metric}@K', fontsize=14)
						ax.set_title(f'{["Image-to-Text", "Text-to-Image"][i]} - {metric}@K', fontsize=16, pad=10)
						
						# Add grid for better readability
						ax.grid(True, linestyle='--', alpha=0.7)
						
						# Set x-ticks and ensure they are integers
						ax.set_xticks(epochs)
						
						# Dynamically adjust y-axis limits based on data
						min_val = min([min([metrics_dict[metric][str(K)] for metrics_dict in task_metrics_list 
															 if metric in metrics_dict and str(K) in metrics_dict[metric]]) 
													for K in topK_values])
						max_val = max([max([metrics_dict[metric][str(K)] for metrics_dict in task_metrics_list 
															 if metric in metrics_dict and str(K) in metrics_dict[metric]]) 
													for K in topK_values])
						ax.set_ylim(bottom=max(0, min_val - 0.05), top=min(1, max_val + 0.05))
						
						# Increase tick label size
						ax.tick_params(axis='both', labelsize=12)

		# Add a shared legend at the top, outside the subplots
		fig.legend(
				legend_handles,
				legend_labels,
				fontsize=14,
				loc='upper center',
				ncol=len(topK_values),
				bbox_to_anchor=(0.5, 0.98),
				bbox_transform=fig.transFigure,
				frameon=True,
				edgecolor='black',
				facecolor='white',
				shadow=True,
				title="Top-K Values",
				title_fontsize=14
		)

		# Adjust layout to prevent overlap
		plt.tight_layout(rect=[0, 0.03, 1, 0.95])
		
		# Save the plot with high resolution
		plt.savefig(fname, dpi=400, bbox_inches='tight')
		plt.close()


def plot_retrieval_metrics(
	image_to_text_metrics_list: List[Dict[str, Dict[str, float]]],
	text_to_image_metrics_list: List[Dict[str, Dict[str, float]]],
	topK_values: List[int],
	fname="Retrieval_Performance_Metrics.png",
	):
	num_epochs = len(image_to_text_metrics_list)
	if num_epochs < 2:
		return
	epochs = range(1, num_epochs + 1)
	metrics = list(image_to_text_metrics_list[0].keys())  # ['mP', 'mAP', 'Recall']
	cmap = plt.get_cmap("tab10")  # Use a colormap with at least 10 colors
	colors = [cmap(i) for i in range(cmap.N)]
	markers = ['o', 's', 'D', 'v', '^', 'P', 'X', 'd', 'H', 'h']  # Different markers for each line
	line_styles = ['-', '--', '-.', ':', '-']  # Different line styles for each metric
	fig, axs = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
	fig.suptitle("Retrieval Performance Metrics: mP@K | mAP@K | Recall@K", fontsize=16)
	# Store legend handles and labels
	legend_handles = []
	legend_labels = []
	for i, task_metrics_list in enumerate([image_to_text_metrics_list, text_to_image_metrics_list]):
		for j, metric in enumerate(metrics):
			ax = axs[i, j]
			for K, color, marker, linestyle in zip(topK_values, colors, markers, line_styles):
				values = []
				for metrics_dict in task_metrics_list:
					if metric in metrics_dict and str(K) in metrics_dict[metric]:
						values.append(metrics_dict[metric][str(K)])
					else:
						values.append(0)
				line, = ax.plot(
					epochs,
					values,
					marker=marker,
					markersize=6,
					linestyle=linestyle,
					label=f'K={K}',
					color=color, 
					alpha=0.7,
					linewidth=2.0,
				)
				# Collect handles and labels for the legend
				if f'K={K}' not in legend_labels:
					legend_handles.append(line)
					legend_labels.append(f'K={K}')
			ax.set_xlabel('Epoch', fontsize=12)
			ax.set_ylabel(f'{metric}@K', fontsize=12)
			ax.set_title(f'{["Image-to-Text", "Text-to-Image"][i]} - {metric}@K', fontsize=14)
			# ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
			ax.grid(True, linestyle='--', alpha=0.7)
			ax.set_xticks(epochs)
			ax.set_ylim(bottom=0.0, top=1.05)
	fig.legend(
		legend_handles,
		legend_labels,
		fontsize=11,
		loc='upper center',
		ncol=len(topK_values),
		bbox_to_anchor=(0.5, 0.96),
		bbox_transform=fig.transFigure,
		frameon=True,
		edgecolor='black',
		facecolor='white',
		shadow=True,
	)
	plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
	plt.savefig(fname, dpi=300, bbox_inches='tight')
	plt.close()

# Sample data for 5 epochs
image_to_text_metrics_list = [
		{
				"mP": {"1": 0.9, "5": 0.7, "10": 0.6},
				"Recall": {"1": 0.8, "5": 0.9, "10": 0.95},
				"mAP": {"1": 0.85, "5": 0.8, "10": 0.75}
		},
		{
				"mP": {"1": 0.92, "5": 0.75, "10": 0.65},
				"Recall": {"1": 0.82, "5": 0.92, "10": 0.96},
				"mAP": {"1": 0.87, "5": 0.83, "10": 0.78}
		},
		{
				"mP": {"1": 0.94, "5": 0.8, "10": 0.7},
				"Recall": {"1": 0.84, "5": 0.94, "10": 0.97},
				"mAP": {"1": 0.89, "5": 0.86, "10": 0.81}
		},
		{
				"mP": {"1": 0.96, "5": 0.85, "10": 0.75},
				"Recall": {"1": 0.86, "5": 0.96, "10": 0.98},
				"mAP": {"1": 0.91, "5": 0.89, "10": 0.84}
		},
		{
				"mP": {"1": 0.98, "5": 0.9, "10": 0.8},
				"Recall": {"1": 0.88, "5": 0.98, "10": 0.99},
				"mAP": {"1": 0.93, "5": 0.92, "10": 0.87}
		}
]

text_to_image_metrics_list = [
		{
				"mP": {"1": 0.8, "5": 0.6, "10": 0.5},
				"Recall": {"1": 0.7, "5": 0.8, "10": 0.85},
				"mAP": {"1": 0.75, "5": 0.7, "10": 0.65}
		},
		{
				"mP": {"1": 0.82, "5": 0.65, "10": 0.55},
				"Recall": {"1": 0.72, "5": 0.82, "10": 0.87},
				"mAP": {"1": 0.77, "5": 0.73, "10": 0.68}
		},
		{
				"mP": {"1": 0.84, "5": 0.7, "10": 0.6},
				"Recall": {"1": 0.74, "5": 0.84, "10": 0.89},
				"mAP": {"1": 0.79, "5": 0.76, "10": 0.71}
		},
		{
				"mP": {"1": 0.86, "5": 0.75, "10": 0.65},
				"Recall": {"1": 0.76, "5": 0.86, "10": 0.91},
				"mAP": {"1": 0.81, "5": 0.79, "10": 0.74}
		},
		{
				"mP": {"1": 0.88, "5": 0.8, "10": 0.7},
				"Recall": {"1": 0.78, "5": 0.88, "10": 0.93},
				"mAP": {"1": 0.83, "5": 0.82, "10": 0.77}
		}
]

# Test the function
topK_values = [1, 5, 10]
plot_retrieval_metrics(image_to_text_metrics_list, text_to_image_metrics_list, topK_values)

plot_retrieval_metrics_grok3(image_to_text_metrics_list, text_to_image_metrics_list, topK_values)


# Test data for Image-to-Text retrieval metrics
image_to_text_metrics_test = {
		"mP": {
				"1": 0.95,
				"5": 0.85,
				"10": 0.75,
				"15": 0.65,
				"20": 0.55
		},
		"Recall": {
				"1": 0.98,
				"5": 0.95,
				"10": 0.90,
				"15": 0.85,
				"20": 0.80
		},
		"mAP": {
				"1": 0.97,
				"5": 0.93,
				"10": 0.89,
				"15": 0.87,
				"20": 0.85
		}
}

# Test data for Text-to-Image retrieval metrics
text_to_image_metrics_test = {
		"mP": {
				"1": 0.88,
				"5": 0.80,
				"10": 0.70,
				"15": 0.60,
				"20": 0.50
		},
		"Recall": {
				"1": 0.92,
				"5": 0.88,
				"10": 0.85,
				"15": 0.80,
				"20": 0.75
		},
		"mAP": {
				"1": 0.90,
				"5": 0.85,
				"10": 0.80,
				"15": 0.75,
				"20": 0.70
		}
}

# Test topK_values
topK_values_test = [1, 5, 10, 15, 20]

# Call the function with test data
plot_retrieval_metrics_best_model(
		image_to_text_metrics_test,
		text_to_image_metrics_test,
		topK_values_test,
		fname="Test_Retrieval_Performance_Metrics_best_model.png"
)