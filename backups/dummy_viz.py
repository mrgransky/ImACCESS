import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib as mpl

# Set a seed for reproducibility
np.random.seed(42)

# Create dummy directory for results
if not os.path.exists("./sample_results"):
		os.makedirs("./sample_results")

# Define constants
topK_values = [1, 5, 10, 15, 20]
dataset_name = "SMU"
models = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
metrics = ["mP", "mAP", "Recall"]

# Generate realistic dummy data
def generate_performance_data(is_finetuned=False, base_performance=None):
		"""
		Generate realistic performance metrics data for CLIP models.
		If is_finetuned is True and base_performance is provided, 
		generate improved metrics based on the base performance.
		"""
		result = {}
		
		for model in models:
				result[model] = {}
				
				# Different base performance for each model
				model_factor = {
						'ViT-B/32': 1.0,
						'ViT-B/16': 1.1,
						'ViT-L/14': 1.2,
						'ViT-L/14@336px': 1.3
				}[model]
				
				for metric in metrics:
						result[model][metric] = {}
						
						# Generate base values for each metric
						if metric == "mP":
								# mP typically decreases as K increases
								base_values = [0.65 * model_factor] + [max(0.05, 0.75 * model_factor / k) for k in topK_values[1:]]
						elif metric == "mAP":
								# mAP tends to increase slightly then plateau
								base_values = [0.6 * model_factor] + [min(0.85, 0.65 * model_factor + 0.05 * i) for i in range(1, len(topK_values))]
						else:  # Recall
								# Recall increases with K
								base_values = [min(0.99, 0.5 * model_factor + 0.1 * k) for k in topK_values]
						
						# Add some randomness
						base_values = [v + random.uniform(-0.05, 0.05) for v in base_values]
						
						# Apply fine-tuning improvement if needed
						if is_finetuned and base_performance:
								# Add improvement percentage (10-30% better)
								improvement_factor = random.uniform(1.15, 1.3)
								values = [min(0.99, v * improvement_factor) for v in base_performance[model][metric].values()]
						else:
								values = base_values
						
						# Store in dictionary
						for i, k in enumerate(topK_values):
								result[model][metric][str(k)] = max(0.01, min(0.99, values[i]))
		
		return result

# Generate data for Image-to-Text retrieval
pretrained_img2txt_dict = generate_performance_data()
finetuned_img2txt_dict = generate_performance_data(is_finetuned=True, base_performance=pretrained_img2txt_dict)

# Generate data for Text-to-Image retrieval (typically lower performance)
pretrained_txt2img_dict = {
		model: {
				metric: {
						str(k): max(0.01, min(0.99, pretrained_img2txt_dict[model][metric][str(k)] * 0.7))
						for k in topK_values
				} for metric in metrics
		} for model in models
}

finetuned_txt2img_dict = {
		model: {
				metric: {
						str(k): max(0.01, min(0.99, v * random.uniform(1.2, 1.4)))
						for k, v in pretrained_txt2img_dict[model][metric].items()
				} for metric in metrics
		} for model in models
}

print("Dummy data generated.")
print(f"pre-trained: {pretrained_img2txt_dict}")
print(f"fine-tuned: {finetuned_img2txt_dict}")

# Function 1: Plot Comparison Metrics
def plot_comparison_metrics(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,  # e.g., 'ViT-B/32'
		topK_values: list,
		results_dir: str,
		figure_size=(15, 10),
		DPI: int=300
):
		metrics = ["mP", "mAP", "Recall"]
		modes = ["Image-to-Text", "Text-to-Image"]
		
		# Create consistent colors for different models
		model_colors = {
				'ViT-B32': 'red',
				'ViT-B16': 'blue',
				'ViT-L14': 'green',
				'ViT-L14336px': 'purple'
		}
		
		# Create figure with 2x3 subplots
		fig, axes = plt.subplots(2, 3, figsize=figure_size, constrained_layout=True)
		fig.suptitle(f"{dataset_name} CLIP Model Performance: Pre-trained vs. Fine-tuned {model_name}", fontsize=16, fontweight='bold')
		
		# Plot data for each mode and metric
		for i, mode in enumerate(modes):
				# Select the appropriate dictionaries
				pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				print(f"mode: {mode}")
				print(f"pretrained_dict: {pretrained_dict}")
				print(f"finetuned_dict: {finetuned_dict}")
				print()

				for j, metric in enumerate(metrics):
						ax = axes[i, j]
						
						# Plot pre-trained model performance
						if model_name in pretrained_dict:
								k_values = sorted([int(k) for k in pretrained_dict[model_name][metric].keys() 
																	if int(k) in topK_values])
								values = [pretrained_dict[model_name][metric][str(k)] for k in k_values]
								
								pretrained_line, = ax.plot(
										k_values,
										values,
										label=f"Pre-trained",
										color=model_colors[model_name],
										marker='o',
										linestyle='--',
										linewidth=2,
										markersize=5,
										alpha=0.7
								)
						
						# Plot fine-tuned model performance
						if model_name in finetuned_dict:
								k_values = sorted([int(k) for k in finetuned_dict[model_name][metric].keys() 
																	if int(k) in topK_values])
								values = [finetuned_dict[model_name][metric][str(k)] for k in k_values]
								
								finetuned_line, = ax.plot(
										k_values,
										values,
										label=f"Fine-tuned",
										color=model_colors[model_name],
										marker='s',
										linestyle='-',
										linewidth=2,
										markersize=5
								)
								
								# Add improvement percentages at key points
								if model_name in pretrained_dict:
										for idx, k in enumerate([1, 10]):
												if k in k_values:
														k_idx = k_values.index(k)
														pre_val = pretrained_dict[model_name][metric][str(k)]
														fine_val = values[k_idx]
														improvement = ((fine_val - pre_val) / pre_val) * 100
														ax.annotate(
																f"+{improvement:.1f}%", 
																xy=(k, fine_val),
																xytext=(5, 5),
																textcoords='offset points',
																fontsize=8,
																fontweight='bold'
														)
						
						# Configure axes
						ax.set_xlabel('K', fontsize=12)
						ax.set_ylabel(f'{metric}@K', fontsize=12)
						ax.set_title(f'{mode} - {metric}@K', fontsize=14)
						ax.grid(True, linestyle='--', alpha=0.7)
						ax.set_xticks(topK_values)
						
						# Set y-axis limits based on data
						all_values = []
						if model_name in pretrained_dict:
								all_values.extend([pretrained_dict[model_name][metric][str(k)] for k in k_values])
						if model_name in finetuned_dict:
								all_values.extend([finetuned_dict[model_name][metric][str(k)] for k in k_values])
						
						if all_values:
								min_val = min(all_values)
								max_val = max(all_values)
								padding = 0.1 * (max_val - min_val) if max_val > min_val else 0.1
								ax.set_ylim(bottom=max(0, min_val - padding), top=min(1.0, max_val + padding))
						
						# Add legend to first subplot only
						if i == 0 and j == 0:
								ax.legend(fontsize=10)
								
		# Save the figure
		plt.savefig(
				os.path.join(results_dir, f"{dataset_name}_{model_name.replace('/', '-')}_comparison.png"), 
				dpi=DPI, 
				bbox_inches='tight'
		)
		plt.close(fig)
		
		return fig

# Function 2: Plot Improvement Summary
def plot_improvement_summary(
		dataset_name: str,
		pretrained_dict: dict,
		finetuned_dict: dict,
		models: list,
		metrics: list,
		k_value: int,  # e.g., 10
		mode: str,  # 'Image-to-Text' or 'Text-to-Image'
		results_dir: str,
		figure_size=(10, 6),
		DPI: int=300
):
		# Calculate improvement percentages
		improvements = []
		model_names = []
		
		for model in models:
				model_improvements = []
				for metric in metrics:
						pre_val = pretrained_dict[model][metric][str(k_value)]
						fine_val = finetuned_dict[model][metric][str(k_value)]
						improvement = ((fine_val - pre_val) / pre_val) * 100
						model_improvements.append(improvement)
				improvements.append(model_improvements)
				model_names.append(model.replace('ViT-', ''))
		
		# Create grouped bar chart
		fig, ax = plt.subplots(figsize=figure_size)
		x = np.arange(len(model_names))
		width = 0.25
		
		# Set color for each metric
		metric_colors = {'mP': 'tab:blue', 'mAP': 'tab:orange', 'Recall': 'tab:green'}
		
		for i, metric in enumerate(metrics):
				bars = ax.bar(x + i*width - width, [imp[i] for imp in improvements], width, 
										 label=f'{metric}@{k_value}', color=metric_colors[metric])
				
				# Add value labels
				for bar in bars:
						height = bar.get_height()
						ax.annotate(f'{height:.1f}%',
											xy=(bar.get_x() + bar.get_width()/2, height),
											xytext=(0, 3),
											textcoords="offset points",
											ha='center', va='bottom',
											fontsize=8)
		
		ax.set_ylabel('Improvement (%)', fontsize=12)
		ax.set_title(f'{dataset_name} {mode} - Performance Improvement at K={k_value}', fontsize=14)
		ax.set_xticks(x)
		ax.set_xticklabels(model_names)
		ax.legend()
		ax.grid(axis='y', linestyle='--', alpha=0.7)
		
		plt.savefig(
				os.path.join(results_dir, f"{dataset_name}_{mode.replace('-', '_')}_improvement_k{k_value}.png"), 
				dpi=DPI, 
				bbox_inches='tight'
		)
		plt.close(fig)
		
		return fig

# Generate plots using our dummy data
# Example 1: Comparison plot for ViT-B/32
comparison_plot = plot_comparison_metrics(
		dataset_name=dataset_name,
		pretrained_img2txt_dict=pretrained_img2txt_dict,
		pretrained_txt2img_dict=pretrained_txt2img_dict,
		finetuned_img2txt_dict=finetuned_img2txt_dict,
		finetuned_txt2img_dict=finetuned_txt2img_dict,
		model_name='ViT-B/32',
		topK_values=topK_values,
		results_dir="./sample_results",
		figure_size=(15, 10),
		DPI=300
)

# Example 2: Improvement summary at K=10 for Image-to-Text
improvement_plot_img2txt = plot_improvement_summary(
		dataset_name=dataset_name,
		pretrained_dict=pretrained_img2txt_dict,
		finetuned_dict=finetuned_img2txt_dict,
		models=models,
		metrics=metrics,
		k_value=10,
		mode='Image-to-Text',
		results_dir="./sample_results",
		figure_size=(12, 7),
		DPI=300
)

# Example 3: Improvement summary at K=10 for Text-to-Image
improvement_plot_txt2img = plot_improvement_summary(
		dataset_name=dataset_name,
		pretrained_dict=pretrained_txt2img_dict,
		finetuned_dict=finetuned_txt2img_dict,
		models=models,
		metrics=metrics,
		k_value=10,
		mode='Text-to-Image',
		results_dir="./sample_results",
		figure_size=(12, 7),
		DPI=300
)

print("Sample plots have been generated in the ./sample_results directory")