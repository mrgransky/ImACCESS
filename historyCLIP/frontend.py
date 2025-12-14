# from backend import run_backend
# import datetime
# import time

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

import numpy as np
import matplotlib.pyplot as plt
import math
import os

def plot_peft_radar_chart(data, feature_labels, save_path):
	"""
	data: Dict[str, List[float]] -> {'Method Name': [value1, value2, ...]}
	feature_labels: List[str] -> Names of axes
	"""
	
	# 1. Normalize Data (Min-Max Scaling) so all axes are 0.0 to 1.0
	# This ensures 'Parameter Efficiency' is comparable to 'mAP'
	raw_values = np.array(list(data.values()))
	min_vals = raw_values.min(axis=0)
	max_vals = raw_values.max(axis=0)
	
	# Avoid division by zero
	normalized_data = {}
	for method, values in data.items():
		norm_vals = (np.array(values) - min_vals) / (max_vals - min_vals + 1e-9)
		# Add a small baseline (0.1) so lines don't disappear at the center
		normalized_data[method] = norm_vals + 0.1
	
	# 2. Setup Plot
	N = len(feature_labels)
	angles = [n / float(N) * 2 * math.pi for n in range(N)]
	angles += angles[:1] # Close the loop
	
	fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
	
	# 3. Visual Styling
	colors = [
		'#1f77b4', 
		'#ff7f0e', 
		'#2ca02c',
		'#d62728', 
		'#9467bd', 
		'#8c564b'
	]
	linestyles = ['-', '--', '-.', ':', '-', '--']
	
	for i, (method, values) in enumerate(normalized_data.items()):
		val_list = values.tolist()
		val_list += val_list[:1] # Close the loop
		
		ax.plot(angles, val_list, linewidth=2, linestyle=linestyles[i], label=method, color=colors[i])
		ax.fill(angles, val_list, color=colors[i], alpha=0.1)
	
	# 4. Axis Formatting
	ax.set_theta_offset(math.pi / 2) # Rotate so first axis is at top
	ax.set_theta_direction(-1)  # Clockwise
	
	plt.xticks(angles[:-1], feature_labels, size=9, rotation=90, ha='right')
	
	# Hide radial labels (the numbers 0.2, 0.4 etc are meaningless after normalization)
	ax.set_yticklabels([])
	
	# Add Legend with some spacing
	plt.legend(fontsize=10, loc="upper left", frameon=False, fancybox=True, shadow=True, edgecolor='black', facecolor='white')
	
	plt.title("Holistic PEFT Evaluation", size=15, weight='bold')
	plt.tight_layout()
	plt.savefig(save_path, dpi=200, bbox_inches='tight')

if __name__ == "__main__":
	# Ensure all metrics are oriented so "Higher is Better"
	# e.g., for Memory, use (1 / VRAM_Usage) or "Memory Savings %"
	RESULTS_DIR = "results"
	os.makedirs(RESULTS_DIR, exist_ok=True)
	metrics = ['I2T mAP', 'T2I mAP', 'Param Efficiency', 'Memory Efficiency', 'Training Speed']
	
	# Dummy data (Replace with your experiment logs!)
	results = {
		'Full Fine-Tuning': [0.85, 0.84, 0.01, 0.10, 0.20], # High Acc, Low Eff
		'LoRA (r=16)':      [0.82, 0.81, 0.80, 0.80, 0.70], # Balanced
		'DoRA (r=16)':      [0.86, 0.85, 0.78, 0.75, 0.65], # High Acc (Good Eff)
		'VeRA':             [0.75, 0.74, 0.99, 0.95, 0.90], # Low Acc, Max Eff
		'Tip-Adapter-F':    [0.79, 0.78, 0.90, 0.85, 0.95], # Good Eff, Mid Acc
	}
	
	plot_peft_radar_chart(results, metrics, "results/radar_chart.png")