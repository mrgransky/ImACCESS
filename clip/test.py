from utils import *

import matplotlib.pyplot as plt

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
    metrics = ['precision', 'map', 'recall']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Retrieval Performance Metrics Over Epochs", fontsize=16)
    
    for i, task_metrics_list in enumerate([image_to_text_metrics_list, text_to_image_metrics_list]):
        for j, metric in enumerate(metrics):
            ax = axs[i, j]
            for K, color in zip(topK_values, colors):
                values = []
                for metrics_dict in task_metrics_list:
                    if metric in metrics_dict and str(K) in metrics_dict[metric]:
                        values.append(metrics_dict[metric][str(K)])
                    else:
                        values.append(0)  # or None if you prefer to show gaps
                ax.plot(epochs, values, marker='o', label=f'K={K}', color=color)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(f'Mean {metric.capitalize()}@K', fontsize=12)
            ax.set_title(f'{["Image-to-Text", "Text-to-Image"][i]} - {metric.capitalize()}@K', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(epochs)
            ax.set_ylim(bottom=0)  # Ensure y-axis starts from 0

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

# Sample data for 5 epochs
image_to_text_metrics_list = [
    {
        "precision": {"1": 0.9, "5": 0.7, "10": 0.6},
        "recall": {"1": 0.8, "5": 0.9, "10": 0.95},
        "map": {"1": 0.85, "5": 0.8, "10": 0.75}
    },
    {
        "precision": {"1": 0.92, "5": 0.75, "10": 0.65},
        "recall": {"1": 0.82, "5": 0.92, "10": 0.96},
        "map": {"1": 0.87, "5": 0.83, "10": 0.78}
    },
    {
        "precision": {"1": 0.94, "5": 0.8, "10": 0.7},
        "recall": {"1": 0.84, "5": 0.94, "10": 0.97},
        "map": {"1": 0.89, "5": 0.86, "10": 0.81}
    },
    {
        "precision": {"1": 0.96, "5": 0.85, "10": 0.75},
        "recall": {"1": 0.86, "5": 0.96, "10": 0.98},
        "map": {"1": 0.91, "5": 0.89, "10": 0.84}
    },
    {
        "precision": {"1": 0.98, "5": 0.9, "10": 0.8},
        "recall": {"1": 0.88, "5": 0.98, "10": 0.99},
        "map": {"1": 0.93, "5": 0.92, "10": 0.87}
    }
]

text_to_image_metrics_list = [
    {
        "precision": {"1": 0.8, "5": 0.6, "10": 0.5},
        "recall": {"1": 0.7, "5": 0.8, "10": 0.85},
        "map": {"1": 0.75, "5": 0.7, "10": 0.65}
    },
    {
        "precision": {"1": 0.82, "5": 0.65, "10": 0.55},
        "recall": {"1": 0.72, "5": 0.82, "10": 0.87},
        "map": {"1": 0.77, "5": 0.73, "10": 0.68}
    },
    {
        "precision": {"1": 0.84, "5": 0.7, "10": 0.6},
        "recall": {"1": 0.74, "5": 0.84, "10": 0.89},
        "map": {"1": 0.79, "5": 0.76, "10": 0.71}
    },
    {
        "precision": {"1": 0.86, "5": 0.75, "10": 0.65},
        "recall": {"1": 0.76, "5": 0.86, "10": 0.91},
        "map": {"1": 0.81, "5": 0.79, "10": 0.74}
    },
    {
        "precision": {"1": 0.88, "5": 0.8, "10": 0.7},
        "recall": {"1": 0.78, "5": 0.88, "10": 0.93},
        "map": {"1": 0.83, "5": 0.82, "10": 0.77}
    }
]

# Test the function
topK_values = [1, 5, 10]
plot_retrieval_metrics(image_to_text_metrics_list, text_to_image_metrics_list, topK_values)