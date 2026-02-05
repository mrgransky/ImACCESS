"""
Generate JSON file for Cluster Analysis Dashboard from clustering CSV output.

Usage:
		python generate_cluster_analysis_json.py --csv clusters_semantic_consolidation_agglomerative.csv --output cluster_analysis.json

This script:
1. Reads the clustering CSV output
2. Computes cluster coherence metrics
3. Calculates intra-cluster distances
4. Generates a JSON file for the dashboard
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os


def compute_cluster_metrics(
		cluster_df: pd.DataFrame,
		embeddings: np.ndarray,
		cluster_id: int
) -> Dict:
		"""
		Compute coherence and distance metrics for a single cluster.
		
		Args:
				cluster_df: DataFrame containing labels for this cluster
				embeddings: Embeddings for all labels in this cluster
				cluster_id: The cluster ID
		
		Returns:
				Dictionary with metrics: coherence, avg_distance, size
		"""
		if len(embeddings) == 1:
				return {
						'coherence': 1.0,
						'avg_distance': 0.0,
						'size': 1
				}
		
		# Compute centroid
		centroid = embeddings.mean(axis=0, keepdims=True)
		
		# Coherence: average cosine similarity to centroid
		similarities = cosine_similarity(centroid, embeddings)[0]
		coherence = float(similarities.mean())
		
		# Average pairwise distance
		pairwise_sims = cosine_similarity(embeddings)
		# Get upper triangle (excluding diagonal)
		n = len(embeddings)
		upper_triangle_indices = np.triu_indices(n, k=1)
		pairwise_distances = 1 - pairwise_sims[upper_triangle_indices]
		avg_distance = float(pairwise_distances.mean()) if len(pairwise_distances) > 0 else 0.0
		
		return {
				'coherence': coherence,
				'avg_distance': avg_distance,
				'size': len(embeddings)
		}


def generate_cluster_analysis_json(
		csv_path: str,
		output_json: str,
		model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
		device: str = None,
		max_labels_per_cluster: int = None,
		verbose: bool = True
):
		"""
		Generate JSON file for cluster analysis dashboard.
		
		Args:
				csv_path: Path to clustering CSV output
				output_json: Path to output JSON file
				model_id: SentenceTransformer model ID (should match clustering model)
				device: Device for embeddings ('cuda:0', 'cpu', etc.)
				max_labels_per_cluster: Maximum labels to include per cluster (None = all)
				verbose: Print progress
		"""
		
		if device is None:
				device = "cuda:0" if torch.cuda.is_available() else "cpu"
		
		print(f"\n{'='*60}")
		print(f"CLUSTER ANALYSIS JSON GENERATOR")
		print(f"{'='*60}")
		print(f"Input CSV: {csv_path}")
		print(f"Output JSON: {output_json}")
		print(f"Model: {model_id}")
		print(f"Device: {device}")
		print(f"{'='*60}\n")
		
		# Load CSV
		if verbose:
				print(f"[1/5] Loading CSV: {csv_path}")
		
		df = pd.read_csv(csv_path)
		
		required_cols = ['label', 'cluster', 'canonical_label']
		missing_cols = [col for col in required_cols if col not in df.columns]
		if missing_cols:
				raise ValueError(f"CSV missing required columns: {missing_cols}")
		
		print(f"  ├─ Total labels: {len(df)}")
		print(f"  ├─ Unique clusters: {df['cluster'].nunique()}")
		print(f"  └─ Columns: {list(df.columns)}")
		
		# Load model and generate embeddings
		if verbose:
				print(f"\n[2/5] Loading SentenceTransformer: {model_id}")
		
		cache_dir = os.path.expanduser("~/.cache/huggingface")
		model = SentenceTransformer(
				model_name_or_path=model_id,
				cache_folder=cache_dir,
				token=os.getenv("HUGGINGFACE_TOKEN"),
		).to(device)
		
		print(f"  └─ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
		
		if verbose:
				print(f"\n[3/5] Encoding {len(df)} labels...")
		
		all_labels = df['label'].tolist()
		embeddings = model.encode(
				all_labels,
				batch_size=512,
				show_progress_bar=verbose,
				convert_to_numpy=True,
				normalize_embeddings=True,
		)
		
		print(f"  └─ Embeddings shape: {embeddings.shape}")
		
		# Process clusters
		if verbose:
				print(f"\n[4/5] Computing cluster metrics...")
		
		clusters_data = []
		cluster_ids = sorted(df['cluster'].unique())
		
		for i, cluster_id in enumerate(cluster_ids):
				if verbose and (i + 1) % 50 == 0:
						print(f"  ├─ Processing cluster {i+1}/{len(cluster_ids)}...")
				
				# Get cluster data
				cluster_mask = df['cluster'] == cluster_id
				cluster_df = df[cluster_mask]
				cluster_embeddings = embeddings[cluster_mask]
				
				# Get labels
				labels_list = cluster_df['label'].tolist()
				canonical = cluster_df['canonical_label'].iloc[0]
				
				# Limit labels if specified
				if max_labels_per_cluster and len(labels_list) > max_labels_per_cluster:
						# Keep canonical + top similar labels
						centroid = cluster_embeddings.mean(axis=0, keepdims=True)
						similarities = cosine_similarity(centroid, cluster_embeddings)[0]
						top_indices = np.argsort(similarities)[::-1][:max_labels_per_cluster]
						labels_list = [labels_list[idx] for idx in top_indices]
						cluster_embeddings = cluster_embeddings[top_indices]
				
				# Compute metrics
				metrics = compute_cluster_metrics(cluster_df, cluster_embeddings, cluster_id)
				
				# Build cluster object
				cluster_obj = {
						'id': int(cluster_id),
						'canonical': canonical,
						'size': int(cluster_mask.sum()),  # Original size
						'labels': labels_list,
						'coherence': round(metrics['coherence'], 4),
						'avgDistance': round(metrics['avg_distance'], 4)
				}
				
				clusters_data.append(cluster_obj)
		
		print(f"  └─ Processed {len(clusters_data)} clusters")
		
		# Generate summary statistics
		total_labels = len(df)
		avg_cluster_size = total_labels / len(clusters_data)
		avg_coherence = np.mean([c['coherence'] for c in clusters_data])
		
		# Build final JSON structure
		output_data = {
				'metadata': {
						'total_clusters': len(clusters_data),
						'total_labels': total_labels,
						'avg_cluster_size': round(avg_cluster_size, 2),
						'avg_coherence': round(avg_coherence, 4),
						'model_id': model_id,
						'source_csv': str(Path(csv_path).name)
				},
				'clusters': clusters_data
		}
		
		# Write JSON
		if verbose:
				print(f"\n[5/5] Writing JSON: {output_json}")
		
		with open(output_json, 'w', encoding='utf-8') as f:
				json.dump(output_data, f, indent=2, ensure_ascii=False)
		
		file_size_mb = Path(output_json).stat().st_size / (1024 * 1024)
		
		print(f"  └─ File size: {file_size_mb:.2f} MB")
		
		print(f"\n{'='*60}")
		print(f"✓ SUCCESS")
		print(f"{'='*60}")
		print(f"Summary:")
		print(f"  ├─ Clusters: {len(clusters_data)}")
		print(f"  ├─ Labels: {total_labels}")
		print(f"  ├─ Avg Size: {avg_cluster_size:.1f}")
		print(f"  ├─ Avg Coherence: {avg_coherence:.4f}")
		print(f"  └─ Output: {output_json}")
		print(f"{'='*60}\n")
		
		return output_data


def main():
		parser = argparse.ArgumentParser(
				description="Generate JSON for Cluster Analysis Dashboard",
				formatter_class=argparse.RawDescriptionHelpFormatter,
				epilog="""
Examples:
	# Basic usage
	python generate_cluster_analysis_json.py --csv clusters_semantic_consolidation_agglomerative.csv
	
	# Specify output file
	python generate_cluster_analysis_json.py --csv clusters.csv --output my_analysis.json
	
	# Use different model (must match clustering model!)
	python generate_cluster_analysis_json.py --csv clusters.csv --model sentence-transformers/all-mpnet-base-v2
	
	# Limit labels per cluster (for large datasets)
	python generate_cluster_analysis_json.py --csv clusters.csv --max-labels 100
				"""
		)
		
		parser.add_argument(
				'--csv',
				type=str,
				required=True,
				help='Path to clustering CSV output file'
		)
		
		parser.add_argument(
				'--output',
				type=str,
				default='cluster_analysis.json',
				help='Output JSON file path (default: cluster_analysis.json)'
		)
		
		parser.add_argument(
				'--model',
				type=str,
				default='sentence-transformers/all-MiniLM-L6-v2',
				help='SentenceTransformer model ID (must match clustering model!)'
		)
		
		parser.add_argument(
				'--device',
				type=str,
				default=None,
				help='Device for embeddings (cuda:0, cpu, etc.). Auto-detect if not specified.'
		)
		
		parser.add_argument(
				'--max-labels',
				type=int,
				default=None,
				help='Maximum labels to include per cluster (default: all)'
		)
		
		parser.add_argument(
				'--quiet',
				action='store_true',
				help='Suppress progress output'
		)
		
		args = parser.parse_args()
		
		# Validate input file
		if not Path(args.csv).exists():
				print(f"ERROR: CSV file not found: {args.csv}")
				return 1
		
		try:
				generate_cluster_analysis_json(
						csv_path=args.csv,
						output_json=args.output,
						model_id=args.model,
						device=args.device,
						max_labels_per_cluster=args.max_labels,
						verbose=not args.quiet
				)
				return 0
		except Exception as e:
				print(f"\nERROR: {e}")
				import traceback
				traceback.print_exc()
				return 1


if __name__ == "__main__":
		exit(main())