import torch
import numpy as np
import pandas as pd
from collections import Counter
import warnings
import os
import ast
import json
import time
import multiprocessing
from sklearn.metrics import (
	silhouette_score, 
	davies_bouldin_score, 
	calinski_harabasz_score
)
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Set, Any, Optional, Union, Callable, Iterable
import matplotlib.pyplot as plt
import seaborn as sns

# For large datasets, use fastcluster if available
try:
	import fastcluster
	use_fastcluster = True
	print("[FASTCLUSTER] Using fastcluster for O(n² log n) performance")
except ImportError:
	use_fastcluster = False
	print("[SCIPY] Using scipy (slower for large n)")

cache_directory = {
	"farid": "/home/farid/datasets/models",
	"alijanif": "/scratch/project_2004072/models",
	"ubuntu": "/media/volume/models",
}

# Global variable for worker processes
canonical_labels_global = None

def init_worker_canonical(canonical_dict):
	global canonical_labels_global
	canonical_labels_global = canonical_dict

def parallel_canonical_mapping(labels_str):
	if isinstance(labels_str, str):
		try:
			labels = ast.literal_eval(labels_str)
		except (ValueError, SyntaxError):
			return []
	elif labels_str is None or (isinstance(labels_str, float) and math.isnan(labels_str)):
		return []
	elif isinstance(labels_str, list):
		labels = labels_str
	else:
		return []

	# # Map to canonical labels using global dict
	# return [canonical_labels_global.get(label, label) for label in labels]

	# Map to canonical labels, SKIPPING labels not in dict
	# (these are labels that were removed as problematic)
	canonical_labels_ = []
	for label in labels:
		if label in canonical_labels_global:
			canonical_labels_.append(canonical_labels_global[label])
		# else: label was removed as problematic, skip it
	
	return canonical_labels_

def get_canonical_labels_parallel(
	labels: List[List[str]],
	label_source: str,          # "llm", "vlm", or "multimodal"
	output_dir: str,
	model_id: str,
	num_workers: int = 4,
	nc: int = None,
	verbose: bool = False,
) -> Tuple[List[List[str]], dict]:
	clusters_fname = os.path.join(output_dir, f"{label_source}_clusters.csv")

	if verbose:
		print(f"\n>> Getting {label_source.upper()} Canonical Labels (Parallel Mapping: nw: {num_workers})")
		print(f"Input: {len(labels)} samples")
		print(f"Examples: {labels[:7]}")
		print(f"clusters_fname: {clusters_fname}")

	clustered_df = cluster(
		labels=labels,
		model_id=model_id,
		nc=nc,
		clusters_fname=clusters_fname,
		verbose=verbose,
	)

	if verbose:
		print(f"[{label_source.upper()}]")
		print(f"  ├─ {len(clustered_df)} unique labels ==>> {clustered_df['cluster'].nunique()} clusters")
		print(clustered_df.head(10))

	canonical_map = clustered_df.set_index('label')['canonical'].to_dict()
	print(f"[{label_source.upper()}] canonical_map: {type(canonical_map)} {len(canonical_map)} entries")

	# Parallel mapping
	chunksize  = max(1, len(labels) // (num_workers * 4))  # 4 chunks per worker
	print(
		f"[{label_source.upper()}] Mapping {len(labels):,} samples → canonical labels "
		f"| workers={num_workers} | chunksize={chunksize}"
	)

	t0 = time.time()
	with multiprocessing.Pool(
		processes=num_workers,
		initializer=init_worker_canonical,	# called ONCE per worker
		initargs=(canonical_map,),					# dict sent ONCE per worker
	) as pool:
		mapped_labels = pool.map(
			parallel_canonical_mapping,
			labels,
			chunksize=chunksize,
		)
	elapsed = time.time() - t0

	print(
		f"[{label_source.upper()}] Mapping done in {elapsed:.2f}s "
		f"({len(labels)/elapsed:,.0f} rows/sec)"
	)

	# Post-processing stats
	missing_labels: set = set()
	none_count     = 0
	empty_count    = 0
	for original, mapped in zip(labels, mapped_labels):
		if mapped is None:
			none_count += 1
			continue
		if len(mapped) == 0:
			empty_count += 1
		
		# Collect labels that were dropped (not in canonical_map)
		if original is not None and isinstance(original, list):
			for lbl in original:
				if lbl not in canonical_map:
					missing_labels.add(lbl)
	
	if verbose:
		print(f"\n[{label_source.upper()}] Mapping summary:")
		print(f"   Total samples   : {len(labels):,}")
		print(f"   None (unparseable): {none_count:,}")
		print(f"   Empty after map : {empty_count:,}")
		print(f"   Labels not in canonical map (removed as problematic): {len(missing_labels):,}")
		if missing_labels:
			print(f"   Sample missing  : {list(missing_labels)[:10]}...")
		print("="*100)

	return mapped_labels, canonical_map

def get_canonical_labels(
	labels: List[List[str]],
	label_source: str,  # "llm", "vlm", or "multimodal"
	output_dir: str,
	model_id: str,
	nc: int = None,
	verbose: bool = False,
) -> Tuple[List[List[str]], dict]:

	print(f"\n>> Getting {label_source} Canonical Labels (Sequential Mapping)")

	clusters_fname = os.path.join(output_dir, f"{label_source}_clusters.csv")
	clustered_df = cluster(
		labels=labels,
		model_id=model_id,
		nc=nc,
		clusters_fname=clusters_fname,
		verbose=verbose,
	)

	if verbose:
		print(
			f"[{label_source.upper()}] Clustered into "
			f"{clustered_df['cluster'].nunique()} clusters "
			f"from {len(clustered_df)} unique labels"
		)
		print(clustered_df.head(10))

	canonical_map = clustered_df.set_index('label')['canonical'].to_dict()
	canonical_labels = []
	missing_labels = set()

	for sample_labels in labels:
		if sample_labels is None:
			canonical_labels.append(None)
			continue
		
		if not isinstance(sample_labels, list):
			if isinstance(sample_labels, str):
				try:
					sample_labels = ast.literal_eval(sample_labels)
				except (ValueError, SyntaxError):
					canonical_labels.append(None)
					continue
			else:
				canonical_labels.append(None)
				continue
		
		mapped = []
		for label in sample_labels:
			if label in canonical_map:
				mapped.append(canonical_map[label])
			else:
				missing_labels.add(label)
		
		# Deduplicate while preserving order
		canonical_labels.append(list(dict.fromkeys(mapped)))
	
	if verbose and missing_labels:
		print(
			f"[{label_source.upper()}] {len(missing_labels)} labels removed "
			f"(not in canonical map): {list(missing_labels)[:10]}..."
		)

	return canonical_labels, canonical_map

def remove_problematic_cluster_labels(
		df,
		embeddings,
		low_cohesion_threshold=0.50,
		poor_canonical_threshold=0.60,
		verbose=True
):
		"""
		Remove ALL labels from problematic clusters.
		
		Removes labels from:
			1. Low-cohesion clusters (intra_sim < threshold)
			2. Poor canonical clusters (canonical_rep < threshold)
		
		This is an aggressive but clean approach that eliminates
		problematic labels entirely rather than trying to fix them.
		
		Parameters
		----------
		df : pd.DataFrame
				Clustering results with ['label', 'cluster', 'canonical']
		embeddings : np.ndarray
				Label embeddings (same order as unique labels)
		low_cohesion_threshold : float
				Intra-similarity threshold for low-cohesion detection (default: 0.50)
		poor_canonical_threshold : float
				Canonical representativeness threshold (default: 0.60)
		verbose : bool
				Print detailed statistics
		
		Returns
		-------
		df_clean : pd.DataFrame
				Cleaned clustering with problematic labels removed
		removed_labels : list
				List of removed labels for reference
		"""
		if verbose:
			print("\nREMOVING PROBLEMATIC CLUSTER LABELS")
			print(f"\tLow-cohesion threshold: {low_cohesion_threshold}")
			print(f"\tPoor canonical threshold: {poor_canonical_threshold}")
			print(df.shape, list(df.columns))
			print(df.head(15))
		
		problematic_cluster_ids = set()
		removed_labels = []
		
		# ========================================================================
		# PART 1: Identify Low-Cohesion Clusters
		# ========================================================================
		
		low_cohesion_clusters = []
		
		for cluster_id in df['cluster'].unique():
				cluster_mask = df['cluster'] == cluster_id
				cluster_labels = df[cluster_mask]['label'].tolist()
				cluster_size = len(cluster_labels)
				
				if cluster_size < 2:
					continue
				
				cluster_indices = df[cluster_mask].index.tolist()
				cluster_embeddings = embeddings[cluster_indices]
				
				# Compute intra-cluster similarity
				sim_matrix = cosine_similarity(cluster_embeddings)
				n = len(cluster_embeddings)
				intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
				
				if intra_sim < low_cohesion_threshold:
					low_cohesion_clusters.append(
						{
							'cluster_id': cluster_id,
							'intra_sim': intra_sim,
							'size': cluster_size,
							'labels': cluster_labels
						}
					)
					problematic_cluster_ids.add(cluster_id)
					removed_labels.extend(cluster_labels)
		
		if verbose:
			print(f"\n[LOW COHESION] Found {len(low_cohesion_clusters)} clusters")
			print(f"  Labels to remove: {sum(c['size'] for c in low_cohesion_clusters)}")
			if low_cohesion_clusters:
				print(f"  Examples:")
				for cluster in low_cohesion_clusters[:15]:
					print(f"    Cluster {cluster['cluster_id']}: {cluster['labels']} (sim={cluster['intra_sim']:.4f})")
		
		# ========================================================================
		# PART 2: Identify Poor Canonical Clusters
		# ========================================================================
		
		poor_canonical_clusters = []
		
		for cluster_id in df['cluster'].unique():
				if cluster_id in problematic_cluster_ids:
					continue  # Already marked for removal
				
				cluster_mask = df['cluster'] == cluster_id
				cluster_labels = df[cluster_mask]['label'].tolist()
				cluster_size = len(cluster_labels)
				
				if cluster_size < 2:
					continue
				
				cluster_indices = df[cluster_mask].index.tolist()
				cluster_embeddings = embeddings[cluster_indices]
				
				# Get current canonical
				current_canonical = df[cluster_mask]['canonical'].iloc[0]
				
				# Check if canonical exists in cluster labels
				if current_canonical not in cluster_labels:
					if verbose:
						print(f"[WARNING] Cluster {cluster_id}: canonical '{current_canonical}' not in labels {cluster_labels}")
					# Skip this cluster or mark as problematic
					problematic_cluster_ids.add(cluster_id)
					removed_labels.extend(cluster_labels)
					poor_canonical_clusters.append({
						'cluster_id': cluster_id,
						'canonical': current_canonical,
						'representativeness': 0.0,
						'size': cluster_size,
						'labels': cluster_labels
					})
					continue
				
				canonical_idx = cluster_labels.index(current_canonical)
				canonical_emb = cluster_embeddings[canonical_idx].reshape(1, -1)
				
				# Compute canonical representativeness
				canonical_rep = cosine_similarity(canonical_emb, cluster_embeddings).mean()
				
				if canonical_rep < poor_canonical_threshold:
						poor_canonical_clusters.append({
								'cluster_id': cluster_id,
								'canonical': current_canonical,
								'representativeness': canonical_rep,
								'size': cluster_size,
								'labels': cluster_labels
						})
						problematic_cluster_ids.add(cluster_id)
						removed_labels.extend(cluster_labels)
		
		if verbose:
				print(f"\n[POOR CANONICAL] Found {len(poor_canonical_clusters)} clusters")
				print(f"  Labels to remove: {sum(c['size'] for c in poor_canonical_clusters)}")
				if poor_canonical_clusters:
						print(f"  Examples:")
						for cluster in poor_canonical_clusters[:5]:
								print(f"    Cluster {cluster['cluster_id']}: {cluster['labels']} (rep={cluster['representativeness']:.4f})")
		
		# PART 3: Remove Problematic Labels		
		if verbose:
			print(f"\n[REMOVAL SUMMARY]")
			print(f"  Total problematic clusters: {len(problematic_cluster_ids)}")
			print(f"  Total labels to remove: {len(removed_labels)}")
			print(f"  Percentage of labels: {len(removed_labels)/len(df)*100:.2f}%")
		
		# Remove labels from problematic clusters
		df_clean = df[~df['cluster'].isin(problematic_cluster_ids)].copy()

		kept_indices = df_clean.index.tolist()
		embeddings_clean = embeddings[kept_indices]
		
		# Re-index cluster IDs to be contiguous
		unique_clusters = sorted(df_clean['cluster'].unique())
		cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
		df_clean['cluster'] = df_clean['cluster'].map(cluster_mapping)

		# Reset index so df_clean indices are 0, 1, 2, ... n-1
		df_clean = df_clean.reset_index(drop=True)

		if verbose:
			print(f"\n[RESULTS]")
			print(f"  df original: {df.shape}")
			print(f"  df clean: {df_clean.shape}")
			print(f"  Removed labels: {len(df) - len(df_clean):,}")
			print(f"  Original embeddings: {embeddings.shape}")
			print(f"  Cleaned embeddings: {embeddings_clean.shape}")

			print(f"  Original clusters: {df['cluster'].nunique():,}")
			print(f"  Cleaned clusters: {df_clean['cluster'].nunique():,}")
			print(f"  Removed clusters: {df['cluster'].nunique() - df_clean['cluster'].nunique():,}")
			
			# Consolidation stats
			original_consolidation = len(df) / df['cluster'].nunique()
			new_consolidation = len(df_clean) / df_clean['cluster'].nunique()
			print(f"\n  Original consolidation: {original_consolidation:.2f}x")
			print(f"  New consolidation: {new_consolidation:.2f}x")
			print(f"  Change: {(new_consolidation - original_consolidation):.2f}x")
			
			print(f"\n{len(removed_labels)} problematic labels removed!")
			print("="*80)
		
		return df_clean, embeddings_clean, removed_labels

def dissolve_low_cohesion_clusters(
		df,
		embeddings,
		threshold=0.5,
		verbose=True
):
		"""
		Dissolve clusters with intra-similarity < threshold.
		Ensures cluster IDs remain contiguous (0, 1, 2, ..., n-1).
		"""
		from sklearn.metrics.pairwise import cosine_similarity
		import numpy as np
		
		if verbose:
				print(f"\n[DISSOLUTION] Analyzing clusters...")
				print(f"  Threshold: {threshold}")
		
		clusters_to_dissolve = []
		
		# Find low cohesion clusters
		for cluster_id in df['cluster'].unique():
				cluster_mask = df['cluster'] == cluster_id
				cluster_size = cluster_mask.sum()
				
				if cluster_size < 2:
						continue
				
				cluster_labels = df[cluster_mask]['label'].tolist()
				cluster_indices = df[cluster_mask].index.tolist()
				cluster_embeddings = embeddings[cluster_indices]
				
				sim_matrix = cosine_similarity(cluster_embeddings)
				n = len(cluster_embeddings)
				intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
				
				if intra_sim < threshold:
						clusters_to_dissolve.append({
								'cluster_id': cluster_id,
								'size': cluster_size,
								'intra_sim': intra_sim,
								'labels': cluster_labels
						})
		
		if verbose:
				print(f"\n[DISSOLUTION] Found {len(clusters_to_dissolve)} low cohesion clusters")
				print(f"  Total labels affected: {sum(c['size'] for c in clusters_to_dissolve)}")
		
		if len(clusters_to_dissolve) == 0:
				print("\n✅ No low cohesion clusters found. Nothing to dissolve.")
				return df
		
		# Get next available cluster ID
		max_cluster_id = df['cluster'].max()
		next_cluster_id = max_cluster_id + 1
		
		if verbose:
				print(f"\n[DISSOLVING] Reassigning labels to new clusters...")
		
		# Dissolve each low cohesion cluster
		for cluster_info in clusters_to_dissolve:
				cluster_id = cluster_info['cluster_id']
				cluster_mask = df['cluster'] == cluster_id
				
				for idx in df[cluster_mask].index:
						label_name = df.loc[idx, 'label']
						df.loc[idx, 'cluster'] = next_cluster_id
						df.loc[idx, 'canonical'] = label_name
						next_cluster_id += 1
		
		if verbose:
				print(f"\n[RE-INDEXING] Making cluster IDs contiguous...")
		
		unique_clusters = sorted(df['cluster'].unique())
		cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
		
		df['cluster'] = df['cluster'].map(cluster_mapping)
		
		# Compute statistics
		old_n_clusters = max_cluster_id + 1
		new_n_clusters = df['cluster'].nunique()
		old_consolidation = len(df) / old_n_clusters
		new_consolidation = len(df) / new_n_clusters
		
		if verbose:
				print(f"\n[RESULTS]")
				print(f"  Old clusters: {old_n_clusters:,}")
				print(f"  New clusters: {new_n_clusters:,}")
				print(f"  Change: +{new_n_clusters - old_n_clusters:,}")
				print(f"  Old consolidation: {old_consolidation:.2f}x")
				print(f"  New consolidation: {new_consolidation:.2f}x")
				print(f"  Cluster ID range: 0 to {df['cluster'].max()} (contiguous: {df['cluster'].max() == new_n_clusters - 1})")
				print(f"\n✅ Dissolution complete!")
		
		return df

def fix_poor_canonical_clusters(
	df,
	embeddings,
	threshold=0.60,
	verbose=True
):
	print(f"\n[FIX POOR CANONICAL] Threshold: {threshold}")
	
	# Identify poor canonical clusters
	poor_canonical_clusters = []
	
	for cluster_id in df['cluster'].unique():
		cluster_mask = df['cluster'] == cluster_id
		cluster_labels = df[cluster_mask]['label'].tolist()
		cluster_size = len(cluster_labels)
		
		if cluster_size < 2:
			continue
		
		cluster_indices = df[cluster_mask].index.tolist()
		cluster_embeddings = embeddings[cluster_indices]
		
		# Get current canonical
		current_canonical = df[cluster_mask]['canonical'].iloc[0]
		canonical_idx = cluster_labels.index(current_canonical)
		canonical_emb = cluster_embeddings[canonical_idx].reshape(1, -1)
		
		# Compute representativeness (avg similarity to all members)
		canonical_rep = cosine_similarity(canonical_emb, cluster_embeddings).mean()
		
		if canonical_rep < threshold:
			poor_canonical_clusters.append(
				{
					'cluster_id': cluster_id,
					'current_canonical': current_canonical,
					'representativeness': canonical_rep,
					'size': cluster_size,
					'labels': cluster_labels
				}
			)
	
	if verbose:
		print(f"  Found {len(poor_canonical_clusters)} clusters with poor canonical")
	
	if len(poor_canonical_clusters) == 0:
		print("  ✓ All canonical labels are representative!")
		return df
	
	# Re-select canonical for each poor cluster
	fixed_count = 0
	
	for cluster_info in poor_canonical_clusters:
		cluster_id = cluster_info['cluster_id']
		cluster_labels = cluster_info['labels']
		old_canonical = cluster_info['current_canonical']
		
		cluster_mask = df['cluster'] == cluster_id
		cluster_indices = df[cluster_mask].index.tolist()
		cluster_embeddings = embeddings[cluster_indices]
		
		# Method 1: Centroid-nearest (most representative)
		centroid = cluster_embeddings.mean(axis=0, keepdims=True)
		similarities = cosine_similarity(centroid, cluster_embeddings)[0]
		best_idx = similarities.argmax()
		new_canonical = cluster_labels[best_idx]
		new_rep = similarities[best_idx]
		
		# Update canonical in dataframe
		df.loc[cluster_mask, 'canonical'] = new_canonical
		
		fixed_count += 1
		
		if verbose:
			print(f"\nCluster {cluster_id} ({len(cluster_labels)} labels):")
			print(f"\tOld: '{old_canonical}' (rep={cluster_info['representativeness']:.4f})")
			print(f"\tNew: '{new_canonical}' (rep={new_rep:.4f})")
			print(f"\tImprovement: {(new_rep - cluster_info['representativeness']):.4f}")
			print(f"\tLabels: {cluster_labels}")
	
	if verbose:
		print(f"\n✓ Fixed {fixed_count} clusters")
	
	return df

def automated_cluster_validation(
	embeddings: np.ndarray,
	labels: np.ndarray,
	cluster_assignments: np.ndarray,
	canonical_labels: Dict[int, str],
	original_label_counts: Optional[Dict[str, int]] = None,
	verbose: bool = True
) -> Dict:
		"""
		Fully automated clustering quality assessment with ZERO human intervention.
		
		Evaluates two critical aspects:
		1. Cluster Quality: Cohesion, separation, size distribution
		2. Canonical Label Quality: Representativeness, generality, frequency alignment
		
		Parameters
		----------
		embeddings : np.ndarray, shape (n_samples, embedding_dim)
				Normalized embeddings of all unique labels
		labels : np.ndarray, shape (n_samples,)
				Original label strings
		cluster_assignments : np.ndarray, shape (n_samples,)
				Cluster ID for each label
		canonical_labels : Dict[int, str]
				Mapping from cluster_id -> canonical label string
		original_label_counts : Dict[str, int], optional
				Frequency of each label in original dataset
		verbose : bool, default=True
				Print detailed analysis
		
		Returns
		-------
		results : Dict
				{
						'cluster_quality': Dict with automated metrics,
						'canonical_quality': Dict with canonical assessment,
						'recommendations': List[str] with actionable insights,
						'overall_score': float in [0, 1] indicating quality,
						'pass_threshold': bool indicating if clustering is acceptable
				}
		"""
		
		n_samples = len(labels)
		n_clusters = len(np.unique(cluster_assignments))
		
		if verbose:
			print("\nAUTOMATED CLUSTER VALIDATION (ZERO HUMAN INTERVENTION)")
			print(f"Dataset: {n_samples:,} unique labels → {n_clusters:,} clusters")
		
		# Create DataFrame for analysis
		df = pd.DataFrame(
			{
				'label': labels,
				'cluster': cluster_assignments
			}
		)
		df['canonical'] = df['cluster'].map(canonical_labels)
		
		results = {}
		
		# =========================================================================
		# PART 1: CLUSTER QUALITY METRICS (Automated)
		# =========================================================================
		if verbose:
			print("\n[PART 1/2] Automated Cluster Quality Assessment")
		
		cluster_quality = {}
		
		# Metric 1.1: Intra-cluster cohesion (higher is better)
		intra_similarities = []
		for cid in range(n_clusters):
				cluster_mask = cluster_assignments == cid
				cluster_embeddings = embeddings[cluster_mask]
				
				if len(cluster_embeddings) > 1:
						sim_matrix = cosine_similarity(cluster_embeddings)
						# Average pairwise similarity (excluding diagonal)
						intra_sim = (sim_matrix.sum() - len(cluster_embeddings)) / (
								len(cluster_embeddings) * (len(cluster_embeddings) - 1)
						)
						intra_similarities.append(intra_sim)
		
		cluster_quality['mean_intra_similarity'] = np.mean(intra_similarities)
		cluster_quality['min_intra_similarity'] = np.min(intra_similarities)
		cluster_quality['std_intra_similarity'] = np.std(intra_similarities)
		
		# Metric 1.2: Inter-cluster separation (higher is better)
		# Compute centroid for each cluster
		cluster_centroids = []
		for cid in range(n_clusters):
				cluster_mask = cluster_assignments == cid
				cluster_embeddings = embeddings[cluster_mask]
				centroid = cluster_embeddings.mean(axis=0)
				cluster_centroids.append(centroid)
		
		cluster_centroids = np.array(cluster_centroids)
		
		# Average pairwise distance between centroids
		if n_clusters > 1:
				centroid_distances = 1 - cosine_similarity(cluster_centroids)
				np.fill_diagonal(centroid_distances, 0)
				inter_cluster_distance = centroid_distances.sum() / (n_clusters * (n_clusters - 1))
				cluster_quality['mean_inter_cluster_distance'] = inter_cluster_distance
		else:
				cluster_quality['mean_inter_cluster_distance'] = 1.0
		
		# Metric 1.3: Dunn Index (ratio of min inter-cluster to max intra-cluster)
		# Higher is better (well-separated, compact clusters)
		if n_clusters > 1:
				min_inter_distance = centroid_distances[centroid_distances > 0].min()
				
				# Max intra-cluster diameter
				max_diameter = 0
				for cid in range(n_clusters):
						cluster_mask = cluster_assignments == cid
						cluster_embeddings = embeddings[cluster_mask]
						if len(cluster_embeddings) > 1:
								pairwise_dists = 1 - cosine_similarity(cluster_embeddings)
								max_diameter = max(max_diameter, pairwise_dists.max())
				
				dunn_index = min_inter_distance / (max_diameter + 1e-10)
				cluster_quality['dunn_index'] = dunn_index
		else:
				cluster_quality['dunn_index'] = 1.0
		
		# Metric 1.4: Cluster size distribution quality
		cluster_sizes = df.groupby('cluster').size().values
		cluster_quality['size_mean'] = cluster_sizes.mean()
		cluster_quality['size_median'] = np.median(cluster_sizes)
		cluster_quality['size_std'] = cluster_sizes.std()
		cluster_quality['size_cv'] = cluster_sizes.std() / (cluster_sizes.mean() + 1e-10)  # Coefficient of variation
		cluster_quality['size_gini'] = _compute_gini(cluster_sizes)  # 0=equal, 1=unequal
		
		# Metric 1.5: Outlier cluster detection (clusters with very low cohesion)
		low_cohesion_threshold = 0.5
		n_low_cohesion = sum(1 for sim in intra_similarities if sim < low_cohesion_threshold)
		cluster_quality['n_low_cohesion_clusters'] = n_low_cohesion
		cluster_quality['pct_low_cohesion'] = n_low_cohesion / n_clusters
		
		# Metric 1.6: Silhouette coefficient (computed earlier, but add for completeness)
		with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				silhouette = silhouette_score(embeddings, cluster_assignments, metric='cosine')
		cluster_quality['silhouette_score'] = silhouette
		
		if verbose:
				print(f"\tIntra-cluster cohesion:")
				print(f"\t\tMean: {cluster_quality['mean_intra_similarity']:.4f}")
				print(f"\t\tMin:  {cluster_quality['min_intra_similarity']:.4f}")
				print(f"\t\tStd:  {cluster_quality['std_intra_similarity']:.4f}")
				print(f"\tInter-cluster separation: {cluster_quality['mean_inter_cluster_distance']:.4f}")
				print(f"\tDunn Index: {cluster_quality['dunn_index']:.4f}")
				print(f"\tSilhouette Score: {cluster_quality['silhouette_score']:.4f}")
				print(f"\tSize distribution:")
				print(f"\t\tMean: {cluster_quality['size_mean']:.1f}")
				print(f"\t\tCV: {cluster_quality['size_cv']:.3f} (lower is more balanced)")
				print(f"\t\tGini: {cluster_quality['size_gini']:.3f} (lower is more equal)")
				print(f"\tLow cohesion clusters: {n_low_cohesion} ({cluster_quality['pct_low_cohesion']*100:.1f}%)")
				print()
		
		results['cluster_quality'] = cluster_quality
		
		# PART 2: CANONICAL LABEL QUALITY METRICS (Automated)
		if verbose:
			print("\n[PART 2/2] Automated Canonical Label Quality Assessment")
		
		canonical_quality = {}
		
		# Metric 2.1: Canonical representativeness (how well canonical represents cluster)
		canonical_representativeness_scores = []
		
		for cid in range(n_clusters):
				cluster_mask = cluster_assignments == cid
				cluster_labels_list = labels[cluster_mask]
				cluster_embeddings = embeddings[cluster_mask]
				canonical = canonical_labels[cid]
				
				# Find canonical's embedding
				canonical_idx = np.where(cluster_labels_list == canonical)[0]
				if len(canonical_idx) > 0:
						canonical_emb = cluster_embeddings[canonical_idx[0]].reshape(1, -1)
						# Average similarity to all cluster members
						similarities = cosine_similarity(canonical_emb, cluster_embeddings)[0]
						avg_similarity = similarities.mean()
						canonical_representativeness_scores.append(avg_similarity)
				else:
						# Canonical not in cluster (shouldn't happen)
						canonical_representativeness_scores.append(0.0)
		
		canonical_quality['mean_representativeness'] = np.mean(canonical_representativeness_scores)
		canonical_quality['min_representativeness'] = np.min(canonical_representativeness_scores)
		canonical_quality['std_representativeness'] = np.std(canonical_representativeness_scores)
		
		# Metric 2.2: Canonical generality (shorter labels are usually more general)
		canonical_lengths = [len(canonical_labels[cid].split()) for cid in range(n_clusters)]
		cluster_avg_lengths = []
		
		for cid in range(n_clusters):
				cluster_mask = cluster_assignments == cid
				cluster_labels_list = labels[cluster_mask]
				avg_length = np.mean([len(lbl.split()) for lbl in cluster_labels_list])
				cluster_avg_lengths.append(avg_length)
		
		# Canonical should be shorter than or equal to cluster average (more general)
		canonical_generality_scores = []
		for can_len, cluster_len in zip(canonical_lengths, cluster_avg_lengths):
				# Score: 1.0 if canonical is shorter, decreases if longer
				generality = max(0, 1 - (can_len - cluster_len) / (cluster_len + 1e-10))
				canonical_generality_scores.append(generality)
		
		canonical_quality['mean_generality_score'] = np.mean(canonical_generality_scores)
		canonical_quality['canonical_avg_length'] = np.mean(canonical_lengths)
		canonical_quality['cluster_avg_length'] = np.mean(cluster_avg_lengths)
		
		# Metric 2.3: Frequency alignment (if label counts available)
		if original_label_counts:
				frequency_alignment_scores = []
				
				for cid in range(n_clusters):
						cluster_mask = cluster_assignments == cid
						cluster_labels_list = labels[cluster_mask]
						canonical = canonical_labels[cid]
						
						# Get frequency of canonical vs cluster average
						canonical_freq = original_label_counts.get(canonical, 0)
						cluster_freqs = [original_label_counts.get(lbl, 0) for lbl in cluster_labels_list]
						cluster_mean_freq = np.mean(cluster_freqs)
						cluster_max_freq = np.max(cluster_freqs)
						
						# Canonical should be among the most frequent in cluster
						# Score: ratio of canonical frequency to max frequency
						if cluster_max_freq > 0:
								freq_alignment = canonical_freq / cluster_max_freq
						else:
								freq_alignment = 1.0
						
						frequency_alignment_scores.append(freq_alignment)
				
				canonical_quality['mean_frequency_alignment'] = np.mean(frequency_alignment_scores)
				canonical_quality['pct_canonical_is_most_frequent'] = sum(
						1 for score in frequency_alignment_scores if score >= 0.99
				) / n_clusters
		else:
				canonical_quality['mean_frequency_alignment'] = None
				canonical_quality['pct_canonical_is_most_frequent'] = None
		
		# Metric 2.4: Canonical uniqueness (each canonical should be unique)
		canonical_values = list(canonical_labels.values())
		n_unique_canonicals = len(set(canonical_values))
		canonical_quality['n_unique_canonicals'] = n_unique_canonicals
		canonical_quality['canonical_uniqueness_ratio'] = n_unique_canonicals / n_clusters
		
		# Metric 2.5: Centroid vs alternatives comparison
		# How much better is centroid-nearest vs random selection?
		centroid_improvement_scores = []
		
		for cid in range(n_clusters):
				cluster_mask = cluster_assignments == cid
				cluster_labels_list = labels[cluster_mask]
				cluster_embeddings = embeddings[cluster_mask]
				canonical = canonical_labels[cid]
				
				if len(cluster_embeddings) < 2:
						continue
				
				# Current canonical representativeness
				canonical_idx = np.where(cluster_labels_list == canonical)[0]
				if len(canonical_idx) > 0:
						canonical_emb = cluster_embeddings[canonical_idx[0]].reshape(1, -1)
						current_score = cosine_similarity(canonical_emb, cluster_embeddings).mean()
				else:
						current_score = 0.0
				
				# Random baseline: average representativeness of random labels
				random_scores = []
				for emb in cluster_embeddings[:min(10, len(cluster_embeddings))]:  # Sample 10
						emb_reshaped = emb.reshape(1, -1)
						random_score = cosine_similarity(emb_reshaped, cluster_embeddings).mean()
						random_scores.append(random_score)
				
				random_baseline = np.mean(random_scores)
				
				# Improvement over random
				if random_baseline > 0:
						improvement = (current_score - random_baseline) / random_baseline
				else:
						improvement = 0.0
				
				centroid_improvement_scores.append(improvement)
		
		canonical_quality['mean_centroid_improvement'] = np.mean(centroid_improvement_scores) if centroid_improvement_scores else 0.0
		
		if verbose:
				print(f"  ✓ Canonical representativeness:")
				print(f"      Mean: {canonical_quality['mean_representativeness']:.4f}")
				print(f"      Min:  {canonical_quality['min_representativeness']:.4f}")
				print(f"      Std:  {canonical_quality['std_representativeness']:.4f}")
				print(f"  ✓ Canonical generality:")
				print(f"      Mean generality score: {canonical_quality['mean_generality_score']:.4f}")
				print(f"      Canonical avg length: {canonical_quality['canonical_avg_length']:.2f} words")
				print(f"      Cluster avg length: {canonical_quality['cluster_avg_length']:.2f} words")
				if canonical_quality['mean_frequency_alignment'] is not None:
						print(f"  ✓ Frequency alignment:")
						print(f"      Mean alignment: {canonical_quality['mean_frequency_alignment']:.4f}")
						print(f"      % canonical is most frequent: {canonical_quality['pct_canonical_is_most_frequent']*100:.1f}%")
				print(f"  ✓ Canonical uniqueness: {n_unique_canonicals}/{n_clusters} ({canonical_quality['canonical_uniqueness_ratio']*100:.1f}%)")
				print(f"  ✓ Centroid improvement over random: {canonical_quality['mean_centroid_improvement']*100:.1f}%")
				print()
		
		results['canonical_quality'] = canonical_quality
		
		# PART 3: OVERALL QUALITY SCORE & AUTOMATED DECISION
		if verbose:
			print("\n[PART 3/3] Overall Quality Score & Automated Decision")
		
		# Compute weighted overall score (0-1 scale)
		scores = {}
		
		# Cluster quality sub-scores
		scores['cohesion'] = cluster_quality['mean_intra_similarity']  # 0-1
		scores['separation'] = cluster_quality['mean_inter_cluster_distance']  # 0-1
		scores['dunn'] = min(cluster_quality['dunn_index'] / 0.5, 1.0)  # Normalize to 0-1
		scores['silhouette'] = (cluster_quality['silhouette_score'] + 1) / 2  # Convert [-1,1] to [0,1]
		scores['size_balance'] = 1 - min(cluster_quality['size_gini'], 1.0)  # Invert Gini
		scores['no_outliers'] = 1 - cluster_quality['pct_low_cohesion']
		
		# Canonical quality sub-scores
		scores['representativeness'] = canonical_quality['mean_representativeness']
		scores['generality'] = canonical_quality['mean_generality_score']
		scores['uniqueness'] = canonical_quality['canonical_uniqueness_ratio']
		scores['centroid_method'] = min(canonical_quality['mean_centroid_improvement'] / 0.2, 1.0)  # 20% improvement = score 1.0
		
		if canonical_quality['mean_frequency_alignment'] is not None:
				scores['frequency'] = canonical_quality['mean_frequency_alignment']
		
		# Weighted overall score
		weights = {
				'cohesion': 0.20,
				'separation': 0.10,
				'dunn': 0.10,
				'silhouette': 0.10,
				'size_balance': 0.05,
				'no_outliers': 0.05,
				'representativeness': 0.20,
				'generality': 0.10,
				'uniqueness': 0.05,
				'centroid_method': 0.05,
		}
		
		if 'frequency' in scores:
				weights['frequency'] = 0.10
				# Renormalize other weights
				total_weight = sum(weights.values())
				for key in weights:
						weights[key] /= total_weight
		
		overall_score = sum(scores[key] * weights[key] for key in scores)
		
		# Quality thresholds
		excellent_threshold = 0.75
		good_threshold = 0.65
		acceptable_threshold = 0.55
		
		if overall_score >= excellent_threshold:
				quality_level = "EXCELLENT"
				decision = "PROCEED"
		elif overall_score >= good_threshold:
				quality_level = "GOOD"
				decision = "PROCEED"
		elif overall_score >= acceptable_threshold:
				quality_level = "ACCEPTABLE"
				decision = "PROCEED WITH CAUTION"
		else:
				quality_level = "POOR"
				decision = "RE-CLUSTER RECOMMENDED"
		
		results['overall_score'] = overall_score
		results['quality_level'] = quality_level
		results['decision'] = decision
		results['pass_threshold'] = overall_score >= acceptable_threshold
		results['component_scores'] = scores
		results['component_weights'] = weights
		
		if verbose:
				print(f"  Component Scores:")
				for key, score in scores.items():
						weight_pct = weights[key] * 100
						weighted_contribution = score * weights[key]
						print(f"    {key:20s}: {score:.3f} (weight: {weight_pct:4.1f}%, contrib: {weighted_contribution:.3f})")
				print()
				print(f"  🎯 OVERALL QUALITY SCORE: {overall_score:.3f} / 1.000")
				print(f"  📊 QUALITY LEVEL: {quality_level}")
				print(f"  ✅ AUTOMATED DECISION: {decision}")
				print()
		
		# PART 4: AUTOMATED RECOMMENDATIONS
		recommendations = []
		
		# Issue 1: Low cluster cohesion
		if cluster_quality['mean_intra_similarity'] < 0.70:
				recommendations.append({
						'issue': 'Low Cluster Cohesion',
						'metric': f"Mean intra-similarity = {cluster_quality['mean_intra_similarity']:.3f} (target: >0.70)",
						'severity': 'HIGH',
						'action': f"Increase n_clusters to {int(n_clusters * 1.5)}-{int(n_clusters * 2)} for tighter clusters"
				})
		
		# Issue 2: Poor inter-cluster separation
		if cluster_quality['mean_inter_cluster_distance'] < 0.20:
				recommendations.append({
						'issue': 'Poor Inter-Cluster Separation',
						'metric': f"Mean inter-distance = {cluster_quality['mean_inter_cluster_distance']:.3f} (target: >0.20)",
						'severity': 'MEDIUM',
						'action': "Consider switching to 'average' linkage method for better separation"
				})
		
		# Issue 3: Unbalanced cluster sizes
		if cluster_quality['size_gini'] > 0.60:
				recommendations.append({
						'issue': 'Highly Unbalanced Cluster Sizes',
						'metric': f"Gini coefficient = {cluster_quality['size_gini']:.3f} (target: <0.60)",
						'severity': 'MEDIUM',
						'action': "Enable split_oversized=True to break up large clusters"
				})
		
		# Issue 4: Many outlier clusters
		if cluster_quality['pct_low_cohesion'] > 0.10:
				recommendations.append({
						'issue': 'Many Outlier Clusters',
						'metric': f"{cluster_quality['pct_low_cohesion']*100:.1f}% of clusters have cohesion <0.5",
						'severity': 'HIGH',
						'action': "Review low-cohesion clusters; may need manual splitting or different distance metric"
				})
		

		# Issue 5: Poor canonical representativeness
		if canonical_quality['mean_representativeness'] < 0.75:
			recommendations.append(
				{
					'issue': 'Poor Canonical Representativeness',
					'metric': f"Mean representativeness = {canonical_quality['mean_representativeness']:.3f} (target: >0.75)",
					'severity': 'HIGH',
					'action': "Switch to frequency-weighted centroid selection or alternative canonical selection method"
				}
			)
		
		# Issue 6: Canonicals too specific (too long)
		if canonical_quality['canonical_avg_length'] > canonical_quality['cluster_avg_length'] * 1.2:
				recommendations.append({
						'issue': 'Canonicals Too Specific',
						'metric': f"Canonical length ({canonical_quality['canonical_avg_length']:.2f}) > cluster avg ({canonical_quality['cluster_avg_length']:.2f})",
						'severity': 'MEDIUM',
						'action': "Prefer shorter labels in canonical selection; add length penalty to selection criteria"
				})
		
		# Issue 7: Low frequency alignment
		if canonical_quality.get('mean_frequency_alignment') and canonical_quality['mean_frequency_alignment'] < 0.50:
				recommendations.append({
						'issue': 'Low Frequency Alignment',
						'metric': f"Mean frequency alignment = {canonical_quality['mean_frequency_alignment']:.3f} (target: >0.50)",
						'severity': 'MEDIUM',
						'action': "Weight canonical selection by label frequency to prefer common terms"
				})
		
		# Issue 8: Centroid method not improving over random
		if canonical_quality['mean_centroid_improvement'] < 0.05:
				recommendations.append({
						'issue': 'Centroid Method Not Effective',
						'metric': f"Improvement over random = {canonical_quality['mean_centroid_improvement']*100:.1f}% (target: >5%)",
						'severity': 'HIGH',
						'action': "Consider alternative canonical selection: medoid, frequency-based, or length-weighted"
				})
		
		# Sort by severity
		severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
		recommendations.sort(key=lambda x: severity_order[x['severity']])
		
		results['recommendations'] = recommendations
		
		if verbose:
			print(f"\nAUTOMATED RECOMMENDATIONS:")
			if not recommendations:
				print(f"[OK] No critical issues detected. Clustering quality is acceptable.")
			else:
				for i, rec in enumerate(recommendations, 1):
					print(f"\n    [{i}] {rec['severity']:6s} | {rec['issue']}")
					print(f"        Metric: {rec['metric']}")
					print(f"        Action: {rec['action']}")
			print()
		
		summary = f"""
			AUTOMATED VALIDATION SUMMARY
		
			OVERALL QUALITY: {overall_score:.3f} / 1.000 ({quality_level})
			DECISION: {decision}

			CLUSTER QUALITY:
				• Cohesion (intra-similarity): {cluster_quality['mean_intra_similarity']:.3f}
				• Separation (inter-distance): {cluster_quality['mean_inter_cluster_distance']:.3f}
				• Dunn Index: {cluster_quality['dunn_index']:.3f}
				• Silhouette: {cluster_quality['silhouette_score']:.3f}
				• Size balance (1-Gini): {1-cluster_quality['size_gini']:.3f}
				• Low cohesion clusters: {cluster_quality['n_low_cohesion_clusters']} ({cluster_quality['pct_low_cohesion']*100:.1f}%)

			CANONICAL QUALITY:
				• Representativeness: {canonical_quality['mean_representativeness']:.3f}
				• Generality score: {canonical_quality['mean_generality_score']:.3f}
				• Uniqueness: {canonical_quality['canonical_uniqueness_ratio']*100:.1f}%
				• Centroid improvement: {canonical_quality['mean_centroid_improvement']*100:.1f}%
				{'• Frequency alignment: ' + f"{canonical_quality['mean_frequency_alignment']:.3f}" if canonical_quality['mean_frequency_alignment'] else ''}

			ISSUES DETECTED: {len(recommendations)}
			{'  • ' + recommendations[0]['issue'] if recommendations else '  • None'}

			RECOMMENDATION: {decision}
		"""
		
		results['summary'] = summary
		
		if verbose:
			print(summary)
		
		return results

def _compute_gini(values):
		"""
		Compute Gini coefficient for measuring inequality.
		0 = perfect equality, 1 = perfect inequality
		"""
		values = np.array(values, dtype=float)
		n = len(values)
		
		if n == 0:
				return 0.0
		
		# Sort values
		sorted_values = np.sort(values)
		
		# Compute Gini
		index = np.arange(1, n + 1)
		gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
		
		return gini

def analyze_cluster_quality(
	embeddings: np.ndarray,
	labels: np.ndarray,
	cluster_assignments: np.ndarray,
	canonical_labels: Dict[int, str],
	original_label_counts: Optional[Dict[str, int]] = None,
	distance_metric: str = 'cosine',
	output_dir: str = "./",
	verbose: bool = True
) -> Dict:

	n_emb = len(embeddings)
	n_labels = len(labels)
	n_assign = len(cluster_assignments)
	if not (n_emb == n_labels == n_assign):
		raise ValueError(
			f"Input size mismatch: embeddings ({n_emb}), labels ({n_labels}), "
			f"and cluster_assignments ({n_assign}) must have the same length."
		)	

	# ========== Now proceed with validated data ==========
	n_samples = len(labels)
	n_clusters = len(np.unique(cluster_assignments))
	
	if n_clusters < 2:
		raise ValueError(f"Need at least 2 clusters for quality analysis, got {n_clusters}")
	
	if verbose:
		print(f"\nANALYZING CLUSTER QUALITY: {n_samples} samples → {n_clusters} clusters (reduction ratio: {n_samples/n_clusters:.2f}x)")
		

	# 1. GLOBAL CLUSTERING METRICS
	if verbose:
		print("\n[1/6] Computing Global Clustering Metrics...")
	
	global_metrics = {}
	
	# Silhouette Score (higher is better, range [-1, 1])
	# Measures how similar objects are to their own cluster vs other clusters
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		if distance_metric == 'cosine':
			distances = cosine_distances(embeddings)
			silhouette = silhouette_score(distances, cluster_assignments, metric='precomputed')
		else:
			silhouette = silhouette_score(embeddings, cluster_assignments, metric='euclidean')
	
	global_metrics['silhouette_score'] = silhouette
	global_metrics['silhouette_interpretation'] = _interpret_silhouette(silhouette)
	
	# Davies-Bouldin Index (lower is better, range [0, ∞))
	# Ratio of within-cluster to between-cluster distances
	db_index = davies_bouldin_score(embeddings, cluster_assignments)
	global_metrics['davies_bouldin_index'] = db_index
	global_metrics['db_interpretation'] = _interpret_db_index(db_index)
	
	# Calinski-Harabasz Index (higher is better, range [0, ∞))
	# Ratio of between-cluster to within-cluster variance
	ch_index = calinski_harabasz_score(embeddings, cluster_assignments)
	global_metrics['calinski_harabasz_index'] = ch_index
	global_metrics['ch_interpretation'] = _interpret_ch_index(ch_index)
	
	if verbose:
		print(f"Silhouette Score:        {silhouette:<15.4f}{global_metrics['silhouette_interpretation']}")
		print(f"Davies-Bouldin Index:    {db_index:<15.4f}{global_metrics['db_interpretation']}")
		print(f"Calinski-Harabasz Index: {ch_index:<15.4f}{global_metrics['ch_interpretation']}")
	
	# 2. PER-CLUSTER QUALITY METRICS
	if verbose:
		print("\n[2/6] Analyzing Per-Cluster Quality...")
	
	cluster_metrics_list = []
	
	for cluster_id in range(n_clusters):
			mask = cluster_assignments == cluster_id
			cluster_embeddings = embeddings[mask]
			cluster_labels = labels[mask]
			cluster_size = len(cluster_labels)
			
			# Skip empty clusters (shouldn't happen but safety check)
			if cluster_size == 0:
				continue
			
			# Canonical label for this cluster
			canonical = canonical_labels.get(cluster_id, "UNKNOWN")
			
			# Intra-cluster cohesion (average cosine similarity within cluster)
			if cluster_size > 1:
					intra_sim = cosine_similarity(cluster_embeddings).mean()
					# Exclude diagonal (self-similarity = 1.0)
					intra_sim_no_diag = (cosine_similarity(cluster_embeddings).sum() - cluster_size) / (cluster_size * (cluster_size - 1))
			else:
					intra_sim = 1.0
					intra_sim_no_diag = 1.0
			
			# Canonical representativeness (how well canonical represents the cluster)
			canonical_idx = np.where(cluster_labels == canonical)[0]
			if len(canonical_idx) > 0:
					canonical_emb = cluster_embeddings[canonical_idx[0]].reshape(1, -1)
					canonical_sim = cosine_similarity(canonical_emb, cluster_embeddings).mean()
			else:
					# Canonical not in cluster (shouldn't happen)
					canonical_sim = 0.0
			
			# Cluster diameter (max pairwise distance)
			if cluster_size > 1:
					pairwise_dists = pdist(cluster_embeddings, metric='cosine')
					diameter = pairwise_dists.max()
					avg_distance = pairwise_dists.mean()
			else:
					diameter = 0.0
					avg_distance = 0.0
			
			# Label diversity (entropy of label distribution)
			if original_label_counts:
					cluster_counts = [original_label_counts.get(lbl, 1) for lbl in cluster_labels]
					total_count = sum(cluster_counts)
					probs = np.array(cluster_counts) / total_count
					label_entropy = entropy(probs)
			else:
					label_entropy = np.log(cluster_size) if cluster_size > 1 else 0.0
			
			# Weighted coverage (if original counts provided)
			if original_label_counts:
					cluster_coverage = sum(original_label_counts.get(lbl, 1) for lbl in cluster_labels)
			else:
					cluster_coverage = cluster_size
			
			cluster_metrics_list.append({
					'cluster_id': cluster_id,
					'size': cluster_size,
					'canonical_label': canonical,
					'intra_cluster_similarity': intra_sim_no_diag,
					'canonical_representativeness': canonical_sim,
					'cluster_diameter': diameter,
					'avg_pairwise_distance': avg_distance,
					'label_entropy': label_entropy,
					'coverage': cluster_coverage
			})
	
	cluster_df = pd.DataFrame(cluster_metrics_list)
	
	# Add quality flags
	cluster_df['quality_flag'] = cluster_df.apply(_flag_cluster_quality, axis=1)
	
	if verbose:
		print(f"\tAnalyzed {n_clusters:,} clusters")
		print(f"\tAvg cluster size: {cluster_df['size'].mean():.1f} (median: {cluster_df['size'].median():.0f})")
		print(f"\tAvg intra-cluster similarity: {cluster_df['intra_cluster_similarity'].mean():.4f}")
		print(f"\tAvg canonical representativeness: {cluster_df['canonical_representativeness'].mean():.4f}")
	
	# 3. IDENTIFY PROBLEMATIC CLUSTERS
	if verbose:
		print("\n[3/6] Identifying Problematic Clusters...")
	
	problematic_clusters = []
	
	# Flag 1: Low cohesion (intra-cluster similarity < 0.5)
	low_cohesion = cluster_df[cluster_df['intra_cluster_similarity'] < 0.5]
	if len(low_cohesion) > 0:
		problematic_clusters.append(
			{
				'issue': 'Low Cohesion',
				'count': len(low_cohesion),
				'cluster_ids': low_cohesion['cluster_id'].tolist(),
				'severity': 'HIGH',
				'description': 'Clusters with low internal similarity (< 0.5). May contain semantically diverse labels.'
			}
		)
		# save into file:
		low_cohesion.to_csv(os.path.join(output_dir, f"low_cohesion_clusters.csv"), index=False)
		low_cohesion_dict = {}
		for _, row in low_cohesion.iterrows():
			cid = int(row['cluster_id'])
			member_labels = labels[cluster_assignments == cid].tolist()
			low_cohesion_dict[cid] = {
				'labels': member_labels,
				'canonical': row['canonical_label'],
				'intra_similarity': float(row['intra_cluster_similarity']),
				'size': int(row['size'])
			}
		
		json_path = os.path.join(output_dir, "low_cohesion_clusters.json")
		with open(json_path, 'w', encoding='utf-8') as f:
			json.dump(low_cohesion_dict, f, indent=2, ensure_ascii=False)
		
		if verbose:
			print(f"\n✓ Exported {len(low_cohesion_dict)} low cohesion clusters to: {json_path}")


	# Flag 2: Poor canonical representativeness (< 0.6)
	poor_canonical = cluster_df[cluster_df['canonical_representativeness'] < 0.6]
	if len(poor_canonical) > 0:
		problematic_clusters.append(
			{
				'issue': 'Poor Canonical Representativeness',
				'count': len(poor_canonical),
				'cluster_ids': poor_canonical['cluster_id'].tolist(),
				'severity': 'MEDIUM',
				'description': 'Canonical label does not represent cluster well (< 0.6 similarity).'
			}
		)
		# save into file:
		poor_canonical.to_csv(os.path.join(output_dir, f"poor_canonical_clusters.csv"), index=False)
	
	# Flag 3: Large diameter (> 0.8 cosine distance)
	large_diameter = cluster_df[cluster_df['cluster_diameter'] > 0.8]
	if len(large_diameter) > 0:
		problematic_clusters.append(
			{
				'issue': 'Large Cluster Diameter',
				'count': len(large_diameter),
				'cluster_ids': large_diameter['cluster_id'].tolist(),
				'severity': 'MEDIUM',
				'description': 'Clusters with large spread (diameter > 0.8). May need splitting.'
			}
		)
		# save into file:
		large_diameter.to_csv(os.path.join(output_dir, f"large_diameter_clusters.csv"), index=False)
	
	# Flag 4: Singleton clusters (size = 1)
	singletons = cluster_df[cluster_df['size'] == 1]
	if len(singletons) > 0:
		problematic_clusters.append(
			{
				'issue': 'Singleton Clusters',
				'count': len(singletons),
				'cluster_ids': singletons['cluster_id'].tolist(),
				'severity': 'LOW',
				'description': 'Clusters with only one label. No consolidation benefit.'
			}
		)
	
	# Flag 5: Very large clusters (size > 95th percentile)
	size_threshold = cluster_df['size'].quantile(0.95)
	very_large = cluster_df[cluster_df['size'] > size_threshold]
	if len(very_large) > 0:
		problematic_clusters.append(
			{
				'issue': 'Very Large Clusters',
				'count': len(very_large),
				'cluster_ids': very_large['cluster_id'].tolist(),
				'severity': 'LOW',
				'description': f'Clusters larger than 95th percentile (> {size_threshold:.0f} labels). May be over-merged.'
			}
		)
		# save into file:
		very_large.to_csv(os.path.join(output_dir, f"very_large_clusters.csv"), index=False)
	
	if verbose:
		if len(problematic_clusters) == 0:
			print("\t[OK] No major problematic clusters detected!")
		else:
			print(f"Found {len(problematic_clusters)} type(s) of problematic clusters:")
			for issue in problematic_clusters:
				print(f"{issue['severity']:10s}{issue['issue']:35s}{issue['count']:4d} clusters")
	
	if verbose and len(low_cohesion) > 0:
		print(f"\n[LOW COHESION DETAIL] {len(low_cohesion)} clusters flagged (intra_sim < 0.5):")
		print(f"{'─' * 60}")
		low_cohesion_sorted = low_cohesion.sort_values('intra_cluster_similarity')
		for _, row in low_cohesion_sorted.iterrows():
			cid = int(row['cluster_id'])
			canonical = row['canonical_label']
			sim = row['intra_cluster_similarity']
			size = int(row['size'])
			# Retrieve all member labels for this cluster
			member_labels = labels[cluster_assignments == cid].tolist()
			print(f"Cluster {cid}: canonical='{canonical}' | sim={sim:.4f} | size={size}")
			print(f"  labels: {member_labels}")
			print()
		print(f"{'─' * 60}")

	if verbose and len(poor_canonical) > 0:
		print(f"\n[POOR CANONICAL DETAIL] {len(poor_canonical)} clusters flagged (canonical_rep < 0.6):")
		print(f"{'─' * 60}")
		for _, row in poor_canonical.iterrows():
			cid = int(row['cluster_id'])
			canonical = row['canonical_label']
			sim = row['canonical_representativeness']
			size = int(row['size'])
			# Retrieve all member labels for this cluster
			member_labels = labels[cluster_assignments == cid].tolist()
			print(f"Cluster {cid}: canonical='{canonical}' | sim={sim:.4f} | size={size}")
			print(f"  labels: {member_labels}")
			print()
		print(f"{'─' * 60}")

	# 4. CONSOLIDATION IMPACT ANALYSIS
	if verbose:
		print("\n[4/6] Analyzing Consolidation Impact...")
	
	consolidation_impact = {
			'original_labels': n_samples,
			'consolidated_labels': n_clusters,
			'reduction_ratio': n_samples / n_clusters,
			'reduction_percentage': (1 - n_clusters / n_samples) * 100,
			'singleton_clusters': len(singletons),
			'singleton_percentage': len(singletons) / n_clusters * 100,
			'avg_cluster_size': cluster_df['size'].mean(),
			'median_cluster_size': cluster_df['size'].median(),
			'max_cluster_size': cluster_df['size'].max(),
			'size_std': cluster_df['size'].std()
	}
	
	if original_label_counts:
			total_original_instances = sum(original_label_counts.values())
			consolidation_impact['total_original_instances'] = total_original_instances
			consolidation_impact['avg_instances_per_original_label'] = total_original_instances / n_samples
			consolidation_impact['avg_instances_per_cluster'] = total_original_instances / n_clusters
	
	if verbose:
			print(f"\tLabel reduction: {n_samples:,} → {n_clusters:,} ({consolidation_impact['reduction_percentage']:.1f}% reduction)")
			print(f"\tAvg consolidation: {consolidation_impact['reduction_ratio']:.2f} labels per cluster")
			print(f"\tSingleton clusters: {len(singletons):,} ({consolidation_impact['singleton_percentage']:.1f}%)")
	
	# 5. CLUSTER SIZE DISTRIBUTION ANALYSIS
	if verbose:
		print("\n[5/6] Cluster Size Distribution...")
	
	size_distribution = {
			'min': int(cluster_df['size'].min()),
			'q25': int(cluster_df['size'].quantile(0.25)),
			'median': int(cluster_df['size'].median()),
			'q75': int(cluster_df['size'].quantile(0.75)),
			'q95': int(cluster_df['size'].quantile(0.95)),
			'max': int(cluster_df['size'].max()),
			'mean': float(cluster_df['size'].mean()),
			'std': float(cluster_df['size'].std())
	}
	
	if verbose:
		print(
			f"\tMin: {size_distribution['min']}, Q25: {size_distribution['q25']}, "
			f"Median: {size_distribution['median']}, Q75: {size_distribution['q75']}, "
			f"Q95: {size_distribution['q95']}, Max: {size_distribution['max']}"
		)
	
	# 6. GENERATE RECOMMENDATIONS
	if verbose:
		print("\n[6/6] Generating Recommendations...")
	
	recommendations = _generate_recommendations(
		global_metrics, 
		cluster_df, 
		problematic_clusters, 
		consolidation_impact
	)
	
	if verbose:
		for i, rec in enumerate(recommendations, 1):
			print(f"\t{i}. {rec}")
	
	summary = _generate_summary(
		global_metrics, 
		consolidation_impact, 
		problematic_clusters,
		n_samples,
		n_clusters
	)

	print(f"\n{summary}\n")

	return {
		'global_metrics': global_metrics,
		'cluster_metrics': cluster_df,
		'problematic_clusters': problematic_clusters,
		'consolidation_impact': consolidation_impact,
		'size_distribution': size_distribution,
		'recommendations': recommendations,
		'summary': summary
	}

def _interpret_ch_index(score: float) -> str:
	# Interpret Calinski-Harabasz index

	if score > 1000:
		return "EXCELLENT"
	elif score > 500:
		return "GOOD"
	elif score > 200:
		return "FAIR"
	else:
		return "WEAK"

def _interpret_db_index(score: float) -> str:
	# Interpret Davies-Bouldin index
	if score < 0.5:
		return "EXCELLENT"
	elif score < 1.0:
		return "GOOD"
	elif score < 1.5:
		return "FAIR"
	else:
		return "POOR"

def _interpret_silhouette(score: float) -> str:
	# Interpret silhouette score
	if score > 0.7:
		return "EXCELLENT"
	elif score > 0.5:
		return "GOOD"
	elif score > 0.3:
		return "FAIR"
	elif score > 0.0:
		return "WEAK"
	else:
		return "POOR"

def _flag_cluster_quality(row: pd.Series) -> str:
	# Flag cluster quality based on metrics
	flags = []
	
	if row['intra_cluster_similarity'] < 0.5:
		flags.append('LOW_COHESION')
	
	if row['canonical_representativeness'] < 0.6:
		flags.append('POOR_CANONICAL')
	
	if row['cluster_diameter'] > 0.8:
		flags.append('LARGE_DIAMETER')
	
	if row['size'] == 1:
		flags.append('SINGLETON')
	
	return '|'.join(flags) if flags else 'OK'

def _generate_recommendations(
	global_metrics: Dict,
	cluster_df: pd.DataFrame,
	problematic_clusters: List[Dict],
	consolidation_impact: Dict
) -> List[str]:
	"""Generate actionable recommendations."""
	recommendations = []
	
	# Recommendation 1: Overall quality
	silhouette = global_metrics['silhouette_score']
	if silhouette < 0.3:
			recommendations.append(
				f"⚠️  LOW OVERALL QUALITY: silhouette={silhouette:.4f}. Either increase n_clusters or use a different linkage method."
			)
	elif silhouette > 0.5:
		recommendations.append(
			"✅ GOOD OVERALL QUALITY: Clustering structure is well-defined."
		)
	
	# Recommendation 2: Problematic clusters
	high_severity_issues = [p for p in problematic_clusters if p['severity'] == 'HIGH']
	if high_severity_issues:
		recommendations.append(
				f"⚠️  REVIEW {len(high_severity_issues)} HIGH-SEVERITY CLUSTERS: "
				f"Check clusters with low cohesion manually."
		)
	
	# Recommendation 3: Singletons
	singleton_pct = consolidation_impact['singleton_percentage']
	if singleton_pct > 20:
		recommendations.append(
				f"⚠️  HIGH SINGLETON RATE ({singleton_pct:.1f}%): "
				f"Consider reducing n_clusters to improve consolidation."
		)
	
	# Recommendation 4: Consolidation effectiveness
	reduction_ratio = consolidation_impact['reduction_ratio']
	if reduction_ratio < 2:
		recommendations.append(
				f"⚠️  LOW CONSOLIDATION ({reduction_ratio:.1f}x): "
				f"Clustering provides minimal label reduction. Consider more aggressive merging."
		)
	elif reduction_ratio > 10:
		recommendations.append(
				f"✅ STRONG CONSOLIDATION ({reduction_ratio:.1f}x): "
				f"Significant label reduction achieved."
		)
	
	# Recommendation 5: Canonical representativeness
	avg_canonical_rep = cluster_df['canonical_representativeness'].mean()
	if avg_canonical_rep < 0.7:
		recommendations.append(
			f"⚠️  CANONICAL LABELS MAY NOT BE OPTIMAL: "
			f"Avg representativeness = {avg_canonical_rep:.3f}. Consider alternative selection strategy."
		)
	
	return recommendations

def _generate_summary(
	global_metrics: Dict,
	consolidation_impact: Dict,
	problematic_clusters: List[Dict],
	n_samples: int,
	n_clusters: int
) -> str:
	
	silhouette = global_metrics['silhouette_score']
	db_index = global_metrics['davies_bouldin_index']
	reduction_ratio = consolidation_impact['reduction_ratio']
	singleton_pct = consolidation_impact['singleton_percentage']
	
	high_severity_count = sum(1 for p in problematic_clusters if p['severity'] == 'HIGH')
	
	summary = f"""
		Clustering consolidated {n_samples:,} unique labels into {n_clusters:,} clusters 
		({reduction_ratio:.2f}x reduction, {consolidation_impact['reduction_percentage']:.1f}% decrease).
		QUALITY ASSESSMENT:
			• Silhouette Score: {silhouette:.4f} ({global_metrics['silhouette_interpretation']})
			• Davies-Bouldin Index: {db_index:.4f} ({global_metrics['db_interpretation']})
			• Singleton Rate: {singleton_pct:.1f}%
		ISSUES DETECTED:
			• {high_severity_count} high-severity clusters requiring review
			• {len(problematic_clusters)} total issue categories identified
		RECOMMENDATION: {'✅ PROCEED with label mapping' if silhouette > 0.4 and high_severity_count == 0 else '⚠️  REVIEW problematic clusters before proceeding'}
	"""
	return summary.strip()

def export_problematic_clusters(
		labels: np.ndarray,
		cluster_assignments: np.ndarray,
		canonical_labels: Dict[int, str],
		problematic_cluster_ids: List[int],
		output_path: str = 'problematic_clusters_review.csv'
) -> None:
		"""
		Export problematic clusters to CSV for manual review.
		
		Parameters
		----------
		labels : np.ndarray
				Original label strings
		cluster_assignments : np.ndarray
				Cluster ID for each label
		canonical_labels : Dict[int, str]
				Mapping from cluster_id -> canonical label
		problematic_cluster_ids : List[int]
				List of cluster IDs flagged as problematic
		output_path : str
				Output CSV file path
		"""
		
		review_data = []
		
		for cluster_id in problematic_cluster_ids:
				mask = cluster_assignments == cluster_id
				cluster_labels = labels[mask]
				canonical = canonical_labels.get(cluster_id, "UNKNOWN")
				
				for label in cluster_labels:
						review_data.append({
								'cluster_id': cluster_id,
								'canonical_label': canonical,
								'original_label': label,
								'is_canonical': label == canonical
						})
		
		df = pd.DataFrame(review_data)
		df.to_csv(output_path, index=False)
		print(f"✅ Exported {len(df)} labels from {len(problematic_cluster_ids)} problematic clusters to: {output_path}")

def get_optimal_super_clusters(
	linkage_matrix,
	embeddings,
	cluster_labels,
	unique_labels,
	linkage_method,
	clusters_fname,
	n_thresholds=50,
	min_clusters=3,
	max_clusters=10,
	verbose=False,
):
	print(f"\n[SUPER-CLUSTERS] Analyzing hierarchy...")

	distances = linkage_matrix[:, 2]
	candidate_distances = np.linspace(distances.min(), distances.max(), n_thresholds)
	best_score = -np.inf
	best_distance = None
	best_n_clusters = None
	print(f"[SUPER-CLUSTERS] Testing {len(candidate_distances)} distance thresholds...")
	for dist in candidate_distances:
			labels = fcluster(linkage_matrix, t=dist, criterion='distance')
			n_clusters = len(np.unique(labels))
			if n_clusters < min_clusters or n_clusters > max_clusters:
					continue
			score = silhouette_score(embeddings, labels, metric='cosine')
			if score > best_score:
					best_score = score
					best_distance = dist
					best_n_clusters = n_clusters
	if best_distance is None:
			# Fallback: choose dist whose n_clusters is closest to min_clusters
			print(f"[SUPER-CLUSTERS] No distance produced n_clusters in [{min_clusters}, {max_clusters}]. Falling back...")
			best_gap = np.inf
			for dist in candidate_distances:
					labels = fcluster(linkage_matrix, t=dist, criterion='distance')
					n_clusters = len(np.unique(labels))
					gap = abs(n_clusters - min_clusters)
					if gap < best_gap:
							best_gap = gap
							best_distance = dist
							best_n_clusters = n_clusters
			print(f"[SUPER-CLUSTERS] Fallback: distance={best_distance:.4f}, n_clusters={best_n_clusters}")
	else:
			print(f"[SUPER-CLUSTERS] Best silhouette: {best_score:.4f} at {best_n_clusters} clusters")

	super_cluster_distance, n_super_clusters = best_distance, best_n_clusters

	print(f"[SUPER-CLUSTERS] Optimal distance: {super_cluster_distance:.4f} ({n_super_clusters} super-clusters)")

	super_cluster_labels = fcluster(linkage_matrix, t=super_cluster_distance, criterion='distance') - 1

	print(f"\n[VERIFICATION] super-cluster alignment...")
	print(f"  ├─ Distance threshold: {super_cluster_distance:.4f}")
	print(f"  ├─ Expected clusters: {n_super_clusters}")

	# Recompute to verify
	labels_check = fcluster(linkage_matrix, t=super_cluster_distance, criterion='distance')
	n_clusters_check = len(np.unique(labels_check))
	print(f"  ├─ Actual clusters from fcluster: {n_clusters_check}")

	if n_clusters_check == n_super_clusters:
		print(f"  └─ Confirmed Alignment: {n_super_clusters} clusters at t={super_cluster_distance:.4f}")
	else:
		print(f"  └─ MISMATCH ALERT: Expected {n_super_clusters}, got {n_clusters_check}")

	# Map fine-grained clusters to super-clusters
	# Get the number of fine-grained clusters dynamically
	n_fine_clusters = len(np.unique(cluster_labels))
	
	# Create mapping: fine_cluster_id -> super_cluster_id
	cluster_to_supercluster = {}
	supercluster_stats = {}
	
	for fine_cluster_id in range(n_fine_clusters):
			# Get all label indices in this fine cluster
			fine_cluster_mask = cluster_labels == fine_cluster_id
			fine_cluster_indices = np.where(fine_cluster_mask)[0]
			
			# Find which super-cluster these labels belong to (majority vote)
			super_ids = super_cluster_labels[fine_cluster_indices]
			super_cluster_id = int(np.bincount(super_ids).argmax())
			
			cluster_to_supercluster[fine_cluster_id] = super_cluster_id
			
			# Track super-cluster stats
			if super_cluster_id not in supercluster_stats:
					supercluster_stats[super_cluster_id] = {
							'fine_clusters': [],
							'total_labels': 0
					}
			supercluster_stats[super_cluster_id]['fine_clusters'].append(fine_cluster_id)
			supercluster_stats[super_cluster_id]['total_labels'] += fine_cluster_mask.sum()
	
	print(f"\n[SUPER-CLUSTER] HIERARCHY")
	print(f"Total fine-grained clusters: {n_fine_clusters}")
	print(f"Total super-clusters: {n_super_clusters}")
	
	for super_id in sorted(supercluster_stats.keys()):
		stats = supercluster_stats[super_id]
		print(f"\n[Super-Cluster {super_id}]")
		print(f"  ├─ Fine clusters: {len(stats['fine_clusters'])} clusters")
		print(f"  ├─ Total labels: {stats['total_labels']} ({stats['total_labels']/len(unique_labels)*100:.1f}%)")
		print(f"  └─ Cluster IDs: {stats['fine_clusters'][:25]}{'...' if len(stats['fine_clusters']) > 25 else ''}")
	
	# 2D cluster visualizations
	plt.figure(figsize=(10, 7))
	dendrogram(
		linkage_matrix, 
		truncate_mode='lastp', 
		p=40, 
		show_leaf_counts=True, 
		color_threshold=super_cluster_distance
	)
	# plt.axhline(
	# 	y=super_cluster_distance, 
	# 	color='#000000', 
	# 	linestyle='--', 
	# 	label=f'Cut at {super_cluster_distance:.4f} ({n_super_clusters} super-clusters)',
	# 	linewidth=2.5,
	# 	zorder=10,
	# )

	plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} Linkage)\n{n_super_clusters} Super-Clusters at distance={super_cluster_distance:.4f}')
	plt.xlabel('Cluster')
	plt.ylabel('Distance')
	plt.legend(loc='upper right', fontsize=12)
	plt.grid(False)
	out_dendogram = clusters_fname.replace(".csv", "_dendrogram.png")
	plt.savefig(out_dendogram, dpi=200, bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(24, 15))
	dendrogram(
		linkage_matrix,
		color_threshold=super_cluster_distance,
		leaf_font_size=8
	)
	plt.axhline(
		y=super_cluster_distance, 
		color='#000000', 
		linestyle='--', 
		linewidth=2.5,
		label=f'Cut at {super_cluster_distance:.4f}',
		zorder=10
	)
	plt.title(f'Full Dendrogram (All {n_fine_clusters} Fine Clusters)')
	plt.xlabel('Fine Cluster ID')
	plt.ylabel('Distance')
	plt.legend()
	
	out_full_dendogram = clusters_fname.replace(".csv", "_dendrogram_full.png")
	plt.savefig(out_full_dendogram, dpi=200, bbox_inches='tight')
	plt.close()

	# PCA
	pca_projection = PCA(n_components=2, random_state=0).fit_transform(embeddings)
	
	# t-SNE (subsample if too large)
	if len(unique_labels) > 10000:
		tsne_indices = np.random.choice(len(unique_labels), 10000, replace=False)
		tsne_projection = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(embeddings[tsne_indices])
		tsne_labels = cluster_labels[tsne_indices]
	else:
		tsne_projection = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(embeddings)
		tsne_labels = cluster_labels
	
	# Color palette
	n_colors = min(len(cluster_labels), 256)
	palette = sns.color_palette('tab20', n_colors) if n_colors <= 20 else sns.color_palette('husl', n_colors)
	colors = [palette[i % len(palette)] for i in cluster_labels]
	# PCA plot
	plt.figure(figsize=(27, 17))
	plt.scatter(*pca_projection.T, s=40, c=colors, alpha=0.6, marker='o')
	plt.title(f"PCA - Agglomerative Clustering ({len(cluster_labels)} clusters, {len(unique_labels)} labels)")
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	out_pca = clusters_fname.replace(".csv", "_pca_agglomerative.png")
	plt.savefig(out_pca, dpi=150, bbox_inches='tight')
	plt.close()
	
	# t-SNE plot
	tsne_colors = [palette[i % len(palette)] for i in tsne_labels]
	plt.figure(figsize=(27, 17))
	plt.scatter(*tsne_projection.T, s=40, c=tsne_colors, alpha=0.6, marker='o')
	plt.title(f"t-SNE - Agglomerative Clustering ({len(cluster_labels)} clusters)")
	plt.xlabel("t-SNE 1")
	plt.ylabel("t-SNE 2")
	out_tsne = clusters_fname.replace(".csv", "_tsne_agglomerative.png")
	plt.savefig(out_tsne, dpi=150, bbox_inches='tight')
	plt.close()

def get_optimal_num_clusters(
	X,
	linkage_matrix,
	min_cluster_size=2,
	merge_singletons=True,
	target_intra_similarity=0.70,
	min_consolidation=4.0,  
	max_consolidation=6.0,
	target_singleton_ratio=0.015,
	quality_vs_consolidation_weight=0.6,
	verbose=True
):
	num_samples = X.shape[0]
	if verbose:
		print("\nADAPTIVE OPTIMAL CLUSTER SELECTION")
		print(f"   ├─ Target intra-cluster similarity: {target_intra_similarity}")
		print(f"   ├─ Min cluster size: {min_cluster_size}")
		print(f"   ├─ Merge singletons: {merge_singletons}")
		print(f"   ├─ Consolidation (Reduction ratio) range: {min_consolidation}x - {max_consolidation}x")
		print(f"   ├─ Required and valid clusters range: {num_samples//max_consolidation} ≤ k ≤ {num_samples//min_consolidation}")
		print(f"   ├─ Target singleton ratio: {target_singleton_ratio}")
		print(f"   ├─ Quality weight: {quality_vs_consolidation_weight*100:.0f}%")
		print(f"   ├─ Dataset: {type(X)} {X.shape} {X.dtype}")
		print(f"   └─ Linkage matrix: {type(linkage_matrix)} {linkage_matrix.shape}")
	
	valid_k_min = int(num_samples // max_consolidation)
	valid_k_max = int(num_samples // min_consolidation)
	if verbose:
		print(f"\n[STAGE 1] COARSE SEARCH - Finding quality plateau: {valid_k_min} ≤ k ≤ {valid_k_max}")
	# Adaptive coarse range based on dataset size
	if num_samples > int(3e4):
		coarse_step = 1000
	elif num_samples > int(1e4):
		coarse_step = 500
	elif num_samples > int(5e3):
		coarse_step = 100
	else:
		coarse_step = max(1, (valid_k_max - valid_k_min) // 10)  # ~10 steps in valid zone
	
	# Extend range to cover from 10% of valid_k_min up to valid_k_max
	coarse_start = max(10, valid_k_min // 2)
	coarse_end = valid_k_max + coarse_step # Use overshoot to ensure valid_k_max is covered and peeked
	coarse_range = range(coarse_start, coarse_end, coarse_step)  # explicit +1, drop overshoot

	if verbose:
		print(f"Testing {len(coarse_range)} configurations: {list(coarse_range)} (step={coarse_step})")
		print(f"\n{'k':<8} {'IntraSim':<12} {'Consol':<10} {'SingleR':<10} {'Status':<50} {'Reason'}")
		print("-" * 200)
	
	coarse_results = []
	best_intra_sim = 0
	plateau_k = None
	
	for n_clusters in coarse_range:
		labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
		
		if len(np.unique(labels)) < 2:
			continue
		
		# Compute mean intra-cluster similarity
		unique_labels = np.unique(labels)
		intra_sims = []
		
		for cid in unique_labels:
			cluster_mask = labels == cid
			cluster_X = X[cluster_mask]
			
			if len(cluster_X) > 1:
				sim_matrix = cosine_similarity(cluster_X)
				n = len(cluster_X)
				intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
				intra_sims.append(intra_sim)
		
		mean_intra_sim = np.mean(intra_sims) if intra_sims else 0
		
		# Cluster statistics
		cluster_sizes = np.bincount(labels)
		n_singletons = np.sum(cluster_sizes == 1)
		singleton_ratio = n_singletons / n_clusters
		consolidation = num_samples / n_clusters
		
		coarse_results.append(
			{
				'k': n_clusters,
				'intra_sim': mean_intra_sim,
				'consolidation': consolidation,
				'singleton_ratio': singleton_ratio,
				'n_singletons': n_singletons
			}
		)
		
		# Check if in target range
		in_consol_range = min_consolidation <= consolidation <= max_consolidation
		in_singleton_range = 0.005 <= singleton_ratio <= 0.03  # 0.5%-3% acceptable

		status = ""
		reason = ""
		if mean_intra_sim >= target_intra_similarity and in_consol_range:
			status = "✓ TARGET REACHED (quality + consolidation)"
			reason = (
				f"IntraSim {mean_intra_sim:.4f} ≥ target {target_intra_similarity:.4f} "
				f"AND consol {consolidation:.1f}x in [{min_consolidation},{max_consolidation}]"
			)
			if plateau_k is None:
				plateau_k = n_clusters
		elif in_consol_range and in_singleton_range:
			status = "✓ OPTIMAL RANGE (consolidation + singletons)"
			reason = (
				f"Consol {consolidation:.1f}x in [{min_consolidation},{max_consolidation}] "
				f"AND singleton {singleton_ratio:.4f} in [0.005,0.03]"
			)
			if plateau_k is None:
				plateau_k = n_clusters
		elif mean_intra_sim > best_intra_sim:
			best_intra_sim = mean_intra_sim
			status = "↑ Improving quality"
			reason = (
				f"IntraSim {mean_intra_sim:.4f} > prev best {best_intra_sim:.4f}"
				+ (f" | consol {consolidation:.1f}x outside [{min_consolidation},{max_consolidation}]" if not in_consol_range else "")
				+ (f" | singleton {singleton_ratio:.4f} outside [0.005,0.03]" if not in_singleton_range else "")
			)
		else:
			status = "→ Plateau region"
			reason = (
				f"IntraSim {mean_intra_sim:.4f} ≤ prev best {best_intra_sim:.4f} (no improvement)"
			)
		
		# Early stopping: If in optimal range for 2 consecutive steps
		if len(coarse_results) >= 2:
			recent = coarse_results[-2:]
			both_optimal = all(
				r['intra_sim'] >= target_intra_similarity * 0.95 and
				min_consolidation <= r['consolidation'] <= max_consolidation
				for r in recent
			)
			if both_optimal and plateau_k is not None:
				if verbose:
					print(f"\n[STAGE 1] Optimal range detected. Moving to fine search.")
				break

		if verbose:
			print(f"{n_clusters:<8} {mean_intra_sim:<12.4f} {consolidation:<10.1f} {singleton_ratio:<10.4f} {status:<50} {reason}")

	if not coarse_results:
		raise ValueError("No valid cluster configurations found in coarse search")
	
	max_observed_intra = max(r['intra_sim'] for r in coarse_results)
	gap = (target_intra_similarity - max_observed_intra) / target_intra_similarity
	if gap > 0.10:  # Target is >10% above what's achievable
		adjusted_target = max_observed_intra * 1.02  # 2% headroom above observed max
		if verbose:
			print(f"\n[STAGE 1] ⚠ Target intra-sim {target_intra_similarity:.2f} unreachable.")
			print(f"           Max observed: {max_observed_intra:.4f} (gap={gap*100:.1f}%)")
			print(f"           Auto-adjusting target → {adjusted_target:.4f}")
		target_intra_similarity = adjusted_target
	else:
		if verbose:
			print(f"Target intra-sim {target_intra_similarity:.2f} is achievable.")

	# Determine search region for Stage 2
	if plateau_k is None:
		# Use k that best balances quality and consolidation
		def score_coarse(r):
			# Penalize if outside consolidation range
			consol_penalty = 1.0
			if r['consolidation'] < min_consolidation:
				consol_penalty = 0.5
			elif r['consolidation'] > max_consolidation:
				consol_penalty = 0.7
			
			# Reward if near target singleton ratio
			singleton_penalty = 1.0 - abs(r['singleton_ratio'] - target_singleton_ratio)
			singleton_penalty = max(0.5, singleton_penalty)
			
			return r['intra_sim'] * consol_penalty * singleton_penalty
		
		best_coarse = max(coarse_results, key=score_coarse)
		plateau_k = best_coarse['k']
	
	if verbose:
		print(f"[DONE STAGE 1] Plateau region centered around k={plateau_k}")
	
	if verbose:
		print(f"\n[STAGE 2] FINE SEARCH - Optimizing around plateau: k={plateau_k}")
	# Fine search range: ±20% around plateau with smaller steps
	fine_min = max(int(plateau_k * 0.8), valid_k_min)  # Never go below valid zone
	_step_proxy = max(1, (valid_k_max - valid_k_min) // 20)  # temporary, only for fine_max
	# Allow exactly 2 fine steps beyond the strict boundary
	fine_max = min(
		int(plateau_k * 1.2), # Strict 20% above plateau
		int(num_samples // min_consolidation) + 2 * _step_proxy  # +2 steps of headroom
	)
	
	if verbose:
		print(f"[PROPOSAL] Fine search range(using Pareto Principle[80:20]): {fine_min} ≤ k ≤ {fine_max}")

	# Guard: if plateau_k was outside valid range, just search the valid range
	if fine_min >= fine_max:
		if verbose:
			print(f"[WARNING] Plateau k={plateau_k} outside valid range: {valid_k_min} ≤ k ≤ {valid_k_max}. Searching valid range only.")
		fine_min = valid_k_min
		fine_max = valid_k_max

	fine_step = max(1, (fine_max - fine_min) // 20)  # ← always ~20 evaluations
	fine_range = range(fine_min, fine_max + 1, fine_step)
	
	if verbose:
		print(f"Testing {len(fine_range)} configurations: {list(fine_range)} (step={fine_step})")
		print(f"\n{'k':<8} {'IntraSim':<12} {'Consol':<10} {'SingleR':<10} {'Score':<10} {'Status':<20} {'Reason'}")
		print("-" * 150)
	
	fine_results = []
	
	for n_clusters in fine_range:
		labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
		
		if len(np.unique(labels)) < 2:
			continue
		
		# Compute intra-cluster similarity
		unique_labels = np.unique(labels)
		intra_sims = []
		
		for cid in unique_labels:
			cluster_mask = labels == cid
			cluster_X = X[cluster_mask]
			
			if len(cluster_X) > 1:
				sim_matrix = cosine_similarity(cluster_X)
				n = len(cluster_X)
				intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
				intra_sims.append(intra_sim)
		
		mean_intra_sim = np.mean(intra_sims) if intra_sims else 0
		
		# Compute statistics
		cluster_sizes = np.bincount(labels)
		n_singletons = np.sum(cluster_sizes == 1)
		singleton_ratio = n_singletons / n_clusters
		consolidation = num_samples / n_clusters
		
		# COMPOSITE SCORING FUNCTION (Key Innovation!)
		# Quality component (normalized to 0-1)
		quality_score = mean_intra_sim / target_intra_similarity
		quality_score = min(1.0, quality_score)  # Cap at 1.0
		
		# Consolidation component (penalize if outside range)
		if consolidation < min_consolidation:
			consol_score = consolidation / min_consolidation  # Penalize low consolidation
		elif consolidation > max_consolidation:
			consol_score = max_consolidation / consolidation  # Penalize high consolidation
		else:
			consol_score = 1.0  # Perfect
		
		# Singleton component (penalize if far from target)
		singleton_error = abs(singleton_ratio - target_singleton_ratio)
		singleton_score = 1.0 - min(1.0, singleton_error / target_singleton_ratio)
		
		# Final composite score
		score = (
				quality_vs_consolidation_weight * quality_score +
				(1 - quality_vs_consolidation_weight) * 0.7 * consol_score +
				(1 - quality_vs_consolidation_weight) * 0.3 * singleton_score
		)
		
		fine_results.append({
				'k': n_clusters,
				'intra_sim': mean_intra_sim,
				'consolidation': consolidation,
				'singleton_ratio': singleton_ratio,
				'n_singletons': n_singletons,
				'quality_score': quality_score,
				'consol_score': consol_score,
				'singleton_score': singleton_score,
				'composite_score': score
		})
		
		# Status + Reason
		status = ""
		reason = ""
		if quality_score >= 0.95 and consol_score >= 0.9:
			status = "EXCELLENT"
			reason = (
				f"QualScore {quality_score:.3f} ≥ 0.95 "
				f"AND ConsolScore {consol_score:.3f} ≥ 0.90 "
				f"| SingletonScore {singleton_score:.3f}"
			)
		elif quality_score >= 0.90 and consol_score >= 0.8:
			status = "✓ GOOD"
			reason = (
				f"QualScore {quality_score:.3f} ≥ 0.90 "
				f"AND ConsolScore {consol_score:.3f} ≥ 0.80 "
				f"| SingletonScore {singleton_score:.3f}"
			)
		elif score >= 0.70:
			status = "→ Acceptable"
			reason = (
				f"CompositeScore {score:.3f} ≥ 0.70 "
				f"| QualScore {quality_score:.3f} ConsolScore {consol_score:.3f} SingletonScore {singleton_score:.3f}"
			)
		else:
			status = "✗ Below target"
			reason = (
				f"CompositeScore {score:.3f} < 0.70 "
				f"| QualScore {quality_score:.3f} ConsolScore {consol_score:.3f} SingletonScore {singleton_score:.3f}"
			)

		if verbose:
			print(f"{n_clusters:<8} {mean_intra_sim:<12.4f} {consolidation:<10.1f} {singleton_ratio:<10.4f} {score:<10.4f} {status:<20}{reason}")
	
	if not fine_results:
		raise ValueError("No valid cluster configurations found in fine search")
	
	# SELECT BEST CONFIGURATION
	# Priority: Highest composite score
	best = max(fine_results, key=lambda x: x['composite_score'])
	
	if verbose:
		print(f"\n[STAGE 2] Selected k={best['k']} (highest composite score)")
		print(f"  ├─ Composite score: {best['composite_score']:.4f}")
		print(f"  ├─ Quality score: {best['quality_score']:.4f}")
		print(f"  ├─ Consolidation score: {best['consol_score']:.4f}")
		print(f"  ├─ Singleton score: {best['singleton_score']:.4f}")
		print(f"  └─ Intra-similarity: {best['intra_sim']:.4f}")
		print()
	
	# STAGE 3: POST-PROCESSING - Merge singletons
	optimal_k = best['k']
	labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust') - 1
	
	if merge_singletons:
			cluster_sizes = np.bincount(labels)
			singleton_ids = np.where(cluster_sizes == 1)[0]
			
			if len(singleton_ids) > 0:
					if verbose:
						print(f"\n[STAGE 3] MERGING SINGLETON CLUSTERS")
						print(f"  Found {len(singleton_ids)} singleton clusters")
					
					unique_labels = np.unique(labels)
					centroids = np.array([X[labels == cid].mean(axis=0) for cid in unique_labels])
					
					new_labels = labels.copy()
					merged_count = 0
					
					for singleton_id in singleton_ids:
							singleton_idx = np.where(labels == singleton_id)[0][0]
							singleton_vec = X[singleton_idx].reshape(1, -1)
							sims = cosine_similarity(singleton_vec, centroids)[0]
							sorted_ids = np.argsort(sims)[::-1]
							
							# Find nearest non-singleton cluster
							for nearest_id in sorted_ids:
									if cluster_sizes[nearest_id] >= min_cluster_size:
											new_labels[singleton_idx] = nearest_id
											merged_count += 1
											if verbose and merged_count <= 5:  # Only print first 5
													print(f"  ├─ Merged singleton {singleton_id} → "
																f"cluster {nearest_id} (sim={sims[nearest_id]:.4f})")
											break
					
					unique_new = np.unique(new_labels)
					label_map = {old: new for new, old in enumerate(unique_new)}
					labels = np.array([label_map[l] for l in new_labels])
					
					if verbose:
						if merged_count > 15:
							print(f"  ├─ ... ({merged_count - 15} more)")
						print(f"  ├─ Total merged: {merged_count} singletons")
						print(f"  └─ Final clusters: {len(unique_new)} (was {len(unique_labels)})")
	
	# FINAL STATISTICS
	final_cluster_sizes = np.bincount(labels)
	final_n_clusters = len(np.unique(labels))
	final_singletons = np.sum(final_cluster_sizes == 1)
	final_max_size = final_cluster_sizes.max()
	
	# Recompute final intra-similarity
	final_intra_sims = []
	for cid in np.unique(labels):
		cluster_X = X[labels == cid]
		if len(cluster_X) > 1:
			sim_matrix = cosine_similarity(cluster_X)
			n = len(cluster_X)
			intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
			final_intra_sims.append(intra_sim)
	
	final_mean_intra_sim = np.mean(final_intra_sims) if final_intra_sims else 0
	
	stats = {
		'n_clusters': final_n_clusters,
		'n_singletons': final_singletons,
		'singleton_ratio': final_singletons / final_n_clusters if final_n_clusters > 0 else 0,
		'max_cluster_size': final_max_size,
		'max_size_ratio': final_max_size / num_samples,
		'mean_cluster_size': num_samples / final_n_clusters,
		'consolidation_ratio': num_samples / final_n_clusters,
		'mean_intra_similarity': final_mean_intra_sim
	}
	
	if verbose:
		print("\n[STATISTICS]")
		print(f"  ├─ Total clusters: {stats['n_clusters']}")
		print(f"  ├─ Singletons: {stats['n_singletons']} ({stats['singleton_ratio']*100:.1f}%)")
		print(f"  ├─ Mean intra-similarity: {stats['mean_intra_similarity']:.4f}")
		print(f"  ├─ Largest cluster: {stats['max_cluster_size']} items ({stats['max_size_ratio']*100:.1f}%)")
		print(f"  ├─ Mean cluster size: {stats['mean_cluster_size']:.2f}")
		print(f"  └─ Consolidation ratio: {stats['consolidation_ratio']:.2f}:1")
		
		if stats['mean_intra_similarity'] >= target_intra_similarity:
			quality_status = "EXCELLENT"
		elif stats['mean_intra_similarity'] >= target_intra_similarity * 0.95:
			quality_status = "GOOD"
		else:
			quality_status = "ACCEPTABLE"
		
		print(f"\n  Quality Status: {quality_status}")
	
	return labels, stats

def cluster_original(
	labels: List[List[str]],
	model_id: str,
	clusters_fname: str,
	batch_size: int = 1024,
	device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
	nc: int = None,
	linkage_method: str = "ward",  # 'average', 'complete', 'single', 'ward'
	distance_metric: str = "euclidean",  # 'cosine', 'euclidean'
	verbose: bool = True,
):
	st_t = time.time()
	if verbose:
		print(f"\n[AGGLOMERATIVE CLUSTERING] {len(labels)} samples")
		print(f"   ├─ {model_id} | {device} | batch_size: {batch_size}")
		print(f"   ├─ linkage: {linkage_method}")
		print(f"   ├─ sample: {labels[:5]}")

		requires_type_exchange = isinstance(labels[0], str)

		print(f"   ├─────> {type(labels[0])} requires_type_exchange: {requires_type_exchange}")
		print(f"   └─ nc: {nc} {f'Manually defined' if nc else '=> Adaptive Search'}")
	
	print(f"\n[DEDUP] {len(labels)} {type(labels)} raw labels")
	documents = []
	for i, doc in enumerate(labels):
		if doc is None:
			# print(f"doc[{i}]: None (skipping)")
			continue
		
		if isinstance(doc, str):
			try:
				doc = ast.literal_eval(doc)
			except (ValueError, SyntaxError):
				print(f"doc[{i}]: Failed to parse '{doc}' (skipping)")
				continue
		
		if not isinstance(doc, list):
			print(f"doc[{i}]: Invalid type {type(doc)} (skipping)")
			continue
		
		# Deduplicate labels within document
		documents.append(list(set(lbl for lbl in doc if lbl is not None)))
	
	# Flatten and deduplicate (deterministic and reproducible)
	unique_labels = sorted(set(label for doc in documents for label in doc))
	
	print(f"Total {type(documents)} documents: {len(documents)}")
	print(f"Unique {type(unique_labels)} labels: {len(unique_labels)}")
	print(f"Sample unique labels: {unique_labels[:15]}")
	print("-"*100)
	
	dtype = torch.float32  # More stable than float16
	if torch.cuda.is_available():
		if torch.cuda.is_bf16_supported():
			dtype = torch.bfloat16  # bfloat16 is more stable than float16
		else:
			dtype = torch.float32
	if verbose:
		print(f"[INFO] {model_id} Dtype selection: {dtype}")

	def _optimal_attn_impl() -> str:
		if not torch.cuda.is_available():
			return "eager"
		
		major, minor = torch.cuda.get_device_capability()
		compute_cap = major + minor / 10
		
		# Flash Attention 2 requires Ampere or newer (compute >= 8.0)
		if compute_cap >= 8.0:
			try:
				import flash_attn
				if verbose:
					print(f"[INFO] Flash Attention 2 available (compute {compute_cap})")
				return "flash_attention_2"
			except ImportError:
				if verbose:
					print(f"[WARN] Flash Attention 2 not installed (pip install flash-attn)")
		
		# For older GPUs (Volta/Turing), use SDPA (PyTorch native, faster than eager)
		if compute_cap >= 7.0:
			if torch.__version__ >= "2.0.0":
				if verbose:
					print(f"[INFO] Using SDPA attention (compute {compute_cap}, PyTorch {torch.__version__})")
				return "sdpa"		

		if verbose:
			print(f"[INFO] Using eager attention (compute {compute_cap})")
		return "eager"

	attn_impl = _optimal_attn_impl()
	if verbose:
		print(f"[INFO] {model_id} with {attn_impl} attention")

	print(f"\n[INIT] Loading Sentence Transformer {model_id}")
	model = SentenceTransformer(
		model_name_or_path=model_id,
		trust_remote_code=True,
		cache_folder=cache_directory[os.getenv('USER')],
		model_kwargs={"attn_implementation": attn_impl, "dtype": dtype} if "Qwen" in model_id else {}, # "Qwen" in model_id else {}",
		token=os.getenv("HUGGINGFACE_TOKEN"),
		tokenizer_kwargs={"padding_side": "left"},
	).to(device)
	
	print(f"[LOADED] {sum(p.numel() for p in model.parameters()):,} parameters")
	
	print(f"\n[ENCODING] {len(unique_labels)} labels | batch_size: {batch_size} | {device}")
	X = model.encode(
		unique_labels,
		batch_size=batch_size,
		show_progress_bar=verbose,
		convert_to_numpy=True,
		normalize_embeddings=True,
		precision='float32', # float32 is more stable than float16 for CPU stability
	)

	# After encoding
	if np.isnan(X).any():
		nan_count = np.isnan(X).sum()
		nan_rows = np.where(np.isnan(X).any(axis=1))[0]
		
		print("\n" + "="*80)
		print(f"❌ ERROR: {nan_count} NaN values in embeddings!")
		print("="*80)
		print(f"Affected labels ({len(nan_rows)} total):")
		for idx in nan_rows[:10]:  # Show first 10
				print(f"  - {unique_labels[idx]}")
		if len(nan_rows) > 10:
				print(f"  ... and {len(nan_rows) - 10} more")
		
		print("\nROOT CAUSE:")
		print("  1. Model dtype is float16 on CPU (use float32)")
		print("  2. Model is too large for available memory")
		print("  3. Numerical instability in model")
		
		print("\nFIX:")
		print("  - Use torch_dtype=torch.float32 when loading model")
		print("  - Use precision='float32' in model.encode()")
		print("  - Or switch to GPU / smaller model")
		
		raise ValueError("Cannot proceed with NaN embeddings")

	if np.isinf(X).any():
		inf_count = np.isinf(X).sum()
		raise ValueError(f"Infinite values detected ({inf_count}) - numerical overflow!")

	if X.shape[0] == 0:
		raise ValueError("No embeddings generated")

	if np.allclose(X, 0):
		raise ValueError("All embeddings are zero vectors")

	print(f"Embeddings {type(X)} {X.shape} {X.dtype}")
	print(f"  ├─ Range: [{X.min():.4f}, {X.max():.4f}]")
	print(f"  ├─ Mean: {X.mean()}")
	print(f"  └─ Std: {X.std()}")

	# Compute linkage matrix
	print(f"[LINKAGE] {linkage_method} Agglomerative Clustering on: {X.shape} embeddings [takes a while...]")

	t0 = time.time()
	# OPTION 1: Ward linkage (RECOMMENDED for preventing mega-clusters)
	if linkage_method == "ward":
		# Ward requires Euclidean distance
		# For normalized embeddings, Euclidean ≈ Cosine
		if use_fastcluster:
			Z = fastcluster.linkage(X, method='ward', metric='euclidean')
		else:
			Z = linkage(X, method='ward', metric='euclidean')
	elif distance_metric == "cosine":
		# Efficient cosine distance computation
		distance_matrix = 1 - (X @ X.T)
		np.fill_diagonal(distance_matrix, 0)
		distance_matrix = np.clip(distance_matrix, 0, 2)
		condensed_dist = squareform(distance_matrix, checks=False)
		if use_fastcluster:
			Z = fastcluster.linkage(condensed_dist, method=linkage_method)
		else:
			Z = linkage(condensed_dist, method=linkage_method)
		print(f"[LINKAGE] Using {linkage_method} linkage with {distance_metric} distance")
	elif distance_metric == "euclidean":
		if use_fastcluster:
			Z = fastcluster.linkage(X, method=linkage_method, metric='euclidean')
		else:
			Z = linkage(X, method=linkage_method, metric='euclidean')
		print(f"[LINKAGE] Using {linkage_method} linkage with Euclidean distance")
	else:
		raise ValueError(f"Unsupported distance metric: {distance_metric}")

	print(f"[LINKAGE] Z[{linkage_method}]: {type(Z)} {Z.shape} {Z.dtype} {Z.strides} {Z.itemsize} {Z.nbytes} | {time.time()-t0:.1f} sec")
	
	# Determine Optimal Number of Clusters
	if nc is None:
		# new dataset:
		cluster_labels, stats = get_optimal_num_clusters(
			X=X,
			linkage_matrix=Z,
			target_intra_similarity=0.69,
			min_consolidation=3.8,
			max_consolidation=5.0,
			target_singleton_ratio=0.015,
			quality_vs_consolidation_weight=0.5,
			merge_singletons=True,
			verbose=verbose,
		)

		# # old dataset:
		# cluster_labels, stats = get_optimal_num_clusters(
		# 	X=X,
		# 	linkage_matrix=Z,
		# 	target_intra_similarity=0.7,
		# 	min_consolidation=4.0,
		# 	max_consolidation=6.0,
		# 	target_singleton_ratio=0.015,
		# 	quality_vs_consolidation_weight=0.6,
		# 	merge_singletons=True,
		# 	verbose=verbose,
		# )

		best_k = stats['n_clusters']
	else:
		best_k = nc
		print(f"\nUsing user-defined k={best_k} for {len(unique_labels)} labels")
	
		print(f"\nCutting dendrogram at k={best_k} for {len(unique_labels)} labels")
		cluster_labels = fcluster(Z, best_k, criterion='maxclust') - 1 # Convert to 0-indexed

	print(f"\n[CLUSTERING] {len(np.unique(cluster_labels))} clusters") 
	print(f"{cluster_labels.shape} {type(cluster_labels)} labels.")
	print(f"(min, max): ({cluster_labels.min()}, {cluster_labels.max()})")

	# get_optimal_super_clusters(
	# 	linkage_matrix=Z, 
	# 	embeddings=X,
	# 	cluster_labels=cluster_labels,
	# 	unique_labels=unique_labels,
	# 	linkage_method=linkage_method,
	# 	clusters_fname=clusters_fname,
	# 	n_thresholds=50,
	# 	min_clusters=3,
	# 	max_clusters=10,
	# 	verbose=verbose
	# )
	
	df = pd.DataFrame(
		{
			'label': unique_labels,
			'cluster': cluster_labels
		}
	)
	
	# cluster_canonicals = {}
	# for cid in sorted(df.cluster.unique()):
	# 	cluster_mask = df.cluster == cid
	# 	cluster_texts = df[cluster_mask]['label'].tolist()
	# 	cluster_indices = df[cluster_mask].index.tolist()
	# 	cluster_embeddings = X[cluster_indices]
		
	# 	# Centroid-nearest
	# 	centroid = cluster_embeddings.mean(axis=0, keepdims=True)
	# 	similarities = cosine_similarity(centroid, cluster_embeddings)[0]
	# 	best_idx = similarities.argmax()
	# 	canonical = cluster_texts[best_idx]
		
	# 	cluster_canonicals[cid] = {
	# 		'canonical': canonical,
	# 		'score': float(similarities[best_idx]),
	# 		'size': len(cluster_texts)
	# 	}

	# 	if verbose:
	# 		print(f"\n[Cluster {cid}] {len(cluster_texts)} labels:\n{cluster_texts}")
	# 		print(f"\tCanonical: {canonical} (sim={similarities[best_idx]:.4f})")

	# Build label frequency dict from documents
	print(f"\n[CLUSTERING] {len(np.unique(cluster_labels))} clusters for {cluster_labels.shape} {type(cluster_labels)} labels. {cluster_labels.min()} {cluster_labels.max()}")
	label_freq_dict = {}
	for doc in documents:
		for label in doc:
			label_freq_dict[label] = label_freq_dict.get(label, 0) + 1
	# print(label_freq_dict)

	original_label_counts = label_freq_dict
	print(f"\tComputed frequencies for {len(original_label_counts)} labels")
	print(f"\tTotal label instances: {sum(original_label_counts.values())}")
	print(f"\tMost frequent: {max(original_label_counts.items(), key=lambda x: x[1])}")
	print('-'*150)

	print(f"\nCanonical labels per cluster")
	cluster_canonicals = {}
	freq_changed_count = 0
	total_sim_loss = []
	total_freq_gain = []
	questionable_examples = []

	t0 = time.time()
	for cid in sorted(df.cluster.unique()):
		cluster_mask = df.cluster == cid
		cluster_texts = df[cluster_mask]['label'].tolist()
		cluster_indices = df[cluster_mask].index.tolist()
		cluster_embeddings = X[cluster_indices]

		if verbose:
			print(f"\n[Cluster {cid}] {len(cluster_texts)} labels:\n{cluster_texts}")
		
		# Compute centroid
		centroid = cluster_embeddings.mean(axis=0, keepdims=True)
		similarities = cosine_similarity(centroid, cluster_embeddings)[0]
		
		# Frequency-Weighted Canonical Selection
		if original_label_counts is not None and len(original_label_counts) > 0 and len(cluster_texts) > 1:
			# Get label frequencies
			label_freqs = np.array([original_label_counts.get(lbl, 1) for lbl in cluster_texts])
			
			# Normalize frequencies to [0, 1] using log-scale
			freq_scores = np.log1p(label_freqs) / np.log1p(label_freqs.max())
			
			# Hybrid scoring: 70% similarity, 30% frequency
			combined_scores = 0.7 * similarities + 0.3 * freq_scores
			best_idx = combined_scores.argmax()
			pure_sim_idx = similarities.argmax()
			
			# Safety check - require minimum 3x gain
			if best_idx != pure_sim_idx:
				freq_gain = label_freqs[best_idx] / max(label_freqs[pure_sim_idx], 1)
				
				# Only override similarity if frequency gain is meaningful (≥3x)
				if freq_gain < 3.0:
					best_idx = pure_sim_idx  # Revert to pure similarity choice
			
			# Track changes (after the safety check)
			if best_idx != pure_sim_idx:
				freq_changed_count += 1
				
				# Calculate impact metrics
				sim_loss = (similarities[pure_sim_idx] - similarities[best_idx]) / similarities[pure_sim_idx]
				freq_gain = label_freqs[best_idx] / max(label_freqs[pure_sim_idx], 1)
				
				total_sim_loss.append(sim_loss)
				total_freq_gain.append(freq_gain)
				
				# Track questionable trades for inspection
				if sim_loss > 0.10 or freq_gain < 3.0:
						questionable_examples.append({
							'cluster_id': cid,
							'pure_choice': cluster_texts[pure_sim_idx],
							'freq_choice': cluster_texts[best_idx],
							'pure_freq': label_freqs[pure_sim_idx],
							'freq_freq': label_freqs[best_idx],
							'pure_sim': similarities[pure_sim_idx],
							'freq_sim': similarities[best_idx],
							'sim_loss': sim_loss,
							'freq_gain': freq_gain,
							'cluster_size': len(cluster_texts),
							'cluster_labels': cluster_texts
						})
				
				if verbose:
					print(f"Frequency weighting changed selection:")
					print(f"  Pure similarity would pick: {cluster_texts[pure_sim_idx]} (sim={similarities[pure_sim_idx]:.4f}, freq={label_freqs[pure_sim_idx]})")
					print(f"  Frequency-weighted picks: {cluster_texts[best_idx]} (sim={similarities[best_idx]:.4f}, freq={label_freqs[best_idx]})")
		else:
			# Fallback: pure similarity (original method)
			best_idx = similarities.argmax()
		
		canonical = cluster_texts[best_idx]
		
		cluster_canonicals[cid] = {
			'canonical': canonical,
			'score': float(similarities[best_idx]),
			'size': len(cluster_texts)
		}
		
		if verbose:
			print(f"\t=> Selected Canonical: {canonical} (sim={similarities[best_idx]:.4f})")
	
	print(f"\n[CLUSTERING] {len(cluster_canonicals)} cluster canonicals computed in {time.time()-t0:.1f} sec.")
	print(f"-"*100)

	print("\nFREQUENCY WEIGHTING IMPACT ANALYSIS")
	total_clusters = len(df.cluster.unique())
	print(f"  Total clusters analyzed: {total_clusters}")
	print(f"  Clusters where frequency changed the canonical: {freq_changed_count} ({freq_changed_count/total_clusters*100:.1f}%)")

	if total_sim_loss:
		print(f"\nSIMILARITY LOSS IMPACT:")
		print(f"  Average  {np.mean(total_sim_loss)*100:.2f}%")
		print(f"  Median   {np.median(total_sim_loss)*100:.2f}%")
		print(f"  Max      {np.max(total_sim_loss)*100:.2f}%")
		print(f"  Min      {np.min(total_sim_loss)*100:.2f}%")
		
		print(f"\nFREQUENCY GAIN BENEFIT:")
		print(f"  Average {np.mean(total_freq_gain):.1f}x")
		print(f"  Median  {np.median(total_freq_gain):.1f}x")
		print(f"  Max     {np.max(total_freq_gain):.1f}x")
		print(f"  Min     {np.min(total_freq_gain):.1f}x")
		
		print(f"\nQUALITY ASSESSMENT:")
		excellent_trades = sum(1 for s, f in zip(total_sim_loss, total_freq_gain) if s < 0.03 and f > 10)
		good_trades = sum(1 for s, f in zip(total_sim_loss, total_freq_gain) if s < 0.05 and f > 5)
		questionable_trades = sum(1 for s, f in zip(total_sim_loss, total_freq_gain) if s > 0.10 or f < 2)
		
		print(f"  Excellent trades (<3% sim loss, >10x freq gain): {excellent_trades:<10} ({excellent_trades/freq_changed_count*100:.1f}%)")
		print(f"  Good trades (<5% sim loss, >5x freq gain):       {good_trades:<10} ({good_trades/freq_changed_count*100:.1f}%)")
		print(f"  Questionable trades (>10% sim loss or <2x gain): {questionable_trades:<10} ({questionable_trades/freq_changed_count*100:.1f}%)")
		
		if questionable_trades > 0 and verbose:
			print(f"\n[WARNING] {questionable_trades} questionable trades detected:")
			print(f"\t=> Consider adjusting weighting (currently 70/30) if this is high\n")
			print(f"{'Cluster':<10} {'Pure Sim Choice':<35} {'Freq Weight Choice':<35} {'Sim Loss(%)':<15} {'Freq Gain'}")
			print("-" * 110)

			for ex in sorted(questionable_examples, key=lambda x: x['sim_loss'], reverse=True):
				pure_label = ex['pure_choice'][:32]
				freq_label = ex['freq_choice'][:32]
				print(f"{ex['cluster_id']:<10} {pure_label:<35} {freq_label:<35} {ex['sim_loss']*100:<15.1f} {ex['freq_gain']:.1f}x")

			# Categorize questionable trades
			high_loss_low_gain = [ex for ex in questionable_examples if ex['sim_loss'] > 0.10 and ex['freq_gain'] < 2]
			high_loss_good_gain = [ex for ex in questionable_examples if ex['sim_loss'] > 0.10 and ex['freq_gain'] >= 2]
			low_loss_low_gain = [ex for ex in questionable_examples if ex['sim_loss'] <= 0.10 and ex['freq_gain'] < 2]
			
			print(f"\nBREAKDOWN OF QUESTIONABLE TRADES:")
			print(f"Type A: High loss (>10%) + Low gain (<2x):   {len(high_loss_low_gain):<10}{len(high_loss_low_gain)/questionable_trades:<10.4f}BAD")
			print(f"Type B: High loss (>10%) + Good gain (≥2x):  {len(high_loss_good_gain):<10}{len(high_loss_good_gain)/questionable_trades:<10.4f}DEBATABLE")
			print(f"Type C: Low loss  (≤10%) + Low gain (<2x):   {len(low_loss_low_gain):<10}{len(low_loss_low_gain)/questionable_trades:<10.4f}UNNECESSARY")
		else:
			print(f"\nAll trades are high-quality!")
		
		# Overall verdict
		avg_sim_loss_pct = np.mean(total_sim_loss) * 100
		avg_freq_gain = np.mean(total_freq_gain)
		
		print(f"\nOVERALL VERDICT:")
		if avg_sim_loss_pct < 3 and avg_freq_gain > 50:
				print(f"  ✅ EXCELLENT: Small quality cost ({avg_sim_loss_pct:.1f}%) for huge frequency benefit ({avg_freq_gain:.0f}x)")
		elif avg_sim_loss_pct < 5 and avg_freq_gain > 10:
				print(f"  ✅ GOOD: Acceptable quality cost ({avg_sim_loss_pct:.1f}%) for strong frequency benefit ({avg_freq_gain:.0f}x)")
		elif avg_sim_loss_pct < 8 and avg_freq_gain > 5:
				print(f"  ⚠️  ACCEPTABLE: Moderate quality cost ({avg_sim_loss_pct:.1f}%) for moderate frequency benefit ({avg_freq_gain:.0f}x)")
		else:
				print(f"  ❌ POOR: High quality cost ({avg_sim_loss_pct:.1f}%) for limited frequency benefit ({avg_freq_gain:.0f}x)")
				print(f"     Consider reducing frequency weight from 0.3 to 0.2")
	else:
		print("\n  ℹ️  Frequency weighting made no changes (all clusters picked highest similarity)")

	print("="*100)

	df['canonical'] = df['cluster'].map(lambda c: cluster_canonicals[c]['canonical'])

	# ── Sanity check: Verify canonical belongs to its cluster ──────────────
	if verbose:
		print("\n[SANITY CHECK] Verifying canonical labels belong to their clusters...")
	
	mismatches = 0
	for cid, info in cluster_canonicals.items():
		cluster_labels_for_cid = df[df['cluster'] == cid]['label'].tolist()
		if info['canonical'] not in cluster_labels_for_cid:
			mismatches += 1
			# Fallback to the first label in the cluster
			fallback = cluster_labels_for_cid[0]
			if verbose:
				print(f"  ⚠️  Cluster {cid}: canonical '{info['canonical']}' not in cluster {cluster_labels_for_cid[:5]}... — using fallback '{fallback}'")
			cluster_canonicals[cid]['canonical'] = fallback
			# Update dataframe
			df.loc[df['cluster'] == cid, 'canonical'] = fallback
	
	if verbose:
		if mismatches > 0:
			print(f"  Fixed {mismatches} canonical mismatches")
		else:
			print(f"  ✅ All canonicals verified")

	df, X_clean, removed_labels = remove_problematic_cluster_labels(
		df=df,
		embeddings=X,
		low_cohesion_threshold=0.50,
		poor_canonical_threshold=0.60,
		verbose=True
	)

	out_csv = clusters_fname.replace(".csv", "_semantic_consolidation_agglomerative.csv")
	df.to_csv(out_csv, index=False)
	
	unique_labels_array = df['label'].values  # 36,657 labels
	cluster_labels = df['cluster'].values      # 36,657 cluster assignments
	canonical_map = df.groupby('cluster')['canonical'].first().to_dict()

	print("\nCOMPREHENSIVE CLUSTER QUALITY")
	print(f"  ├─ Updated cluster_labels: {len(np.unique(cluster_labels))} unique clusters")
	print(f"  ├─ Updated canonical_map: {len(canonical_map)} mappings")
	print(f"  ├─ unique_labels_array: {type(unique_labels_array)} {unique_labels_array.shape}")
	print(f"  ├─ cluster_labels: {type(cluster_labels)} {cluster_labels.shape}")
	print(f"  ├─ label_freq_dict: {len(label_freq_dict)} labels with frequencies")
	print(f"  ├─ df reports: {df['cluster'].nunique()} clusters")
	print(f"  └─ cluster_labels reports: {len(np.unique(cluster_labels))} clusters")
	
	if df['cluster'].nunique() != len(np.unique(cluster_labels)):
		print(f"[WARNING] Mismatch detected! Analysis may be stale!")
	else:
		print(f"All consistent!")

	results = analyze_cluster_quality(
		embeddings=X_clean,									# 36,657 embeddings (matches!)
		labels=unique_labels_array,					# 36,657 labels
		cluster_assignments=cluster_labels,	# 36,657 assignments
		canonical_labels=canonical_map,
		original_label_counts=label_freq_dict,
		distance_metric='cosine',
		output_dir=os.path.dirname(clusters_fname),
		verbose=verbose,
	)

	cluster_quality_csv = clusters_fname.replace(".csv", "_cluster_quality_metrics.csv")
	results['cluster_metrics'].to_csv(cluster_quality_csv, index=False)

	if results['problematic_clusters']:
		if verbose:
			print(f"\n[WARNING] {len(results['problematic_clusters'])} types of problematic clusters detected! => Exporting for manual review")
		all_problematic_ids = []
		for issue in results['problematic_clusters']:
			if issue['severity'] in ['HIGH', 'MEDIUM']:
				all_problematic_ids.extend(issue['cluster_ids'])
	
		if all_problematic_ids:
			problematic_csv = clusters_fname.replace(".csv", "_problematic_clusters_review.csv")
			export_problematic_clusters(
				labels=unique_labels_array,
				cluster_assignments=cluster_labels,
				canonical_labels=canonical_map,
				problematic_cluster_ids=list(set(all_problematic_ids)),
				output_path=problematic_csv
			)

	if verbose:
		print(f"Clustered {len(df)} labels into {df['cluster'].nunique()} clusters")
		print(f"{type(df)} {df.shape} {list(df.columns)}")
		print(df.head(15))
		print(df.info())
		print(f"[CLUSTERING] Total Elapsed Time: {time.time()-st_t:.1f} sec")
		print("="*60)

	return df

def assign_canonical_labels(
	df: pd.DataFrame,
	X: np.ndarray,
	model,
	original_label_counts: Dict[str, int],
	verbose: bool = True,
) -> Tuple[Dict[int, Dict], int, int, List[float], List[float], List[Dict]]:
	"""
	Assign a canonical label to every cluster using a five-signal composite
	score, with optional virtual hypernym synthesis for modifier-only clusters.

	The core problem this solves
	----------------------------
	A cluster like ['black aircraft', 'white aircraft', 'yellow aircraft'] has
	its centroid in "coloured-aircraft" embedding space.  Pure centroid-nearest
	therefore picks a colour variant rather than 'aircraft'.

	Fix: derive a *virtual hypernym* ('aircraft') from the shared token core of
	the cluster, encode it on-the-fly, and let it compete alongside the real
	labels.  Five signals then vote:

	  1. Cosine similarity to centroid          (w=0.30)
	  2. Corpus frequency, log-normalised       (w=0.15)
	  3. Head-noun dominance across the cluster (w=0.20)
	  4. Lexical containment / hypernym-ness    (w=0.25)
	  5. Brevity (shorter → more general)       (w=0.10)

	Only the canonical *assignment* changes; the clustering itself is untouched.

	Parameters
	----------
	df : pd.DataFrame
		Must have columns ['label', 'cluster'] with contiguous cluster IDs.
	X : np.ndarray, shape (n_unique_labels, d)
		L2-normalised embeddings; row order matches df['label'].
	model : SentenceTransformer
		Already-loaded model, used only to encode virtual hypernyms (cheap,
		one short string per cluster that needs it).
	original_label_counts : Dict[str, int]
		Corpus frequency of every label across all documents.
	verbose : bool
		Print per-cluster decisions when True.

	Returns
	-------
	cluster_canonicals : Dict[int, Dict]
		{cid: {'canonical': str, 'score': float, 'size': int, 'virtual': bool}}
	virtual_used_count : int
		Number of clusters where a virtual hypernym was chosen.
	freq_changed_count : int
		Number of clusters where composite scoring overrode pure centroid-sim.
	total_sim_loss : List[float]
		Fractional similarity loss for each freq-changed cluster.
	total_freq_gain : List[float]
		Frequency multiplier for each freq-changed cluster.
	questionable_examples : List[Dict]
		Trades with >10% sim loss or <3x freq gain, for inspection.
	"""

	# ── Internal utilities ────────────────────────────────────────────────────

	def _shared_token_core(lbls: List[str], min_support: float = 0.5) -> List[str]:
		"""
		Return ordered tokens shared by >= min_support fraction of labels.

		Examples
		--------
		['black aircraft', 'white aircraft', 'yellow aircraft'] -> ['aircraft']
		['red cross badge', 'red cross banner', 'red cross sign'] -> ['red', 'cross']
		['bridge club', 'club', 'glee club'] -> ['club']
		"""
		n = len(lbls)
		threshold = max(2, int(np.ceil(min_support * n)))
		token_support: dict = {}
		for lbl in lbls:
			for tok in set(lbl.split()):
				token_support[tok] = token_support.get(tok, 0) + 1
		shared = {tok for tok, cnt in token_support.items() if cnt >= threshold}
		if not shared:
			return []
		# Recover left-to-right order from the longest label (most informative anchor)
		anchor = max(lbls, key=lambda l: len(l.split()))
		return [tok for tok in anchor.split() if tok in shared]

	def _virtual_hypernym(lbls: List[str], min_support: float = 0.5) -> Optional[str]:
		"""
		Synthesise a hypernym string from the shared token core, or return None
		if no useful core exists or the result is already a cluster member.
		"""
		core = _shared_token_core(lbls, min_support=min_support)
		if not core:
			return None
		candidate = " ".join(core)
		if candidate == max(lbls, key=len):   # no reduction achieved
			return None
		return candidate

	def _containment_scores(candidates: List[str], cluster_lbls: List[str]) -> np.ndarray:
		"""
		For each candidate, fraction of cluster members whose token set is a
		superset of the candidate's tokens.  Direct measure of hypernym-ness:
		'bridge' scores 1.0 in a cluster of 'X bridge' labels.
		"""
		cluster_token_sets = [set(lbl.split()) for lbl in cluster_lbls]
		scores = []
		for cand in candidates:
			cand_tokens = set(cand.split())
			subsumers = sum(1 for ts in cluster_token_sets if cand_tokens.issubset(ts))
			scores.append(subsumers / max(len(cluster_lbls), 1))
		return np.array(scores)

	# ── Main loop ─────────────────────────────────────────────────────────────

	print(f"\nCanonical labels per cluster")
	cluster_canonicals    = {}
	virtual_used_count    = 0
	freq_changed_count    = 0
	total_sim_loss        = []
	total_freq_gain       = []
	questionable_examples = []

	for cid in sorted(df.cluster.unique()):
		cluster_mask       = df.cluster == cid
		cluster_texts      = df[cluster_mask]['label'].tolist()
		cluster_indices    = df[cluster_mask].index.tolist()
		cluster_embeddings = X[cluster_indices]   # shape (n, d), L2-normalised
		cluster_size       = len(cluster_texts)

		if verbose:
			print(f"\n[Cluster {cid}] {cluster_size} labels:\n{cluster_texts}")

		# Centroid is always computed from real members only
		centroid = cluster_embeddings.mean(axis=0)

		# ── Build candidate pool ──────────────────────────────────────────
		# Real labels first; virtual hypernym appended if one can be derived.
		virtual_hypernym = None
		if cluster_size >= 3:
			vh = _virtual_hypernym(cluster_texts, min_support=0.5)
			if vh is not None and vh not in cluster_texts:
				virtual_hypernym = vh

		candidates    = cluster_texts + ([virtual_hypernym] if virtual_hypernym else [])
		virtual_flags = [False] * cluster_size + ([True] if virtual_hypernym else [])

		# Encode virtual hypernym on-the-fly (one short string — cheap)
		if virtual_hypernym is not None:
			vh_emb = model.encode(
				[virtual_hypernym],
				batch_size=1,
				convert_to_numpy=True,
				normalize_embeddings=True,
				precision='float32',
			)[0]
			all_embeddings = np.vstack([cluster_embeddings, vh_emb[np.newaxis, :]])
		else:
			all_embeddings = cluster_embeddings

		# ── Score 1: cosine similarity to centroid ────────────────────────
		similarities = cosine_similarity(centroid.reshape(1, -1), all_embeddings)[0]

		if original_label_counts and cluster_size > 1:

			# ── Score 2: frequency (log-normalised; virtual gets 0) ──────
			label_freqs = np.array([
				original_label_counts.get(c, 0) if not virtual_flags[i] else 0
				for i, c in enumerate(candidates)
			], dtype=float)
			freq_scores = np.log1p(label_freqs) / np.log1p(label_freqs.max() + 1e-12)

			# ── Score 3: head-noun dominance (proportional) ───────────────
			real_heads: dict = {}
			for lbl in cluster_texts:
				h = lbl.split()[-1]
				real_heads[h] = real_heads.get(h, 0) + 1
			head_scores = np.array([
				real_heads.get(c.split()[-1], 0) / cluster_size
				for c in candidates
			])

			# ── Score 4: containment / hypernym-ness ─────────────────────
			cont_scores = _containment_scores(candidates, cluster_texts)
			if virtual_hypernym is not None:
				# Virtual hypernym is contained in >= 50% of members by construction
				cont_scores[-1] = max(cont_scores[-1], 0.5)

			# ── Score 5: brevity (shorter = more general) ─────────────────
			token_lengths  = np.array([len(c.split()) for c in candidates], dtype=float)
			brevity_scores = 1.0 - (token_lengths - 1.0) / max(token_lengths.max(), 1)

			# ── Composite score ───────────────────────────────────────────
			# sim weight (0.30) is intentionally lower than the naive approach
			# (0.35+) because over-trusting the centroid is what caused the
			# colour-variant canonical problem in the first place.
			combined_scores = (
				0.30 * similarities
				+ 0.15 * freq_scores
				+ 0.20 * head_scores
				+ 0.25 * cont_scores
				+ 0.10 * brevity_scores
			)

			best_idx     = int(combined_scores.argmax())
			pure_sim_idx = int(similarities[:cluster_size].argmax())  # real labels only

			# Safety: only allow a real label to override pure-sim when freq gain >= 3x
			if best_idx != pure_sim_idx and not virtual_flags[best_idx]:
				real_freqs = np.array([original_label_counts.get(t, 1) for t in cluster_texts])
				if real_freqs[best_idx] / max(real_freqs[pure_sim_idx], 1) < 3.0:
					best_idx = pure_sim_idx

			# ── Bookkeeping ───────────────────────────────────────────────
			is_virtual_pick = virtual_flags[best_idx]

			if is_virtual_pick:
				virtual_used_count += 1
			elif best_idx != pure_sim_idx:
				freq_changed_count += 1
				real_freqs = np.array([original_label_counts.get(t, 1) for t in cluster_texts])
				sim_loss   = (similarities[pure_sim_idx] - similarities[best_idx]) / (similarities[pure_sim_idx] + 1e-12)
				freq_gain  = real_freqs[best_idx] / max(real_freqs[pure_sim_idx], 1)
				total_sim_loss.append(sim_loss)
				total_freq_gain.append(freq_gain)
				if sim_loss > 0.10 or freq_gain < 3.0:
					questionable_examples.append({
						'cluster_id':    cid,
						'pure_choice':   cluster_texts[pure_sim_idx],
						'freq_choice':   candidates[best_idx],
						'pure_freq':     real_freqs[pure_sim_idx],
						'freq_freq':     real_freqs[best_idx],
						'pure_sim':      similarities[pure_sim_idx],
						'freq_sim':      similarities[best_idx],
						'sim_loss':      sim_loss,
						'freq_gain':     freq_gain,
						'cluster_size':  cluster_size,
						'cluster_labels': cluster_texts,
					})

			if verbose:
				if virtual_hypernym is not None:
					print(f"  Virtual hypernym candidate: '{virtual_hypernym}'")
				if is_virtual_pick:
					print(f"  => Virtual hypernym selected!")
				elif best_idx != pure_sim_idx:
					print(f"  Score-based selection changed canonical:")
					print(f"    Pure similarity: {cluster_texts[pure_sim_idx]} (sim={similarities[pure_sim_idx]:.4f})")
					print(f"    Score-weighted:  {candidates[best_idx]} (sim={similarities[best_idx]:.4f})")
		else:
			# Singleton or no frequency data: fall back to pure centroid similarity
			best_idx        = int(similarities[:cluster_size].argmax())
			is_virtual_pick = False

		canonical = candidates[best_idx]
		cluster_canonicals[cid] = {
			'canonical': canonical,
			'score':     float(similarities[best_idx]),
			'size':      cluster_size,
			'virtual':   virtual_flags[best_idx],
		}

		if verbose:
			tag = " [VIRTUAL]" if virtual_flags[best_idx] else ""
			print(f"\t=> Selected Canonical: {canonical} (sim={similarities[best_idx]:.4f}){tag}")

	return (
		cluster_canonicals,
		virtual_used_count,
		freq_changed_count,
		total_sim_loss,
		total_freq_gain,
		questionable_examples,
	)

def cluster(
	labels: List[List[str]],
	model_id: str,
	clusters_fname: str,
	batch_size: int = 1024,
	device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
	nc: int = None,
	linkage_method: str = "ward",
	distance_metric: str = "euclidean",
	verbose: bool = True,
):
	st_t = time.time()
	if verbose:
		print(f"\n[AGGLOMERATIVE CLUSTERING] {len(labels)} samples")
		print(f"   ├─ {model_id} | {device} | batch_size: {batch_size}")
		print(f"   ├─ linkage: {linkage_method}")
		print(f"   ├─ sample: {labels[:5]}")
		requires_type_exchange = isinstance(labels[0], str)
		print(f"   ├─────> {type(labels[0])} requires_type_exchange: {requires_type_exchange}")
		print(f"   └─ nc: {nc} {f'Manually defined' if nc else '=> Adaptive Search'}")

	# =========================================================================
	# STEP 1: DEDUP + FLATTEN
	# =========================================================================
	print(f"\n[DEDUP] {len(labels)} {type(labels)} raw labels")
	documents = []
	for i, doc in enumerate(labels):
		if doc is None:
			continue
		if isinstance(doc, str):
			try:
				doc = ast.literal_eval(doc)
			except (ValueError, SyntaxError):
				print(f"doc[{i}]: Failed to parse '{doc}' (skipping)")
				continue
		if not isinstance(doc, list):
			print(f"doc[{i}]: Invalid type {type(doc)} (skipping)")
			continue
		documents.append(list(set(lbl for lbl in doc if lbl is not None)))

	unique_labels = sorted(set(label for doc in documents for label in doc))

	print(f"Total {type(documents)} documents: {len(documents)}")
	print(f"Unique {type(unique_labels)} labels: {len(unique_labels)}")
	print(f"Sample unique labels: {unique_labels[:15]}")
	print("-" * 100)

	# =========================================================================
	# STEP 2: LOAD MODEL + ENCODE
	# =========================================================================
	dtype = torch.float32
	if torch.cuda.is_available():
		dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
	if verbose:
		print(f"[INFO] {model_id} Dtype selection: {dtype}")

	def _optimal_attn_impl() -> str:
		if not torch.cuda.is_available():
			return "eager"
		major, minor = torch.cuda.get_device_capability()
		compute_cap = major + minor / 10
		if compute_cap >= 8.0:
			try:
				import flash_attn
				if verbose:
					print(f"[INFO] Flash Attention 2 available (compute {compute_cap})")
				return "flash_attention_2"
			except ImportError:
				if verbose:
					print(f"[WARN] Flash Attention 2 not installed (pip install flash-attn)")
		if compute_cap >= 7.0 and torch.__version__ >= "2.0.0":
			if verbose:
				print(f"[INFO] Using SDPA attention (compute {compute_cap}, PyTorch {torch.__version__})")
			return "sdpa"
		if verbose:
			print(f"[INFO] Using eager attention (compute {compute_cap})")
		return "eager"

	attn_impl = _optimal_attn_impl()
	if verbose:
		print(f"[INFO] {model_id} with {attn_impl} attention")

	print(f"\n[INIT] Loading Sentence Transformer {model_id}")
	model = SentenceTransformer(
		model_name_or_path=model_id,
		trust_remote_code=True,
		cache_folder=cache_directory[os.getenv('USER')],
		model_kwargs={"attn_implementation": attn_impl, "dtype": dtype} if "Qwen" in model_id else {},
		token=os.getenv("HUGGINGFACE_TOKEN"),
		tokenizer_kwargs={"padding_side": "left"},
	).to(device)

	print(f"[LOADED] {sum(p.numel() for p in model.parameters()):,} parameters")

	print(f"\n[ENCODING] {len(unique_labels)} labels | batch_size: {batch_size} | {device}")
	X = model.encode(
		unique_labels,
		batch_size=batch_size,
		show_progress_bar=verbose,
		convert_to_numpy=True,
		normalize_embeddings=True,
		precision='float32',
	)

	if np.isnan(X).any():
		nan_rows = np.where(np.isnan(X).any(axis=1))[0]
		print(f"\n❌ ERROR: {np.isnan(X).sum()} NaN values in embeddings!")
		for idx in nan_rows[:10]:
			print(f"  - {unique_labels[idx]}")
		raise ValueError("Cannot proceed with NaN embeddings")
	if np.isinf(X).any():
		raise ValueError(f"Infinite values detected ({np.isinf(X).sum()}) - numerical overflow!")
	if X.shape[0] == 0:
		raise ValueError("No embeddings generated")
	if np.allclose(X, 0):
		raise ValueError("All embeddings are zero vectors")

	print(f"Embeddings {type(X)} {X.shape} {X.dtype}")
	print(f"  ├─ Range: [{X.min():.4f}, {X.max():.4f}]")
	print(f"  ├─ Mean: {X.mean()}")
	print(f"  └─ Std: {X.std()}")

	# =========================================================================
	# STEP 3: LINKAGE MATRIX
	# =========================================================================
	print(f"[LINKAGE] {linkage_method} Agglomerative Clustering on: {X.shape} embeddings [takes a while...]")
	t0 = time.time()
	if linkage_method == "ward":
		Z = fastcluster.linkage(X, method='ward', metric='euclidean') if use_fastcluster \
			else linkage(X, method='ward', metric='euclidean')
	elif distance_metric == "cosine":
		distance_matrix = np.clip(1 - (X @ X.T), 0, 2)
		np.fill_diagonal(distance_matrix, 0)
		condensed_dist = squareform(distance_matrix, checks=False)
		Z = fastcluster.linkage(condensed_dist, method=linkage_method) if use_fastcluster \
			else linkage(condensed_dist, method=linkage_method)
		print(f"[LINKAGE] Using {linkage_method} linkage with {distance_metric} distance")
	elif distance_metric == "euclidean":
		Z = fastcluster.linkage(X, method=linkage_method, metric='euclidean') if use_fastcluster \
			else linkage(X, method=linkage_method, metric='euclidean')
		print(f"[LINKAGE] Using {linkage_method} linkage with Euclidean distance")
	else:
		raise ValueError(f"Unsupported distance metric: {distance_metric}")

	print(f"[LINKAGE] Z[{linkage_method}]: {type(Z)} {Z.shape} {Z.dtype} {Z.strides} {Z.itemsize} {Z.nbytes} | {time.time()-t0:.1f} sec")

	# =========================================================================
	# STEP 4: OPTIMAL NUMBER OF CLUSTERS
	# =========================================================================
	if nc is None:
		cluster_labels, stats = get_optimal_num_clusters(
			X=X,
			linkage_matrix=Z,
			target_intra_similarity=0.69,
			min_consolidation=3.8,
			max_consolidation=5.0,
			target_singleton_ratio=0.015,
			quality_vs_consolidation_weight=0.5,
			merge_singletons=True,
			verbose=verbose,
		)
		best_k = stats['n_clusters']
	else:
		best_k = nc
		print(f"\nUsing user-defined k={best_k} for {len(unique_labels)} labels")
		cluster_labels = fcluster(Z, best_k, criterion='maxclust') - 1

	print(f"\n[CLUSTERING] {len(np.unique(cluster_labels))} clusters")
	print(f"{cluster_labels.shape} {type(cluster_labels)} labels.")
	print(f"(min, max): ({cluster_labels.min()}, {cluster_labels.max()})")

	df = pd.DataFrame({'label': unique_labels, 'cluster': cluster_labels})

	# =========================================================================
	# STEP 5: LABEL FREQUENCY DICT
	# =========================================================================
	print(f"\n[CLUSTERING] {len(np.unique(cluster_labels))} clusters for {cluster_labels.shape} {type(cluster_labels)} labels. {cluster_labels.min()} {cluster_labels.max()}")
	label_freq_dict: dict = {}
	for doc in documents:
		for label in doc:
			label_freq_dict[label] = label_freq_dict.get(label, 0) + 1

	print(f"\tComputed frequencies for {len(label_freq_dict)} labels")
	print(f"\tTotal label instances: {sum(label_freq_dict.values())}")
	print(f"\tMost frequent: {max(label_freq_dict.items(), key=lambda x: x[1])}")
	print('-' * 150)

	# =========================================================================
	# STEP 6: CANONICAL SELECTION (with virtual hypernym synthesis)
	# =========================================================================
	t0 = time.time()
	(
		cluster_canonicals,
		virtual_used_count,
		freq_changed_count,
		total_sim_loss,
		total_freq_gain,
		questionable_examples,
	) = assign_canonical_labels(
		df=df,
		X=X,
		model=model,
		original_label_counts=label_freq_dict,
		verbose=verbose,
	)
	print(f"\n[CLUSTERING] {len(cluster_canonicals)} cluster canonicals computed in {time.time()-t0:.1f} sec.")
	print("-" * 100)

	# =========================================================================
	# STEP 7: IMPACT ANALYSIS
	# =========================================================================
	total_clusters = len(df.cluster.unique())
	print("\nFREQUENCY WEIGHTING IMPACT ANALYSIS")
	print(f"  Total clusters analyzed: {total_clusters}")
	print(f"  Virtual hypernym used as canonical: {virtual_used_count} ({virtual_used_count/total_clusters*100:.1f}%)")
	print(f"  Clusters where score changed the canonical: {freq_changed_count} ({freq_changed_count/total_clusters*100:.1f}%)")

	if total_sim_loss:
		print(f"\nSIMILARITY LOSS IMPACT:")
		print(f"  Average  {np.mean(total_sim_loss)*100:.2f}%")
		print(f"  Median   {np.median(total_sim_loss)*100:.2f}%")
		print(f"  Max      {np.max(total_sim_loss)*100:.2f}%")
		print(f"  Min      {np.min(total_sim_loss)*100:.2f}%")

		print(f"\nFREQUENCY GAIN BENEFIT:")
		print(f"  Average {np.mean(total_freq_gain):.1f}x")
		print(f"  Median  {np.median(total_freq_gain):.1f}x")
		print(f"  Max     {np.max(total_freq_gain):.1f}x")
		print(f"  Min     {np.min(total_freq_gain):.1f}x")

		print(f"\nQUALITY ASSESSMENT:")
		excellent_trades    = sum(1 for s, f in zip(total_sim_loss, total_freq_gain) if s < 0.03 and f > 10)
		good_trades         = sum(1 for s, f in zip(total_sim_loss, total_freq_gain) if s < 0.05 and f > 5)
		questionable_trades = sum(1 for s, f in zip(total_sim_loss, total_freq_gain) if s > 0.10 or f < 2)
		print(f"  Excellent trades (<3% sim loss, >10x freq gain): {excellent_trades:<10} ({excellent_trades/freq_changed_count*100:.1f}%)")
		print(f"  Good trades (<5% sim loss, >5x freq gain):       {good_trades:<10} ({good_trades/freq_changed_count*100:.1f}%)")
		print(f"  Questionable trades (>10% sim loss or <2x gain): {questionable_trades:<10} ({questionable_trades/freq_changed_count*100:.1f}%)")

		if questionable_trades > 0 and verbose:
			print(f"\n[WARNING] {questionable_trades} questionable trades detected:")
			print(f"\t=> Consider adjusting weighting if this is high\n")
			print(f"{'Cluster':<10} {'Pure Sim Choice':<35} {'Score-Weighted Choice':<35} {'Sim Loss(%)':<15} {'Freq Gain'}")
			print("-" * 110)
			for ex in sorted(questionable_examples, key=lambda x: x['sim_loss'], reverse=True):
				print(f"{ex['cluster_id']:<10} {ex['pure_choice'][:32]:<35} {ex['freq_choice'][:32]:<35} {ex['sim_loss']*100:<15.1f} {ex['freq_gain']:.1f}x")

			high_loss_low_gain  = [ex for ex in questionable_examples if ex['sim_loss'] > 0.10 and ex['freq_gain'] < 2]
			high_loss_good_gain = [ex for ex in questionable_examples if ex['sim_loss'] > 0.10 and ex['freq_gain'] >= 2]
			low_loss_low_gain   = [ex for ex in questionable_examples if ex['sim_loss'] <= 0.10 and ex['freq_gain'] < 2]
			print(f"\nBREAKDOWN OF QUESTIONABLE TRADES:")
			print(f"Type A: High loss (>10%) + Low gain (<2x):   {len(high_loss_low_gain):<10}{len(high_loss_low_gain)/questionable_trades:<10.4f}BAD")
			print(f"Type B: High loss (>10%) + Good gain (>=2x): {len(high_loss_good_gain):<10}{len(high_loss_good_gain)/questionable_trades:<10.4f}DEBATABLE")
			print(f"Type C: Low loss  (<=10%) + Low gain (<2x):  {len(low_loss_low_gain):<10}{len(low_loss_low_gain)/questionable_trades:<10.4f}UNNECESSARY")
		else:
			print(f"\nAll trades are high-quality!")

		avg_sim_loss_pct = np.mean(total_sim_loss) * 100
		avg_freq_gain    = np.mean(total_freq_gain)
		print(f"\nOVERALL VERDICT:")
		if avg_sim_loss_pct < 3 and avg_freq_gain > 50:
			print(f"  ✅ EXCELLENT: Small quality cost ({avg_sim_loss_pct:.1f}%) for huge frequency benefit ({avg_freq_gain:.0f}x)")
		elif avg_sim_loss_pct < 5 and avg_freq_gain > 10:
			print(f"  ✅ GOOD: Acceptable quality cost ({avg_sim_loss_pct:.1f}%) for strong frequency benefit ({avg_freq_gain:.0f}x)")
		elif avg_sim_loss_pct < 8 and avg_freq_gain > 5:
			print(f"  ⚠️  ACCEPTABLE: Moderate quality cost ({avg_sim_loss_pct:.1f}%) for moderate frequency benefit ({avg_freq_gain:.0f}x)")
		else:
			print(f"  ❌ POOR: High quality cost ({avg_sim_loss_pct:.1f}%) for limited frequency benefit ({avg_freq_gain:.0f}x)")
			print(f"     Consider reducing frequency weight")
	else:
		print("\n  ℹ️  Score-based selection made no changes (all clusters picked highest similarity)")

	print("=" * 100)

	# =========================================================================
	# STEP 8: MAP CANONICALS + CLEAN PROBLEMATIC CLUSTERS
	# =========================================================================
	df['canonical'] = df['cluster'].map(lambda c: cluster_canonicals[c]['canonical'])

	# # ── Sanity check: Verify canonical belongs to its cluster ──────────────
	# if verbose:
	# 	print("\n[SANITY CHECK] Verifying canonical labels belong to their clusters...")
	
	# mismatches = 0
	# for cid, info in cluster_canonicals.items():
	# 	cluster_labels_for_cid = df[df['cluster'] == cid]['label'].tolist()
	# 	if info['canonical'] not in cluster_labels_for_cid:
	# 		mismatches += 1
	# 		# Fallback to the first label in the cluster
	# 		fallback = cluster_labels_for_cid[0]
	# 		if verbose:
	# 			print(f"[WARNING] Cluster {cid}: canonical '{info['canonical']}' not in cluster {cluster_labels_for_cid}\n\tfallback '{fallback}'")
	# 		cluster_canonicals[cid]['canonical'] = fallback
	# 		# Update dataframe
	# 		df.loc[df['cluster'] == cid, 'canonical'] = fallback
	
	# if verbose:
	# 	if mismatches > 0:
	# 		print(f"  Fixed {mismatches} canonical mismatches")
	# 	else:
	# 		print(f"[OK] All canonicals verified")


	df, X_clean, removed_labels = remove_problematic_cluster_labels(
		df=df,
		embeddings=X,
		low_cohesion_threshold=0.50,
		poor_canonical_threshold=0.60,
		verbose=True
	)


	df, X_clean, removed_labels = remove_problematic_cluster_labels(
		df=df,
		embeddings=X,
		low_cohesion_threshold=0.50,
		poor_canonical_threshold=0.60,
		verbose=True,
	)

	out_csv = clusters_fname.replace(".csv", "_semantic_consolidation_agglomerative.csv")
	df.to_csv(out_csv, index=False)

	unique_labels_array = df['label'].values
	cluster_labels      = df['cluster'].values
	canonical_map       = df.groupby('cluster')['canonical'].first().to_dict()

	# =========================================================================
	# STEP 9: COMPREHENSIVE CLUSTER QUALITY
	# =========================================================================
	print("\nCOMPREHENSIVE CLUSTER QUALITY")
	print(f"  ├─ Updated cluster_labels: {len(np.unique(cluster_labels))} unique clusters")
	print(f"  ├─ Updated canonical_map: {len(canonical_map)} mappings")
	print(f"  ├─ unique_labels_array: {type(unique_labels_array)} {unique_labels_array.shape}")
	print(f"  ├─ cluster_labels: {type(cluster_labels)} {cluster_labels.shape}")
	print(f"  ├─ label_freq_dict: {len(label_freq_dict)} labels with frequencies")
	print(f"  ├─ df reports: {df['cluster'].nunique()} clusters")
	print(f"  └─ cluster_labels reports: {len(np.unique(cluster_labels))} clusters")

	if df['cluster'].nunique() != len(np.unique(cluster_labels)):
		print(f"[WARNING] Mismatch detected! Analysis may be stale!")
	else:
		print(f"All consistent!")

	results = analyze_cluster_quality(
		embeddings=X_clean,
		labels=unique_labels_array,
		cluster_assignments=cluster_labels,
		canonical_labels=canonical_map,
		original_label_counts=label_freq_dict,
		distance_metric='cosine',
		output_dir=os.path.dirname(clusters_fname),
		verbose=verbose,
	)

	cluster_quality_csv = clusters_fname.replace(".csv", "_cluster_quality_metrics.csv")
	results['cluster_metrics'].to_csv(cluster_quality_csv, index=False)

	if results['problematic_clusters']:
		if verbose:
			print(f"\n[WARNING] {len(results['problematic_clusters'])} types of problematic clusters detected! => Exporting for manual review")
		all_problematic_ids = []
		for issue in results['problematic_clusters']:
			if issue['severity'] in ['HIGH', 'MEDIUM']:
				all_problematic_ids.extend(issue['cluster_ids'])
		if all_problematic_ids:
			problematic_csv = clusters_fname.replace(".csv", "_problematic_clusters_review.csv")
			export_problematic_clusters(
				labels=unique_labels_array,
				cluster_assignments=cluster_labels,
				canonical_labels=canonical_map,
				problematic_cluster_ids=list(set(all_problematic_ids)),
				output_path=problematic_csv,
			)

	if verbose:
		print(f"Clustered {len(df)} labels into {df['cluster'].nunique()} clusters")
		print(f"{type(df)} {df.shape} {list(df.columns)}")
		print(df.head(15))
		print(df.info())
		print(f"[CLUSTERING] Total Elapsed Time: {time.time()-st_t:.1f} sec")
		print("=" * 60)

	return df
