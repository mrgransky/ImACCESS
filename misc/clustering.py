import torch
import numpy as np
import pandas as pd
from collections import Counter
import warnings
import os
import ast
import json
import time
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
				print("\n" + "="*80)
				print("AUTOMATED CLUSTER VALIDATION (ZERO HUMAN INTERVENTION)")
				print("="*80)
				print(f"Dataset: {n_samples:,} unique labels → {n_clusters:,} clusters")
				print("="*80 + "\n")
		
		# Create DataFrame for analysis
		df = pd.DataFrame({
				'label': labels,
				'cluster': cluster_assignments
		})
		df['canonical'] = df['cluster'].map(canonical_labels)
		
		results = {}
		
		# =========================================================================
		# PART 1: CLUSTER QUALITY METRICS (Automated)
		# =========================================================================
		if verbose:
				print("📊 [PART 1/2] Automated Cluster Quality Assessment")
				print("-" * 80)
		
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
				print(f"  ✓ Intra-cluster cohesion:")
				print(f"      Mean: {cluster_quality['mean_intra_similarity']:.4f}")
				print(f"      Min:  {cluster_quality['min_intra_similarity']:.4f}")
				print(f"      Std:  {cluster_quality['std_intra_similarity']:.4f}")
				print(f"  ✓ Inter-cluster separation: {cluster_quality['mean_inter_cluster_distance']:.4f}")
				print(f"  ✓ Dunn Index: {cluster_quality['dunn_index']:.4f}")
				print(f"  ✓ Silhouette Score: {cluster_quality['silhouette_score']:.4f}")
				print(f"  ✓ Size distribution:")
				print(f"      Mean: {cluster_quality['size_mean']:.1f}")
				print(f"      CV: {cluster_quality['size_cv']:.3f} (lower is more balanced)")
				print(f"      Gini: {cluster_quality['size_gini']:.3f} (lower is more equal)")
				print(f"  ✓ Low cohesion clusters: {n_low_cohesion} ({cluster_quality['pct_low_cohesion']*100:.1f}%)")
				print()
		
		results['cluster_quality'] = cluster_quality
		
		# =========================================================================
		# PART 2: CANONICAL LABEL QUALITY METRICS (Automated)
		# =========================================================================
		if verbose:
				print("🏷️  [PART 2/2] Automated Canonical Label Quality Assessment")
				print("-" * 80)
		
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
		
		# =========================================================================
		# PART 3: OVERALL QUALITY SCORE & AUTOMATED DECISION
		# =========================================================================
		if verbose:
				print("🎯 [PART 3/3] Overall Quality Score & Automated Decision")
				print("-" * 80)
		
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
		
		# =========================================================================
		# PART 4: AUTOMATED RECOMMENDATIONS
		# =========================================================================
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
				recommendations.append({
						'issue': 'Poor Canonical Representativeness',
						'metric': f"Mean representativeness = {canonical_quality['mean_representativeness']:.3f} (target: >0.75)",
						'severity': 'HIGH',
						'action': "Switch to frequency-weighted centroid selection or alternative canonical selection method"
				})
		
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
			print(f"  📋 AUTOMATED RECOMMENDATIONS:")
			if not recommendations:
				print(f"    ✅ No critical issues detected. Clustering quality is acceptable.")
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
	verbose: bool = True
) -> Dict:
		"""
		Comprehensive cluster quality analysis for semantic label consolidation.
		
		This function evaluates clustering quality from multiple perspectives:
		1. Global clustering metrics (silhouette, DB index, CH index)
		2. Per-cluster quality metrics (cohesion, separation, size distribution)
		3. Semantic quality (intra-cluster similarity, canonical representativeness)
		4. Label consolidation impact (coverage, reduction ratio)
		5. Anomaly detection (outlier clusters, suspicious merges)
		
		Parameters
		----------
		embeddings : np.ndarray, shape (n_samples, embedding_dim)
				Normalized embeddings of all unique labels
		labels : np.ndarray, shape (n_samples,)
				Original label strings corresponding to embeddings
		cluster_assignments : np.ndarray, shape (n_samples,)
				Cluster ID for each label (from agglomerative clustering)
		canonical_labels : Dict[int, str]
				Mapping from cluster_id -> canonical label string
		original_label_counts : Dict[str, int], optional
				Original frequency of each label in the full dataset
				(before deduplication). Used for weighted analysis.
		distance_metric : str, default='cosine'
				Distance metric for quality evaluation ('cosine' or 'euclidean')
		verbose : bool, default=True
				Print detailed analysis report
		
		Returns
		-------
		analysis_results : Dict
				Comprehensive dictionary containing:
				- 'global_metrics': Overall clustering quality scores
				- 'cluster_metrics': Per-cluster quality DataFrame
				- 'problematic_clusters': List of clusters needing review
				- 'consolidation_impact': Label reduction statistics
				- 'recommendations': Actionable insights
				- 'summary': Executive summary string
		
		Example
		-------
		>>> results = analyze_cluster_quality(
		...     embeddings=embeddings,
		...     labels=unique_labels,
		...     cluster_assignments=cluster_ids,
		...     canonical_labels=canonical_map,
		...     original_label_counts=label_freq_dict
		... )
		>>> print(results['summary'])
		>>> problematic = results['problematic_clusters']
		>>> cluster_df = results['cluster_metrics']
		"""
		
		n_samples = len(labels)
		n_clusters = len(np.unique(cluster_assignments))
		
		if verbose:
				print("\n" + "="*80)
				print("CLUSTER QUALITY ANALYSIS REPORT")
				print("="*80)
				print(f"Dataset: {n_samples:,} unique labels → {n_clusters:,} clusters")
				print(f"Reduction ratio: {n_samples/n_clusters:.2f}x")
				print("="*80 + "\n")
		
		# -------------------------------------------------------------------------
		# 1. GLOBAL CLUSTERING METRICS
		# -------------------------------------------------------------------------
		if verbose:
				print("📊 [1/6] Computing Global Clustering Metrics...")
		
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
				print(f"  ✓ Silhouette Score:        {silhouette:.4f}  [{global_metrics['silhouette_interpretation']}]")
				print(f"  ✓ Davies-Bouldin Index:    {db_index:.4f}  [{global_metrics['db_interpretation']}]")
				print(f"  ✓ Calinski-Harabasz Index: {ch_index:.2f}  [{global_metrics['ch_interpretation']}]")
		
		# -------------------------------------------------------------------------
		# 2. PER-CLUSTER QUALITY METRICS
		# -------------------------------------------------------------------------
		if verbose:
				print("\n🔍 [2/6] Analyzing Per-Cluster Quality...")
		
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
				print(f"  ✓ Analyzed {n_clusters:,} clusters")
				print(f"  ✓ Avg cluster size: {cluster_df['size'].mean():.1f} (median: {cluster_df['size'].median():.0f})")
				print(f"  ✓ Avg intra-cluster similarity: {cluster_df['intra_cluster_similarity'].mean():.4f}")
				print(f"  ✓ Avg canonical representativeness: {cluster_df['canonical_representativeness'].mean():.4f}")
		
		# -------------------------------------------------------------------------
		# 3. IDENTIFY PROBLEMATIC CLUSTERS
		# -------------------------------------------------------------------------
		if verbose:
				print("\n⚠️  [3/6] Identifying Problematic Clusters...")
		
		problematic_clusters = []
		
		# Flag 1: Low cohesion (intra-cluster similarity < 0.5)
		low_cohesion = cluster_df[cluster_df['intra_cluster_similarity'] < 0.5]
		if len(low_cohesion) > 0:
				problematic_clusters.append({
						'issue': 'Low Cohesion',
						'count': len(low_cohesion),
						'cluster_ids': low_cohesion['cluster_id'].tolist(),
						'severity': 'HIGH',
						'description': 'Clusters with low internal similarity (< 0.5). May contain semantically diverse labels.'
				})
		
		# Flag 2: Poor canonical representativeness (< 0.6)
		poor_canonical = cluster_df[cluster_df['canonical_representativeness'] < 0.6]
		if len(poor_canonical) > 0:
				problematic_clusters.append({
						'issue': 'Poor Canonical Representativeness',
						'count': len(poor_canonical),
						'cluster_ids': poor_canonical['cluster_id'].tolist(),
						'severity': 'MEDIUM',
						'description': 'Canonical label does not represent cluster well (< 0.6 similarity).'
				})
		
		# Flag 3: Large diameter (> 0.8 cosine distance)
		large_diameter = cluster_df[cluster_df['cluster_diameter'] > 0.8]
		if len(large_diameter) > 0:
				problematic_clusters.append({
						'issue': 'Large Cluster Diameter',
						'count': len(large_diameter),
						'cluster_ids': large_diameter['cluster_id'].tolist(),
						'severity': 'MEDIUM',
						'description': 'Clusters with large spread (diameter > 0.8). May need splitting.'
				})
		
		# Flag 4: Singleton clusters (size = 1)
		singletons = cluster_df[cluster_df['size'] == 1]
		if len(singletons) > 0:
				problematic_clusters.append({
						'issue': 'Singleton Clusters',
						'count': len(singletons),
						'cluster_ids': singletons['cluster_id'].tolist(),
						'severity': 'LOW',
						'description': 'Clusters with only one label. No consolidation benefit.'
				})
		
		# Flag 5: Very large clusters (size > 95th percentile)
		size_threshold = cluster_df['size'].quantile(0.95)
		very_large = cluster_df[cluster_df['size'] > size_threshold]
		if len(very_large) > 0:
				problematic_clusters.append({
						'issue': 'Very Large Clusters',
						'count': len(very_large),
						'cluster_ids': very_large['cluster_id'].tolist(),
						'severity': 'LOW',
						'description': f'Clusters larger than 95th percentile (> {size_threshold:.0f} labels). May be over-merged.'
				})
		
		if verbose:
				if len(problematic_clusters) == 0:
						print("  ✓ No major issues detected!")
				else:
						for issue in problematic_clusters:
								print(f"  ⚠️  {issue['severity']:6s} | {issue['issue']:30s} | {issue['count']:4d} clusters")
		
		# -------------------------------------------------------------------------
		# 4. CONSOLIDATION IMPACT ANALYSIS
		# -------------------------------------------------------------------------
		if verbose:
				print("\n📉 [4/6] Analyzing Consolidation Impact...")
		
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
				print(f"  ✓ Label reduction: {n_samples:,} → {n_clusters:,} ({consolidation_impact['reduction_percentage']:.1f}% reduction)")
				print(f"  ✓ Avg consolidation: {consolidation_impact['reduction_ratio']:.2f} labels per cluster")
				print(f"  ✓ Singleton clusters: {len(singletons):,} ({consolidation_impact['singleton_percentage']:.1f}%)")
		
		# -------------------------------------------------------------------------
		# 5. CLUSTER SIZE DISTRIBUTION ANALYSIS
		# -------------------------------------------------------------------------
		if verbose:
				print("\n📊 [5/6] Cluster Size Distribution...")
		
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
				print(f"  ✓ Min: {size_distribution['min']}, Q25: {size_distribution['q25']}, "
							f"Median: {size_distribution['median']}, Q75: {size_distribution['q75']}, "
							f"Q95: {size_distribution['q95']}, Max: {size_distribution['max']}")
		
		# -------------------------------------------------------------------------
		# 6. GENERATE RECOMMENDATIONS
		# -------------------------------------------------------------------------
		if verbose:
				print("\n💡 [6/6] Generating Recommendations...")
		
		recommendations = _generate_recommendations(
				global_metrics, 
				cluster_df, 
				problematic_clusters, 
				consolidation_impact
		)
		
		if verbose:
				for i, rec in enumerate(recommendations, 1):
						print(f"  {i}. {rec}")
		
		# -------------------------------------------------------------------------
		# GENERATE EXECUTIVE SUMMARY
		# -------------------------------------------------------------------------
		summary = _generate_summary(
				global_metrics, 
				consolidation_impact, 
				problematic_clusters,
				n_samples,
				n_clusters
		)
		
		if verbose:
				print("\n" + "="*80)
				print("EXECUTIVE SUMMARY")
				print("="*80)
				print(summary)
				print("="*80 + "\n")
		
		# -------------------------------------------------------------------------
		# RETURN COMPREHENSIVE RESULTS
		# -------------------------------------------------------------------------
		return {
				'global_metrics': global_metrics,
				'cluster_metrics': cluster_df,
				'problematic_clusters': problematic_clusters,
				'consolidation_impact': consolidation_impact,
				'size_distribution': size_distribution,
				'recommendations': recommendations,
				'summary': summary
		}

def _interpret_silhouette(score: float) -> str:
		"""Interpret silhouette score."""
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

def _interpret_db_index(score: float) -> str:
		"""Interpret Davies-Bouldin index."""
		if score < 0.5:
				return "EXCELLENT"
		elif score < 1.0:
				return "GOOD"
		elif score < 1.5:
				return "FAIR"
		else:
				return "POOR"

def _interpret_ch_index(score: float) -> str:
		"""Interpret Calinski-Harabasz index."""
		if score > 1000:
				return "EXCELLENT"
		elif score > 500:
				return "GOOD"
		elif score > 200:
				return "FAIR"
		else:
				return "WEAK"

def _flag_cluster_quality(row: pd.Series) -> str:
		"""Flag cluster quality based on metrics."""
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
						"⚠️  LOW OVERALL QUALITY: Consider increasing n_clusters or using a different linkage method."
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
		"""Generate executive summary."""
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

def eval_clusters(df, X):
		"""
		Comprehensive quality evaluation.
		"""
		cluster_sizes = df.groupby('cluster').size()
		
		metrics = {
				'n_clusters': len(cluster_sizes),
				'n_singletons': (cluster_sizes == 1).sum(),
				'singleton_ratio': (cluster_sizes == 1).sum() / len(cluster_sizes),
				'mean_size': cluster_sizes.mean(),
				'median_size': cluster_sizes.median(),
				'max_size': cluster_sizes.max(),
				'min_size': cluster_sizes.min(),
				'consolidation_ratio': len(df) / len(cluster_sizes),
		}
		
		# Intra-cluster similarity
		intra_sim = []
		for cid in df.cluster.unique():
				cluster_mask = df.cluster == cid
				if cluster_mask.sum() < 2:
						continue
				cluster_indices = df[cluster_mask].index.tolist()
				cluster_embeddings = X[cluster_indices]
				
				# Average pairwise cosine similarity
				sim_matrix = cosine_similarity(cluster_embeddings)
				avg_sim = (sim_matrix.sum() - len(cluster_indices)) / (len(cluster_indices) * (len(cluster_indices) - 1))
				intra_sim.append(avg_sim)
		
		metrics['mean_intra_similarity'] = np.mean(intra_sim)
		
		print("\n[QUALITY METRICS]")
		for k, v in metrics.items():
				print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
		
		return metrics

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
	plt.figure(figsize=(24, 15))
	dendrogram(
		linkage_matrix, 
		truncate_mode='lastp', 
		p=30, 
		show_leaf_counts=True, 
		color_threshold=super_cluster_distance
	)
	plt.axhline(
		y=super_cluster_distance, 
		color='#000000', 
		linestyle='--', 
		label=f'Cut at {super_cluster_distance:.4f} ({n_super_clusters} super-clusters)',
		linewidth=2.5,
		zorder=10

	)

	plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} Linkage)\n{n_super_clusters} Super-Clusters at distance={super_cluster_distance:.4f}')
	plt.xlabel('Cluster')
	plt.ylabel('Distance')
	plt.legend(loc='upper right', fontsize=12)
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

def get_optimal_num_clusters_old(
	X,
	linkage_matrix,
	max_cluster_size_ratio,
	min_cluster_size=2,
	merge_singletons=True,
	split_oversized=True,
	verbose=True
):
	if verbose:
		print("\nOPTIMAL CLUSTER SELECTION WITH ENHANCED SCORING")
		print(f"max_cluster_size_ratio: {max_cluster_size_ratio}")
		print(f"merge_singletons: {merge_singletons}")
		print(f"split_oversized: {split_oversized}")
		print(f"min_cluster_size: {min_cluster_size}")
		print(f"X: {type(X)} {X.shape} {X.dtype} {X.min():.1f} {X.max():.1f} {X.mean():.1f} {X.std():.1f}")
		print(f"Z: {type(linkage_matrix)} {linkage_matrix.shape} {linkage_matrix.dtype} {linkage_matrix.min():.1f} {linkage_matrix.max():.1f} {linkage_matrix.mean():.1f} {linkage_matrix.std():.1f}")

	# Adaptive range based on dataset size
	num_samples = X.shape[0]
	if num_samples > int(3e4):
		range_n_clusters = range(50, min(4001, num_samples // 15), 50)
	elif num_samples > int(2e4):
		range_n_clusters = range(20, min(751, num_samples // 25), 25)
	elif num_samples > int(5e3):
		range_n_clusters = range(10, min(251, num_samples // 15), 10)
	else:
		# extremely small datasets: more conservative range
		range_n_clusters = range(5, 161, 1)

	if verbose:
		print(f"\n[OPTIMAL K] Testing {len(range_n_clusters)} cluster configurations: {range_n_clusters}")
		print(f"\n{'k':<6} {'Sil':<8} {'CH':<8} {'DB':<8} {'Single':<8} {'MaxSize':<8} {'Score':<8}")
		print("=" * 70)

	results = []
	for n_clusters in range_n_clusters:
		labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
		if len(np.unique(labels)) < 2:
			continue
		# Compute clustering quality metrics
		sil = silhouette_score(X, labels, metric='cosine')
		ch = calinski_harabasz_score(X, labels)
		db = davies_bouldin_score(X, labels)
		# Cluster size statistics
		cluster_sizes = np.bincount(labels)
		n_singletons = np.sum(cluster_sizes == 1)
		n_small = np.sum(cluster_sizes < min_cluster_size)
		singleton_ratio = n_singletons / n_clusters
		small_ratio = n_small / n_clusters
		# Max cluster size penalty
		max_cluster_size = cluster_sizes.max()
		max_size_ratio = max_cluster_size / num_samples
		max_size_penalty = max(0, (max_size_ratio - max_cluster_size_ratio) / max_cluster_size_ratio)
		results.append(
			{
				'k': n_clusters,
				'sil': sil,
				'ch': ch,
				'db': db,
				'singletons': n_singletons,
				'singleton_ratio': singleton_ratio,
				'small_ratio': small_ratio,
				'max_size': max_cluster_size,
				'max_size_ratio': max_size_ratio,
				'max_size_penalty': max_size_penalty,
			}
		)
	if not results:
		raise ValueError("No valid cluster configurations found")

	# Normalize metrics to [0, 1]
	sil_scores = np.array([r['sil'] for r in results])
	ch_scores = np.array([r['ch'] for r in results])
	db_scores = np.array([r['db'] for r in results])
	sil_norm = (sil_scores - sil_scores.min()) / (sil_scores.max() - sil_scores.min() + 1e-10)
	ch_norm = (ch_scores - ch_scores.min()) / (ch_scores.max() - ch_scores.min() + 1e-10)
	db_norm = 1 - (db_scores - db_scores.min()) / (db_scores.max() - db_scores.min() + 1e-10)

	# Compute composite score with enhanced penalties
	for i, r in enumerate(results):
		# Weighted average of normalized metrics
		base_score = (
			0.25 * sil_norm[i] +
			0.25 * ch_norm[i] +
			0.20 * db_norm[i]
		)
		# Cap penalties to prevent negative scores
		singleton_penalty_capped = min(0.7 * r['singleton_ratio'], 0.6)
		small_penalty_capped = min(0.4 * r['small_ratio'], 0.4)
		max_size_penalty_capped = min(0.8 * r['max_size_penalty'], 0.8)
		# Final composite score
		r['composite'] = base_score * (1 - singleton_penalty_capped) * (1 - small_penalty_capped) * (1 - max_size_penalty_capped)
		if verbose:
			print(
				f"{r['k']:<6} {r['sil']:<8.4f} {r['ch']:<8.2f} {r['db']:<8.4f} "
				f"{r['singletons']:<8} {r['max_size']:<8} {r['composite']:<8.4f}"
			)

	# Select best composite score
	best = max(results, key=lambda x: x['composite'])
	if verbose:
		print(f"\n[OPTIMAL K] Selected: {best['k']} clusters")
		print(f"  ├─ Silhouette: {best['sil']:.4f}")
		print(f"  ├─ Calinski-Harabasz: {best['ch']:.2f}")
		print(f"  ├─ Davies-Bouldin: {best['db']:.4f}")
		print(f"  ├─ Singletons: {best['singletons']} ({best['singleton_ratio']*100:.1f}%)")
		print(f"  ├─ Max cluster size: {best['max_size']} ({best['max_size_ratio']*100:.1f}%)")
		print(f"  └─ Composite score: {best['composite']:.4f}")
	# Get initial labels
	labels = fcluster(linkage_matrix, best['k'], criterion='maxclust') - 1
	# STAGE 2: Merge singleton clusters into nearest neighbors
	if merge_singletons:
		if verbose:
			print("\nMERGING SINGLETON CLUSTERS")
		cluster_sizes = np.bincount(labels)
		singleton_ids = np.where(cluster_sizes == 1)[0]
		if len(singleton_ids) > 0:
			if verbose:
				print(f"\n[MERGE] Found {len(singleton_ids)} singleton clusters to merge...")
			# Compute cluster centroids
			unique_labels = np.unique(labels)
			centroids = np.array(
				[
					X[labels == cid].mean(axis=0)
					for cid in unique_labels
				]
			)
			# For each singleton, find nearest non-singleton cluster
			new_labels = labels.copy()
			merged_count = 0
			for singleton_id in singleton_ids:
				singleton_idx = np.where(labels == singleton_id)[0][0]
				singleton_vec = X[singleton_idx].reshape(1, -1)
				# Compute similarity to all cluster centroids
				sims = cosine_similarity(singleton_vec, centroids)[0]
				# Find nearest non-singleton cluster
				sorted_ids = np.argsort(sims)[::-1]
				for nearest_id in sorted_ids:
					if cluster_sizes[nearest_id] >= min_cluster_size:
						new_labels[singleton_idx] = nearest_id
						merged_count += 1
						if verbose:
							print(f"  ├─ Merged singleton {singleton_id} → cluster {nearest_id} (sim={sims[nearest_id]:.4f})")
						break
			# Relabel to remove gaps
			unique_new = np.unique(new_labels)
			label_map = {old: new for new, old in enumerate(unique_new)}
			labels = np.array([label_map[l] for l in new_labels])
			if verbose:
				print(f"\n[MERGE] Merged {merged_count} singletons")
				print(f"[MERGE] Reduced from {len(unique_labels)} to {len(unique_new)} clusters")
		else:
			if verbose:
				print("\n[MERGE] No singleton clusters found. Skipping...")
	# Split oversized clusters
	if split_oversized:
			if verbose:
				print("\n" + "="*80)
				print(f"SPLITTING OVERSIZED CLUSTERS (> {max_cluster_size_ratio*100:.0f}% of total)")
				print("="*80)
			cluster_sizes = np.bincount(labels)
			max_size_threshold = num_samples * max_cluster_size_ratio
			oversized = np.where(cluster_sizes > max_size_threshold)[0]
			if len(oversized) > 0:
					if verbose:
						print(f"\n[SPLIT] Found {len(oversized)} oversized clusters (>{max_size_threshold:.0f} items)...")
					final_labels = labels.copy()
					next_label = labels.max() + 1
					for cid in oversized:
							cluster_mask = labels == cid
							cluster_indices = np.where(cluster_mask)[0]
							cluster_X = X[cluster_indices]
							# Target: ~20-30 items per subcluster
							n_sub = max(2, int(cluster_sizes[cid] / 25))
							if verbose:
								print(f"\n  ├─ Splitting cluster {cid} ({cluster_sizes[cid]} items) → {n_sub} subclusters")
							# Re-cluster this subset
							sub_clusterer = AgglomerativeClustering(
									n_clusters=n_sub,
									metric='cosine',
									linkage='average'
							)
							sub_labels = sub_clusterer.fit_predict(cluster_X)
							# Relabel subclusters
							for sub_id in range(n_sub):
									sub_mask = sub_labels == sub_id
									sub_size = sub_mask.sum()
									final_labels[cluster_indices[sub_mask]] = next_label
									if verbose:
											print(f"  │  ├─ Subcluster {next_label}: {sub_size} items")
									next_label += 1
					labels = final_labels
					if verbose:
							print(f"\n[SPLIT] Final cluster count: {len(np.unique(labels))}")
			else:
					if verbose:
							print(f"\n[SPLIT] No oversized clusters found (max size: {cluster_sizes.max()}/{max_size_threshold:.0f})")
	# FINAL STATISTICS
	final_cluster_sizes = np.bincount(labels)
	final_n_clusters = len(np.unique(labels))
	final_singletons = np.sum(final_cluster_sizes == 1)
	final_max_size = final_cluster_sizes.max()
	stats = {
		'n_clusters': final_n_clusters,
		'n_singletons': final_singletons,
		'singleton_ratio': final_singletons / final_n_clusters,
		'max_cluster_size': final_max_size,
		'max_size_ratio': final_max_size / num_samples,
		'mean_cluster_size': num_samples / final_n_clusters,
		'consolidation_ratio': num_samples / final_n_clusters,
	}
	if verbose:
		print("\nCLUSTERING STATISTICS")
		print(f"  ├─ Total clusters: {stats['n_clusters']}")
		print(f"  ├─ Singletons: {stats['n_singletons']} ({stats['singleton_ratio']*100:.1f}%)")
		print(f"  ├─ Largest cluster: {stats['max_cluster_size']} items ({stats['max_size_ratio']*100:.1f}%)")
		print(f"  ├─ Mean cluster size: {stats['mean_cluster_size']:.1f}")
		print(f"  └─ Consolidation ratio: {stats['consolidation_ratio']:.1f}:1")
	return labels, stats

def get_optimal_num_clusters(
	X,
	linkage_matrix,
	max_cluster_size_ratio=0.025,
	min_cluster_size=2,
	merge_singletons=True,
	split_oversized=True,
	target_intra_similarity=0.82,  # Target quality
	min_consolidation=5.0,         # Minimum reduction ratio
	verbose=True
):
	"""
	Adaptive two-stage cluster selection optimized for semantic label clustering.
	
	Parameters:
	-----------
	target_intra_similarity : float
			Target mean intra-cluster similarity (default: 0.82)
			Algorithm stops when this is reached or improvement plateaus
	min_consolidation : float
			Minimum label reduction ratio (default: 5.0x)
			Prevents over-clustering
	"""
	
	if verbose:
		print("ADAPTIVE OPTIMAL CLUSTER SELECTION (TWO-STAGE)")
		print(f"   ├─ Target intra-cluster similarity: {target_intra_similarity:.3f}")
		print(f"   ├─ Minimum consolidation ratio: {min_consolidation:.1f}x")
		print(f"   ├─ Dataset: {type(X)} {X.shape} {X.dtype}")
		print(f"   └─ Linkage matrix: {type(linkage_matrix)} {linkage_matrix.shape} {linkage_matrix.dtype}")
	
	num_samples = X.shape[0]
	
	# STAGE 1: COARSE SEARCH - Find plateau region
	if verbose:
			print("\n[STAGE 1] COARSE SEARCH - Finding quality plateau")
			print("-" * 80)
	
	# Adaptive coarse range based on dataset size
	if num_samples > int(3e4):
			# Large datasets: test k = 1000, 2000, 3000, ..., 10000
			coarse_range = range(1000, min(10001, num_samples // 3), 1000)
	elif num_samples > int(1e4):
			# Medium datasets: test k = 500, 1000, 1500, ..., 5000
			coarse_range = range(500, min(5001, num_samples // 2), 500)
	elif num_samples > int(5e3):
			# Small datasets: test k = 100, 200, 300, ..., 1000
			coarse_range = range(100, min(1001, num_samples // 5), 100)
	else:
			# Very small datasets: test k = 20, 40, 60, ..., 200
			coarse_range = range(20, min(201, num_samples // 25), 20)
	
	if verbose:
			print(f"Testing {len(coarse_range)} configurations: {list(coarse_range)}")
			print(f"\n{'k':<8} {'IntraSim':<12} {'Consol':<10} {'MaxSize':<10} {'Status':<20}")
			print("-" * 80)
	
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
							# Compute pairwise similarities within cluster
							sim_matrix = cosine_similarity(cluster_X)
							# Mean of off-diagonal elements
							n = len(cluster_X)
							intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
							intra_sims.append(intra_sim)
			
			mean_intra_sim = np.mean(intra_sims) if intra_sims else 0
			
			# Cluster statistics
			cluster_sizes = np.bincount(labels)
			max_size = cluster_sizes.max()
			consolidation = num_samples / n_clusters
			
			coarse_results.append({
					'k': n_clusters,
					'intra_sim': mean_intra_sim,
					'consolidation': consolidation,
					'max_size': max_size
			})
			
			# Check if we've reached target or plateau
			status = ""
			if mean_intra_sim >= target_intra_similarity:
					status = "✓ TARGET REACHED"
					if plateau_k is None:
							plateau_k = n_clusters
			elif mean_intra_sim > best_intra_sim:
					best_intra_sim = mean_intra_sim
					status = "↑ Improving"
			elif mean_intra_sim < best_intra_sim - 0.005:
					status = "↓ Declining (stop soon)"
			else:
					status = "→ Plateau region"
					if plateau_k is None and mean_intra_sim > 0.80:
							plateau_k = n_clusters
			
			if verbose:
					print(f"{n_clusters:<8} {mean_intra_sim:<12.4f} {consolidation:<10.1f} "
								f"{max_size:<10} {status:<20}")
			
			# Early stopping: If we've been in plateau for 2 steps and met min consolidation
			if len(coarse_results) >= 3:
					recent_sims = [r['intra_sim'] for r in coarse_results[-3:]]
					if max(recent_sims) - min(recent_sims) < 0.01 and consolidation >= min_consolidation:
							if verbose:
									print(f"\n[STAGE 1] Plateau detected at k≈{n_clusters}. Moving to fine search.")
							break
	
	if not coarse_results:
			raise ValueError("No valid cluster configurations found in coarse search")
	
	# Determine search region for Stage 2
	if plateau_k is None:
			# Use k with best intra-similarity
			best_coarse = max(coarse_results, key=lambda x: x['intra_sim'])
			plateau_k = best_coarse['k']
	
	if verbose:
			print(f"\n[STAGE 1] Complete. Plateau region centered around k={plateau_k}")
	
	# STAGE 2: FINE SEARCH - Optimize within plateau region		
	if verbose:
		print("\n[STAGE 2] FINE SEARCH - Optimizing within plateau region")
		print("-" * 80)
	
	# Adaptive fine range based on dataset size
	if num_samples < 1000:
		# Small dataset: test around plateau_k with small steps
		fine_min = max(2, int(plateau_k * 0.5))
		fine_max = min(int(plateau_k * 2.0), num_samples // 2)
		fine_step = max(1, (fine_max - fine_min) // 10)  # ~10 test points
	else:
		# Large dataset: your original logic
		fine_min = max(int(plateau_k * 0.7), 100)
		fine_max = min(int(plateau_k * 1.3), num_samples // 3)
		fine_step = max(50, plateau_k // 50)

	fine_range = range(fine_min, fine_max + 1, fine_step)
	
	if verbose:
			print(f"Testing {len(fine_range)} configurations: {fine_min} to {fine_max} (step={fine_step})")
			print(f"\n{'k':<8} {'IntraSim':<12} {'Represent':<12} {'FreqAlign':<12} {'Score':<10}")
			print("-" * 80)
	
	fine_results = []
	
	for n_clusters in fine_range:
			labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
			
			if len(np.unique(labels)) < 2:
					continue
			
			# Compute detailed metrics
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
			
			# Compute global metrics
			sil = silhouette_score(X, labels, metric='cosine') if len(unique_labels) > 1 else 0
			
			# Cluster size statistics
			cluster_sizes = np.bincount(labels)
			max_size = cluster_sizes.max()
			consolidation = num_samples / n_clusters
			
			# Composite score: Prioritize intra-similarity, penalize over-consolidation
			score = mean_intra_sim * (1.0 if consolidation >= min_consolidation else 0.5)
			
			fine_results.append({
					'k': n_clusters,
					'intra_sim': mean_intra_sim,
					'silhouette': sil,
					'consolidation': consolidation,
					'max_size': max_size,
					'score': score
			})
			
			if verbose:
					print(f"{n_clusters:<8} {mean_intra_sim:<12.4f} {sil:<12.4f} "
								f"{consolidation:<12.1f} {score:<10.4f}")
	
	if not fine_results:
			raise ValueError("No valid cluster configurations found in fine search")
	
	# Select best configuration
	# Priority: intra_sim > target, then maximize score
	target_met = [r for r in fine_results if r['intra_sim'] >= target_intra_similarity]
	
	if target_met:
			# Among those meeting target, prefer higher consolidation (fewer clusters)
			best = min(target_met, key=lambda x: x['k'])
			if verbose:
					print(f"\n[STAGE 2] Selected k={best['k']} (meets target, minimal k)")
	else:
			# Otherwise, select highest intra-similarity
			best = max(fine_results, key=lambda x: x['intra_sim'])
			if verbose:
					print(f"\n[STAGE 2] Selected k={best['k']} (highest intra-similarity)")
	
	# STAGE 3: POST-PROCESSING (unchanged)
	optimal_k = best['k']
	labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust') - 1
	
	if verbose:
			print("\n" + "="*80)
			print(f"OPTIMAL K SELECTED: {optimal_k}")
			print(f"  ├─ Intra-cluster similarity: {best['intra_sim']:.4f}")
			print(f"  ├─ Silhouette score: {best['silhouette']:.4f}")
			print(f"  ├─ Consolidation ratio: {best['consolidation']:.1f}x")
			print(f"  └─ Max cluster size: {best['max_size']}")
			print("="*80)
	
	# Merge singletons (existing code)
	if merge_singletons:
			if verbose:
					print("\nMERGING SINGLETON CLUSTERS")
			cluster_sizes = np.bincount(labels)
			singleton_ids = np.where(cluster_sizes == 1)[0]
			
			if len(singleton_ids) > 0:
					if verbose:
							print(f"[MERGE] Found {len(singleton_ids)} singleton clusters to merge...")
					
					unique_labels = np.unique(labels)
					centroids = np.array([X[labels == cid].mean(axis=0) for cid in unique_labels])
					
					new_labels = labels.copy()
					merged_count = 0
					
					for singleton_id in singleton_ids:
							singleton_idx = np.where(labels == singleton_id)[0][0]
							singleton_vec = X[singleton_idx].reshape(1, -1)
							sims = cosine_similarity(singleton_vec, centroids)[0]
							sorted_ids = np.argsort(sims)[::-1]
							
							for nearest_id in sorted_ids:
									if cluster_sizes[nearest_id] >= min_cluster_size:
											new_labels[singleton_idx] = nearest_id
											merged_count += 1
											if verbose:
													print(f"  ├─ Merged singleton {singleton_id} → cluster {nearest_id} (sim={sims[nearest_id]:.4f})")
											break
					
					unique_new = np.unique(new_labels)
					label_map = {old: new for new, old in enumerate(unique_new)}
					labels = np.array([label_map[l] for l in new_labels])
					
					if verbose:
							print(f"[MERGE] Merged {merged_count} singletons")
							print(f"[MERGE] Reduced from {len(unique_labels)} to {len(unique_new)} clusters")
	
	# Split oversized clusters (existing code - optional, can be removed)
	if split_oversized:
			if verbose:
					print(f"\nSPLITTING OVERSIZED CLUSTERS (> {max_cluster_size_ratio*100:.0f}% of total)")
			
			cluster_sizes = np.bincount(labels)
			max_size_threshold = num_samples * max_cluster_size_ratio
			oversized = np.where(cluster_sizes > max_size_threshold)[0]
			
			if len(oversized) > 0:
					if verbose:
							print(f"[SPLIT] Found {len(oversized)} oversized clusters")
					
					final_labels = labels.copy()
					next_label = labels.max() + 1
					
					for cid in oversized:
							cluster_mask = labels == cid
							cluster_indices = np.where(cluster_mask)[0]
							cluster_X = X[cluster_indices]
							n_sub = max(2, int(cluster_sizes[cid] / 25))
							
							if verbose:
									print(f"  ├─ Splitting cluster {cid} ({cluster_sizes[cid]} items) → {n_sub} subclusters")
							
							sub_clusterer = AgglomerativeClustering(
									n_clusters=n_sub,
									metric='cosine',
									linkage='average'
							)
							sub_labels = sub_clusterer.fit_predict(cluster_X)
							
							for sub_id in range(n_sub):
									sub_mask = sub_labels == sub_id
									sub_size = sub_mask.sum()
									final_labels[cluster_indices[sub_mask]] = next_label
									if verbose:
											print(f"  │  ├─ Subcluster {next_label}: {sub_size} items")
									next_label += 1
					
					labels = final_labels
					if verbose:
							print(f"[SPLIT] Final cluster count: {len(np.unique(labels))}")
	
	# Final statistics
	final_cluster_sizes = np.bincount(labels)
	final_n_clusters = len(np.unique(labels))
	final_singletons = np.sum(final_cluster_sizes == 1)
	final_max_size = final_cluster_sizes.max()
	
	stats = {
			'n_clusters': final_n_clusters,
			'n_singletons': final_singletons,
			'singleton_ratio': final_singletons / final_n_clusters,
			'max_cluster_size': final_max_size,
			'max_size_ratio': final_max_size / num_samples,
			'mean_cluster_size': num_samples / final_n_clusters,
			'consolidation_ratio': num_samples / final_n_clusters,
	}
	
	if verbose:
			print("\nFINAL CLUSTERING STATISTICS")
			print(f"  ├─ Total clusters: {stats['n_clusters']}")
			print(f"  ├─ Singletons: {stats['n_singletons']} ({stats['singleton_ratio']*100:.1f}%)")
			print(f"  ├─ Largest cluster: {stats['max_cluster_size']} items ({stats['max_size_ratio']*100:.1f}%)")
			print(f"  ├─ Mean cluster size: {stats['mean_cluster_size']:.1f}")
			print(f"  └─ Consolidation ratio: {stats['consolidation_ratio']:.1f}:1")
	
	return labels, stats

def cluster(
	labels: List[List[str]],
	model_id: str,
	batch_size: int = 1024,
	device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
	clusters_fname: str = "clusters.csv",
	nc: int = None,
	linkage_method: str = "ward",  # 'average', 'complete', 'single', 'ward'
	distance_metric: str = "euclidean",  # 'cosine', 'euclidean'
	verbose: bool = True,
):	
	if verbose:
		print(f"\n[AGGLOMERATIVE CLUSTERING] {len(labels)} documents")
		print(f"   ├─ {model_id} | {device} | batch_size: {batch_size}")
		print(f"   ├─ linkage: {linkage_method}")
		# print(f"   ├─ distance: {distance_metric}")
		print(f"   ├─ sample: {labels[:5]}")
		# Parse first label if it's a string, otherwise use as-is
		requires_type_exchange = isinstance(labels[0], str)
		print(f"   ├─────> {type(labels[0])} requires_type_exchange: {requires_type_exchange}")
		print(f"   └─ nc: {nc} {f'Manually defined' if nc else '=> Adaptive Search'}")
	
	print(f"\n[DEDUP] {len(labels)} raw labels")
	documents = []
	for doc in labels:
		if isinstance(doc, str):
			doc = ast.literal_eval(doc)
		documents.append(list(set(lbl for lbl in doc)))
	
	# Flatten and deduplicate (deterministic and reproducible)
	unique_labels = sorted(set(label for doc in documents for label in doc))
	
	print(f"Total {type(documents)} documents: {len(documents)}")
	print(f"Unique {type(unique_labels)} labels: {len(unique_labels)}")
	print(f"Sample unique labels: {unique_labels[:15]}")
	
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
		model_kwargs={"attn_implementation": attn_impl, "dtype": dtype},
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
		cluster_labels, stats = get_optimal_num_clusters(
			X=X,
			linkage_matrix=Z,
			max_cluster_size_ratio=0.10,  # Max 10% of data in one cluster
			min_cluster_size=2,
			merge_singletons=True,
			split_oversized=False,
			verbose=verbose,
		)
		best_k = stats['n_clusters']
		
		print(f"\n[OPTIMAL K] Using optimized k={best_k} clusters")
		print(f"  ├─ Consolidation: {stats['consolidation_ratio']:.1f}:1")
		print(f"  ├─ Singletons: {stats['n_singletons']} ({stats['singleton_ratio']*100:.1f}%)")
		print(f"  └─ Max cluster size: {stats['max_cluster_size']} ({stats['max_size_ratio']*100:.1f}%)")
	else:
		best_k = nc
		print(f"\nUsing user-defined k={best_k} for {len(unique_labels)} labels")
	
		print(f"\nCutting dendrogram at k={best_k} for {len(unique_labels)} labels")
		cluster_labels = fcluster(Z, best_k, criterion='maxclust') - 1 # Convert to 0-indexed

	print(f"\n[CLUSTERING] {len(np.unique(cluster_labels))} clusters for {cluster_labels.shape} {type(cluster_labels)} labels. {cluster_labels.min()} {cluster_labels.max()}")

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
	
	print(f"\nCanonical labels per cluster")
	df = pd.DataFrame(
		{
			'label': unique_labels,
			'cluster': cluster_labels
		}
	)
	
	cluster_canonicals = {}
	
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
	print(label_freq_dict)

	original_label_counts = label_freq_dict
	print(f"\tComputed frequencies for {len(original_label_counts)} labels")
	print(f"\tTotal label instances: {sum(original_label_counts.values())}")
	print(f"\tMost frequent: {max(original_label_counts.items(), key=lambda x: x[1])}")
	print('-'*150)

	print(f"\nCanonical labels per cluster")
	cluster_canonicals = {}
	for cid in sorted(df.cluster.unique()):
		cluster_mask = df.cluster == cid
		cluster_texts = df[cluster_mask]['label'].tolist()
		cluster_indices = df[cluster_mask].index.tolist()
		cluster_embeddings = X[cluster_indices]
		
		# Compute centroid
		centroid = cluster_embeddings.mean(axis=0, keepdims=True)
		similarities = cosine_similarity(centroid, cluster_embeddings)[0]
		
		# Frequency-Weighted Canonical Selection
		if original_label_counts is not None and len(original_label_counts) > 0 and len(cluster_texts) > 1:
			# Get label frequencies
			label_freqs = np.array([original_label_counts.get(lbl, 1) for lbl in cluster_texts])
			
			# Normalize frequencies to [0, 1] using log-scale (handles extreme distributions)
			freq_scores = np.log1p(label_freqs) / np.log1p(label_freqs.max())
			
			# Hybrid scoring: 70% similarity, 30% frequency
			combined_scores = 0.7 * similarities + 0.3 * freq_scores
			best_idx = combined_scores.argmax()
			
			if verbose:
				pure_sim_idx = similarities.argmax()
				if best_idx != pure_sim_idx:
					print(f"\n[Cluster {cid}] Frequency weighting changed selection:")
					print(f"  Pure similarity would pick: {cluster_texts[pure_sim_idx]} (sim={similarities[pure_sim_idx]:.4f}, freq={label_freqs[pure_sim_idx]})")
					print(f"  Frequency-weighted picks: {cluster_texts[best_idx]} (sim={similarities[best_idx]:.4f}, freq={label_freqs[best_idx]})")
		else:
				# Fallback: pure similarity (original method)
				best_idx = similarities.argmax()
		# ========================================================================
		
		canonical = cluster_texts[best_idx]
		
		cluster_canonicals[cid] = {
			'canonical': canonical,
			'score': float(similarities[best_idx]),
			'size': len(cluster_texts)
		}
		
		if verbose:
			print(f"\n[Cluster {cid}] {len(cluster_texts)} labels:\n{cluster_texts}")
			print(f"\tCanonical: {canonical} (sim={similarities[best_idx]:.4f})")

	df['canonical'] = df['cluster'].map(lambda c: cluster_canonicals[c]['canonical'])
	
	out_csv = clusters_fname.replace(".csv", "_semantic_consolidation_agglomerative.csv")
	df.to_csv(out_csv, index=False)
	try:
		df.to_excel(out_csv.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	
	eval_clusters(df, X)

	print("\nRUNNING COMPREHENSIVE CLUSTER QUALITY ANALYSIS\n")

	# Prepare data for analysis function
	unique_labels_array = np.array(unique_labels)  # Convert list to numpy array

	# Extract simple canonical mapping (cluster_id -> canonical_label_string)
	canonical_map = {
		cid: info['canonical'] 
		for cid, info in cluster_canonicals.items()
	}

	print(f"Prepared data for analysis:")
	print(f"  ├─ unique_labels_array: {type(unique_labels_array)} {unique_labels_array.shape}")
	print(f"  ├─ cluster_labels: {type(cluster_labels)} {cluster_labels.shape}")
	print(f"  ├─ canonical_map: {len(canonical_map)} mappings")
	print(f"  └─ label_freq_dict: {len(label_freq_dict)} labels with frequencies")

	# Run comprehensive analysis
	results = analyze_cluster_quality(
		embeddings=X,
		labels=unique_labels_array,  # FIXED: Use numpy array
		cluster_assignments=cluster_labels,  # FIXED: Use correct variable name
		canonical_labels=canonical_map,  # FIXED: Use simple dict
		original_label_counts=label_freq_dict,  # FIXED: Use computed frequencies
		distance_metric='cosine',
		verbose=True
	)	
	cluster_quality_csv = clusters_fname.replace(".csv", "_cluster_quality_metrics.csv")
	results['cluster_metrics'].to_csv(cluster_quality_csv, index=False)

	# Export problematic clusters if any
	if results['problematic_clusters']:
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

	automated_cluster_validation(
		embeddings=X,
		labels=unique_labels_array,
		cluster_assignments=cluster_labels,
		canonical_labels=canonical_map,
		original_label_counts=label_freq_dict,
		verbose=True
	)

	if verbose and "discharge" in unique_labels and "hospital discharge" in unique_labels:
		# Check if "discharge" and "hospital discharge" are in the SAME cluster
		discharge_cluster = df[df['label'] == 'discharge']['cluster'].iloc[0]
		hospital_discharge_cluster = df[df['label'] == 'hospital discharge']['cluster'].iloc[0]

		print(f"discharge cluster: {discharge_cluster}")
		print(f"hospital discharge cluster: {hospital_discharge_cluster}")
		print(f"Same cluster? {discharge_cluster == hospital_discharge_cluster}")

		if discharge_cluster == hospital_discharge_cluster:
			print("❌ THEY ARE STILL TOGETHER!")
			canonical = df[df['cluster'] == discharge_cluster]['canonical'].iloc[0]
			print(f"Canonical: {canonical}")

	return df