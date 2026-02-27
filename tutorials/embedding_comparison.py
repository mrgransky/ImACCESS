import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to the JSON file exported by analyze_cluster_quality()
LOW_COHESION_JSON = "/scratch/project_2004072/ImACCESS/_WW_DATASETs/HISTORY_X4/outputs/low_cohesion_clusters.json"

# Models to test
MODELS_TO_TEST = {
	'MPNet-Base': 'sentence-transformers/all-mpnet-base-v2',          # Current (768-dim)
	'E5-Large-v2': 'intfloat/e5-large-v2',                            # Strong semantic (1024-dim)
	'E5-Mistral': 'intfloat/e5-mistral-7b-instruct',                  # SOTA (4096-dim)
	'BGE-Large': 'BAAI/bge-large-en-v1.5',                            # Chinese SOTA (1024-dim)
	'MiniLM-L12': "sentence-transformers/all-MiniLM-L12-v2",
	'MiniLM-L6': "sentence-transformers/all-MiniLM-L6-v2",
	# 'Jina-v3': 'jinaai/jina-embeddings-v3',                           # Long context (1024-dim)
	'Nomic-v1.5': 'nomic-ai/nomic-embed-text-v1.5',                   # Vision-compatible (768-dim)
	'Qwen-Embedding-0.6B': 'Qwen/Qwen3-Embedding-0.6B',                   # 0.6B parameters (1024-dim)
}

# Cache directory
CACHE_DIR = "/scratch/project_2004072/models"

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# 1. LOAD LOW COHESION CLUSTERS FROM JSON
# ============================================================================

print("="*80)
print("LOADING LOW COHESION CLUSTERS FROM JSON")
print("="*80)

with open(LOW_COHESION_JSON, 'r', encoding='utf-8') as f:
		low_cohesion_data = json.load(f)

# Convert to simpler format: {cluster_id: [labels]}
LOW_COHESION_CLUSTERS = {
		int(cid): data['labels'] 
		for cid, data in low_cohesion_data.items()
}

print(f"\n✓ Loaded {len(LOW_COHESION_CLUSTERS)} low-cohesion clusters")
print(f"  Total labels: {sum(len(labels) for labels in LOW_COHESION_CLUSTERS.values())}")
print(f"  Avg cluster size: {np.mean([len(labels) for labels in LOW_COHESION_CLUSTERS.values()]):.1f}")
print(f"  Device: {DEVICE}")

# Show first 5 examples
print("\nFirst 5 clusters:")
for i, (cid, labels) in enumerate(list(LOW_COHESION_CLUSTERS.items())[:5], 1):
		print(f"  {i}. Cluster {cid}: {labels}")

# ============================================================================
# 2. LOAD EMBEDDING MODELS
# ============================================================================

print("\n" + "="*80)
print("LOADING EMBEDDING MODELS")
print("="*80)

models = {}
for model_name, model_id in MODELS_TO_TEST.items():
		try:
				print(f"\n[{model_name}] Loading {model_id}...")
				model = SentenceTransformer(
					model_id,
					trust_remote_code=True,
					device=DEVICE,
					cache_folder=CACHE_DIR,
					model_kwargs={'dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32}
				)
				
				# # Special handling for large models
				# if 'mistral' in model_id.lower():
				# 		model = SentenceTransformer(
				# 				model_id,
				# 				trust_remote_code=True,
				# 				device=DEVICE,
				# 				cache_folder=CACHE_DIR,
				# 				model_kwargs={'dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32}
				# 		)
				# else:
				# 		model = SentenceTransformer(
				# 				model_id,
				# 				trust_remote_code=True,
				# 				device=DEVICE,
				# 				cache_folder=CACHE_DIR
				# 		)
				
				models[model_name] = model
				dim = model.get_sentence_embedding_dimension()
				print(f"  ✓ Loaded. Embedding dim: {dim}")
		
		except Exception as e:
				print(f"  ✗ Failed: {e}")
				print(f"  Skipping {model_name}")

print(f"\n✓ Successfully loaded {len(models)}/{len(MODELS_TO_TEST)} models")

# ============================================================================
# 3. ANALYSIS FUNCTION
# ============================================================================

def analyze_cluster_with_model(cluster_id, labels, model, model_name):
		"""Analyze a single cluster with a specific embedding model."""
		try:
				# Encode labels
				embeddings = model.encode(
						labels,
						convert_to_numpy=True,
						normalize_embeddings=True,
						show_progress_bar=False
				)
				
				# Compute similarity matrix
				sim_matrix = cosine_similarity(embeddings)
				
				# Compute metrics
				n = len(labels)
				if n > 1:
						# Intra-cluster similarity (mean of off-diagonal)
						intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
						
						# Min and max (excluding diagonal)
						mask = ~np.eye(n, dtype=bool)
						min_sim = sim_matrix[mask].min()
						max_sim = sim_matrix[mask].max()
						std_sim = sim_matrix[mask].std()
				else:
						intra_sim = 1.0
						min_sim = 1.0
						max_sim = 1.0
						std_sim = 0.0
				
				return {
						'cluster_id': cluster_id,
						'model': model_name,
						'n_labels': n,
						'intra_sim': intra_sim,
						'min_sim': min_sim,
						'max_sim': max_sim,
						'std_sim': std_sim,
						'labels': labels
				}
		
		except Exception as e:
				print(f"  Error analyzing cluster {cluster_id} with {model_name}: {e}")
				return None

# ============================================================================
# 4. RUN ANALYSIS ON ALL CLUSTERS
# ============================================================================

print("\n" + "="*80)
print("ANALYZING LOW-COHESION CLUSTERS")
print("="*80)

results = []
total_clusters = len(LOW_COHESION_CLUSTERS)

for idx, (cluster_id, labels) in enumerate(LOW_COHESION_CLUSTERS.items(), 1):
		# Show progress every 50 clusters
		if idx % 50 == 0 or idx == 1:
				print(f"\n[{idx}/{total_clusters}] Processing cluster {cluster_id}...")
		
		cluster_results = {}
		
		for model_name, model in models.items():
				result = analyze_cluster_with_model(cluster_id, labels, model, model_name)
				if result:
						results.append(result)
						cluster_results[model_name] = result['intra_sim']
		
		# Show details for first 10 clusters
		if idx <= 10:
				print(f"  Cluster {cluster_id}: {labels}")
				for model_name, sim in cluster_results.items():
						print(f"    {model_name:<15} intra-sim: {sim:.4f}")
				
				# Find best model (LOWEST similarity = best separation)
				if cluster_results:
						best_model = min(cluster_results.items(), key=lambda x: x[1])
						print(f"    → Best (lowest): {best_model[0]} ({best_model[1]:.4f}) ✓")

print(f"\n✓ Analysis complete!")

# ============================================================================
# 5. CREATE SUMMARY DATAFRAME
# ============================================================================

print("\n" + "="*80)
print("CREATING SUMMARY")
print("="*80)

df_results = pd.DataFrame(results)

# Pivot for comparison
pivot = df_results.pivot(
		index='cluster_id',
		columns='model',
		values='intra_sim'
)

# Add metadata columns
first_results = df_results.drop_duplicates('cluster_id').set_index('cluster_id')
pivot['n_labels'] = first_results['n_labels']
pivot['labels_preview'] = first_results['labels'].apply(
		lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '')
)

# Add analysis columns
pivot['MPNet_sim'] = pivot.get('MPNet-Base', np.nan)
pivot['Best_sim'] = pivot[list(models.keys())].min(axis=1)
pivot['Best_model'] = pivot[list(models.keys())].idxmin(axis=1)
pivot['Improvement_over_MPNet'] = ((pivot['MPNet_sim'] - pivot['Best_sim']) / pivot['MPNet_sim'] * 100)

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\n1. AVERAGE INTRA-SIMILARITY (Lower = Better Separation)")
print("-" * 80)
for model_name in models.keys():
		if model_name in pivot.columns:
				avg_sim = pivot[model_name].mean()
				status = "✅ EXCELLENT" if avg_sim < 0.45 else "✓ GOOD" if avg_sim < 0.55 else "⚠️ MODERATE" if avg_sim < 0.65 else "❌ POOR"
				print(f"  {model_name:<15} {avg_sim:.4f}  {status}")

print("\n2. WINNER COUNT (Most clusters with lowest similarity)")
print("-" * 80)
winner_counts = pivot['Best_model'].value_counts()
for model_name, count in winner_counts.items():
		pct = count / len(pivot) * 100
		print(f"  {model_name:<15} {count:3d} wins ({pct:5.1f}%)")

print("\n3. IMPROVEMENT OVER MPNet-Base")
print("-" * 80)
avg_improvement = pivot['Improvement_over_MPNet'].mean()
print(f"  Average improvement: {avg_improvement:.2f}%")
print(f"  Max improvement: {pivot['Improvement_over_MPNet'].max():.2f}%")
print(f"  Min improvement: {pivot['Improvement_over_MPNet'].min():.2f}%")
print(f"  Clusters with >10% improvement: {(pivot['Improvement_over_MPNet'] > 10).sum()}")
print(f"  Clusters with >20% improvement: {(pivot['Improvement_over_MPNet'] > 20).sum()}")

# ============================================================================
# 7. CATEGORY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CATEGORY ANALYSIS (by MPNet-Base similarity)")
print("="*80)

mpnet_sim = pivot['MPNet_sim']
categories = {
		'Extremely Low (< 0.40)': mpnet_sim < 0.40,
		'Very Low (0.40-0.45)': (mpnet_sim >= 0.40) & (mpnet_sim < 0.45),
		'Low (0.45-0.50)': (mpnet_sim >= 0.45) & (mpnet_sim < 0.50),
}

for category_name, mask in categories.items():
		n_clusters = mask.sum()
		if n_clusters == 0:
				continue
		
		print(f"\n{category_name}: {n_clusters} clusters")
		print("-" * 80)
		
		category_data = pivot[mask]
		
		# Best model for this category
		best_models = category_data['Best_model'].value_counts()
		print("  Best model distribution:")
		for model, count in best_models.items():
				pct = count / n_clusters * 100
				print(f"    {model:<15} {count:3d} clusters ({pct:5.1f}%)")
		
		# Average improvement
		avg_imp = category_data['Improvement_over_MPNet'].mean()
		print(f"  Average improvement over MPNet: {avg_imp:.2f}%")

# ============================================================================
# 8. DETAILED EXAMPLES (Top 20 Worst MPNet-Base Performers)
# ============================================================================

print("\n" + "="*80)
print("TOP 20 WORST MPNet-BASE PERFORMERS")
print("="*80)

# Sort by MPNet similarity (highest = worst separation)
worst_clusters = pivot.nlargest(20, 'MPNet_sim')

for idx, (cluster_id, row) in enumerate(worst_clusters.iterrows(), 1):
		labels = LOW_COHESION_CLUSTERS[cluster_id]
		
		print(f"\n{idx}. Cluster {cluster_id}: {labels}")
		print("-" * 80)
		
		for model_name in models.keys():
				if model_name in row:
						sim = row[model_name]
						is_best = (model_name == row['Best_model'])
						marker = " ← BEST" if is_best else ""
						print(f"  {model_name:<15} {sim:.4f}{marker}")
		
		improvement = row['Improvement_over_MPNet']
		print(f"  Improvement: {improvement:.2f}%")

# ============================================================================
# 9. EXPORT RESULTS
# ============================================================================

output_csv = LOW_COHESION_JSON.replace('.json', '_model_comparison.csv')
pivot.to_csv(output_csv)
print(f"\n✓ Results exported to: {output_csv}")

# Also export detailed results
detailed_csv = LOW_COHESION_JSON.replace('.json', '_model_comparison_detailed.csv')
df_results.to_csv(detailed_csv, index=False)
print(f"✓ Detailed results exported to: {detailed_csv}")

# ============================================================================
# 10. FINAL RECOMMENDATION
# ============================================================================

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

# Find overall best model
best_model_name = winner_counts.idxmax()
best_model_wins = winner_counts.max()
best_model_pct = best_model_wins / len(pivot) * 100
best_model_avg_sim = pivot[best_model_name].mean()
mpnet_avg_sim = pivot['MPNet_sim'].mean()

print(f"\nBest Performing Model: {best_model_name}")
print(f"  Wins: {best_model_wins}/{len(pivot)} clusters ({best_model_pct:.1f}%)")
print(f"  Avg intra-sim: {best_model_avg_sim:.4f} (MPNet: {mpnet_avg_sim:.4f})")
print(f"  Avg improvement: {avg_improvement:.2f}%")
print(f"  Separation gain: {(mpnet_avg_sim - best_model_avg_sim) / mpnet_avg_sim * 100:.1f}%")

# Decision logic
if best_model_name != 'MPNet-Base':
		if avg_improvement > 15 and best_model_pct > 60:
				print(f"\n✅ STRONG RECOMMENDATION: Switch to {best_model_name}")
				print(f"   - Dominates {best_model_pct:.0f}% of problematic clusters")
				print(f"   - Average {avg_improvement:.1f}% improvement in separation")
				print(f"   - Will significantly reduce low-cohesion clusters")
				print(f"\n   NEXT STEPS:")
				print(f"   1. Re-run clustering with {best_model_name}")
				print(f"   2. Expected: <100 low-cohesion clusters (vs current 300)")
				print(f"   3. No need for dissolution or aggressive filtering")
		elif avg_improvement > 8 and best_model_pct > 50:
				print(f"\n⚠️  MODERATE RECOMMENDATION: Consider switching to {best_model_name}")
				print(f"   - Notable improvement ({avg_improvement:.1f}%)")
				print(f"   - Wins {best_model_pct:.0f}% of problematic clusters")
				print(f"   - Worth re-clustering if time permits")
		else:
				print(f"\n⚠️  MARGINAL IMPROVEMENT: {best_model_name} is slightly better")
				print(f"   - Small improvement ({avg_improvement:.1f}%)")
				print(f"   - May not justify re-clustering effort")
				print(f"   - Consider keeping MPNet-Base and using dissolution + filtering")
else:
		print(f"\n✓ MPNet-Base is already optimal for these clusters")
		print(f"   - No significant improvement from other models")
		print(f"   - Proceed with dissolution + filtering strategy")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)