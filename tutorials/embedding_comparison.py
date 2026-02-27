import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch

# ============================================================================
# 1. LOAD ALL THREE MODELS
# ============================================================================

print("Loading embedding models...")
print("="*80)

# Model 1: Your current model (4096-dim)
print("\n[1/3] Loading Qwen3-Embedding-8B (4096-dim)...")
model_qwen = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    trust_remote_code=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"✓ Loaded. Embedding dim: {model_qwen.get_sentence_embedding_dimension()}")

# Model 2: E5-large-v2 (1024-dim, strong semantic understanding)
print("\n[2/3] Loading intfloat/e5-large-v2 (1024-dim)...")
model_e5 = SentenceTransformer(
    "intfloat/e5-large-v2",
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"✓ Loaded. Embedding dim: {model_e5.get_sentence_embedding_dimension()}")

# Model 3: MPNet (768-dim, balanced performance)
print("\n[3/3] Loading all-mpnet-base-v2 (768-dim)...")
model_mpnet = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"✓ Loaded. Embedding dim: {model_mpnet.get_sentence_embedding_dimension()}")

print("\n" + "="*80)
print("All models loaded successfully!\n")

# ============================================================================
# 2. DEFINE YOUR PROBLEMATIC CLUSTERS
# ============================================================================

problematic_clusters = {
    "Cluster 3187 (Random Objects)": [
        'candlestick', 'ticker tape', 'tweezer'
    ],

    "Cluster 587 (Bow Homonym)": [
        'arrow', 'arrowhead', 'bow', 'bow section', 'bowtie', 'yellow arrow'
    ],

    "Cluster 1825 (Sports Mixed)": [
        'tack', 'tackle', 'tackle block', 'tackling', 'touchdown'
    ],

    "Cluster 984 (Spider/Crawler)": [
        'cobweb', 'crawler tractor', 'crawling', 'spider', 'spider trap'
    ],

    "Cluster 1767 (Random Short Words)": [
        'oar', 'ore', 'outrigger', 'urn'
    ],

    "Cluster 1741 (Alphabetical Soup)": [
        'delouser', 'lace', 'lad', 'landau', 'lap', 'lar', 'lei',
        'leper', 'levy', 'lichen', 'lying'
    ],

    "Cluster 1829 (T-words)": [
        'target tug', 'tiller', 'toga', 'tosspot', 'tug', 'tuxedo'
    ],

    "Cluster 1813 (B-words)": [
        'abri', 'barging', 'berg', 'blubber', 'bluff', 'boister', 'brig'
    ],

    "Cluster 2035 (Random Short)": [
        'ale', 'ant', 'ash', 'fin', 'fir', 'inn', 'ink', 'safe', 'tern'
    ],

    "Cluster 1060 (RAAF/Raft)": [
        'raaf', 'raf pilot', 'raft', 'target raft'
    ],

    "Cluster 981 (Mosquito)": [
        'butterfly', 'mosquito', 'mosquito aircraft', 'mosquito bomber'
    ],
}

# ============================================================================
# 3. ANALYSIS FUNCTION
# ============================================================================

def analyze_cluster(cluster_name, terms, model, model_name):
    """
    Analyze a cluster with a specific embedding model.
    Returns intra-cluster similarity and similarity matrix.
    """
    # Encode terms
    embeddings = model.encode(terms, convert_to_numpy=True, normalize_embeddings=True)

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Compute intra-cluster similarity (mean of off-diagonal elements)
    n = len(terms)
    if n > 1:
        intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
    else:
        intra_sim = 1.0

    return {
        'model': model_name,
        'cluster': cluster_name,
        'n_terms': n,
        'intra_similarity': intra_sim,
        'sim_matrix': sim_matrix,
        'terms': terms
    }

# ============================================================================
# 4. RUN ANALYSIS ON ALL CLUSTERS WITH ALL MODELS
# ============================================================================

print("Analyzing problematic clusters with all models...")
print("="*80)

results = []

for cluster_name, terms in problematic_clusters.items():
    print(f"\n{cluster_name}")
    print(f"Terms: {terms[:3]}..." if len(terms) > 3 else f"Terms: {terms}")
    print("-" * 80)

    # Analyze with each model
    qwen_result = analyze_cluster(cluster_name, terms, model_qwen, "Qwen3-8B")
    e5_result = analyze_cluster(cluster_name, terms, model_e5, "E5-Large-v2")
    mpnet_result = analyze_cluster(cluster_name, terms, model_mpnet, "MPNet-Base")

    results.extend([qwen_result, e5_result, mpnet_result])

    # Print comparison
    print(f"  Qwen3-8B      intra-sim: {qwen_result['intra_similarity']:.4f}")
    print(f"  E5-Large-v2   intra-sim: {e5_result['intra_similarity']:.4f}")
    print(f"  MPNet-Base    intra-sim: {mpnet_result['intra_similarity']:.4f}")

    # Determine best model (LOWER is better for problematic clusters)
    best_model = min([qwen_result, e5_result, mpnet_result],
                     key=lambda x: x['intra_similarity'])

    print(f"  → Best (lowest): {best_model['model']} ✓")

# ============================================================================
# 5. SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: Intra-Cluster Similarity by Model")
print("="*80)
print("(LOWER is BETTER for problematic clusters - indicates better separation)\n")

# Create summary DataFrame
summary_data = []
for result in results:
    summary_data.append({
        'Cluster': result['cluster'],
        'Model': result['model'],
        'Intra-Similarity': result['intra_similarity'],
        'N_Terms': result['n_terms']
    })

df_summary = pd.DataFrame(summary_data)

# Pivot for easier comparison
pivot = df_summary.pivot(index='Cluster', columns='Model', values='Intra-Similarity')
pivot = pivot[['Qwen3-8B', 'E5-Large-v2', 'MPNet-Base']]  # Order columns

# Add "Winner" column (lowest similarity)
pivot['Best_Model'] = pivot.idxmin(axis=1)
pivot['Separation_Gain'] = pivot['Qwen3-8B'] - pivot[['E5-Large-v2', 'MPNet-Base']].min(axis=1)

print(pivot.to_string())

# ============================================================================
# 6. DETAILED PAIRWISE ANALYSIS FOR WORST CLUSTERS
# ============================================================================

print("\n" + "="*80)
print("DETAILED PAIRWISE ANALYSIS: Worst Problematic Clusters")
print("="*80)

worst_clusters = [
    ("Cluster 3187", ['candlestick', 'ticker tape', 'tweezer']),
    ("Cluster 587", ['arrow', 'bow', 'bowtie']),  # Key problematic trio
    ("Cluster 1767", ['oar', 'ore', 'outrigger', 'urn']),
]

for cluster_name, terms in worst_clusters:
    print(f"\n{cluster_name}: {terms}")
    print("-" * 80)

    for model, model_name in [(model_qwen, "Qwen3-8B"),
                               (model_e5, "E5-Large-v2"),
                               (model_mpnet, "MPNet-Base")]:
        print(f"\n{model_name}:")
        embeddings = model.encode(terms, convert_to_numpy=True, normalize_embeddings=True)
        sim_matrix = cosine_similarity(embeddings)

        # Print pairwise similarities
        for i in range(len(terms)):
            for j in range(i+1, len(terms)):
                sim = sim_matrix[i, j]

                # Status based on similarity
                if sim > 0.80:
                    status = "❌ TOO HIGH"
                elif sim > 0.70:
                    status = "⚠️  HIGH"
                elif sim > 0.60:
                    status = "✓ OK"
                else:
                    status = "✓✓ GOOD"

                print(f"  {terms[i]:20s} ↔ {terms[j]:20s}: {sim:.3f}  {status}")

# ============================================================================
# 7. SPECIFIC HOMONYM TEST (bow weapon vs bowtie)
# ============================================================================

print("\n" + "="*80)
print("HOMONYM DISAMBIGUATION TEST: 'bow' (weapon) vs 'bowtie' (clothing)")
print("="*80)

homonym_terms = {
    'archery_context': ['arrow', 'arrowhead', 'bow', 'quiver', 'target'],
    'fashion_context': ['bowtie', 'tie', 'suit', 'tuxedo', 'collar'],
}

for model, model_name in [(model_qwen, "Qwen3-8B"),
                           (model_e5, "E5-Large-v2"),
                           (model_mpnet, "MPNet-Base")]:
    print(f"\n{model_name}:")

    # Encode all terms
    all_terms = homonym_terms['archery_context'] + homonym_terms['fashion_context']
    embeddings = model.encode(all_terms, convert_to_numpy=True, normalize_embeddings=True)

    # Get 'bow' and 'bowtie' embeddings
    bow_idx = all_terms.index('bow')
    bowtie_idx = all_terms.index('bowtie')

    bow_emb = embeddings[bow_idx:bow_idx+1]
    bowtie_emb = embeddings[bowtie_idx:bowtie_idx+1]

    # Similarity between bow and bowtie
    bow_bowtie_sim = cosine_similarity(bow_emb, bowtie_emb)[0, 0]

    # Average similarity of 'bow' to archery terms
    archery_embs = embeddings[:5]  # First 5 are archery
    bow_archery_sim = cosine_similarity(bow_emb, archery_embs)[0].mean()

    # Average similarity of 'bowtie' to fashion terms
    fashion_embs = embeddings[5:]  # Last 5 are fashion
    bowtie_fashion_sim = cosine_similarity(bowtie_emb, fashion_embs)[0].mean()

    print(f"  bow ↔ bowtie:           {bow_bowtie_sim:.3f}  {'❌ TOO HIGH' if bow_bowtie_sim > 0.70 else '✓ OK'}")
    print(f"  bow → archery context:  {bow_archery_sim:.3f}  (should be high)")
    print(f"  bowtie → fashion context: {bowtie_fashion_sim:.3f}  (should be high)")
    print(f"  Disambiguation quality: {'✓ GOOD' if bow_bowtie_sim < bow_archery_sim - 0.1 else '⚠️ POOR'}")

# ============================================================================
# 8. ALPHABETICAL CLUSTERING TEST
# ============================================================================

print("\n" + "="*80)
print("ALPHABETICAL CLUSTERING TEST: Random 'la-' words")
print("="*80)

alphabetical_terms = ['lace', 'lad', 'lap', 'lei', 'levy', 'lichen']

for model, model_name in [(model_qwen, "Qwen3-8B"),
                           (model_e5, "E5-Large-v2"),
                           (model_mpnet, "MPNet-Base")]:
    print(f"\n{model_name}:")
    embeddings = model.encode(alphabetical_terms, convert_to_numpy=True, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings)

    # Mean intra-cluster similarity
    n = len(alphabetical_terms)
    mean_sim = (sim_matrix.sum() - n) / (n * (n - 1))

    print(f"  Mean similarity: {mean_sim:.3f}")

    if mean_sim < 0.30:
        print(f"  ✓✓ EXCELLENT - Model resists alphabetical clustering")
    elif mean_sim < 0.50:
        print(f"  ✓ GOOD - Some resistance to alphabetical clustering")
    elif mean_sim < 0.65:
        print(f"  ⚠️  MODERATE - Weak resistance")
    else:
        print(f"  ❌ POOR - Falls for alphabetical clustering")

# ============================================================================
# 9. FINAL RECOMMENDATION
# ============================================================================

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

# Count wins per model
wins = df_summary.groupby('Model')['Intra-Similarity'].apply(
    lambda x: (x == x.min()).sum() if len(x) > 0 else 0
)

print("\nModel Performance Summary:")
print(f"  Qwen3-8B:      {wins.get('Qwen3-8B', 0)} clusters with lowest similarity")
print(f"  E5-Large-v2:   {wins.get('E5-Large-v2', 0)} clusters with lowest similarity")
print(f"  MPNet-Base:    {wins.get('MPNet-Base', 0)} clusters with lowest similarity")

# Average separation gain
avg_gain = pivot['Separation_Gain'].mean()
best_alternative = 'E5-Large-v2' if pivot['E5-Large-v2'].mean() < pivot['MPNet-Base'].mean() else 'MPNet-Base'

print(f"\nAverage separation gain over Qwen3-8B: {avg_gain:.4f}")

if avg_gain > 0.05:
    print(f"\n✅ RECOMMENDATION: Switch to {best_alternative}")
    print(f"   Expected improvement: {avg_gain*100:.1f}% better separation")
elif avg_gain > 0.02:
    print(f"\n⚠️  RECOMMENDATION: Consider switching to {best_alternative}")
    print(f"   Marginal improvement: {avg_gain*100:.1f}% better separation")
else:
    print(f"\n❌ RECOMMENDATION: Stay with Qwen3-8B")
    print(f"   Alternative models don't provide significant improvement")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

model_qwen = SentenceTransformer("Qwen/Qwen3-Embedding-8B", trust_remote_code=True)
model_e5 = SentenceTransformer("intfloat/e5-large-v2")
model_mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

print("All models loaded!\n")

# ============================================================================
# DEFINE CLUSTERS TO TEST
# ============================================================================

# Mix of PROBLEMATIC clusters (should show MPNet wins)
# and WELL-FORMED clusters (should show all models do well)

test_clusters = {
    # ========================================================================
    # PROBLEMATIC CLUSTERS (from previous analysis)
    # ========================================================================
    "❌ Random Objects": [
        'candlestick', 'ticker tape', 'tweezer'
    ],

    "❌ Bow Homonym": [
        'arrow', 'arrowhead', 'bow', 'bow section', 'bowtie', 'yellow arrow'
    ],

    "❌ Random Short Words": [
        'oar', 'ore', 'outrigger', 'urn'
    ],

    "❌ Alphabetical Soup": [
        'delouser', 'lace', 'lad', 'landau', 'lap', 'lar', 'lei',
        'leper', 'levy', 'lichen', 'lying'
    ],

    "❌ RAAF/Raft": [
        'raaf', 'raf pilot', 'raft', 'target raft'
    ],

    # ========================================================================
    # WELL-FORMED CLUSTERS (from your new output)
    # ========================================================================
    "✅ Merchant Ships": [
        'merchant marine', 'merchant marine vessel', 'merchant navy',
        'merchant ship', 'merchantman'
    ],

    "✅ Naval Vessels": [
        'military ship', 'naval ship', 'naval vessel',
        'navy ship', 'navy vessel'
    ],

    "✅ Steamships": [
        'steam ship', 'steamboat', 'steamship', 'steamer'
    ],

    "✅ Sailing Ships": [
        'sailing ship', 'sailing vessel', 'sailship',
        'schooner', 'sloop'
    ],

    "✅ Troopships": [
        'troop ship', 'troop transport ship', 'troopship'
    ],

    "✅ Air Squadron": [
        'air squadron', 'aircraft squadron', 'fighter squadron',
        'squadron', 'combat squadron'
    ],

    "✅ Fighter Groups": [
        'fighter group', 'fighter wing', 'combat wing'
    ],

    "✅ Air Force Units": [
        'air force', 'air force squadron', 'air force unit',
        'tactical air command'
    ],

    "✅ Supply Crates": [
        'supply crate', 'supply crates', 'packing crate'
    ],

    "✅ Containers": [
        'container', 'supply container', 'packing box', 'equipment box'
    ],
}

# ============================================================================
# ANALYSIS FUNCTION
# ============================================================================

def analyze_cluster(cluster_name, terms, model, model_name):
    """Analyze cluster quality with a specific model."""
    embeddings = model.encode(terms, convert_to_numpy=True, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings)

    n = len(terms)
    if n > 1:
        intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
    else:
        intra_sim = 1.0

    # Additional metrics
    min_sim = np.min(sim_matrix + np.eye(n) * 10)  # Exclude diagonal
    max_sim = np.max(sim_matrix - np.eye(n) * 10)  # Exclude diagonal
    std_sim = np.std(sim_matrix[np.triu_indices(n, k=1)])

    return {
        'model': model_name,
        'cluster': cluster_name,
        'n_terms': n,
        'intra_similarity': intra_sim,
        'min_similarity': min_sim,
        'max_similarity': max_sim,
        'std_similarity': std_sim,
        'sim_matrix': sim_matrix,
        'terms': terms
    }

# ============================================================================
# RUN ANALYSIS
# ============================================================================

print("Analyzing clusters with all models...")
print("="*80)

results = []

for cluster_name, terms in test_clusters.items():
    is_problematic = cluster_name.startswith("❌")
    cluster_type = "PROBLEMATIC" if is_problematic else "WELL-FORMED"

    print(f"\n{cluster_name} [{cluster_type}]")
    print(f"Terms: {terms[:3]}..." if len(terms) > 3 else f"Terms: {terms}")
    print("-" * 80)

    # Analyze with each model
    qwen_result = analyze_cluster(cluster_name, terms, model_qwen, "Qwen3-8B")
    e5_result = analyze_cluster(cluster_name, terms, model_e5, "E5-Large-v2")
    mpnet_result = analyze_cluster(cluster_name, terms, model_mpnet, "MPNet-Base")

    results.extend([qwen_result, e5_result, mpnet_result])

    # Print comparison
    print(f"  Qwen3-8B      intra: {qwen_result['intra_similarity']:.4f}  "
          f"range: [{qwen_result['min_similarity']:.3f}, {qwen_result['max_similarity']:.3f}]")
    print(f"  E5-Large-v2   intra: {e5_result['intra_similarity']:.4f}  "
          f"range: [{e5_result['min_similarity']:.3f}, {e5_result['max_similarity']:.3f}]")
    print(f"  MPNet-Base    intra: {mpnet_result['intra_similarity']:.4f}  "
          f"range: [{mpnet_result['min_similarity']:.3f}, {mpnet_result['max_similarity']:.3f}]")

    # Determine best model based on cluster type
    if is_problematic:
        # For problematic clusters, LOWER is better
        best_model = min([qwen_result, e5_result, mpnet_result],
                        key=lambda x: x['intra_similarity'])
        metric = "lowest (best for separation)"
    else:
        # For well-formed clusters, HIGHER is better
        best_model = max([qwen_result, e5_result, mpnet_result],
                        key=lambda x: x['intra_similarity'])
        metric = "highest (best cohesion)"

    print(f"  → Best ({metric}): {best_model['model']} ✓")

# ============================================================================
# SUMMARY TABLES
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: Performance by Cluster Type")
print("="*80)

# Create summary DataFrame
summary_data = []
for result in results:
    is_problematic = result['cluster'].startswith("❌")
    summary_data.append({
        'Cluster': result['cluster'],
        'Type': 'PROBLEMATIC' if is_problematic else 'WELL-FORMED',
        'Model': result['model'],
        'Intra-Similarity': result['intra_similarity'],
        'Min-Sim': result['min_similarity'],
        'Max-Sim': result['max_similarity'],
        'Std-Sim': result['std_similarity']
    })

df_summary = pd.DataFrame(summary_data)

# ============================================================================
# ANALYSIS 1: Problematic Clusters (LOWER similarity is BETTER)
# ============================================================================

print("\n" + "="*80)
print("PROBLEMATIC CLUSTERS ANALYSIS")
print("="*80)
print("(LOWER intra-similarity = BETTER separation)\n")

df_problematic = df_summary[df_summary['Type'] == 'PROBLEMATIC'].copy()
pivot_prob = df_problematic.pivot(index='Cluster', columns='Model', values='Intra-Similarity')
pivot_prob = pivot_prob[['Qwen3-8B', 'E5-Large-v2', 'MPNet-Base']]

# Calculate Best_Score first, only from the numeric columns
pivot_prob['Best_Score'] = pivot_prob.min(axis=1)

# Then add the Winner column based on idxmin
pivot_prob['Winner'] = pivot_prob[['Qwen3-8B', 'E5-Large-v2', 'MPNet-Base']].idxmin(axis=1)

# Then calculate Qwen_Improvement
pivot_prob['Qwen_Improvement'] = ((pivot_prob['Qwen3-8B'] - pivot_prob['Best_Score']) /
                                   pivot_prob['Qwen3-8B'] * 100)

print(pivot_prob.to_string())

# Count wins
prob_wins = pivot_prob['Winner'].value_counts()
print(f"\n📊 Winner Count (Problematic Clusters):")
for model, count in prob_wins.items():
    print(f"  {model}: {count} wins")

avg_improvement = pivot_prob['Qwen_Improvement'].mean()
print(f"\n💡 Average improvement over Qwen3-8B: {avg_improvement:.1f}%")

# ============================================================================
# ANALYSIS 2: Well-Formed Clusters (HIGHER similarity is BETTER)
# ============================================================================

print("\n" + "="*80)
print("WELL-FORMED CLUSTERS ANALYSIS")
print("="*80)
print("(HIGHER intra-similarity = BETTER cohesion)\n")

df_wellformed = df_summary[df_summary['Type'] == 'WELL-FORMED'].copy()
pivot_well = df_wellformed.pivot(index='Cluster', columns='Model', values='Intra-Similarity')
pivot_well = pivot_well[['Qwen3-8B', 'E5-Large-v2', 'MPNet-Base']]

# Calculate Best_Score first, only from the numeric columns
pivot_well['Best_Score'] = pivot_well.max(axis=1)

# Then add the Winner column (highest for well-formed)
pivot_well['Winner'] = pivot_well[['Qwen3-8B', 'E5-Large-v2', 'MPNet-Base']].idxmax(axis=1)

# Then calculate Qwen_Difference
pivot_well['Qwen_Difference'] = pivot_well['Qwen3-8B'] - pivot_well['Best_Score']

print(pivot_well.to_string())

# Count wins
well_wins = pivot_well['Winner'].value_counts()
print(f"\n📊 Winner Count (Well-Formed Clusters):")
for model, count in well_wins.items():
    print(f"  {model}: {count} wins")

avg_diff = pivot_well['Qwen_Difference'].mean()
if avg_diff < 0:
    print(f"\n⚠️  Qwen3-8B is {abs(avg_diff):.4f} LOWER on average (cohesion loss)")
else:
    print(f"\n✓ Qwen3-8B is {avg_diff:.4f} HIGHER on average (cohesion maintained)")

# ============================================================================
# ANALYSIS 3: Overall Performance Score
# ============================================================================

print("\n" + "="*80)
print("OVERALL PERFORMANCE SCORE")
print("="*80)

# Score = (Problematic wins × 2) + (Well-formed wins × 1)
# Problematic clusters are weighted 2x because they're more critical

overall_scores = {}
for model in ['Qwen3-8B', 'E5-Large-v2', 'MPNet-Base']:
    prob_score = prob_wins.get(model, 0) * 2  # Weight problematic 2x
    well_score = well_wins.get(model, 0) * 1
    total_score = prob_score + well_score
    overall_scores[model] = {
        'Problematic_Wins': prob_wins.get(model, 0),
        'WellFormed_Wins': well_wins.get(model, 0),
        'Total_Score': total_score
    }

print("\nModel Performance:")
print(f"{'Model':<20} {'Prob. Wins':<12} {'Well Wins':<12} {'Total Score':<12}")
print("-" * 60)
for model, scores in sorted(overall_scores.items(), key=lambda x: x[1]['Total_Score'], reverse=True):
    print(f"{model:<20} {scores['Problematic_Wins']:<12} "
          f"{scores['WellFormed_Wins']:<12} {scores['Total_Score']:<12}")

winner = max(overall_scores.items(), key=lambda x: x[1]['Total_Score'])
print(f"\n🏆 OVERALL WINNER: {winner[0]}")

# ============================================================================
# ANALYSIS 4: Detailed Quality Metrics
# ============================================================================

print("\n" + "="*80)
print("DETAILED QUALITY METRICS")
print("="*80)

print("\nProblematic Clusters - Target: Low intra-sim, high separation")
print("-" * 80)
for model in ['Qwen3-8B', 'E5-Large-v2', 'MPNet-Base']:
    model_data = df_problematic[df_problematic['Model'] == model]
    avg_intra = model_data['Intra-Similarity'].mean()
    avg_std = model_data['Std-Sim'].mean()

    print(f"{model:<15} Avg Intra: {avg_intra:.4f}  Avg Std: {avg_std:.4f}  "
          f"{'✅ GOOD' if avg_intra < 0.65 else '⚠️ MODERATE' if avg_intra < 0.75 else '❌ POOR'}")

print("\nWell-Formed Clusters - Target: High intra-sim, low variance")
print("-" * 80)
for model in ['Qwen3-8B', 'E5-Large-v2', 'MPNet-Base']:
    model_data = df_wellformed[df_wellformed['Model'] == model]
    avg_intra = model_data['Intra-Similarity'].mean()
    avg_std = model_data['Std-Sim'].mean()

    print(f"{model:<15} Avg Intra: {avg_intra:.4f}  Avg Std: {avg_std:.4f}  "
          f"{'✅ EXCELLENT' if avg_intra > 0.90 else '✓ GOOD' if avg_intra > 0.85 else '⚠️ OK'}")

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

# Decision logic
mpnet_prob_wins = prob_wins.get('MPNet-Base', 0)
mpnet_well_wins = well_wins.get('MPNet-Base', 0)
total_prob = len(pivot_prob)
total_well = len(pivot_well)

mpnet_prob_rate = mpnet_prob_wins / total_prob * 100
mpnet_well_rate = mpnet_well_wins / total_well * 100

print(f"\nMPNet-Base Performance:")
print(f"  Problematic clusters: {mpnet_prob_wins}/{total_prob} wins ({mpnet_prob_rate:.0f}%)")
print(f"  Well-formed clusters: {mpnet_well_wins}/{total_well} wins ({mpnet_well_rate:.0f}%)")
print(f"  Overall score: {overall_scores['MPNet-Base']['Total_Score']}")

qwen_prob_wins = prob_wins.get('Qwen3-8B', 0)
qwen_well_wins = well_wins.get('Qwen3-8B', 0)
qwen_prob_rate = qwen_prob_wins / total_prob * 100
qwen_well_rate = qwen_well_wins / total_well * 100

print(f"\nQwen3-8B Performance:")
print(f"  Problematic clusters: {qwen_prob_wins}/{total_prob} wins ({qwen_prob_rate:.0f}%)")
print(f"  Well-formed clusters: {qwen_well_wins}/{total_well} wins ({qwen_well_rate:.0f}%)")
print(f"  Overall score: {overall_scores['Qwen3-8B']['Total_Score']}")

# Decision
if mpnet_prob_rate >= 80 and overall_scores['MPNet-Base']['Total_Score'] > overall_scores['Qwen3-8B']['Total_Score']:
    print("\n✅ STRONG RECOMMENDATION: Switch to MPNet-Base")
    print(f"   - Dominates problematic clusters ({mpnet_prob_rate:.0f}%)")
    print(f"   - {'Maintains' if mpnet_well_rate >= 40 else 'Acceptable'} well-formed quality ({mpnet_well_rate:.0f}%)")
    print(f"   - Higher overall score ({overall_scores['MPNet-Base']['Total_Score']} vs {overall_scores['Qwen3-8B']['Total_Score']})")

elif mpnet_prob_rate >= 60:
    print("\n⚠️  MODERATE RECOMMENDATION: Consider switching to MPNet-Base")
    print(f"   - Strong on problematic clusters ({mpnet_prob_rate:.0f}%)")
    print(f"   - Trade-off on well-formed clusters ({mpnet_well_rate:.0f}%)")

else:
    print("\n❌ RECOMMENDATION: Stay with Qwen3-8B")
    print(f"   - MPNet-Base doesn't show sufficient improvement")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
