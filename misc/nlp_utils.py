import os
import re
import torch
import pickle
import ast

import nltk

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score

from sentence_transformers import SentenceTransformer
import numpy as np
import math
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict, Set, Any, Optional, Union, Callable, Iterable
# Try relative import first, fallback to absolute
import string
import hdbscan

from joblib import Parallel, delayed
# from sklearn.metrics.pairwise import cosine_similarity_chunked
# import multiprocessing as mp

# Install: pip install lingua-language-detector
from lingua import Language, LanguageDetectorBuilder, IsoCode639_1

# suppress warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings(
	"ignore",
	# message=".*number of unique classes.*",
	category=UserWarning,
	module="sklearn.metrics"
)

MISC_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTHONHASHSEED"] = "0"

nltk_modules = [
	'punkt',
	'punkt_tab',
	'wordnet',
	'averaged_perceptron_tagger',
	'averaged_perceptron_tagger_eng',
	'omw-1.4',
	'stopwords',
]

# check if nltk_data exists:
try:
	nltk.data.find('tokenizers/punkt')
except LookupError:
	print("Downloading NLTK data...")
	# Download only the required components
	nltk.download(
		nltk_modules,
		quiet=False,
		raise_on_error=True,
	)

cache_directory = {
	"farid": "/home/farid/datasets/models",
	"alijanif": "/scratch/project_2004072/models",
	"ubuntu": "/media/volume/models",
}

# STOPWORDS = set(nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())) # all languages
STOPWORDS = set(nltk.corpus.stopwords.words('english')) # english only
# custom_stopwords_list = requests.get("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/refs/heads/master/stopwords-en.txt").content
# stopwords = set(custom_stopwords_list.decode().splitlines())
meaningless_words_path = os.path.join(MISC_DIR, 'meaningless_words.txt')
with open(meaningless_words_path, 'r') as file_:
	stopwords = set([line.strip().lower() for line in file_])
STOPWORDS.update(stopwords)

geographic_references_path = os.path.join(MISC_DIR, 'geographic_references.txt')
with open(geographic_references_path, 'r') as file_:
	geographic_references = set([line.strip().lower() for line in file_ if line.strip()])
STOPWORDS.update(geographic_references)

# This DRASTICALLY improves accuracy on short text.
languages_to_check = [
	IsoCode639_1.EN, # English
	IsoCode639_1.DE, # German
	IsoCode639_1.FR, # French
	IsoCode639_1.ES, # Spanish
	IsoCode639_1.IT, # Italian
	# IsoCode639_1.NL, # Dutch
	IsoCode639_1.PT, # Portuguese
	IsoCode639_1.SV, # Swedish
	IsoCode639_1.FI, # Finnish
	IsoCode639_1.DA, # Danish
	IsoCode639_1.NB, # Norwegian Bokmål
	IsoCode639_1.NN, # Norwegian Nynorsk
	IsoCode639_1.PL, # Polish
	IsoCode639_1.RU, # Russian
	IsoCode639_1.HU, # Hungarian
	IsoCode639_1.CS, # Czech
	IsoCode639_1.SK, # Slovak
	IsoCode639_1.EL, # Greek
	IsoCode639_1.BG, # Bulgarian
	IsoCode639_1.RO, # Romanian
]

# Preload the shortlisted detector
detector_shortlist = (
	LanguageDetectorBuilder
	.from_iso_codes_639_1(*languages_to_check)
	.with_preloaded_language_models()
	.build()
)

# Preload the full detector (all languages)
detector_all = (
	LanguageDetectorBuilder
	.from_all_languages()
	.with_preloaded_language_models()
	.build()
)

def autotune_hdbscan_params(X, n_bootstrap=5, n_jobs=-1):
	test_params = []
	for mcs in [2, 3, 5, 7, 10, 15, 20]:
		for ms in [1, 2, 3, 4, 5]:
			for method in ['eom', 'leaf']:  # Try both selection methods
				params = {
					'min_cluster_size': mcs,
					'min_samples': ms,
					'metric': 'euclidean',
					'core_dist_n_jobs': n_jobs,
					'cluster_selection_method': method
				}
				test_params.append(params)
	
	best_score = -1
	best_result = None
	candidates = []
	
	for params in test_params:
		agreements = []
		
		for seed in range(n_bootstrap):
			idx = resample(np.arange(X.shape[0]), n_samples=int(0.8*X.shape[0]), random_state=seed)
			X_boot = X[idx]
			hdb = hdbscan.HDBSCAN(**params)
			boot_labels = hdb.fit_predict(X_boot)
			full_labels = np.full(X.shape[0], -1)
			full_labels[idx] = boot_labels
			agreements.append(full_labels)
		
		agreements = np.array(agreements)
		noise_rates = [(a == -1).mean() for a in agreements]
		avg_noise = np.mean(noise_rates)
		coverage = 1 - avg_noise
		
		# Skip if coverage too low (can't run KMeans on <10% of data)
		if coverage < 0.15:
			continue
		
		# Calculate stability only on non-noise points
		stability_scores = []
		for i in range(n_bootstrap):
			for j in range(i+1, n_bootstrap):
				mask = (agreements[i] != -1) & (agreements[j] != -1)
				if mask.sum() > 10:
					# ARI on points clustered in both runs
					stability_scores.append(adjusted_rand_score(agreements[i][mask], agreements[j][mask]))
		
		stability = np.mean(stability_scores) if stability_scores else 0
		
		# Balance stability and coverage
		# Use harmonic mean (F1-style) to require both to be good
		if stability + coverage > 0:
			score = 2 * (stability * coverage) / (stability + coverage)
		else:
			score = 0
		
		result = {
			'params': params,
			'stability': float(stability),
			'coverage': float(coverage),
			'score': float(score),
			'n_clusters': len(set(agreements[0])) - (1 if -1 in agreements[0] else 0)
		}
		candidates.append(result)
		
		print(
			f"mcs={params['min_cluster_size']:2d}/ms={params['min_samples']:2d}/{params['cluster_selection_method']:<10}"
			f"Coverage: {coverage:<10.1%}Stability: {stability:<10.3f}"
			f"Clusters: {result['n_clusters']:<10}Score: {score:.3f}"
		)
		
		if score > best_score:
			best_score = score
			# Refit on full data
			hdb_full = hdbscan.HDBSCAN(**params)
			hdb_labels = hdb_full.fit_predict(X)
			best_result = {
				**result,
				'hdb_labels': hdb_labels,
				'noise_rate': float((hdb_labels == -1).mean()),
				'core_indices': np.where(hdb_labels != -1)[0].tolist(),
				'n_clusters': len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
			}
	
	if best_result is None:
		# Ultimate fallback: pick highest coverage
		best = max(candidates, key=lambda x: x['coverage'])
		params = best['params']
		hdb_full = hdbscan.HDBSCAN(**params)
		hdb_labels = hdb_full.fit_predict(X)
		best_result = {
			**best,
			'hdb_labels': hdb_labels,
			'core_indices': np.where(hdb_labels != -1)[0].tolist(),
			'n_clusters': len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0),
			'fallback': True
		}
	
	print(f"\n[HDBSCAN AUTO-TUNED PARAMS] {best_result['params']}")
	print(
		f"{best_result['n_clusters']} clusters, "
		f"{best_result.get('coverage', 1-best_result['noise_rate']):.1%} coverage, "
		f"stability={best_result['stability']:.3f}"
	)
	
	return best_result

def autotune_hdbscan_params_opt(
	X,
	n_bootstrap=5,
	n_jobs=-1,
	early_stop_threshold=0.95,
	batch_size=None,
	sample_frac=0.8,
	min_overlap=50,          # require enough shared points to trust ARI
	min_coverage=0.15,       # skip configs that label too little data
	large_n=int(1e4),
):
	n = X.shape[0]
	if batch_size is None:
		# Keep batches small-ish for early stop responsiveness
		batch_size = 2 if n > 20000 else 5
	
	# Build param grid
	if n > large_n:
		print("[LARGE DATASET MODE] Using reduced parameter grid")
		mcs_list = [3, 5, 10, 15, 20]
		ms_list = [2, 3, 4, 5]
		methods = ["eom"]
	else:
		print("[STANDARD MODE] Using full parameter grid")
		mcs_list = [2, 3, 5, 7, 9, 11, 15, 17, 20, 25]
		ms_list = [1, 2, 3, 4, 5, 6, 7, 8]
		methods = ["eom", "leaf"]
	
	test_params = []
	for mcs in mcs_list:
		for ms in ms_list:
			for method in methods:
				test_params.append(
					dict(
						min_cluster_size=mcs,
						min_samples=ms,
						metric="euclidean",
						cluster_selection_method=method,
					)
				)
	
	# Order params to find good ones earlier (helps early stop)
	def priority(p):
		method_pri = 0 if p["cluster_selection_method"] == "leaf" else 1
		return (p["min_cluster_size"], p["min_samples"], method_pri)

	test_params = sorted(test_params, key=priority)
	print(f"[AUTO-TUNE] Testing up to {len(test_params)} parameter combinations")
	print(f"[AUTO-TUNE] Bootstrap iterations: {n_bootstrap}")
	print(f"[AUTO-TUNE] Parallel jobs: {'all cores' if n_jobs == -1 else n_jobs}")
	print(f"[AUTO-TUNE] Early stop threshold: {early_stop_threshold}")
	
	def run_one_config(params):
		# Store (idx, labels) per bootstrap
		runs = []
		noise_rates = []
		n_sub = int(sample_frac * n)
		for seed in range(n_bootstrap):
			idx = resample(np.arange(n), n_samples=n_sub, random_state=seed)
			with warnings.catch_warnings():
				warnings.filterwarnings(
					"ignore",
					message=".*force_all_finite.*",
					category=FutureWarning,
				)
				hdb = hdbscan.HDBSCAN(**params, core_dist_n_jobs=1)
				lbl = hdb.fit_predict(X[idx])
			runs.append((idx, lbl))
			noise_rates.append((lbl == -1).mean())

		coverage = 1.0 - float(np.mean(noise_rates))
		if coverage < min_coverage:
			return None
		
		# Stability: ARI on shared points clustered in both runs
		ari_scores = []
		for i in range(n_bootstrap):
			idx_i, lab_i = runs[i]
			# map index -> label for non-noise only
			mask_i = lab_i != -1
			map_i = dict(zip(idx_i[mask_i], lab_i[mask_i]))
			for j in range(i + 1, n_bootstrap):
				idx_j, lab_j = runs[j]
				mask_j = lab_j != -1
				map_j = dict(zip(idx_j[mask_j], lab_j[mask_j]))
				# intersection of clustered points
				common = set(map_i.keys()) & set(map_j.keys())
				if len(common) < min_overlap:
					continue
				common = np.fromiter(common, dtype=np.int64)
				a = np.array([map_i[k] for k in common], dtype=np.int64)
				b = np.array([map_j[k] for k in common], dtype=np.int64)

				with warnings.catch_warnings():
					warnings.filterwarnings(
						"ignore",
						message=".*number of unique classes.*",
						category=UndefinedMetricWarning,
					)
					ari_scores.append(adjusted_rand_score(a, b))
		
		stability = float(np.mean(ari_scores)) if ari_scores else 0.0
		if stability + coverage > 0:
			score = float(2.0 * (stability * coverage) / (stability + coverage))
		else:
			score = 0.0
		
		# estimate clusters from first run (rough)
		_, lab0 = runs[0]
		n_clusters = int(len(set(lab0)) - (1 if -1 in lab0 else 0))

		return dict(
			params=params,
			stability=stability,
			coverage=coverage,
			score=score,
			n_clusters=n_clusters,
		)

	best = None
	best_score = -np.inf
	candidates = []
	print("\n" + "=" * 100)
	print(f"{'Params':<22} {'Coverage':<10} {'Stability':<10} {'Clusters':<10} {'Score':<8}")
	print("=" * 100)
	for start in range(0, len(test_params), batch_size):
		batch = test_params[start : start + batch_size]
		# Outer parallelism only
		batch_results = Parallel(n_jobs=n_jobs, backend="loky")(
			delayed(run_one_config)(p) for p in batch
		)

		for r in batch_results:
			if r is None:
				continue
			candidates.append(r)
			p = r["params"]
			pstr = f"mcs: {p['min_cluster_size']:2d} ms: {p['min_samples']:2d} {p['cluster_selection_method']:<3}"
			print(f"{pstr:<22} {r['coverage']:<10.1%} {r['stability']:<10.3f} {r['n_clusters']:<10d} {r['score']:<8.3f}")
			if r["score"] > best_score:
				best_score = r["score"]
				best = r
				print(f"{'':<22} {'':<10} {'':<10} {'':<10} {'NEW BEST':<8}")

		if best is not None and best_score >= early_stop_threshold:
			remaining = len(test_params) - (start + batch_size)
			print(f"\n[EARLY STOP] score={best_score:.3f} >= {early_stop_threshold:.3f}; skipping {remaining} configs")
			break

	if best is None:
		raise ValueError("No valid parameter combinations found (all configs below min_coverage).")
	
	# Final refit on full data
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			message=".*force_all_finite.*",
			category=FutureWarning,
		)
		hdb_full = hdbscan.HDBSCAN(**best["params"], core_dist_n_jobs=n_jobs)
		hdb_labels = hdb_full.fit_predict(X)
	noise_rate = float((hdb_labels == -1).mean())
	core_indices = np.where(hdb_labels != -1)[0].tolist()
	n_clusters_full = int(len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0))
	best_result = dict(
		**best,
		hdb_labels=hdb_labels,
		noise_rate=noise_rate,
		core_indices=core_indices,
		n_tested=len(candidates),
	)
	best_result['n_clusters'] = n_clusters_full
	
	print(f"\n[HDBSCAN AUTO-TUNED PARAMS] {best_result['params']}")
	print(
		f"{best_result['n_clusters']} clusters, "
		f"{(1-noise_rate):.1%} coverage, "
		f"stability={best_result['stability']:.3f}, "
		f"score={best_result['score']:.3f}"
	)

	return best_result

def _clustering_hdbscan(
	labels: List[List[str]],
	model_id: str,
	device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
	clusters_fname: str = "clusters.csv",
	nc: int = None,
	n_jobs: int = -1,
	auto_tune: bool = True,
	verbose: bool = True,
):
	if verbose:
		print(f"\n[CLUSTERING] {len(labels)} {type(labels)} {type(labels[0])} labels")
		print(f"   ├─ model_id: {model_id}")
		print(f"   ├─ device: {device}")
		print(f"   └─ {labels[:5]}")

	print("\n[STEP 1] Deduplicating labels")
	documents = list()
	for doc in labels:
		if isinstance(doc, str):
			doc = ast.literal_eval(doc)
		documents.append(list(set(lbl for lbl in doc)))

	# make results deterministic & reproducible
	all_labels = sorted(set(label for doc in documents for label in doc))

	print(f"Total samples: {type(documents)} {len(documents)} {type(documents[0])}")
	print(f"Unique labels: {type(all_labels)} {len(all_labels)} {type(all_labels[0])}")
	print(f"Sample labels: {all_labels[:15]}")

	print(f"\n[STEP 2] Loading SentenceTransformer {model_id}")
	model = SentenceTransformer(
		model_name_or_path=model_id,
		cache_folder=cache_directory[os.getenv('USER')],
		token=os.getenv("HUGGINGFACE_TOKEN"),
	).to(device)

	print(f"Model loaded: {model_id} Parameters: {sum(p.numel() for p in model.parameters()):,}")

	print(f"\n[STEP 3] Encoding {len(all_labels)} labels into semantic space")
	X = model.encode(
		all_labels, 
		batch_size = 256 if len(all_labels) > 1000 else 32,
		show_progress_bar=False,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)
	print(f"Embedding: {type(X)} {X.shape} sparsity: {np.count_nonzero(X) / np.prod(X.shape):.4f}")

	print(f"\n[STEP 4] Density-based clustering with HDBSCAN on semantic space for {X.shape[0]} labels")
	if auto_tune:
		print(f"Auto-tuning HDBSCAN parameters")
		n_boot = 3 if len(all_labels) > int(1e4) else 5
		result = autotune_hdbscan_params_opt(X=X, n_bootstrap=n_boot, n_jobs=n_jobs)
		hdb_labels = result['hdb_labels']
		min_cluster_size = result['params']['min_cluster_size']
		min_samples = result['params']['min_samples']
		cluster_selection_method = result['params']['cluster_selection_method']
		metric = result['params']['metric']
	else:
		min_cluster_size = 5
		min_samples = 2
		cluster_selection_method = "eom"
		metric = "euclidean"
	
	print(f"[HDBSCAN] input arguments:")
	print(f"   ├─ {type(X)} {X.shape} {X.dtype} {X.strides} {X.itemsize} {X.nbytes}")
	print(f"   ├─ min_cluster_size={min_cluster_size}")
	print(f"   ├─ min_samples={min_samples}")
	print(f"   ├─ cluster_selection_method={cluster_selection_method}")
	print(f"   └─ metric={metric}")

	hdb = hdbscan.HDBSCAN(
		min_cluster_size=min_cluster_size,
		min_samples=min_samples,
		cluster_selection_method=cluster_selection_method,
		metric=metric,
		core_dist_n_jobs=n_jobs,
	)

	clusterer = hdb.fit(X)
	hdb_labels = clusterer.labels_
	hdb_probs = clusterer.probabilities_
	print(f"[HDBSCAN] labels: {type(hdb_labels)} {hdb_labels.shape} {set(hdb_labels)}")
	print(f"[HDBSCAN] probs: {type(hdb_probs)} {hdb_probs.shape} {set(hdb_probs)}")
	cluster_counts = {int(k): v for k, v in Counter(hdb_labels).items()}
	print(f"[HDBSCAN] {len(np.unique(hdb_labels))} cluster counts:\n{json.dumps(cluster_counts, indent=2, ensure_ascii=False)}")
	num_noise = np.sum(hdb_labels == -1) # outlier
	num_core = len(hdb_labels) - num_noise
	
	print(f"[HDBSCAN] core: {num_core}/{len(all_labels)} ({num_core / len(all_labels):.2%}) | noise: {num_noise}/{len(all_labels)} ({num_noise / len(all_labels):.2%})")
	
	core_indices = np.where(hdb_labels != -1)[0]
	noise_indices = np.where(hdb_labels == -1)[0]
	
	X_core = X[core_indices]
	core_labels = [all_labels[i] for i in core_indices]
	noise_labels = [all_labels[i] for i in noise_indices]
	
	print(f"{len(core_labels)} CORE labels:")
	for _, v in enumerate(sorted(core_labels)):
		print(f"{v}")
	print("-"*140)

	print(f"{len(noise_labels)} NOISE labels:")
	for _, v in enumerate(sorted(noise_labels)):
		print(f"{v}")
	print("-"*140)

	print(f"\n[STEP 4.1] Visualizing HDBSCAN clusters in 2D")
	tsne_projection = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(X)
	pca_projection = PCA(n_components=2).fit_transform(X)
	print(f"TSNE: {type(tsne_projection)}, {tsne_projection.shape}")
	print(f"PCA: {type(pca_projection)}, {pca_projection.shape}")

	max_label = np.max(hdb_labels)
	palette_size = max(max_label + 1, 12)
	color_palette = sns.color_palette('Paired', palette_size)
	cluster_colors = [
		color_palette[x] 
		if x >= 0 else (0.5, 0.5, 0.5)
		for x in hdb_labels
	]
	cluster_member_colors = [
		sns.desaturate(x, p) 
		for x, p in zip(cluster_colors, hdb_probs)
	]

	plt.figure(figsize=(27, 17))
	plt.scatter(*pca_projection.T, s=40, linewidth=1.8, c=cluster_member_colors, alpha=0.8, marker="o")
	plt.title(f"PCA HDBSCAN ({len(np.unique(hdb_labels))} clusters) Noise: {len(noise_labels)} Core: {len(core_labels)}")
	out_cluster_fig_fpath = clusters_fname.replace(".csv", "_pca_hdb_clusters.png")
	plt.savefig(out_cluster_fig_fpath, dpi=150, bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(27, 17))
	plt.scatter(*tsne_projection.T, s=40, linewidth=1.8, c=cluster_member_colors, alpha=0.8, marker="o")
	plt.title(f"TSNE HDBSCAN ({len(np.unique(hdb_labels))} clusters) Noise: {len(noise_labels)} Core: {len(core_labels)}")
	out_cluster_fig_fpath = clusters_fname.replace(".csv", "_tsne_hdb_clusters.png")
	plt.savefig(out_cluster_fig_fpath, dpi=150, bbox_inches='tight')
	plt.close()

	print(f"\n[STEP 5.1] Silhouette analysis for KMeans clustering on {len(core_labels)} semantic cores")
	if nc is None:
		if len(core_labels) > 4000:
			range_n_clusters = range(50, min(3001, len(core_labels) // 10), 50)
		else:
			range_n_clusters = range(10, min(401, len(core_labels) // 2), 5)
		silhouette_scores = []
		print(f"Searching for optimal cluster count {range_n_clusters}...")
		for k in range_n_clusters:
			km = KMeans(n_clusters=k, n_init="auto", random_state=0)
			preds = km.fit_predict(X_core)
			score = silhouette_score(X_core, preds, metric="euclidean", random_state=0)
			silhouette_scores.append(score)
			print(f"\tk: {k:<6} silhouette: {score:.4f}")
		best_k = range_n_clusters[np.argmax(silhouette_scores)]
		print(f"Optimal k selected: {best_k}")
	else:
		best_k = nc
		print(f"Using user-defined k: {best_k}")

	print(f"\n[STEP 5.2] KMeans clustering on {type(X_core)} {X_core.shape} semantic cores with k={best_k}")
	kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=0)
	core_cluster_ids = kmeans.fit_predict(X_core)


	df_core = pd.DataFrame(
		{
			"label": core_labels,
			"cluster": core_cluster_ids,
		}
	)
	
	print(f"\n[STEP 6] Canonical label induction per cluster on {len(core_labels)} semantic cores")
	cluster_canonicals = {}

	def get_centroid_canonical(cluster_embeddings, cluster_labels):
		# Compute centroid (mean of all embeddings)
		centroid = cluster_embeddings.mean(axis=0, keepdims=True) # (1, embedding_dim)		
		
		# similarity of each label to centroid
		similarities = cosine_similarity(centroid, cluster_embeddings)[0] # (n_samples,)
		
		# label with highest similarity
		best_idx = similarities.argmax() # index of max similarity

		return cluster_labels[best_idx], similarities[best_idx]

	for cid in sorted(df_core.cluster.unique()):
		# Get labels and their embeddings for this cluster
		cluster_mask = df_core.cluster == cid
		cluster_texts = df_core[cluster_mask]["label"].tolist()

		# Get embeddings for this cluster (from X_core)
		cluster_indices = df_core[cluster_mask].index.tolist()
		cluster_embeddings = X_core[cluster_indices]

		# centroid-nearest label
		canonical, score = get_centroid_canonical(cluster_embeddings, cluster_texts)

		cluster_canonicals[cid] = {
			"canonical": canonical,
			"score": score,
			"size": len(cluster_texts),
		}

		print(f"\n[Cluster {cid}] contains {len(cluster_texts)} samples:\n{cluster_texts}")
		print(f">> Canonical (centroid-nearest, sim={score:.4f}): {canonical}")
	
	print("\n[STEP 7] Saving results")
	df_clusters = pd.DataFrame(
		{
			"label": core_labels + noise_labels,
			"cluster": list(core_cluster_ids) + [-1] * len(noise_labels),
			"canonical_label": ([cluster_canonicals[c]["canonical"] for c in core_cluster_ids] + [None] * len(noise_labels))
		}
	)
	out_csv = clusters_fname.replace(".csv", "_semantic_consolidation.csv")
	df_clusters.to_csv(out_csv, index=False)
	print(f"Saved consolidated labels → {out_csv}")
	print("\n[PIPELINE COMPLETE]")
	print("=" * 120)

	return df_clusters

def get_num_clusters_agglomerative(
	X,
	linkage_matrix,
	metric='silhouette',
):
	num_samples, embedding_dim = X.shape
	sample_size = int(3e4) if num_samples > int(3e4) else num_samples
	print(f"Auto-tuning number of clusters for X: {type(X)} {X.shape}")

	# Subsample for large datasets
	if num_samples > sample_size:
		indices = np.random.choice(num_samples, sample_size, replace=False)
		X_sample = X[indices]
		print(f"[OPTIMAL K] Subsampling {sample_size}/{num_samples} points for speed")
	else:
		X_sample = X
		indices = np.arange(num_samples)
	
	if num_samples > int(2e4):
		range_n_clusters = range(100, min(1501, num_samples // 20), 100)
	elif num_samples > int(5e3):
		range_n_clusters = range(20, min(301, num_samples // 15), 20)
	else:
		range_n_clusters = range(5, min(201, num_samples // 2), 5)

	print(f"\n[OPTIMAL K] Testing {len(range_n_clusters)} clusters {range_n_clusters} counts using {metric} score...")
	print(f"{'n_clusters':<12} {metric:<15}")
	print("=" * 30)
	
	scores = []
	for n_clusters in range_n_clusters:
		# Cut dendrogram at this height to get n_clusters
		labels_full = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
		labels_sample = labels_full[indices]
		
		# Skip if only 1 cluster or all singletons
		if len(np.unique(labels_sample)) < 2:
			print(f"{n_clusters:<12} {0.0:<15.4f} (skipped)")
			continue
		
		score = -np.inf
		if metric == 'silhouette':
			score = silhouette_score(X_sample, labels_sample, metric='cosine')
		elif metric == 'davies_bouldin':
			score = davies_bouldin_score(X_sample, labels_sample)
		
		scores.append((n_clusters, score))
		print(f"{n_clusters:<12} {score:<15.4f}")
	
	if not scores:
		raise ValueError("No valid cluster configurations found")
	
	# Select best
	if metric == 'silhouette':
		best_k = max(scores, key=lambda x: x[1])[0]
	else:  # davies_bouldin (lower is better)
		best_k = min(scores, key=lambda x: x[1])[0]
	
	print(f"\n[OPTIMAL K] Selected: {best_k} clusters")

	return best_k

def _clustering_(
	labels: List[List[str]],
	model_id: str,
	device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
	clusters_fname: str = "clusters.csv",
	nc: int = None,
	linkage_method: str = "average",  # 'average', 'complete', 'ward'
	distance_metric: str = "cosine",  # 'cosine', 'euclidean'
	verbose: bool = True,
):	
	if verbose:
			print(f"\n[CLUSTERING - AGGLOMERATIVE] {len(labels)} documents")
			print(f"   ├─ model_id: {model_id}")
			print(f"   ├─ device: {device}")
			print(f"   ├─ linkage: {linkage_method}")
			print(f"   ├─ distance: {distance_metric}")
			print(f"   └─ sample: {labels[:5]}")
	
	# ========== STEP 1: Deduplicate ==========
	print("\n[STEP 1] Deduplicating labels")
	documents = []
	for doc in labels:
			if isinstance(doc, str):
					doc = ast.literal_eval(doc)
			documents.append(list(set(lbl for lbl in doc)))
	
	all_labels = sorted(set(label for doc in documents for label in doc))
	
	print(f"Total documents: {len(documents)}")
	print(f"Unique labels: {len(all_labels)}")
	print(f"Sample labels: {all_labels[:15]}")
	
	# ========== STEP 2: Load Model ==========
	print(f"\n[STEP 2] Loading SentenceTransformer {model_id}")
	
	model = SentenceTransformer(
		model_name_or_path=model_id,
		cache_folder=cache_directory[os.getenv('USER')],
		token=os.getenv("HUGGINGFACE_TOKEN"),
	).to(device)
	
	print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
	
	# ========== STEP 3: Encode ==========
	print(f"\n[STEP 3] Encoding {len(all_labels)} labels")
	X = model.encode(
		all_labels,
		batch_size=256 if len(all_labels) > 1000 else 32,
		show_progress_bar=False,
		convert_to_numpy=True,
		normalize_embeddings=True,  # Critical for cosine distance
	)
	print(f"Embeddings: {X.shape} {X.dtype}")
	
	# ========== STEP 4: Agglomerative Clustering ==========
	print(f"\n[STEP 4] Agglomerative Clustering on {X.shape[0]} labels")
	
	# For large datasets, use fastcluster if available
	try:
			import fastcluster
			use_fastcluster = True
			print("[FASTCLUSTER] Using fastcluster for O(n² log n) performance")
	except ImportError:
			use_fastcluster = False
			print("[SCIPY] Using scipy (slower for large n)")
	
	# Compute linkage matrix
	print(f"[LINKAGE] Computing {linkage_method} linkage with {distance_metric} distance...")
	
	if distance_metric == "cosine":
		# Convert to distance matrix (1 - cosine_similarity)
		# For normalized vectors, cosine distance = 1 - dot product
		distance_matrix = 1 - (X @ X.T)
		np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is exactly 0
		distance_matrix = np.clip(distance_matrix, 0, 2)  # Numerical stability
		
		# Convert to condensed form (upper triangle)
		condensed_dist = squareform(distance_matrix, checks=False)
		
		if use_fastcluster:
			Z = fastcluster.linkage(condensed_dist, method=linkage_method)
		else:
			Z = linkage(condensed_dist, method=linkage_method)
	elif distance_metric == "euclidean":
		if use_fastcluster:
			Z = fastcluster.linkage(X, method=linkage_method, metric='euclidean')
		else:
			Z = linkage(X, method=linkage_method, metric='euclidean')
	else:
		raise ValueError(f"Unsupported distance metric: {distance_metric}")
	
	print(f"[LINKAGE] Complete. Linkage matrix shape: {Z.shape}")
	
	# STEP 5: Determine Optimal Number of Clusters
	if nc is None:
		best_k = get_num_clusters_agglomerative(
			X=X,
			linkage_matrix=Z,
			metric='silhouette',
		)
	else:
		best_k = nc
		print(f"Using {'user-defined' if nc else 'heuristic'} k={best_k} for {len(all_labels)} labels")
	
	# STEP 6: Cut Dendrogram
	print(f"\n[STEP 6] Cutting dendrogram at k={best_k} for {len(all_labels)} labels")
	cluster_labels = fcluster(Z, best_k, criterion='maxclust')
	
	# Convert to 0-indexed
	cluster_labels = cluster_labels - 1
	
	cluster_counts = Counter(cluster_labels)
	print(f"[CLUSTERS] {len(cluster_counts)} clusters formed")
	print(
		f"[CLUSTERS] Size distribution: min={min(cluster_counts.values())}, "
		f"max={max(cluster_counts.values())}, "
		f"mean={np.mean(list(cluster_counts.values())):.1f}"
	)
	
	# 2D cluster visualizations
	# PCA
	pca_projection = PCA(n_components=2, random_state=0).fit_transform(X)
	
	# t-SNE (subsample if too large)
	if len(all_labels) > 10000:
		tsne_indices = np.random.choice(len(all_labels), 10000, replace=False)
		tsne_projection = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(X[tsne_indices])
		tsne_labels = cluster_labels[tsne_indices]
	else:
		tsne_projection = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(X)
		tsne_labels = cluster_labels
	
	# Color palette
	n_colors = min(len(cluster_counts), 256)
	palette = sns.color_palette('tab20', n_colors) if n_colors <= 20 else sns.color_palette('husl', n_colors)
	colors = [palette[i % len(palette)] for i in cluster_labels]
	
	# PCA plot
	plt.figure(figsize=(27, 17))
	plt.scatter(*pca_projection.T, s=40, c=colors, alpha=0.6, marker='o')
	plt.title(f"PCA - Agglomerative Clustering ({len(cluster_counts)} clusters, {len(all_labels)} labels)")
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	out_pca = clusters_fname.replace(".csv", "_pca_agglomerative.png")
	plt.savefig(out_pca, dpi=150, bbox_inches='tight')
	plt.close()
	
	# t-SNE plot
	tsne_colors = [palette[i % len(palette)] for i in tsne_labels]
	plt.figure(figsize=(27, 17))
	plt.scatter(*tsne_projection.T, s=40, c=tsne_colors, alpha=0.6, marker='o')
	plt.title(f"t-SNE - Agglomerative Clustering ({len(cluster_counts)} clusters)")
	plt.xlabel("t-SNE 1")
	plt.ylabel("t-SNE 2")
	out_tsne = clusters_fname.replace(".csv", "_tsne_agglomerative.png")
	plt.savefig(out_tsne, dpi=150, bbox_inches='tight')
	plt.close()
	
	# ========== STEP 8: Canonical Label Selection ==========
	print(f"\n[STEP 8] Selecting canonical labels per cluster")
	
	df = pd.DataFrame(
		{
			'label': all_labels,
			'cluster': cluster_labels
		}
	)
	
	cluster_canonicals = {}
	
	for cid in sorted(df.cluster.unique()):
		cluster_mask = df.cluster == cid
		cluster_texts = df[cluster_mask]['label'].tolist()
		cluster_indices = df[cluster_mask].index.tolist()
		cluster_embeddings = X[cluster_indices]
		
		# Centroid-nearest
		centroid = cluster_embeddings.mean(axis=0, keepdims=True)
		similarities = cosine_similarity(centroid, cluster_embeddings)[0]
		best_idx = similarities.argmax()
		canonical = cluster_texts[best_idx]
		
		cluster_canonicals[cid] = {
			'canonical': canonical,
			'score': float(similarities[best_idx]),
			'size': len(cluster_texts)
		}
		
		if verbose:
			print(f"\n[Cluster {cid}] {len(cluster_texts)} labels:\n{cluster_texts}")
			print(f"\tCanonical: {canonical} (sim={similarities[best_idx]:.4f})")
	
	print(f"\n[STEP 9] Saving results")
	
	df['canonical_label'] = df['cluster'].map(lambda c: cluster_canonicals[c]['canonical'])
	
	out_csv = clusters_fname.replace(".csv", "_semantic_consolidation_agglomerative.csv")
	df.to_csv(out_csv, index=False)
	try:
		df.to_excel(out_csv.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	
	print(f"\n[SUMMARY]")
	print(f"  Total labels: {len(all_labels)}")
	print(f"  Clusters: {len(cluster_counts)}")
	print(f"  Consolidation ratio: {len(all_labels) / len(cluster_counts):.1f}:1")
	print(f"  Largest cluster: {max(cluster_counts.values())} labels")
	print(f"  Smallest cluster: {min(cluster_counts.values())} labels")

	return df

def _post_process_(
	labels_list: List[List[str]], 
	min_kw_ch_length: int = 3,
	max_kw_word_length: int = 5,
	verbose: bool = False
) -> List[List[str]]:
	"""
	Cleans, normalizes, and lemmatizes label lists.
	1. Handles parsing (str -> list).
	2. Lowercases and strips quotes/brackets.
	3. Lemmatizes each word in phrases (e.g., "tool pushers" -> "tool pusher").
	4. Protects abbreviations (NAS, WACs) from lemmatization.
	5. Protects quantified plurals (e.g., "two women") from lemmatization.
	6. Protects title-case phrases (e.g., "As You Like It") from lemmatization.
	7. Filters out keywords shorter than min_kw_ch_length (except abbreviations).
	8. Deduplicates within the sample (post-lemmatization).
	
	Args:
		labels_list: List of label lists to process
		min_kw_ch_length: Minimum character length for keywords (default: 2)
		verbose: Enable detailed logging
	"""
	# Number words for quantified plural detection
	NUMBER_WORDS = {
		"one", "two", "three", "four", "five",
		"six", "seven", "eight", "nine", "ten"
	}

	def is_named_facility(original_phrase: str) -> bool:
		"""
		Check if phrase is a named facility/location.
		These are proper nouns and should not be lemmatized.
		"""
		phrase_lower = original_phrase.lower()
		
		# Common facility/location keywords
		facility_keywords = {
			'air force base', 'naval air station', 'army depot', 'navy yard',
			'national park', 'state park', 'memorial', 'monument',
			'air station', 'naval base', 'military base', 'fort',
			'airport', 'airfield', 'field', 'station'
		}
		
		return any(keyword in phrase_lower for keyword in facility_keywords)

	def is_quantified_plural(original_phrase: str) -> bool:
		tokens = original_phrase.lower().split()
		if len(tokens) < 2:
			return False
		
		is_number = tokens[0].isdigit() or tokens[0] in NUMBER_WORDS
		# Check for standard 's' ending OR common irregular plurals
		is_plural = tokens[1].endswith("s") or tokens[1] in {"men", "women", "children", "people"}
		
		return is_number and is_plural

	def is_title_like(original_phrase: str) -> bool:
		"""
		Check if phrase looks like a title or proper name.
		If 60%+ of tokens start with uppercase, treat as title.
		"""
		tokens = original_phrase.split()
		if len(tokens) < 2:
			return False
		capitalized = sum(1 for t in tokens if t and t[0].isupper())
		return capitalized / len(tokens) >= 0.6
	
	def is_abbreviation(original_phrase: str) -> bool:
		"""
		Check if phrase is an abbreviation or model code.
		Abbreviations should be protected from lemmatization and length filtering.
		"""
		return (
			original_phrase.isupper()
			or "." in original_phrase
			or any(c.isdigit() for c in original_phrase)
		)

	def is_adjectival_phrase(original_phrase: str) -> bool:
		"""
		Detect descriptive adjectival phrases like:
		'newly built', 'recently completed', 'partially destroyed'
		"""
		tokens = original_phrase.lower().split()
		if len(tokens) < 2:
				return False
		return tokens[0].endswith("ly")

	def is_activity_gerund(original_phrase: str) -> bool:
		"""
		Detect single-word activity nouns like:
		'snowshoeing', 'skiing', 'fishing'
		"""
		return (
				" " not in original_phrase
				and original_phrase.lower().endswith("ing")
		)

	def is_event_gerund_phrase(original_phrase: str) -> bool:
			"""
			Detect event phrases like:
			'flag raising', 'ship launching', 'troop landing'
			"""
			tokens = original_phrase.lower().split()
			return (
					len(tokens) >= 2
					and tokens[-1].endswith("ing")
					and not tokens[0].endswith("ly")  # excludes 'newly built'
			)

	def is_phrasal_verb(lemma: str) -> bool:
			tokens = lemma.split()
			if len(tokens) < 2:
					return False

			# verb + particle/preposition
			return (
					tokens[0] not in STOPWORDS and
					tokens[1] in STOPWORDS
			)

	if verbose:
		print(f"Starting post-processing")
		print(f"\tInput {type(labels_list)} length: {len(labels_list) if labels_list else 0}")
		print(f"\tStopwords loaded: {len(STOPWORDS)}")
		print(f"\tMinimum keyword length: {min_kw_ch_length}")
	
	if not labels_list:
		if verbose:
			print("\tEmpty input, returning as-is")
		return labels_list

	lemmatizer = nltk.stem.WordNetLemmatizer()

	def get_wordnet_pos(treebank_tag):
		"""Convert Penn Treebank POS tag to WordNet POS tag"""
		if treebank_tag.startswith('J'):
			return nltk.corpus.wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return nltk.corpus.wordnet.VERB
		elif treebank_tag.startswith('N'):
			return nltk.corpus.wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return nltk.corpus.wordnet.ADV
		else:
			return nltk.corpus.wordnet.NOUN  # Default to noun

	def lemmatize_phrase(phrase: str, original_phrase: str) -> str:
		tokens = phrase.split()
		original_tokens = original_phrase.split()
		
		# Get POS tags for the phrase
		pos_tags = nltk.pos_tag(tokens)
		lemmatized_tokens = []

		for i, (token, pos) in enumerate(pos_tags):
			original_token = original_tokens[i] if i < len(original_tokens) else token
			is_abbr = original_token.isupper() or '.' in original_token
			
			if is_abbr:
				lemmatized_tokens.append(token)  # Keep as-is
			else:
				# For multi-word phrases, treat non-final words as nouns to preserve compound nouns
				# This prevents "diving board" → "dive board", "shipping container" → "ship container"
				if len(tokens) > 1 and i < len(tokens) - 1:
					wordnet_pos = nltk.corpus.wordnet.NOUN
				else:
					wordnet_pos = get_wordnet_pos(pos)

				candidate = lemmatizer.lemmatize(token, pos=wordnet_pos)
				# Reject WordNet bug: boss → bos, pass → pas, glass → glas, grass → gras
				if token.endswith("ss") and candidate == token[:-1]:
					lemmatized_tokens.append(token)
				else:
					lemmatized_tokens.append(candidate)

		return ' '.join(lemmatized_tokens)
	
	processed_batch = []

	for idx, labels in enumerate(labels_list):
		if labels is None:
			processed_batch.append(None)
			continue

		if isinstance(labels, float) and math.isnan(labels):
			processed_batch.append(None)
			continue

		# labels must be list:
		if not isinstance(labels, list):
			# raise ValueError(f"labels must be list, got {type(labels)} {labels}")
			# use eval for str to list conversion:
			try:
				labels = eval(labels)
			except Exception as e:
				print(f"Failed to convert {labels} to list: {e}")
				raise e
				# processed_batch.append(None)
				# continue

		if verbose:
			print(f"\n[Sample {idx+1}/{len(labels_list)}]")
			print(f"{len(labels)} {type(labels)} {type(labels).__name__} {labels}")

		# --- 1. Standardization: Ensure we have a list of strings ---
		current_items = []
		if labels is None:
			if verbose:
				print(f"  → None detected, appending None to output")
			processed_batch.append(None)
			continue
		elif isinstance(labels, list):
			current_items = labels
			if verbose:
				print(f"  → Already a list with {len(current_items)} items")
		elif isinstance(labels, str):
			if verbose:
				print(f"  → String detected, attempting to parse...")
			try:
				parsed = ast.literal_eval(labels)
				if isinstance(parsed, list):
					current_items = parsed
					if verbose:
						print(f"  → Successfully parsed to list with {len(current_items)} items")
				else:
					current_items = [str(parsed)]
					if verbose:
						print(f"  → Parsed to non-list type ({type(parsed)}), wrapping in list")
			except Exception as e:
				current_items = [labels] # Fallback for non-list strings
				if verbose:
					print(f"  → Parse failed ({type(e).__name__}), treating as single-item list")
		else:
			# Numeric or other types
			current_items = [str(labels)]
			if verbose:
				print(f"  → Non-standard type ({type(labels)}), converting to string and wrapping")

		if verbose:
			print(f"  Current items after standardization: {current_items}")

		# --- 2. Normalization & Lemmatization ---
		clean_set = set() # Use set for automatic deduplication
		
		if verbose:
			print(f"  Processing {len(current_items)} items...")
		
		for item_idx, item in enumerate(current_items):
			if verbose:
				print(f"    [{item_idx+1}] Original: {repr(item)} (type: {type(item).__name__})")
			
			if not item:
				if verbose:
					print(f"        → Empty/falsy, skipping")
				continue
			
			if str(item).isupper():
				if verbose:
					print(f"        → All uppercase detected, skipping")
				continue

			# Store original before lowercasing (for abbreviation detection)
			original = str(item).strip()
			
			# String conversion & basic cleanup
			s = original.lower()
			if verbose:
				print(f"        → After str/strip/lower: {repr(s)}")

			# Strip quotes and brackets
			s = s.strip('"').strip("'").strip('()').strip('[]')
			original_cleaned = original.strip('"').strip("'").strip('()').strip('[]')

			# Collapse accidental extra whitespace
			s = ' '.join(s.split())
			original_cleaned = ' '.join(original_cleaned.split())

			if verbose:
				print(f"        → After quote/bracket removal: {repr(s)}")
			
			if not s:
				if verbose:
					print(f"        → Empty after cleanup, skipping")
				continue

			# --- Lemmatization with guards ---
			if is_quantified_plural(original_cleaned):
				lemma = s  # Preserve "two women", "three soldiers"
				if verbose:
					print(f"        → Quantified plural detected, preserving: {repr(lemma)}")
			elif is_title_like(original_cleaned):
				lemma = s  # Preserve "As You Like It", "Gone With the Wind"
				if verbose:
					print(f"        → Title-like phrase detected, preserving: {repr(lemma)}")
			elif is_named_facility(original_cleaned):
				lemma = s  # Preserve "Pease Air Force Base", "Truax Field"
				if verbose:
					print(f"        → Named facility detected, preserving: {repr(lemma)}")	
			elif is_adjectival_phrase(original_cleaned):
				lemma = s  # Preserve "newly built", "recently completed"
				if verbose:
					print(f"        → Adjectival phrase detected, preserving: {repr(lemma)}")
			elif is_activity_gerund(original_cleaned):
				lemma = s  # Preserve "snowshoeing", "skiing", "fishing"
				if verbose:
					print(f"        → Activity gerund detected, preserving: {repr(lemma)}")
			elif is_event_gerund_phrase(original_cleaned):
				lemma = s  # Preserve "flag raising", "ship launching", "troop landing"
				if verbose:
					print(f"        → Event gerund phrase detected, preserving: {repr(lemma)}")
			else:
				# Lemmatize each word in the phrase (with abbreviation protection)
				lemma = lemmatize_phrase(s, original_cleaned)
				if verbose:
					if lemma != s:
						print(f"        → {repr(s)}: Lemmatized → {repr(lemma)} (changed)")
					else:
						print(f"        → {repr(s)}: Lemmatized → {repr(lemma)} (unchanged)")
			
			# Check minimum length (but exempt abbreviations)
			if lemma.isupper():
				if verbose:
					print(f"        → {lemma} All uppercase detected, skipping")
				continue

			if (
				len(lemma) < min_kw_ch_length
				# and not is_abbreviation(original_cleaned) # SMU, NAS
			):
				if verbose:
					print(f"        → {lemma} Too short and not abbreviation (len={len(lemma)} < {min_kw_ch_length}), skipping")
				continue

			if len(lemma.split()) > max_kw_word_length:
				if verbose:
					print(f"        → {lemma} Too short and not abbreviation (len={len(lemma)} < {max_kw_word_length}), skipping")
				continue

			# Replace & with and and remove extra spaces:
			lemma = re.sub(r'\s&\s', ' and ', lemma).strip() # Replace & with and and remove extra spaces

			# check if digit is in the lemma:
			if any(c.isdigit() for c in lemma):
				if verbose:
					print(f"        → {lemma} Digit detected, skipping")
				continue

			# Check if lemma is a number
			if lemma.isdigit():
				if verbose:
					print(f"        → {lemma} Number detected, skipping")
				continue

			if re.match(r'^number\s\d+$', lemma):
				if verbose:
					print(f"        → {lemma} Number detected, skipping")
				continue

			# check for phrasal verbs or words containing prepositions: (dangerous)
			if any(lm in STOPWORDS for lm in lemma.split()):
				if verbose:
					print(f"        → {lemma} Stopword detected, skipping")
				continue

			if is_phrasal_verb(lemma):
				if verbose:
					print(f"        → {lemma} Phrasal verb detected, skipping")
				continue

			# if (
			# 	all(lm in STOPWORDS for lm in lemma.split()) 
			# 	or lemma in STOPWORDS
			# ):
			# 	if verbose:
			# 		print(f"        → {lemma} All words are stopwords or stopword, skipping")
			# 	continue

			# only No. NNNNN ex) No. X1657 or No. 1657
			if re.match(r"^No\.\s\w+$", lemma, re.IGNORECASE):
				if verbose:
					print(f"        → {lemma} Only No. NNNNN detected, skipping")
				continue

			if re.match(r'^\d+\sfeet$', lemma, re.IGNORECASE) or re.match(r'^\d+\sft$', lemma, re.IGNORECASE):
				if verbose:
					print(f"        → {lemma} Only NNNNN feet/ft detected, skipping")
				continue

			if re.match(r'^\d+\sfoot$', lemma, re.IGNORECASE):
				if verbose:
					print(f"        → {lemma} Only NNNNN foot detected, skipping")
				continue

			if any(ch in string.punctuation for ch in lemma):
				if verbose:
					print(f"        → Punctuation detected in {lemma}, skipping")
				continue

			# entire string must consist only of uppercase/lowercase English letters and spaces
			if not re.match(r'^[a-zA-Z\s]+$', lemma):
				if verbose:
					print(f"        → {lemma} Non-alphabetic character detected, skipping")
				continue

			# Check duplicates
			if lemma in clean_set:
				if verbose:
					print(f"        → {lemma} Duplicate detected, skipping")
			else:
				clean_set.add(lemma)
				if verbose:
					print(f"        → {lemma} Added to clean set")

		# Convert back to list
		result = list(clean_set)
		processed_batch.append(result)
		
		if verbose:
			print(f"  Final output for sample {idx+1}: {type(result)} {len(result)}: {result}")
			print(f"  Items: {len(current_items)} → {len(result)} (removed {len(current_items) - len(result)})")
	
	if verbose:
		print(f"\n{'='*80}")
		print(f"Completed post-processing")
		print(f"\tOutput {type(processed_batch)} {len(processed_batch)}")
		print(f"\tNone values: {sum(1 for x in processed_batch if x is None)}")
		print(f"\tEmpty lists: {sum(1 for x in processed_batch if x is not None and len(x) == 0)}")
		print(f"{'='*80}\n")
	
	return processed_batch

def is_english(
	text: str,
	confidence_threshold: float = 0.05,
	use_shortlist: bool = True,  # use shortlist of European languages for detection
	verbose: bool = False,
) -> bool:
	"""
	Check if the given text is in English.
	
	Args:
			text: The text to check
			confidence_threshold: Minimum confidence score to consider text as English
			use_shortlist: If True, use shortlisted European languages for detection.
										If False, use all available languages.
			verbose: Print detailed detection information
	
	Returns:
			True if text is detected as English with confidence above threshold
	"""
	if not text or not str(text).strip():
			return False
	
	# Select detector based on use_shortlist flag
	detector = detector_shortlist if use_shortlist else detector_all
	
	if verbose:
		detector_type = "shortlisted languages" if use_shortlist else "all languages"
		print(f"Checking if text is in English (using {detector_type}):\n{text}\n")
	
	try:
		cleaned_text = " ".join(str(text).split())
		results = detector.compute_language_confidence_values(cleaned_text)
		
		if verbose:
			print(f"All detected languages:")
			for res in results:
				print(f"  {res.language.name:<15} {res.value:.4f}")
		
		if not results:
			return False
		
		for res in results:
			if res.language == Language.ENGLISH:
				score = res.value
				if verbose:
					print(f"\nEnglish confidence: {score:.4f}")
					print(f"Threshold: {confidence_threshold}")
					print(f"Is English: {score > confidence_threshold}")
				
				if score > confidence_threshold:
					return True
		
		return False
	except Exception as e:
		if verbose:
			print(f"Error: {e}")
		return False

def basic_clean(txt: str):
	if (
		not txt
		or not isinstance(txt, str)
		# or "[sic]" in txt
	):
		return ""

	# Step 1: PROTECT real apostrophes FIRST (most important!)
	txt = re.sub(r"(\w)'(\w)", r"\1__APOSTROPHE__\2", txt)
	# This safely protects: don't → don__APOSTROPHE__t, John's → John__APOSTROPHE__s

	# Step 2: Remove known junk/phrase patterns
	junk_phrases = [
		r'view from upstream side of ',
		r"view+\s+looking+\s+\w+\s+\w+\s+",
		r"this is a general view of ",
		r"this is a view of ",
		r"close up view of ",
		r'View from atop ',
		r"another view of ",
		r'full view of ',
		r"rear view of ",
		r"front view of ",
		r"Street View of ",
		r"night view of ",
		r'partial view of ',
		r"panoramic view of ",
		r"downstream view of ",
		r"\s+All+\s+are+\s+unidentified",
		r"general+\s+view+\s+\w+\s+",
		r'here is a view of ',
		r"This item includes\s\w+\s\w+\sof\s",
		r'This item is a photo depicting ',
		r'Source:\s\d+\sPalm Beach city directory',
		r"This item is a photograph depicting ",
		r"This item consists of a photograph of ",
		r'The original finding aid described this item as:',
		r"This photograph includes the following: ",
		r"this photograph is a view of ",
		r"View of bottom, showing ",
		r"Steinheimer note",
		r'World travel.',
		r'Original caption on envelope: ',
		r"In the photo, ",
		r'Historical Miscellaneous -',
		r'\[Translated Title\]',
		r'\[Photograph by: Unknown\]',
		r'Date Month: \[Blank\]',
		r'Date Day: \[Blank\]',
		r'Date Year: \[Blank\]',
		r'Subcategory: \[BLANK\]',
		r'Subcategory: Unidentified',
		r'Category: Miscellaneous ',
		r'Date+\s+Month:+\s+\w+',
		r'Date+\s+Day:+\s+\w+',
		r'Date+\s+Year:+\s+\w+',
		r"This is an image of ",
		r'\[blank\]',
		r'\[sic\]',
		r'\[arrow symbol\]',
		r'as seen from below',
		r'Black-and-white snapshot of ',
		r'An informal color snapshot of ',
		r'Blurry Snapshot of ',
		r'\w+ faded snapshot of ',
		r"This photograph depicts ",
		r'This is a photograph of ',
		r'Photography presents ',
		r"WBP Digitization Studio",
		r'Note on negative envelope',
		r'photo from the photo album ',
		r'The digitalisat was made by the original album.',
		r'The following geographic information is associated with this record: ',
		r'The photo was taken in 1954 on a pilgrimage to World War I sites. From: James K. Monteith, Clayton, Missouri, 35th Division, 128th Field Artillery Regiment.',
		r'From an album of Lorain H. Cunningham, who served in the 129th Field Artillery during World War I and was a friend of Harry S. Truman.',
		r'From album created by Allied Reparations Committee, headed by Ambassador Edwin W\. Pauley, 1945-47\.',
		r'Snapshot of France during World War I, probably taken by\s\w+\s\w+\swho was stationed there one year.',
		r'The information about the photograph was provided by the creator of the collection, Mr. Dan Hadani',
		r'General view of p\.\s\d+\sin the photo album of the NEF with photos.',
		r'The album was probably for the Soviet superior in the NEF.',
		r'State digitization program Saxony: Postcard publisher Brück und Sohn \(digitization\)',
		r'DFG project: worldviews \(2015-2017\), record author: Deutsche Fotothek\/SLUB Dresden \(DF\)',
		r'DFG project: worldviews \(2015-2017\), record author: Deutsche Fotothek\/SLUB Dresden \(\)\)',
		r'DFG project: worldviews \(2015-2017\), record author: Deutsche Fotothek\/SLUB Dresden \(\:\)',
		r'Included in the file is a copy of ',
		r'Description: Imagery taken during the ',
		r'The original finding aid described this photograph as:',
		r'The original finding aid described this as:',
		r'The original database describes this as:',
		r'This image is one of a series of\s\d+\snegatives showing\s',
		r'This image is part of a series of \d+ images taken for the',
		r'Law Title taken from similar image in this series.',
		r'The photographer’s notes from this negative series indicate ',
		r"The photographer's notes from this negative series indicate that ",
		r'The photo is accompanied by a typescript with a description',
		r'The following information was provided by digitizing partner Fold3:',
		r'It was subsequently published in conjunction with an article.',
		r'Original photograph is in a photo album of inaugural events.',
		r'Type: C-N \(Color Negative\) C-P \(Color Print\) ',
		r'Picture documentation (small picture slideshow) about ',
		r'Misc. shots of ',
		r'Original caption: Miscellaneous',
		r'Original caption: Photograph Of ',
		r"Captured Japanese Photograph of ",
		r"The photographer's notes indicate ",
		r'This photograph is of a location in',
		r'This is a photograph from ',
		r'Photograph Relating to ',
		r"This photograph is of ",
		r'This image is part of ',
		r'This image is one of ',
		r'According to Shaffer: ',
		r'Photo album with photo',
		r'Photographs from ',
		r'A+\s+photograph+\s+obtained+\s+by+\s+\w+\s+\w+\s+from film\s+\w+.',
		r'A photograph obtained by ',
		r"This photograph shows ",
		r'The photograph shows ',
		r'The photo shows ',
		r"This photo shows ",
		r'This image shows ',
		r'The image shows ',
		r"This photograph is ",
		r'Photograph Showing ',
		r'Text on the card: ',
		r'The picture shows ',
		r'The photo was taken ',
		r"View is of ",
		r'Photograph taken ',
		r'Original caption:',
		r'Caption: ',
		r'uncaptioned ',
		r'In the picture are ',
		r'In the photograph ',
		r'This photograph of ',
		r'This Photo Of ',
		r'This image depicts ',
		r'Text on the back',
		r"A B\/W photo of ",
		r'black and white',
		r'Photographn of ',
		r'In the photo ',
		r"Photographer:; ",
		r'\[No title entered\]',
		r'\[No description entered\]',
		r'\[No caption entered\]',
		r'Original Title: ',
		r'Other Projects',
		r'Other Project ',
		r'View across ',
		r'view over ',
		r"Unknown Male",
		r"Unknown Man",
		r"Unknown Female",
		r"Unknown Woman",
		r'Pictures of ',
		r'index to ',
		r'Phot. of ',
		r'Slideshow of plastic in color',
		r'color photo',
		r'Colored photo',
		r"color copies.",
		r'in color, broad',
		r"photo in color",
		r"slide copy",
		r'Country: Unknown',
		r'Electronic ed.',
		r'press image',
		r'press photograph',
		r"Placeholder",
		r"No description",
		r'Photograph: ',
		r'Image: ',
		r'Wash. D\.C\.',
		r'File Record',
		r'Original negative.',
		r'Description: ',
		r'- Types -',
		r' - Groups - ',
		r'- Miscellaneous',
		r'Steinheimer+\s\w+\s+note',
		r"\(steinheimer+\s\w+\s\w+\s+note\).",
		r"^Unknown$", # when the whole string is exactly “Unknown”
		r"^\bPhotograph+\s+of+\s+", # Photograph of powerhouse
		r"^\bPhotographs+\s+of+\s+", # Photographs of Admiral Chester
		# r"looking+\s+upstream+\s+\w+\s+",
		# r"looking+\s+downstream+\s+\w+\s+",
	]

	for pattern in junk_phrases:
		txt = re.sub(pattern, ' ', txt, flags=re.IGNORECASE)

	# === REMOVE ARCHIVAL METADATA KEY-VALUE PAIRS (NARA/USAF style) ===
	metadata_patterns = [
		r'https?://\S+|www\.\S+', # URLs
		# r'\bCategory\s*:\s*.+?(?=\n|$)',                     # Category: Aircraft, Ground
		# r'\bSubcategory\s*:\s*.+?(?=\n|$)',                  # Subcategory: Consolidated
		# r'\bSubjects\s*:\s*.+?(?=\n|$)',                     # Subjects: BURMA & INDIA,RECREATION
		# r'\bWar Theater\s*:\s*.+?(?=\n|$)',                  # War Theater: Burma-India
		# r'\bPlace\s*:\s*.+?(?=\n|$)',                        # Place: Burma-India
		r'\bWar Theater(?: Number)?\s*:\s*.+?(?=\n|$)',      	# War Theater Number: 20
		r'\bPhoto Series\s*:\s*.+?(?=\n|$)',                 	# Photo Series: WWII
		r'\bUS Air Force Reference Number\s*:\s*[A-Z0-9]+',  	# US Air Force Reference Number: 74399AC
		r'\bReference Number\s*:\s*[A-Z0-9]+',               	# fallback
		r'\bConsolidated Subjects\s*:?\s*', 									# Consolidated Subjects:
		r'\bProperty Number\s*:?.*', 													# Property Number: X 12345 
		r'\bDFG project\s*:?\s*.*?(?:,|\.|\n|$)',
		r'\bworldviews\s*(?:\(?\d{4}-\d{4}\)?)?',
		r'^Image\s+[A-Z]\b',  # Image A (only removes "Image A", "Image B", etc.)
		r'Europeana\s+Collections\s+\d{4}(?:-\d{4})?',
		r'(?i)^Project\s+.*?\s-\s',
		r'(?i)(?:Series of |a series of |Group of |Collection of )(\d+\s*\w+)',
		r'Part of the documentary ensemble:\s\w+',
		r'History:.*?(?=\bCategory:)',					# History: 8" x 10" print received 19 Jan. 1949 from Air Historical Group, AF, 386th Bomb Group, England. Copied 9 March 1949.
		r'State:\s*[^.]+\.?', 									# State: New York.
		# r'no\.\s\w+\s-\w+',											# No. 43 -C, no. 43 -C, no. 43-122
		# r'no\.\s*\d+(?:-\d+)?', 								# no. 123, no. 123-125
		r'\[No\.\s\w+\]', 											# [No. 123]
		r'vol\.\s\d+',                          # Vol. 5,
		r'vol+\s+\d+',                          # Vol 5,
		r'issue\s\d+',													# issue 1
		r'part\s\d+',														# part 1
		r'picture\s\d+\.',											# picture 125.
		r"^\bView\sof\s", 											# View of powerhouse
		r"^\bCopy\sof\s", 											# Copy of photo
		r'(?:Neg\.|Negative)\s*#\s*\d+',           # Neg. # 6253
		r'(?:CONT|Contract)\s*#\s*\d+',            # CONT # 6715
		r'(?:Reference|Ref\.?)\s*#\s*\d+',         # Reference # 123
		r'(?:Item|Record|File)\s*#\s*\d+',         # Item # 456
		r'(?:Photo|Picture)\s*#\s*\d+',            # Photo # 789
		r"one\sof\sthe\s\w+\sphotographs\sof the\sinventory\sunit\s\d+\/\w\.",
		r"U.S.\sAir\sForce\sNumber\s\w\d+\w+",
		r'Most\s\d+\sera.',
		r"^(\d+)\s-\s",
		r'\d+-\w+-\d+\w-\d+',
		r'-\s+\w\s+thru\s\w\s-',
		r'AS\d+-\d+-\d+\s-\s',
		r"color\sphoto\s\d+",
		r'(?:^|[,\s])\+(?!\d)[A-Za-z0-9]+[.,]?', # remove +B09. but not +123
		r"\sW\.Nr\.\s\d+\s\d+\s", # W.Nr. 4920 3000
		r"\sW\.Nr\.\s\d+\s", # W.Nr. 4920
		r'\bWWII\b',
		r'\bWorld War II\b',
		r'\bWW2\b',
		r'\bWorld War 2\b',
	]

	for pattern in metadata_patterns:
		txt = re.sub(pattern, '   ', txt, flags=re.IGNORECASE)

	# Also catch any remaining lines that are ALL CAPS + colon + value (common in archives)
	txt = re.sub(r'(?m)^[A-Z\s&]{5,}:.*$', '', txt)

	txt = re.sub(r'\\\s*[nrt]', ' ', txt, flags=re.IGNORECASE) # \n, \r, \t
	txt = re.sub(r'\\+[nrt]', ' ', txt, flags=re.IGNORECASE)  # \n, \r, \t
	txt = re.sub(r'\\+', ' ', txt) # remove any stray back‑slashes (e.g. "\ -")

	# # === REMOVE DOCUMENT SERIAL NUMBERS / ARCHIVE IDs ===
	# # Common trailing IDs in parentheses
	# # txt = re.sub(r'\s*\([^()]*\b(?:number|no\.?|photo|negative|item|record|file|usaf|usaaf|nara|gp-|aal-)[^()]*\)\s*$', '', txt, flags=re.IGNORECASE) # (color photo)
	# # txt = re.sub(r'\s*\([^()]*[A-Za-z]{0,4}\d{5,}[A-Za-z]?\)\s*$', '', txt)   # B25604AC, 123456, etc.
	# # Only delete parentheses that consist of *just* an ID (optional 0‑4 letters + 5+ digits)
	# txt = re.sub(r'\s*$$\s*[A-Za-z]{0,4}\d{5,}[A-Za-z]?\s*$$\s*', ' ', txt)

	# txt = re.sub(r'\s*\([^()]*\d{5,}[A-Za-z]?\)\s*$', '', txt)              # pure long numbers
	
	# # Also catch them anywhere if they contain trigger words
	# txt = re.sub(r'\s*\([^()]*\b(?:number|no\.?|photo|negative|item|record|file)[^()]*\)', ' ', txt, flags=re.IGNORECASE)

	# Step 3: Handle newlines/tabs → space
	txt = txt.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')

	# Step 4: Remove quotation marks (single and double)
	# First: remove 'quoted text' style (with possible spaces)
	txt = re.sub(r'''\s*'\s*''', ' ', txt)
	txt = re.sub(r"^'\s*|\s*'$", ' ', txt)
	# Then double quotes
	txt = txt.replace('""', '"').replace('"', '')
	txt = txt.replace("„", " ") # low double quotation mark (unicode: \u201e)	
	# txt = re.sub(r'["“”„]', ' ', txt) # all double quotes
	txt = txt.replace("‘", " ") # left single quotation mark (unicode: \u2018)	
	
	# txt = txt.replace(r'#', ' ') # not always safe!
	# txt = txt.replace(',', ' ')

	# remove everything inside parantheses
	txt = re.sub(r'\([^)]*\)', ' ', txt)

	# # remove everything inside brackets
	# txt = re.sub(r'\[[^\]]*\]', ' ', txt)

	txt = re.sub(r'-{2,}', ' ', txt)   # multiple dashes
	txt = re.sub(r'\.{2,}', '.', txt)  # ellipses ...
	txt = re.sub(r'[\[\]]', ' ', txt)  # square brackets
	txt = re.sub(r'[\{\}]', ' ', txt)  # curly braces
	txt = re.sub(r'[\(\)]', ' ', txt)  # parentheses

	txt = re.sub(r'\[\?\]', ' ', txt) # remove [?]
	txt = re.sub(r'\s+', ' ', txt) # Collapse all whitespace
	txt = txt.replace("'", "") # stray leftover single quotes (should be none, but safe)
	txt = txt.replace("__APOSTROPHE__", "'") # RESTORE real apostrophes
	txt = txt.strip() # Remove leading/trailing whitespace

	return txt

def get_enriched_description(
	df: pd.DataFrame, 
	check_english: bool=False, 
	min_length: int=3,
	verbose: bool=False
)-> pd.DataFrame:
	if verbose:
		print(f"\nGenerating enriched_document_description for {type(df)} {df.shape}...")
		print(f"\t{list(df.columns)}")
		print(f"\tcheck_english: {check_english} min_length: {min_length}")

	# check if title and description are in df.columns:
	if "title" not in df.columns:
		raise ValueError("title column not found in df")
	if "description" not in df.columns:
		raise ValueError("description column not found in df")

	# check if how many empty(Nones) exist in title and description:
	if verbose:
		print(f"\tEmpty title: {df['title'].isna().sum()}/{df.shape[0]} "
			f"({df['title'].isna().sum()/df.shape[0]*100:.2f}%)"
		)
		print(f"\tEmpty description: {df['description'].isna().sum()}/{df.shape[0]} "
			f"({df['description'].isna().sum()/df.shape[0]*100:.2f}%)"
		)

	# safety check if enriched_document_description already exists in df.columns:
	if "enriched_document_description" in df.columns:
		df = df.drop(columns=['enriched_document_description'])
		if verbose:
			print("enriched_document_description column already exists. Dropped it...")
			print(f"df: {df.shape} {type(df)} {list(df.columns)}")

	df_enriched = df.copy(deep=True)
	
	if verbose:
		print(f"df_enriched: {df_enriched.shape} {type(df_enriched)} {list(df_enriched.columns)}")

	df_enriched['enriched_document_description'] = df_enriched.apply(
		lambda row: ". ".join(
			filter(
				None, 
				[
					basic_clean(str(row['title'])) if pd.notna(row['title']) and str(row['title']).strip() else None, 
					basic_clean(str(row['description'])) if pd.notna(row['description']) and str(row['description']).strip() else None,
					# basic_clean(str(row['keywords'])) if 'keywords' in df_enriched.columns and pd.notna(row['keywords']) and str(row['keywords']).strip() else None
				]
			)
		),
		axis=1
	)
	
	# Filter out samples with text < min_length
	df_enriched['enriched_document_description'] = df_enriched['enriched_document_description'].apply(
		lambda x: x if x and len(x.strip()) >= min_length else None
	)

	# Ensure proper ending
	df_enriched['enriched_document_description'] = df_enriched['enriched_document_description'].apply(
		lambda x: x.rstrip('.') + '.' if x and not x.endswith('.') else x
	)

	# length = 0 => None
	df_enriched['enriched_document_description'] = df_enriched['enriched_document_description'].apply(
		lambda x: x if x and x.strip() and x.strip() != '.' else None
	)

	# exclude texts that are not English:
	if check_english:
		df_enriched['enriched_document_description'] = df_enriched['enriched_document_description'].apply(
			lambda x: x if is_english(text=x, confidence_threshold=0.01, use_shortlist=True, verbose=verbose) else None
		)

	# Filter out samples with text < min_length
	df_enriched['enriched_document_description'] = df_enriched['enriched_document_description'].apply(
		lambda x: x if x and len(x.strip()) >= min_length else None
	)
		
	if verbose:
		print(f"Samples filtered (too short): {df_enriched['enriched_document_description'].isna().sum()}")
		
	if verbose:
		print(
			f"Number of empty enriched_document_description: "
			f"{df_enriched['enriched_document_description'].isna().sum()} "
			f"out of {df_enriched.shape[0]} total samples "
			f"({df_enriched['enriched_document_description'].isna().sum()/df_enriched.shape[0]*100:.2f}%) "
		)
		print(f"{type(df_enriched)} {df_enriched.shape} {list(df_enriched.columns)}")

	return df_enriched

def validate_cleaning_quality(df: pd.DataFrame, text_column: str = 'enriched_document_description', top_n: int = 100):
		"""
		Improved validation that distinguishes between:
		- Metadata artifacts (BAD)
		- Natural language patterns (GOOD)
		"""
		from collections import Counter
		from nltk import ngrams
		from nltk.tokenize import word_tokenize
		
		print(f"Analyzing {len(df)} samples for cleaning quality...")
		
		all_text = ' '.join(df[text_column].dropna().astype(str).tolist())
		tokens = word_tokenize(all_text.lower())
		
		results = {
				'total_samples': len(df),
				'warnings': [],
				'informational': [],  # NEW: Non-problematic patterns
				'recommendations': [],
				'suspicious_patterns': {}
		}
		
		# === 1. Check for metadata field names (ACTUAL PROBLEMS) ===
		metadata_indicators = [
				'category:', 'subcategory:', 'subjects:', 'war theater:', 'photo series:',
				'property number:', 'reference number:', 'photographer:', 'history:',
				'original caption:', 'description:', 'project:', 'credit:',
		]
		
		found_metadata = []
		for indicator in metadata_indicators:
				count = all_text.lower().count(indicator)
				if count > len(df) * 0.01:  # Appears in >1% of samples
						found_metadata.append((indicator, count))
						results['warnings'].append(f"⚠️ Found '{indicator}' {count} times ({count/len(df)*100:.1f}% of samples)")
		
		if found_metadata:
				results['suspicious_patterns']['metadata_fields'] = found_metadata
		
		# === 2. Bigram analysis - FILTER OUT NATURAL LANGUAGE ===
		bigrams = list(ngrams(tokens, 2))
		bigram_freq = Counter(bigrams)
		top_bigrams = bigram_freq.most_common(top_n)
		
		# Define natural English stopword bigrams (NOT problems)
		natural_bigrams = {
				'of the', 'in the', 'at the', 'on the', 'to the', 'for the',
				'from the', 'by the', 'with the', 'as the', 'is the', 'was the',
				'and the', 'or the', 'that the', 'this the', 'which the',
		}
		
		suspicious_bigrams = []
		informational_bigrams = []
		
		for bigram, count in top_bigrams[:50]:
				phrase = ' '.join(bigram)
				frequency_pct = (count / len(df)) * 100
				
				# Skip natural language patterns
				if phrase in natural_bigrams:
						if frequency_pct > 5:
								informational_bigrams.append((phrase, count, frequency_pct, "Natural English"))
						continue
				
				# Flag entity names separately (informational, not problems)
				if any(word[0].isupper() for word in bigram):  # Contains capitalized words
						if frequency_pct > 5:
								informational_bigrams.append((phrase, count, frequency_pct, "Entity/Place name"))
						continue
				
				# Now flag actual suspicious patterns
				if frequency_pct > 8:  # Raised threshold for non-natural patterns
						suspicious_bigrams.append((phrase, count, frequency_pct))
		
		if suspicious_bigrams:
				results['suspicious_patterns']['frequent_bigrams'] = suspicious_bigrams
				print("\n⚠️ Suspiciously frequent bigrams (non-natural, appear in >8% of samples):")
				for phrase, count, pct in suspicious_bigrams:
						print(f"   '{phrase}': {count} times ({pct:.1f}%)")
		
		if informational_bigrams:
				print("\n✅ Informational patterns (expected in historical dataset):")
				for phrase, count, pct, reason in informational_bigrams[:10]:
						print(f"   '{phrase}': {count} times ({pct:.1f}%) - {reason}")
		
		# === 3. Generate smart recommendations ===
		if not results['warnings']:
				results['recommendations'].append("✅ Text appears well-cleaned!")
				results['recommendations'].append("📊 Frequent patterns are natural language or entity names")
		else:
				results['recommendations'].append(f"❌ Found {len(results['warnings'])} potential issues")
				results['recommendations'].append("🔧 Consider adding these patterns to basic_clean():")
				
				for indicator, count in found_metadata:
						results['recommendations'].append(f"   r'\\b{re.escape(indicator)}\\b'")
		
		return results

def detect_outlier_samples(df: pd.DataFrame, text_column: str = 'enriched_document_description'):
		"""
		Find samples that are statistical outliers (likely have cleaning issues).
		"""		
		df_analysis = df.copy()
		
		# Calculate text statistics
		df_analysis['text_length'] = df_analysis[text_column].str.len()
		df_analysis['word_count'] = df_analysis[text_column].str.split().str.len()
		df_analysis['uppercase_ratio'] = df_analysis[text_column].apply(
				lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
		)
		df_analysis['digit_ratio'] = df_analysis[text_column].apply(
				lambda x: sum(1 for c in str(x) if c.isdigit()) / max(len(str(x)), 1)
		)
		df_analysis['special_char_ratio'] = df_analysis[text_column].apply(
				lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace()) / max(len(str(x)), 1)
		)
		
		# Detect outliers using IQR method
		outliers = {}
		
		for col in ['text_length', 'uppercase_ratio', 'digit_ratio', 'special_char_ratio']:
				Q1 = df_analysis[col].quantile(0.25)
				Q3 = df_analysis[col].quantile(0.75)
				IQR = Q3 - Q1
				
				lower_bound = Q1 - 3 * IQR
				upper_bound = Q3 + 3 * IQR
				
				outlier_mask = (df_analysis[col] < lower_bound) | (df_analysis[col] > upper_bound)
				outlier_indices = df_analysis[outlier_mask].index.tolist()
				
				if outlier_indices:
						outliers[col] = outlier_indices
						print(f"\n⚠️ Found {len(outlier_indices)} outliers in {col}")
						print(f"   Normal range: {lower_bound:.3f} - {upper_bound:.3f}")
						print(f"   Sample outliers:")
						for idx in outlier_indices[:3]:
								if idx in df_analysis.index:
										col_value = df_analysis.loc[idx, col]
										text_value = df_analysis.loc[idx, text_column]
										if col_value is not None and text_value is not None and isinstance(text_value, str):
												print(f"      [{idx}] {col}={col_value:.3f}")
												print(f"          Text preview: {text_value[:150]}...")
										else:
												print(f"      [{idx}] {col}={col_value if col_value is not None else 'N/A'} - [No valid text]")
								else:
										print(f"      [{idx}] [Index not found]")
		
		return outliers, df_analysis

def find_repeated_substrings(df: pd.DataFrame, text_column: str = 'enriched_document_description', min_length: int = 10, min_frequency: int = 100):
		"""
		Find substrings that appear verbatim across many samples.
		These are likely metadata artifacts or boilerplate text.
		"""
		from collections import defaultdict
		
		print(f"Searching for repeated substrings (min_length={min_length}, min_frequency={min_frequency})...")
		
		# Extract all substrings of sufficient length
		substring_counts = defaultdict(int)
		
		for text in df[text_column].dropna():
				text_str = str(text)
				# Generate all substrings of min_length
				for i in range(len(text_str) - min_length + 1):
						substring = text_str[i:i+min_length]
						# Only count if it's not just whitespace or punctuation
						if len(substring.strip()) >= min_length - 2:
								substring_counts[substring] += 1
		
		# Filter by frequency
		frequent_substrings = {
				substring: count 
				for substring, count in substring_counts.items() 
				if count >= min_frequency
		}
		
		# Sort by frequency
		sorted_substrings = sorted(frequent_substrings.items(), key=lambda x: x[1], reverse=True)
		
		print(f"\nFound {len(sorted_substrings)} repeated substrings:")
		for substring, count in sorted_substrings[:20]:
				frequency_pct = (count / len(df)) * 100
				print(f"   '{substring}': {count} times ({frequency_pct:.1f}%)")
		
		return sorted_substrings

def sample_based_quality_check(df: pd.DataFrame, text_column: str = 'enriched_document_description', sample_size: int = 100):
		"""
		Random sample inspection with automated checks.
		"""
		import random
		
		sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
		
		issues_found = {
				'has_metadata_fields': [],
				'has_long_numbers': [],
				'has_bracketed_content': [],
				'too_short': [],
				'too_many_special_chars': [],
				'mostly_uppercase': []
		}
		
		for idx, row in sample_df.iterrows():
				text = str(row[text_column])
				
				# Check 1: Metadata field names
				if re.search(r'\b(category|subcategory|subjects|war theater|photo series|property number):', text, re.IGNORECASE):
						issues_found['has_metadata_fields'].append(idx)
				
				# Check 2: Long numbers (likely IDs)
				if re.search(r'\b\d{6,}\b', text):
						issues_found['has_long_numbers'].append(idx)
				
				# Check 3: Bracketed content
				if re.search(r'\[[^\]]{3,}\]', text):
						issues_found['has_bracketed_content'].append(idx)
				
				# Check 4: Too short (< 20 chars)
				if len(text.strip()) < 20:
						issues_found['too_short'].append(idx)
				
				# Check 5: Too many special characters
				special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
				if special_char_ratio > 0.15:
						issues_found['too_many_special_chars'].append(idx)
				
				# Check 6: Mostly uppercase (likely metadata)
				if len(text) > 20:
						uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
						if uppercase_ratio > 0.3:
								issues_found['mostly_uppercase'].append(idx)
		
		# Print results
		print(f"\nQuality check on {sample_size} random samples:")
		print("="*80)
		
		total_issues = sum(len(issues) for issues in issues_found.values())
		
		if total_issues == 0:
				print("✅ No issues found! Text appears well-cleaned.")
		else:
				print(f"⚠️ Found issues in {total_issues} samples:\n")
				
				for issue_type, indices in issues_found.items():
						if indices:
								percentage = (len(indices) / sample_size) * 100
								print(f"   {issue_type}: {len(indices)} samples ({percentage:.1f}%)")
								
								# Show example
								if indices:
										example_idx = indices[0]
										# Check if index exists in dataframe
										if example_idx in df.index:
												example_text = df.loc[example_idx, text_column]
												if example_text is not None and isinstance(example_text, str):
														print(f"      Example [{example_idx}]: {example_text[:200]}...")
												else:
														print(f"      Example [{example_idx}]: [No valid text content]")
										else:
												print(f"      Example [{example_idx}]: [Index not found in DataFrame]")
										print()
		
		return issues_found

def compare_cleaning_versions(df: pd.DataFrame, before_col: str = 'description', after_col: str = 'enriched_document_description', n_samples: int = 10):
		"""
		Display side-by-side comparison of text before and after cleaning.
		"""
		import textwrap
		
		print("BEFORE/AFTER CLEANING COMPARISON")
		print("="*120)
		
		sample_df = df.sample(n=n_samples, random_state=42)
		
		for idx, row in sample_df.iterrows():
				before_val = row.get(before_col)
				after_val = row.get(after_col)
				
				before = str(before_val)[:300] if before_val is not None else "[No content]"
				after = str(after_val)[:300] if after_val is not None else "[No content]"
				
				print(f"\nSample #{idx}:")
				print("-"*120)
				print("BEFORE:")
				print(textwrap.fill(before, width=110))
				print("\nAFTER:")
				print(textwrap.fill(after, width=110))
				print("-"*120)

def validate_text_cleaning_pipeline(df: pd.DataFrame, text_column: str = 'enriched_document_description'):
	print("\nTEXT CLEANING QUALITY VALIDATION PIPELINE")
	
	# Step 1: N-gram analysis
	print("\n[1/5] Running N-gram frequency analysis...")
	validation_results = validate_cleaning_quality(df, text_column)
	
	# Step 2: Outlier detection
	print("\n[2/5] Detecting statistical outliers...")
	outliers, stats_df = detect_outlier_samples(df, text_column)
	
	# Step 3: Repeated substring search
	print("\n[3/5] Searching for repeated substrings...")
	repeated_patterns = find_repeated_substrings(df, text_column, min_length=15, min_frequency=50)
	
	# Step 4: Sample-based quality check
	print("\n[4/5] Running sample-based quality checks...")
	quality_issues = sample_based_quality_check(df, text_column, sample_size=200)
	
	# Step 5: Generate cleaning recommendations
	print("\n[5/5] Generating recommendations...")
	
	recommendations = []
	
	# Based on n-gram analysis
	if 'frequent_bigrams' in validation_results.get('suspicious_patterns', {}):
		recommendations.append("\n📝 Recommended additions to basic_clean() based on bigrams:")
		for phrase, count, pct in validation_results['suspicious_patterns']['frequent_bigrams'][:10]:
			recommendations.append(f"   r'\\b{phrase}\\b',  # Appears in {pct:.1f}% of samples")
	
	# Based on repeated substrings
	if repeated_patterns:
		recommendations.append("\n📝 Recommended additions based on repeated substrings:")
		for substring, count in repeated_patterns[:5]:
			frequency_pct = (count / len(df)) * 100
			# Clean the substring for use in regex
			cleaned = re.escape(substring.strip())
			recommendations.append(f"   r'{cleaned}',  # Appears {count} times ({frequency_pct:.1f}%)")
	
	print("\nVALIDATION SUMMARY\n")
	
	if not validation_results['warnings'] and not quality_issues:
		print("✅ Text cleaning quality is GOOD - safe to proceed with LLM extraction")
	else:
		print("⚠️ Text cleaning quality needs improvement")
		print(f"\nFound {len(validation_results['warnings'])} warnings")
		print("\nRecommended actions:")
		for rec in recommendations:
			print(rec)
	
	results = {
		'validation_results': validation_results,
		'outliers': outliers,
		'repeated_patterns': repeated_patterns,
		'quality_issues': quality_issues,
		'recommendations': recommendations
	}

	print(json.dumps(results, indent=2, ensure_ascii=False))

	return results
