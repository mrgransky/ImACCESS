import argparse
import random
import torch
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# local:
# python viz.py --audit_jsonl /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_mlm_cot_modality_conflict_audit.jsonl --freq_json /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/outputs/metadata_multi_label_mlm_cot_modality_conflict_audit_global_label_frequency.json --stage4_parquet /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/outputs/metadata_multi_label_mlm_cot_modality_conflict_audit_auditable_matrix.parquet --out_dir /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/outputs --n_raw_concepts 172

REGIME_COLORS = {
	"AGREEMENT":        "#028307",
	"SOFT_CONFLICT":    "#F8AB37",
	"HARD_CONFLICT":    "#F44336",
	"MISSING_MODALITY": "#141414",
	"UNKNOWN":          "#607D8B",
}

def _load_jsonl(path: str) -> list[dict]:
	records = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				try:
					records.append(json.loads(line))
				except json.JSONDecodeError:
					pass
	return records

def _to_list(x) -> list:
    """Safely convert a parquet list cell (may be np.ndarray, list, None, NaN) to a plain Python list."""
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return []

def _save(fig, out_path: Optional[str], name: str):
	if out_path:
		p = Path(out_path) / f"{name}.png"
		fig.savefig(p, dpi=150, bbox_inches="tight")
		print(f"[VIZ] Saved → {p}")
	# plt.show()
	plt.close(fig)

# V1 — Regime Distribution Donut
def viz_regime_distribution(
	audit_jsonl: str,
	out_dir: Optional[str] = None,
):
	"""
	Donut chart of regime distribution + bar of absolute counts.
	Reads Stage 2 Evidence Receipt JSONL.
	"""
	records = _load_jsonl(audit_jsonl)
	regime_counts = Counter(r.get("regime", "UNKNOWN") for r in records)
	regimes = list(regime_counts.keys())
	counts  = [regime_counts[r] for r in regimes]
	colors  = [REGIME_COLORS.get(r, "#607D8B") for r in regimes]
	total   = sum(counts)
	fig, axes = plt.subplots(1, 2, figsize=(13, 6))
	fig.suptitle("Stage 2 — Regime Distribution", fontsize=14, fontweight="bold")

	# Donut
	wedges, texts, autotexts = axes[0].pie(
		counts, 
		labels=None, 
		colors=colors,
		autopct=lambda p: f"{p:.1f}%\n({int(round(p*total/100)):,})",
		startangle=90, 
		pctdistance=0.75,
		wedgeprops=dict(width=0.5, edgecolor="#FFFFFF", linewidth=2),
	)

	for at in autotexts:
		at.set_fontsize(9)
	axes[0].legend(
		wedges, 
		[f"{r}  ({regime_counts[r]:,})" for r in regimes],
		loc="lower center", 
		bbox_to_anchor=(0.5, -0.12),
		fontsize=9, 
		frameon=False,
	)
	axes[0].set_title(f"Total samples: {total:,}", fontsize=11)
	
	# Bar
	bars = axes[1].barh(regimes, counts, color=colors, edgecolor="white", height=0.55)
	for bar, cnt in zip(bars, counts):
		axes[1].text(
			bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
			f"{cnt:,}  ({cnt/total*100:.1f}%)",
			va="center", 
			fontsize=9,
		)
	axes[1].set_xlabel("Sample count")
	axes[1].set_title("Absolute counts by regime")
	axes[1].invert_yaxis()
	axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
	axes[1].spines[["top", "right"]].set_visible(False)
	
	plt.tight_layout()
	_save(fig, out_dir, "V1_regime_distribution")

# V2 — Cross-Modal Similarity Histogram by Regime
def viz_similarity_histograms(
	audit_jsonl: str,
	sim_key: str = "metrics.set_similarity",       # key inside each receipt
	out_dir: Optional[str] = None,
):
	"""
	Overlapping histograms of cross-modal cosine similarity, one curve per regime.
	Reveals whether your routing thresholds are well-placed.
	"""
	records = _load_jsonl(audit_jsonl)
	regime_sims: dict[str, list[float]] = defaultdict(list)
	for r in records:
		regime = r.get("regime", "UNKNOWN")
		if "." in sim_key:
			outer, inner = sim_key.split(".", 1)
			sim = r.get(outer, {}).get(inner)
		else:
			sim = r.get(sim_key)
		if sim is not None:
			try:
				regime_sims[regime].append(float(sim))
			except (TypeError, ValueError):
				pass
	
	if not regime_sims:
		print(
			f"[VIZ][WARN] No '{sim_key}' field found in receipts. "
			"Check the key name in your Stage 2 output."
		)
		return

	fig, ax = plt.subplots(figsize=(14, 7))
	fig.suptitle("Stage 2 — Cross-Modal Cosine Similarity by Regime",
							 fontsize=14, fontweight="bold")
	bins = np.linspace(0, 1, 41)
	for regime, sims in sorted(regime_sims.items()):
			ax.hist(
					sims, bins=bins, alpha=0.55,
					color=REGIME_COLORS.get(regime, "#607D8B"),
					label=f"{regime}  (n={len(sims):,})",
					density=True, edgecolor="none",
			)
			ax.axvline(np.median(sims), color=REGIME_COLORS.get(regime, "#607D8B"),
								 linestyle="--", linewidth=1.2, alpha=0.85)
	ax.set_xlabel("Cosine similarity (text ↔ visual concept sets)")
	ax.set_ylabel("Density")
	ax.legend(fontsize=9, frameon=False)
	ax.spines[["top", "right"]].set_visible(False)
	plt.tight_layout()
	_save(fig, out_dir, "V2_similarity_histograms")

# V3 — Concept Frequency Rank-Log Plot (Zipf curve)
def viz_zipf_curve(
		freq_json: str,
		top_n_labels: int = 30,
		out_dir: Optional[str] = None,
):
		"""
		Log-log rank vs frequency plot (Zipf) for the raw concept vocabulary.
		Annotates the top-N most frequent concepts.
		"""
		with open(freq_json, encoding="utf-8") as f:
				freq_dict: dict = json.load(f)

		sorted_freqs = sorted(freq_dict.values(), reverse=True)
		sorted_items = sorted(freq_dict.items(), key=lambda x: -x[1])
		ranks = np.arange(1, len(sorted_freqs) + 1)

		fig, ax = plt.subplots(figsize=(15, 8))
		fig.suptitle("Bridge — Raw Concept Vocabulary: Zipf Distribution",
								 fontsize=14, fontweight="bold")

		ax.loglog(ranks, sorted_freqs, color="#1565C0", linewidth=1.8, alpha=0.85)
		ax.fill_between(ranks, sorted_freqs, alpha=0.08, color="#1565C0")

		# Annotate top-N
		for rank, (label, freq) in enumerate(sorted_items[:top_n_labels], start=1):
				ax.annotate(
						label, xy=(rank, freq),
						xytext=(rank * 1.3, freq * 1.15),
						fontsize=6.5, color="#333",
						arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6),
				)

		# Singleton line
		singleton_rank = next(
				(i + 1 for i, v in enumerate(sorted_freqs) if v == 1), len(sorted_freqs)
		)
		ax.axvline(singleton_rank, color="#E53935", linestyle=":", linewidth=1.2,
							 label=f"Singletons start (rank {singleton_rank:,})")

		ax.set_xlabel("Rank (log scale)")
		ax.set_ylabel("Frequency (log scale)")
		ax.legend(fontsize=9, frameon=False)
		ax.spines[["top", "right"]].set_visible(False)

		plt.tight_layout()
		_save(fig, out_dir, "V3_zipf_curve")

# V4 — Vocabulary Funnel (Waterfall bar)
def viz_vocabulary_funnel(
		n_raw_concepts: int,
		n_after_clustering: int,
		n_after_audit: int,
		n_target_vocab: int,
		n_emb_cache: int,
		out_dir: Optional[str] = None,
):
		"""
		Horizontal waterfall showing how many concepts survive each Bridge gate.
		Pass the numbers directly from your Bridge printout.

		Example:
				viz_vocabulary_funnel(
						n_raw_concepts      = 172,
						n_after_clustering  = 45,
						n_after_audit       = 38,
						n_target_vocab      = 35,
						n_emb_cache         = 207,
				)
		"""
		stages = [
				"Raw VLM concepts\n(post-normalisation)",
				"After clustering\n(optimal k)",
				"After cluster audit\n(Step 8a/8c)",
				"Target vocabulary |V|\n(canonical labels)",
				"emb_cache\n(raw + canonical)",
		]
		values = [
				n_raw_concepts,
				n_after_clustering,
				n_after_audit,
				n_target_vocab,
				n_emb_cache,
		]
		colors = ["#1565C0", "#1976D2", "#42A5F5", "#4CAF50", "#8BC34A"]

		fig, ax = plt.subplots(figsize=(14, 8))
		fig.suptitle("Bridge — Vocabulary Funnel", fontsize=14, fontweight="bold")

		bars = ax.barh(stages[::-1], values[::-1], color=colors[::-1],
									 edgecolor="white", height=0.55)
		for bar, val in zip(bars, values[::-1]):
				ax.text(
						bar.get_width() + max(values) * 0.01,
						bar.get_y() + bar.get_height() / 2,
						f"{val:,}", va="center", fontsize=10, fontweight="bold",
				)

		# Reduction annotations
		for i in range(len(values) - 2):  # skip emb_cache (it grows)
				reduction = (values[i] - values[i + 1]) / max(values[i], 1) * 100
				if reduction > 0:
						y_pos = len(stages) - 1 - i - 0.5
						ax.text(
								max(values) * 0.5, y_pos,
								f"▼ {reduction:.0f}% reduction",
								va="center", ha="center", fontsize=8,
								color="#E53935", style="italic",
						)

		ax.set_xlabel("Number of concepts / labels")
		ax.spines[["top", "right"]].set_visible(False)
		ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

		plt.tight_layout()
		_save(fig, out_dir, "V4_vocabulary_funnel")

# V5 — CGD Score Heatmap per Regime
def viz_cgd_heatmap(
		stage3_jsonl: str,
		out_dir: Optional[str] = None,
):
		"""
		Heatmap of mean C, G, D scores broken down by regime.
		Reads Stage 3 Micro-CGD Audit output JSONL.
		Expected fields per record: regime, cgd_scores: {label: {C, G, D}}
		"""
		records = _load_jsonl(stage3_jsonl)

		regime_cgd: dict[str, dict[str, list]] = defaultdict(lambda: {"C": [], "G": [], "D": []})

		for rec in records:
				regime = rec.get("regime", "UNKNOWN")
				cgd    = rec.get("cgd_scores", {})
				for label_scores in cgd.values():
						for dim in ("C", "G", "D"):
								val = label_scores.get(dim)
								if val is not None:
										try:
												regime_cgd[regime][dim].append(float(val))
										except (TypeError, ValueError):
												pass

		if not regime_cgd:
				print("[VIZ][WARN] No cgd_scores found in Stage 3 JSONL.")
				return

		regimes = sorted(regime_cgd.keys())
		dims    = ["C", "G", "D"]
		matrix  = np.array(
			[
				[np.mean(regime_cgd[r][d]) if regime_cgd[r][d] else np.nan for d in dims]
				for r in regimes
			]
		)

		fig, ax = plt.subplots(figsize=(7, max(3, len(regimes) * 1.2)))
		fig.suptitle("Stage 3 — Mean CGD Scores by Regime", fontsize=14, fontweight="bold")

		im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
		ax.set_xticks(range(len(dims)))
		ax.set_xticklabels(["Coverage (C)", "Grounding (G)", "Density (D)"], fontsize=10)
		ax.set_yticks(range(len(regimes)))
		ax.set_yticklabels(regimes, fontsize=10)

		for i in range(len(regimes)):
				for j in range(len(dims)):
						val = matrix[i, j]
						if not np.isnan(val):
								ax.text(j, i, f"{val:.3f}", ha="center", va="center",
												fontsize=11, fontweight="bold",
												color="black" if 0.3 < val < 0.8 else "white")

		plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Mean score [0, 1]")
		plt.tight_layout()
		_save(fig, out_dir, "V5_cgd_heatmap")

# V6 — Supervision Weight Distribution (w_pos, w_neg)
def viz_supervision_weights(
		stage4_parquet: str,
		out_dir: Optional[str] = None,
):
		"""
		KDE + histogram of w_pos and w_neg, faceted by regime.
		Reads the auditable_supervision_matrix.parquet produced by Stage 4.
		Expected columns: regime, w_pos, w_neg
		"""
		df = pd.read_parquet(stage4_parquet, columns=["regime", "w_pos", "w_neg"])

		regimes = sorted(df["regime"].unique())
		fig, axes = plt.subplots(
				len(regimes), 2,
				figsize=(12, 3 * len(regimes)),
				sharex=False, sharey=False,
		)
		if len(regimes) == 1:
				axes = [axes]

		fig.suptitle("Stage 4 — Supervision Weight Distributions by Regime",
								 fontsize=14, fontweight="bold")

		bins = np.linspace(0, 1, 31)
		for row_idx, regime in enumerate(regimes):
				sub = df[df["regime"] == regime]
				color = REGIME_COLORS.get(regime, "#607D8B")

				for col_idx, (weight_col, label) in enumerate(
						[("w_pos", "w_pos (positive weight)"), ("w_neg", "w_neg (hard-negative weight)")]
				):
						ax = axes[row_idx][col_idx]
						vals = sub[weight_col].dropna().values
						ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor="white")
						ax.axvline(np.mean(vals), color="black", linestyle="--",
											 linewidth=1.2, label=f"mean={np.mean(vals):.3f}")
						ax.axvline(np.median(vals), color="gray", linestyle=":",
											 linewidth=1.2, label=f"median={np.median(vals):.3f}")
						ax.set_title(f"{regime} — {label}  (n={len(vals):,})", fontsize=9)
						ax.legend(fontsize=8, frameon=False)
						ax.spines[["top", "right"]].set_visible(False)
						ax.set_xlabel(weight_col)
						ax.set_ylabel("Count")

		plt.tight_layout()
		_save(fig, out_dir, "V6_supervision_weights")

# V7 — Regime × Top-K Label Co-occurrence Heatmap
def viz_regime_label_heatmap(
		stage4_parquet: str,
		top_k: int = 30,
		out_dir: Optional[str] = None,
):
	"""
	Heatmap: rows = regimes, columns = top-K canonical labels by frequency.
	Cell value = fraction of samples in that regime that contain the label.
	"""
	df = pd.read_parquet(stage4_parquet, columns=["regime", "positive_targets"])
	df = df.rename(columns={"positive_targets": "pos_targets"})
	# Normalize: convert every cell to a plain Python list (handles np.ndarray from parquet)
	df["pos_targets"] = df["pos_targets"].map(_to_list)

	# Flatten to find global top-K labels
	all_labels = [lbl for targets in df["pos_targets"] for lbl in targets]
	top_labels = [lbl for lbl, _ in Counter(all_labels).most_common(top_k)]

	regimes = sorted(df["regime"].unique())
	matrix  = np.zeros((len(regimes), len(top_labels)))

	for r_idx, regime in enumerate(regimes):
			sub = df[df["regime"] == regime]
			n   = max(len(sub), 1)
			for c_idx, label in enumerate(top_labels):
					count = sub["pos_targets"].apply(lambda t: label in t).sum()
					matrix[r_idx, c_idx] = count / n

	fig, ax = plt.subplots(figsize=(max(14, top_k * 0.45), max(4, len(regimes) * 1.4)))
	fig.suptitle(f"Stage 4 — Regime × Top-{top_k} Label Co-occurrence",
							 fontsize=14, fontweight="bold")

	im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max(), aspect="auto")
	ax.set_xticks(range(len(top_labels)))
	ax.set_xticklabels(top_labels, rotation=60, ha="right", fontsize=8)
	ax.set_yticks(range(len(regimes)))
	ax.set_yticklabels(regimes, fontsize=10)

	plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
							 label="Fraction of regime samples containing label")
	plt.tight_layout()
	_save(fig, out_dir, "V7_regime_label_heatmap")

# V8 — Per-Sample Label Count Distribution
def viz_label_count_distribution(
		stage4_parquet: str,
		out_dir: Optional[str] = None,
):
		"""
		Histogram of number of positive labels per sample, faceted by regime.
		Reveals over/under-labeling and regime-specific label density.
		"""
		df = pd.read_parquet(stage4_parquet, columns=["regime", "positive_targets", "hard_negatives"])
		df = df.rename(columns={"positive_targets": "pos_targets", "hard_negatives": "hn_targets"})
		# Normalize: convert every cell to a plain Python list (handles np.ndarray from parquet)
		df["pos_targets"] = df["pos_targets"].map(_to_list)
		df["hn_targets"]  = df["hn_targets"].map(_to_list)
		df["n_pos"] = df["pos_targets"].apply(len)
		df["n_hn"]  = df["hn_targets"].apply(len)

		regimes = sorted(df["regime"].unique())
		fig, axes = plt.subplots(
				1, len(regimes),
				figsize=(5 * len(regimes), 5),
				sharey=False,
		)
		if len(regimes) == 1:
				axes = [axes]

		fig.suptitle("Stage 4 — Per-Sample Label Count Distribution by Regime",
								 fontsize=14, fontweight="bold")

		for ax, regime in zip(axes, regimes):
				sub   = df[df["regime"] == regime]
				color = REGIME_COLORS.get(regime, "#607D8B")

				max_labels = max(sub["n_pos"].max(), sub["n_hn"].max(), 1)
				bins = np.arange(0, max_labels + 2) - 0.5

				ax.hist(
					sub["n_pos"], 
					bins=bins, 
					color=color, 
					alpha=0.85,
					label=f"pos_targets (μ={sub['n_pos'].mean():.2f})", 
					edgecolor="#ECF2F7"
				)
				ax.hist(
					sub["n_hn"], 
					bins=bins, 
					color="#200100", 
					alpha=0.4,
					label=f"hn_targets  (μ={sub['n_hn'].mean():.2f})", 
					edgecolor="#ECF2F7"
				)

				ax.set_title(f"{regime}\n(n={len(sub):,})", fontsize=10)
				ax.set_xlabel("Labels per sample")
				ax.set_ylabel("Sample count")
				ax.legend(fontsize=8, frameon=False)
				ax.spines[["top", "right"]].set_visible(False)

		plt.tight_layout()
		_save(fig, out_dir, "V8_label_count_distribution")

def set_seeds(seed: int = 42):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
				torch.cuda.manual_seed_all(seed)

def main():
	parser = argparse.ArgumentParser(
		description="Regime-Aware Consolidation Visualization Suite (V1–V8)",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--audit_jsonl", "-a",
		type=str,
		required=True,
		help="Path to Stage 2 Evidence Receipt JSONL (e.g. test_mlm_cot_modality_conflict_audit.jsonl)",
	)

	# ── Bridge funnel numbers (paste from Bridge printout) ────────────────────
	parser.add_argument(
		"--n_raw_concepts",
		type=int,
		default=0,
		metavar="N",
		help="Bridge Step 2: unique concepts post-normalisation.",
	)
	parser.add_argument(
		"--n_after_clustering",
		type=int,
		default=0,
		metavar="N",
		help="Bridge Step 5: optimal k (number of clusters).",
	)
	parser.add_argument(
		"--n_after_audit",
		type=int,
		default=0,
		metavar="N",
		help="Bridge Step 8c: df_clean size after vocab gate.",
	)
	parser.add_argument(
		"--n_target_vocab",
		type=int,
		default=0,
		metavar="N",
		help="Bridge Step 10: |V| target vocabulary size.",
	)
	parser.add_argument(
		"--n_emb_cache",
		type=int,
		default=0,
		metavar="N",
		help="Bridge Step 11: total emb_cache entries (raw + canonical).",
	)

	# Misc
	parser.add_argument(
		"--select", "-s",
		type=str,
		default=None,
		metavar="V1,V2,...",
		help="Comma-separated list of visualizations to run (e.g. --select V1,V3,V4). Runs all by default.",
	)
	parser.add_argument(
		"--sim_key",
		type=str,
		default="metrics.set_similarity",
		help="Field name for cross-modal cosine similarity in Stage 2 receipts (V2).",
	)
	parser.add_argument(
		"--top_k_labels",
		type=int,
		default=30,
		metavar="K",
		help="Top-K canonical labels shown in V7 regime×label heatmap.",
	)
	parser.add_argument(
		"--top_n_zipf",
		type=int,
		default=30,
		metavar="N",
		help="Number of concept labels annotated on the Zipf curve (V3).",
	)
	parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity")
	args = parser.parse_args()
	print(args)
	set_seeds(seed=42)

	DATASET_DIR = os.path.dirname(args.audit_jsonl)
	print(DATASET_DIR)
	AUDIT_FILE = os.path.basename(args.audit_jsonl)
	print(AUDIT_FILE)

	OUTPUT_DIR = os.path.join(DATASET_DIR, "outputs")
	print(OUTPUT_DIR)
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	VIZ_DIR = os.path.join(OUTPUT_DIR, "viz")
	print(VIZ_DIR)
	os.makedirs(VIZ_DIR, exist_ok=True)
	
	# Resolve which visualizations to run
	all_viz = {"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"}
	if args.select:
		requested = {v.strip().upper() for v in args.select.split(",")}
		invalid   = requested - all_viz
		if invalid:
			parser.error(f"Unknown visualization(s): {invalid}. Choose from {all_viz}.")
		run_set = requested
	else:
		run_set = all_viz
	
	# Guard: skip stage3/4 plots if paths not provided
	stage3_jsonl = os.path.join(OUTPUT_DIR, AUDIT_FILE.replace(".jsonl", "_auditable_cgd.jsonl"))
	if not os.path.exists(stage3_jsonl) and "V5" in run_set:
		print(f"[VIZ][SKIP] stage3_jsonl: {stage3_jsonl} not provided → skipping V5 (CGD heatmap).")
		run_set.discard("V5")

	stage4_parquet = os.path.join(OUTPUT_DIR, AUDIT_FILE.replace(".jsonl", "_auditable_matrix.parquet"))
	if not os.path.exists(stage4_parquet):
		for v in ("V6", "V7", "V8"):
			if v in run_set:
				print(f"[VIZ][SKIP] --stage4_parquet: {stage4_parquet} not provided → skipping {v}.")
				run_set.discard(v)

	print(f"\n[VIZ] Running: {sorted(run_set)}\n")

	if "V1" in run_set:
		print("[VIZ] V1 — Regime Distribution")
		viz_regime_distribution(args.audit_jsonl, out_dir=VIZ_DIR)
	
	if "V2" in run_set:
		print("[VIZ] V2 — Similarity Histograms")
		viz_similarity_histograms(args.audit_jsonl, out_dir=VIZ_DIR)
	
	if "V3" in run_set:
		freq_json_path = os.path.join(OUTPUT_DIR, AUDIT_FILE.replace(".jsonl", "_global_label_frequency.json"))
		print(f"freq_json_path: {freq_json_path}")
		print("[VIZ] V3 — Zipf Curve")
		viz_zipf_curve(freq_json_path, top_n_labels=args.top_n_zipf, out_dir=VIZ_DIR)
	
	if "V4" in run_set:
		print("[VIZ] V4 — Vocabulary Funnel")
		viz_vocabulary_funnel(
			n_raw_concepts      = args.n_raw_concepts,
			n_after_clustering  = args.n_after_clustering,
			n_after_audit       = args.n_after_audit,
			n_target_vocab      = args.n_target_vocab,
			n_emb_cache         = args.n_emb_cache,
			out_dir             = VIZ_DIR,
		)
	
	if "V5" in run_set and os.path.exists(stage3_jsonl):
		print(f"[VIZ] V5 — CGD Heatmap: {stage3_jsonl}")
		viz_cgd_heatmap(stage3_jsonl, out_dir=VIZ_DIR)
	
	if "V6" in run_set:
		print("[VIZ] V6 — Supervision Weights")
		viz_supervision_weights(stage4_parquet, out_dir=VIZ_DIR)
	
	if "V7" in run_set:
		print("[VIZ] V7 — Regime × Label Heatmap")
		viz_regime_label_heatmap(stage4_parquet, top_k=args.top_k_labels, out_dir=VIZ_DIR)
	
	if "V8" in run_set:
		print("[VIZ] V8 — Label Count Distribution")
		viz_label_count_distribution(stage4_parquet, out_dir=VIZ_DIR)

if __name__ == "__main__":
	main()