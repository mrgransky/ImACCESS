import argparse
import random
import torch
import json
import os
import numpy as np
import pandas as pd
import joblib
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# local:
# python viz.py --audit_jsonl /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_mlm_cot_modality_conflict_audit.jsonl --n_raw_concepts 172

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
		top_n_labels: int = 10,
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

		fig, ax = plt.subplots(figsize=(12, 8))
		fig.suptitle("Bridge — Raw Concept Vocabulary: Zipf Distribution", fontsize=14, fontweight="bold")

		ax.loglog(ranks, sorted_freqs, color="#1565C0", linewidth=1.8, alpha=0.85)
		ax.fill_between(ranks, sorted_freqs, alpha=0.08, color="#90EB19")

		# Annotate top-N
		for rank, (label, freq) in enumerate(sorted_items[:top_n_labels], start=1):
			ax.annotate(
				label, 
				xy=(rank, freq),
				xytext=(rank * 1.3, freq * 1.15),
				fontsize=5.5, 
				color="#333555",
				arrowprops=dict(arrowstyle="-", color="#161624", lw=0.6),
			)

		# Singleton line
		singleton_rank = next(
			(i + 1 for i, v in enumerate(sorted_freqs) if v == 1), 
			len(sorted_freqs)
		)
		ax.axvline(
			singleton_rank, 
			color="#E53935", 
			linestyle=":", 
			linewidth=1.2,
			label=f"Singletons start (rank {singleton_rank:,})"
		)

		ax.set_xlabel("Rank (log scale)")
		ax.set_ylabel("Frequency (log scale)")
		ax.legend(fontsize=9, frameon=False)
		ax.spines[["top", "right"]].set_visible(False)

		plt.tight_layout()
		_save(fig, out_dir, "V3_zipf_curve")

# V4 — Vocabulary Funnel (Waterfall bar)
def viz_vocabulary_funnel(
	audit_jsonl: str,
	n_after_clustering: int,
	n_after_audit: int,
	n_target_vocab: int,
	n_emb_cache: int,
	out_dir: Optional[str]=None,
):
	"""
	Horizontal waterfall showing how many concepts survive each Bridge gate.
	Pass the numbers directly from your Bridge printout.
	Example:
		viz_vocabulary_funnel(
			n_after_clustering  = 45,
			n_after_audit       = 38,
			n_target_vocab      = 35,
			n_emb_cache         = 207,
		)
	"""
	def infer_n_raw_concepts(
		jsonl_path: str,
		column: str = "mlm_cot_raw",
		include_fused: bool = False,
		normalize: bool = True,
	) -> int:
		"""
		Infer the number of unique raw concepts across a dataset.
		- Reads JSONL where each line has a dict containing `column`
			with keys: text_concepts, visual_concepts, fused_concepts.
		- Returns UNIQUE concept count (dataset-level vocabulary size).
		"""
		def _norm(s: str) -> str:
				s = s.strip()
				if normalize:
						s = s.lower()
				return s
		vocab = set()
		with open(jsonl_path, "r", encoding="utf-8") as f:
				for line in f:
						line = line.strip()
						if not line:
								continue
						try:
								rec = json.loads(line)
						except Exception:
								continue
						blob = rec.get(column) or {}
						if not isinstance(blob, dict):
								continue
						for key in ("text_concepts", "visual_concepts"):
								concepts = blob.get(key) or []
								if isinstance(concepts, list):
										for c in concepts:
												if isinstance(c, str) and c.strip():
														vocab.add(_norm(c))
						if include_fused:
								concepts = blob.get("fused_concepts") or []
								if isinstance(concepts, list):
										for c in concepts:
												if isinstance(c, str) and c.strip():
														vocab.add(_norm(c))
		return len(vocab)

	n_raw_concepts = infer_n_raw_concepts(audit_jsonl, column="mlm_cot_raw", include_fused=False)

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
	bars = ax.barh(
		stages[::-1], 
		values[::-1], 
		color=colors[::-1],
		edgecolor="#F7F1F1",
		height=0.55
	)
	
	for bar, val in zip(bars, values[::-1]):
		ax.text(
			bar.get_width() + max(values) * 0.01,
			bar.get_y() + bar.get_height() / 2,
			f"{val:,}", 
			va="center", 
			fontsize=10, 
			fontweight="bold",
		)
	
	# Reduction annotations
	for i in range(len(values) - 2):  # skip emb_cache (it grows)
		reduction = (values[i] - values[i + 1]) / max(values[i], 1) * 100
		if reduction > 0:
			y_pos = len(stages) - 1 - i - 0.5
			ax.text(
				max(values) * 0.5, y_pos,
				f"▼ {reduction:.0f}% reduction",
				va="center", 
				ha="center", 
				fontsize=8,
				color="#E53935", 
				style="italic",
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

# V9 — Semantic Asymmetry & NLI Entailment Analysis
def viz_semantic_asymmetry(
	audit_jsonl: str,
	tau_asym:   float = 0.25,   # Stage 2 default: |gap| >= tau_asym → SOFT_CONFLICT
	tau_orphan: float = 0.60,   # Stage 2 default: orphan_ratio >= tau_orphan → HARD_CONFLICT (NLI bypassed)
	out_dir: Optional[str] = None,
):
	"""
	Four-panel deep-dive into Stage 2 Asymmetric NLI signals.
	Panel A — Entailment Scatter (V→T vs T→V)
			Each dot is one sample where NLI was computed (nli_bypassed=False).
			Color = regime. Diagonal = perfect symmetry (gap=0).
			Shaded band = |gap| < tau_asym (AGREEMENT zone).
			Points above diagonal → VISUAL denser; below → TEXT denser.
	Panel B — Signed Asymmetry Gap Distribution
			Histogram of asymmetry_gap per regime (NLI-computed samples only).
			Vertical lines at ±tau_asym mark the SOFT_CONFLICT threshold.
			Reveals whether conflicts are systematically visual-heavy or text-heavy.
	Panel C — Denser Modality Breakdown per Regime
			Stacked horizontal bar: fraction of samples in each regime that are
			VISUAL-denser / TEXT-denser / EQUAL / NLI_BYPASSED.
			Exposes how the orphan_ratio gate interacts with the NLI gate.
	Panel D — NLI Coverage & Pair-Count Distribution
			Left y-axis: stacked bar of nli_computed vs nli_bypassed per regime.
			Right y-axis (twin): box/strip of nli_computed_on (pair count) per regime,
			showing how many concept pairs were evaluated when NLI ran.
	Reads: Stage 2 Evidence Receipt JSONL.
	Required fields per record:
			regime, metrics.{entail_V_to_T, entail_T_to_V, asymmetry_gap,
												denser_modality, nli_bypassed, nli_computed_on}
	"""
	records = _load_jsonl(audit_jsonl)
	rows = []
	for rec in records:
			m = rec.get("metrics") or {}
			rows.append({
					"regime":          rec.get("regime", "UNKNOWN"),
					"V_entails_T":     m.get("entail_V_to_T"),
					"T_entails_V":     m.get("entail_T_to_V"),
					"gap":             m.get("asymmetry_gap"),
					"denser":          m.get("denser_modality"),
					"nli_bypassed":    m.get("nli_bypassed", True),
					"computed_on":     m.get("nli_computed_on", 0),
					"orphan_ratio":    m.get("orphan_ratio"),
			})
	df = pd.DataFrame(rows)
	df_nli = df[df["nli_bypassed"] == False].copy()   # samples where NLI ran
	regimes_all = sorted(df["regime"].unique())
	if df_nli.empty:
			print("[VIZ][WARN] No NLI-computed samples found in receipts (all nli_bypassed=True). "
						"V9 requires at least some AGREEMENT or SOFT_CONFLICT samples.")
			return
	fig = plt.figure(figsize=(13, 8), constrained_layout=True)
	fig.suptitle(
		"Stage 2 — Semantic Asymmetry & Asymmetric NLI Entailment Analysis",
		fontsize=13, 
		fontweight="bold",
	)
	gs = fig.add_gridspec(2, 2)
	ax_scatter = fig.add_subplot(gs[0, 0])
	ax_hist    = fig.add_subplot(gs[0, 1])
	ax_stack   = fig.add_subplot(gs[1, 0])
	ax_cov     = fig.add_subplot(gs[1, 1])
	# ── Panel A: Entailment Scatter ───────────────────────────────────────────
	ax = ax_scatter
	ax.set_title("Entailment Scatter (V→T vs. T→V)", fontsize=10, fontweight="bold")
	# AGREEMENT zone band: |gap| < tau_asym  ↔  |V→T − T→V| < tau_asym
	x_band = np.linspace(0, 1, 200)
	ax.fill_between(
			x_band,
			np.clip(x_band - tau_asym, 0, 1),
			np.clip(x_band + tau_asym, 0, 1),
			alpha=0.10, 
			color="#4CAF50", 
			label=f"AGREEMENT band (|gap|<{tau_asym})",
	)
	# Symmetry diagonal
	ax.plot([0, 1], [0, 1], color="#555", linewidth=1.0, linestyle="--", alpha=0.6, label="gap = 0 (perfect symmetry)")
	for regime in regimes_all:
			sub = df_nli[df_nli["regime"] == regime].dropna(subset=["V_entails_T", "T_entails_V"])
			if sub.empty:
					continue
			ax.scatter(
					sub["T_entails_V"], sub["V_entails_T"],
					c=REGIME_COLORS.get(regime, "#607D8B"),
					label=f"{regime}  (n={len(sub):,})",
					alpha=0.55, s=22, edgecolors="none",
			)
	ax.set_xlabel("T→V entailment (text entails visual)", fontsize=9)
	ax.set_ylabel("V→T entailment (visual entails text)", fontsize=9)
	ax.set_xlim(-0.02, 1.02)
	ax.set_ylim(-0.02, 1.02)
	ax.legend(fontsize=7.5, frameon=False, loc="best", ncol=2)
	ax.spines[["top", "right"]].set_visible(False)
	# Quadrant annotations
	ax.text(0.82, 0.08, "TEXT\ndenser", fontsize=7, color="#1565C0", ha="center", style="italic")
	ax.text(0.08, 0.82, "VISUAL\ndenser", fontsize=7, color="#E53935", ha="center", style="italic")
	# ── Panel B: Signed Gap Histogram ─────────────────────────────────────────
	ax = ax_hist
	ax.set_title("Signed Asymmetry Gap Distribution", fontsize=11, fontweight="bold")
	bins = np.linspace(-1, 1, 41)
	for regime in regimes_all:
			sub = df_nli[df_nli["regime"] == regime].dropna(subset=["gap"])
			if sub.empty:
					continue
			ax.hist(
					sub["gap"], bins=bins,
					color=REGIME_COLORS.get(regime, "#607D8B"),
					alpha=0.55, label=f"{regime}  (n={len(sub):,})",
					density=True, edgecolor="none",
			)
			med = sub["gap"].median()
			ax.axvline(med, color=REGIME_COLORS.get(regime, "#607D8B"),
								 linestyle=":", linewidth=1.1, alpha=0.85)
	# Threshold lines
	ax.axvline( tau_asym, color="#E53935", linewidth=1.4, linestyle="--",
							label=f"+τ_asym = +{tau_asym}")
	ax.axvline(-tau_asym, color="#1565C0", linewidth=1.4, linestyle="--",
							label=f"−τ_asym = −{tau_asym}")
	ax.axvline(0, color="#555", linewidth=0.8, linestyle="-", alpha=0.5)
	# Zone labels
	ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
	ax.text( 0.60, ymax * 0.88, "VISUAL\ndenser", fontsize=7.5,
					 color="#E53935", ha="center", style="italic")
	ax.text(-0.60, ymax * 0.88, "TEXT\ndenser",   fontsize=7.5,
					 color="#1565C0", ha="center", style="italic")
	ax.set_xlabel("asymmetry_gap  =  V→T  −  T→V", fontsize=9)
	ax.set_ylabel("Density", fontsize=9)
	ax.legend(fontsize=7.5, frameon=False)
	ax.spines[["top", "right"]].set_visible(False)
	# ── Panel C: Denser Modality Stacked Bar ──────────────────────────────────
	ax = ax_stack
	ax.set_title("Denser Modality Breakdown per Regime", fontsize=11, fontweight="bold")
	denser_cats  = ["VISUAL", "TEXT", "EQUAL", "NLI_BYPASSED"]
	denser_colors = ["#E53935", "#1565C0", "#4CAF50", "#9E9E9E"]
	fracs = {cat: [] for cat in denser_cats}
	for regime in regimes_all:
			sub_all = df[df["regime"] == regime]
			n = max(len(sub_all), 1)
			bypassed_n = sub_all["nli_bypassed"].sum()
			fracs["NLI_BYPASSED"].append(bypassed_n / n)
			sub_nli = sub_all[sub_all["nli_bypassed"] == False]
			for cat in ("VISUAL", "TEXT", "EQUAL"):
					fracs[cat].append((sub_nli["denser"] == cat).sum() / n)
	lefts = np.zeros(len(regimes_all))
	for cat, color in zip(denser_cats, denser_colors):
			vals = np.array(fracs[cat])
			bars = ax.barh(regimes_all, vals, left=lefts, color=color,
										 label=cat, height=0.55, edgecolor="white")
			for bar, val in zip(bars, vals):
					if val > 0.04:
							ax.text(
									bar.get_x() + bar.get_width() / 2,
									bar.get_y() + bar.get_height() / 2,
									f"{val:.0%}", va="center", ha="center",
									fontsize=7.5, color="white", fontweight="bold",
							)
			lefts += vals
	ax.set_xlabel("Fraction of samples", fontsize=9)
	ax.set_xlim(0, 1)
	ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
	ax.legend(fontsize=7, frameon=False, loc="best")
	ax.spines[["top", "right"]].set_visible(False)
	# ── Panel D: NLI Coverage & Pair-Count Distribution ───────────────────────
	ax = ax_cov
	ax.set_title("NLI Coverage & Evaluated Pair Count", fontsize=11, fontweight="bold")
	computed_counts = []
	bypassed_counts = []
	for regime in regimes_all:
			sub = df[df["regime"] == regime]
			computed_counts.append((sub["nli_bypassed"] == False).sum())
			bypassed_counts.append(sub["nli_bypassed"].sum())
	y = np.arange(len(regimes_all))
	bar_h = 0.45
	b1 = ax.barh(
		y + bar_h / 2, 
		computed_counts, 
		height=bar_h,
		color="#4CAF50", 
		alpha=0.85, 
		label="NLI computed", 
		edgecolor="white"
	)
	b2 = ax.barh(
		y - bar_h / 2, 
		bypassed_counts, 
		height=bar_h,
		color="#9E9E9E", 
		alpha=0.75, 
		label="NLI bypassed", 
		edgecolor="white"
	)
	for bar, val in zip(list(b1) + list(b2), computed_counts + bypassed_counts):
		if val > 0:
			ax.text(
				bar.get_width() + max(computed_counts + bypassed_counts) * 0.01,
				bar.get_y() + bar.get_height() / 2,
				f"{val:,}", 
				va="center", 
				fontsize=8
			)
	ax.set_yticks(y)
	ax.set_yticklabels(regimes_all, fontsize=9)
	ax.set_xlabel("Sample count", fontsize=9)
	ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
	ax.legend(fontsize=8, frameon=False)
	ax.spines[["top", "right"]].set_visible(False)
	# Overlay: mean nli_computed_on (pair count) as text annotation per regime
	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks([])
	for i, regime in enumerate(regimes_all):
		sub_nli = df[(df["regime"] == regime) & (df["nli_bypassed"] == False)]
		if not sub_nli.empty:
			mean_pairs = sub_nli["computed_on"].mean()
			ax.text(
				max(computed_counts + bypassed_counts) * 0.5, i + bar_h / 2,
				f"avg {mean_pairs:.1f} pairs/sample",
				va="center", 
				ha="center", 
				fontsize=7,
				color="#0B0B0C", 
				fontweight="bold",
			)
	_save(fig, out_dir, "V9_semantic_asymmetry")

# V10 — GMM Feature-Space Map (2D/3D) + Learned Ellipses
def viz_gmm_feature_space(
	audit_jsonl: str,
	gmm_pkl:     str,
	out_dir:     Optional[str] = None,
	grid_resolution: int = 200,
	ellipse_sigmas: tuple = (1.0, 2.0),
):
	"""
	Visualises the GMM's learned geometry in the feature space used at Bridge time.
	Layout depends on feature_dim stored in the GMM payload:
			feature_dim == 2  →  single scatter panel (set_sim vs orphan_ratio)
													 + GMM decision-region background
													 + covariance ellipses (1σ, 2σ) per component
			feature_dim == 3  →  3 scatter panels (all pairwise projections):
														 (set_sim vs orphan_ratio)
														 (set_sim vs abs_asym_gap)
														 (orphan_ratio vs abs_asym_gap)
													 + centroids projected onto each 2D plane
													 (decision-region background only for 2D case)
	Each point is colored TWICE via a twin-axis trick:
			Left  panel(s) → color by heuristic_regime (Stage 2)
			Right panel(s) → color by GMM-assigned regime (argmax of gmm.probabilities)
	Reads:
			audit_jsonl  — Stage 2 Evidence Receipt JSONL
										 (fields: regime, heuristic_regime, metrics.{set_similarity,
											orphan_ratio, asymmetry_gap})
			gmm_pkl      — Bridge GMM payload (joblib pickle)
										 (keys: gmm_model, scaler, class_mapping, feature_dim,
											feature_names, centroids_unscaled, bic_scores, optimal_k_bic,
											orphan_gap)
	"""

	# ── Load GMM payload ──────────────────────────────────────────────────────
	try:
			payload = joblib.load(gmm_pkl)
	except Exception as e:
			print(f"[VIZ][V10][ERROR] Failed to load GMM pkl: {e}")
			return
	gmm           = payload["gmm_model"]
	scaler        = payload["scaler"]
	class_mapping = payload["class_mapping"]          # {cluster_idx: regime_name}
	feature_dim   = payload.get("feature_dim", 3)
	feature_names = payload.get(
		"feature_names",
		["set_similarity", "orphan_ratio", "abs_asym_gap"][:feature_dim],
	)

	centroids_unscaled = payload.get("centroids_unscaled", {})  # {regime: [f0, f1, ...]}
	bic_scores         = payload.get("bic_scores", {})
	optimal_k_bic      = payload.get("optimal_k_bic", "N/A")
	orphan_gap         = payload.get("orphan_gap", float("nan"))
	n_samples_fit      = payload.get("n_samples_fit", "N/A")

	# Load Stage 2 receipts → feature matrix
	records = _load_jsonl(audit_jsonl)
	rows = []
	for rec in records:
		m              = rec.get("metrics") or {}
		heuristic_reg  = rec.get("heuristic_regime") or rec.get("regime", "UNKNOWN")
		set_sim        = m.get("set_similarity")
		orphan_ratio   = m.get("orphan_ratio")
		asym_gap       = m.get("asymmetry_gap")
		# Only include samples that were valid GMM inputs at Bridge time
		if set_sim is None or orphan_ratio is None:
				continue
		if heuristic_reg == "MISSING_MODALITY":
				continue
		if feature_dim == 3 and asym_gap is None:
				continue  # 3D GMM: skip NLI-bypassed samples (same logic as Bridge)
		rows.append(
			{
				"heuristic_regime": heuristic_reg,
				"set_similarity":   float(set_sim),
				"orphan_ratio":     float(orphan_ratio),
				"abs_asym_gap":     abs(float(asym_gap)) if asym_gap is not None else None,
			}
		)
	
	if not rows:
		print("[VIZ][V10][WARN] No valid feature rows found in audit JSONL.")
		return

	df = pd.DataFrame(rows)
	# Build raw feature matrix (same shape as Bridge training)
	if feature_dim == 2:
		feat_raw = df[["set_similarity", "orphan_ratio"]].values
	else:
		feat_raw = df[["set_similarity", "orphan_ratio", "abs_asym_gap"]].values
	
	# Scale and run GMM predict_proba to get per-sample GMM regime
	feat_scaled = scaler.transform(feat_raw)
	probs       = gmm.predict_proba(feat_scaled)          # [N, 3]
	best_idx    = np.argmax(probs, axis=1)                # [N]
	df["gmm_regime"]   = [class_mapping[i] for i in best_idx]
	df["gmm_conf"]     = probs[np.arange(len(probs)), best_idx]
	df["gmm_override"] = df["gmm_regime"] != df["heuristic_regime"]

	# ── draw covariance ellipse for one GMM component ─────────────────
	def _draw_ellipse(ax, mean_2d, cov_2d, sigma, color, alpha=0.18, lw=1.5):
			"""Draw a sigma-level confidence ellipse from a 2×2 covariance matrix."""
			vals, vecs = np.linalg.eigh(cov_2d)
			order      = vals.argsort()[::-1]
			vals, vecs = vals[order], vecs[:, order]
			angle      = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
			width, height = 2 * sigma * np.sqrt(np.abs(vals))
			ell = Ellipse(
					xy=mean_2d, width=width, height=height, angle=angle,
					edgecolor=color, facecolor=color, alpha=alpha,
					linewidth=lw, linestyle="--",
			)
			ax.add_patch(ell)

	# scatter + ellipses for one 2D projection
	def _scatter_panel(ax, df_plot, x_col, y_col, color_col, title_suffix, show_decision_bg=False):
			"""
			Scatter plot colored by `color_col` (regime column name).
			Overlays GMM centroids and covariance ellipses projected onto (x_col, y_col).
			Optionally draws a decision-region background (2D only, slow for 3D).
			"""
			# Axis indices in the feature vector
			col_to_idx = {n: i for i, n in enumerate(feature_names)}
			xi = col_to_idx.get(x_col, 0)
			yi = col_to_idx.get(y_col, 1)
			# Decision-region background (only for 2D feature space)
			if show_decision_bg and feature_dim == 2:
					x_min, x_max = df_plot[x_col].min() - 0.05, df_plot[x_col].max() + 0.05
					y_min, y_max = df_plot[y_col].min() - 0.05, df_plot[y_col].max() + 0.05
					xx, yy = np.meshgrid(
							np.linspace(x_min, x_max, grid_resolution),
							np.linspace(y_min, y_max, grid_resolution),
					)
					grid_raw    = np.c_[xx.ravel(), yy.ravel()]
					grid_scaled = scaler.transform(grid_raw)
					grid_probs  = gmm.predict_proba(grid_scaled)
					grid_labels = np.argmax(grid_probs, axis=1)
					grid_conf   = grid_probs[np.arange(len(grid_probs)), grid_labels]
					# Map cluster idx → regime → color
					grid_colors = np.array([
							REGIME_COLORS.get(class_mapping[l], "#607D8B")
							for l in grid_labels
					])
					# Convert hex colors to RGBA and modulate alpha by confidence
					from matplotlib.colors import to_rgba
					rgba = np.array([to_rgba(c) for c in grid_colors])
					rgba[:, 3] = np.clip(grid_conf * 0.30, 0.04, 0.30)  # soft alpha
					ax.imshow(
							rgba.reshape(grid_resolution, grid_resolution, 4),
							origin="lower",
							extent=[x_min, x_max, y_min, y_max],
							aspect="auto",
							interpolation="bilinear",
					)
			# Scatter points
			for regime in sorted(df_plot[color_col].unique()):
					sub   = df_plot[df_plot[color_col] == regime]
					color = REGIME_COLORS.get(regime, "#607D8B")
					ax.scatter(
							sub[x_col], sub[y_col],
							c=color, label=f"{regime}  (n={len(sub):,})",
							alpha=0.45, s=18, edgecolors="none", zorder=3,
					)
			# GMM centroids + ellipses (projected onto this 2D plane)
			means_scaled_all = gmm.means_          # [K, feature_dim]
			covs_scaled_all  = gmm.covariances_    # [K, feature_dim, feature_dim]
			for k in range(gmm.n_components):
					regime_name = class_mapping[k]
					color       = REGIME_COLORS.get(regime_name, "#607D8B")
					# Project centroid to unscaled space for plotting
					mean_unscaled = scaler.inverse_transform(means_scaled_all[k:k+1])[0]
					cx, cy = float(mean_unscaled[xi]), float(mean_unscaled[yi])
					# Centroid marker
					ax.scatter(
							cx, cy,
							marker="*", s=260, c=color,
							edgecolors="white", linewidths=1.2,
							zorder=6, label=f"{regime_name}",
					)
					ax.annotate(
							regime_name.replace("_", "\n"),
							xy=(cx, cy), xytext=(cx + 0.02, cy + 0.02),
							fontsize=7, color=color, fontweight="bold",
							arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
					)
					# Project 2×2 covariance sub-matrix onto (xi, yi) axes
					cov_full = covs_scaled_all[k]                    # [feature_dim, feature_dim]
					cov_2d   = cov_full[np.ix_([xi, yi], [xi, yi])] # [2, 2]
					# Covariance ellipses are in scaled space → need to unscale axes
					# Approximate: scale the covariance by the scaler's std for each axis
					std_xi = scaler.scale_[xi]
					std_yi = scaler.scale_[yi]
					scale_mat = np.diag([std_xi, std_yi])
					cov_2d_unscaled = scale_mat @ cov_2d @ scale_mat.T
					for sigma, alpha in zip(ellipse_sigmas, [0.20, 0.10]):
						_draw_ellipse(ax, [cx, cy], cov_2d_unscaled, sigma, color, alpha=alpha)
			ax.set_xlabel(x_col.replace("_", " "), fontsize=9)
			ax.set_ylabel(y_col.replace("_", " "), fontsize=9)
			ax.set_title(title_suffix, fontsize=9, fontweight="bold")
			ax.spines[["top", "right"]].set_visible(False)
			# Deduplicate legend (centroids + scatter)
			handles, labels = ax.get_legend_handles_labels()
			seen_labels = {}
			for h, l in zip(handles, labels):
				if l not in seen_labels:
					seen_labels[l] = h
			ax.legend(
					seen_labels.values(), 
					seen_labels.keys(),
					ncol=2,
					fontsize=7,
					frameon=False, 
					loc="best",
			)
	# ── Build figure layout ───────────────────────────────────────────────────
	# 2D: 1 row × 2 cols (heuristic | GMM)
	# 3D: 3 rows × 2 cols (3 projections × heuristic | GMM)
	n_proj = 1 if feature_dim == 2 else 3
	projections = (
			[("set_similarity", "orphan_ratio")]
			if feature_dim == 2
			else [
					("set_similarity",  "orphan_ratio"),
					("set_similarity",  "abs_asym_gap"),
					("orphan_ratio",    "abs_asym_gap"),
			]
	)
	fig = plt.figure(figsize=(16, 7 * n_proj), constrained_layout=True)
	fig.suptitle(
			f"GMM Feature-Space "
			f"(feature_dim={feature_dim}, K=3, "
			f"n_fit={n_samples_fit:,}, "
			f"BIC-optimal K={optimal_k_bic}, "
			f"orphan_gap={orphan_gap:.3f})",
			fontsize=10, 
			fontweight="bold",
	)

	gs = gridspec.GridSpec(n_proj, 2, figure=fig)
	for row_idx, (x_col, y_col) in enumerate(projections):
		ax_heur = fig.add_subplot(gs[row_idx, 0])
		ax_gmm  = fig.add_subplot(gs[row_idx, 1])
		_scatter_panel(
			ax_heur, 
			df, 
			x_col, 
			y_col,
			color_col="heuristic_regime",
			title_suffix=f"Heuristic regime [{x_col} vs {y_col}]",
			show_decision_bg=(feature_dim == 2),
		)
		_scatter_panel(
			ax_gmm, 
			df, 
			x_col, 
			y_col,
			color_col="gmm_regime",
			title_suffix=f"GMM-assigned regime [{x_col} vs {y_col}]",
			show_decision_bg=(feature_dim == 2),
		)

	# ── Annotation strip: BIC curve + override stats ──────────────────────────
	# Add a narrow text box below the figure with key GMM diagnostics
	n_override = int(df["gmm_override"].sum())
	n_total    = len(df)
	override_pct = n_override / max(n_total, 1) * 100
	mean_conf    = df["gmm_conf"].mean()
	diag_lines = [
			f"Samples plotted: {n_total:,}  |  "
			f"GMM overrides (heuristic ≠ GMM): {n_override:,} ({override_pct:.1f}%)  |  "
			f"Mean GMM confidence: {mean_conf:.3f}  |  "
			f"Ellipses: {ellipse_sigmas[0]}σ / {ellipse_sigmas[1]}σ",
	]
	if bic_scores:
			bic_str = "  BIC: " + "  ".join(
					f"K={k}:{v:.0f}{'★' if k == optimal_k_bic else ''}"
					for k, v in sorted(bic_scores.items())
			)
			diag_lines.append(bic_str)
	# fig.text(
	# 		0.5, -0.01,
	# 		"\n".join(diag_lines),
	# 		ha="center", 
	# 		va="top", 
	# 		fontsize=8,
	# 		color="#5A5A7A", 
	# 		style="italic",
	# 		transform=fig.transFigure,
	# )
	_save(fig, out_dir, "V10_gmm_feature_space")
	print(
			f"[VIZ][V10] Done. "
			f"feature_dim={feature_dim} | "
			f"n={n_total:,} | "
			f"overrides={n_override:,} ({override_pct:.1f}%) | "
			f"mean_conf={mean_conf:.3f}"
	)

# V11 — BIC/AIC Model-Selection Curve (Bridge Step 1B Audit)
def viz_bic_aic(
		gmm_pkl:  str,
		out_dir:  Optional[str] = None,
):
		"""
		Renders the BIC and AIC model-selection curves from the GMM payload
		produced by Bridge Step 1B (global_aggregation.py).

		Layout  (single figure, 2 panels side-by-side):
				Left  — BIC(K) and AIC(K) line curves for K∈{2,…,6}
									• Vertical dashed line at K=3 (design choice)
									• Star markers at BIC-optimal and AIC-optimal K
									• Δ-annotation: how far BIC(3) is from BIC(optimal)
				Right — Δ-BIC and Δ-AIC bar chart (relative to K=3)
									• Bars above zero → K is worse than K=3
									• Bars below zero → K is better than K=3
									• Colour-coded: green = better, red = worse

		Footer strip:
				GMM diagnostics: feature_dim, n_samples_fit, converged K,
				orphan_gap, optimal_k_bic, optimal_k_aic, design K=3 verdict.

		Reads:
				gmm_pkl — Bridge GMM payload (joblib pickle)
									Required keys: bic_scores, aic_scores,
																 optimal_k_bic, optimal_k_aic,
																 feature_dim, n_samples_fit, orphan_gap
		"""
		import joblib

		# ── Load payload ──────────────────────────────────────────────────────────
		try:
				payload = joblib.load(gmm_pkl)
		except Exception as e:
				print(f"[VIZ][V11][ERROR] Failed to load GMM pkl: {e}")
				return

		bic_scores    = payload.get("bic_scores",    {})
		aic_scores    = payload.get("aic_scores",    {})
		optimal_k_bic = payload.get("optimal_k_bic", None)
		optimal_k_aic = payload.get("optimal_k_aic", None)
		feature_dim   = payload.get("feature_dim",   "?")
		n_samples_fit = payload.get("n_samples_fit", "?")
		orphan_gap    = payload.get("orphan_gap",    float("nan"))
		centroids     = payload.get("centroids_unscaled", {})
		class_mapping = payload.get("class_mapping", {})

		if not bic_scores or not aic_scores:
				print("[VIZ][V11][WARN] bic_scores / aic_scores missing from payload. Nothing to plot.")
				return

		ks      = sorted(bic_scores.keys())
		bic_vals = [bic_scores[k] for k in ks]
		aic_vals = [aic_scores[k] for k in ks]

		DESIGN_K = 3

		# ── Δ relative to design K=3 ──────────────────────────────────────────────
		bic_at_3 = bic_scores.get(DESIGN_K, None)
		aic_at_3 = aic_scores.get(DESIGN_K, None)
		delta_bic = (
				[bic_scores[k] - bic_at_3 for k in ks]
				if bic_at_3 is not None else None
		)
		delta_aic = (
				[aic_scores[k] - aic_at_3 for k in ks]
				if aic_at_3 is not None else None
		)

		# ── Verdict string ────────────────────────────────────────────────────────
		if optimal_k_bic == DESIGN_K and optimal_k_aic == DESIGN_K:
				verdict = "✓  BIC and AIC both confirm K=3 — data-driven support for three conflict regimes."
				verdict_color = "#2e7d32"
		elif optimal_k_bic == DESIGN_K:
				verdict = f"△  BIC confirms K=3; AIC prefers K={optimal_k_aic}. K=3 used by design."
				verdict_color = "#e65100"
		elif optimal_k_aic == DESIGN_K:
				verdict = f"△  AIC confirms K=3; BIC prefers K={optimal_k_bic}. K=3 used by design."
				verdict_color = "#e65100"
		else:
				verdict = (
						f"✗  Neither BIC (K={optimal_k_bic}) nor AIC (K={optimal_k_aic}) "
						f"selects K=3. K=3 used by design — justify via domain alignment in paper."
				)
				verdict_color = "#c62828"

		# ── Figure ────────────────────────────────────────────────────────────────
		fig, (ax_curve, ax_delta) = plt.subplots(
				1, 2,
				figsize=(13, 5),
				constrained_layout=True,
		)
		fig.suptitle(
				f"V11 — BIC / AIC Model-Selection Curve  "
				f"(Bridge Step 1B  |  feature_dim={feature_dim}  |  "
				f"n_fit={n_samples_fit:,}  |  orphan_gap={orphan_gap:.4f})",
				fontsize=12, fontweight="bold",
		)

		# ── Left panel: raw BIC / AIC curves ─────────────────────────────────────
		ax_curve.plot(
				ks, bic_vals,
				color="#1565C0", marker="o", linewidth=2.2,
				markersize=8, label="BIC", zorder=4,
		)
		ax_curve.plot(
				ks, aic_vals,
				color="#AD1457", marker="s", linewidth=2.2,
				markersize=8, linestyle="--", label="AIC", zorder=4,
		)

		# Design K=3 vertical line
		ax_curve.axvline(
				DESIGN_K, color="#37474F", linewidth=1.4,
				linestyle=":", zorder=2, label=f"Design K={DESIGN_K}",
		)

		# BIC-optimal star
		if optimal_k_bic is not None and optimal_k_bic in bic_scores:
				ax_curve.scatter(
						[optimal_k_bic], [bic_scores[optimal_k_bic]],
						marker="*", s=320, color="#1565C0",
						edgecolors="white", linewidths=1.0,
						zorder=6, label=f"BIC-optimal K={optimal_k_bic}",
				)

		# AIC-optimal star
		if optimal_k_aic is not None and optimal_k_aic in aic_scores:
				ax_curve.scatter(
						[optimal_k_aic], [aic_scores[optimal_k_aic]],
						marker="*", s=320, color="#AD1457",
						edgecolors="white", linewidths=1.0,
						zorder=6, label=f"AIC-optimal K={optimal_k_aic}",
				)

		# Δ annotation: BIC(3) − BIC(optimal)
		if bic_at_3 is not None and optimal_k_bic is not None and optimal_k_bic != DESIGN_K:
				delta_val = bic_at_3 - bic_scores[optimal_k_bic]
				mid_y = (bic_at_3 + bic_scores[optimal_k_bic]) / 2
				ax_curve.annotate(
						f"ΔBIC = +{delta_val:.1f}\n(K=3 vs K={optimal_k_bic})",
						xy=(optimal_k_bic, bic_scores[optimal_k_bic]),
						xytext=(optimal_k_bic + 0.35, mid_y),
						fontsize=8, color="#1565C0",
						arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.0),
				)

		ax_curve.set_xlabel("Number of GMM components  K", fontsize=10)
		ax_curve.set_ylabel("Score (lower is better)", fontsize=10)
		ax_curve.set_title("BIC and AIC vs K", fontsize=10, fontweight="bold")
		ax_curve.set_xticks(ks)
		ax_curve.legend(fontsize=8, frameon=False)
		ax_curve.spines[["top", "right"]].set_visible(False)

		# ── Right panel: Δ-BIC / Δ-AIC bar chart relative to K=3 ─────────────────
		bar_width = 0.35
		x_pos     = np.arange(len(ks))

		if delta_bic is not None:
				bic_colors = ["#2e7d32" if d < 0 else "#c62828" for d in delta_bic]
				bars_bic = ax_delta.bar(
						x_pos - bar_width / 2, delta_bic,
						width=bar_width, color=bic_colors,
						alpha=0.80, label="ΔBIC  (vs K=3)", zorder=3,
				)
				# Value labels on bars
				for bar, val in zip(bars_bic, delta_bic):
						y_off = 5 if val >= 0 else -18
						ax_delta.text(
								bar.get_x() + bar.get_width() / 2,
								bar.get_height() + y_off,
								f"{val:+.0f}",
								ha="center", va="bottom", fontsize=7, color="#333",
						)

		if delta_aic is not None:
				aic_colors = ["#2e7d32" if d < 0 else "#c62828" for d in delta_aic]
				bars_aic = ax_delta.bar(
						x_pos + bar_width / 2, delta_aic,
						width=bar_width, color=aic_colors,
						alpha=0.55, label="ΔAIC  (vs K=3)",
						hatch="//", edgecolor="white", zorder=3,
				)
				for bar, val in zip(bars_aic, delta_aic):
						y_off = 5 if val >= 0 else -18
						ax_delta.text(
								bar.get_x() + bar.get_width() / 2,
								bar.get_height() + y_off,
								f"{val:+.0f}",
								ha="center", va="bottom", fontsize=7, color="#555",
						)

		ax_delta.axhline(0, color="#37474F", linewidth=1.2, linestyle="--", zorder=2)

		# Shade the K=3 column
		k3_idx = ks.index(DESIGN_K) if DESIGN_K in ks else None
		if k3_idx is not None:
				ax_delta.axvspan(
						k3_idx - 0.5, k3_idx + 0.5,
						alpha=0.08, color="#37474F", zorder=1,
						label=f"Design K={DESIGN_K}",
				)

		ax_delta.set_xlabel("Number of GMM components  K", fontsize=10)
		ax_delta.set_ylabel("Δ Score relative to K=3  (↓ better than K=3)", fontsize=10)
		ax_delta.set_title("Relative BIC / AIC  (Δ vs design K=3)", fontsize=10, fontweight="bold")
		ax_delta.set_xticks(x_pos)
		ax_delta.set_xticklabels([str(k) for k in ks])
		ax_delta.legend(fontsize=8, frameon=False)
		ax_delta.spines[["top", "right"]].set_visible(False)

		# ── Footer: GMM diagnostics + centroid table ──────────────────────────────
		centroid_lines = []
		if centroids and class_mapping:
				feature_names = payload.get(
						"feature_names",
						["set_similarity", "orphan_ratio", "abs_asym_gap"][:int(feature_dim)]
						if isinstance(feature_dim, int) else [],
				)
				header = "  ".join(f"{n:>18}" for n in feature_names)
				centroid_lines.append(f"{'Regime':<20}  {header}")
				for cluster_idx, regime_name in sorted(class_mapping.items()):
						if regime_name in centroids:
								vals = "  ".join(f"{v:>18.4f}" for v in centroids[regime_name])
								centroid_lines.append(f"{regime_name:<20}  {vals}")

		footer_parts = [
				f"feature_dim={feature_dim}  |  "
				f"n_fit={n_samples_fit:,}  |  "
				f"BIC-optimal K={optimal_k_bic}  |  "
				f"AIC-optimal K={optimal_k_aic}  |  "
				f"orphan_gap={orphan_gap:.4f}",
				verdict,
		]
		if centroid_lines:
				footer_parts += ["Centroids (unscaled):"] + centroid_lines

		fig.text(
				0.5, -0.02,
				"\n".join(footer_parts),
				ha="center", va="top", fontsize=7.5,
				color=verdict_color if len(footer_parts) <= 2 else "#444",
				style="italic",
				transform=fig.transFigure,
		)
		# Verdict in a coloured box just below the title
		fig.text(
				0.5, 0.01,
				verdict,
				ha="center", 
				va="bottom", 
				fontsize=9,
				color=verdict_color, 
				fontweight="bold",
				transform=fig.transFigure,
		)

		_save(fig, out_dir, "V11_bic_aic_model_selection")
		print(
				f"[VIZ][V11] Done. "
				f"BIC-optimal K={optimal_k_bic} | "
				f"AIC-optimal K={optimal_k_aic} | "
				f"Design K=3 | "
				f"Verdict: {verdict}"
		)

# V12 — Heuristic vs GMM Regime Confusion + Override Matrix
def viz_heuristic_gmm_confusion(
		stage4_parquet: str,
		out_dir:        Optional[str] = None,
		conf_threshold: float = 0.60,
):
		"""
		Four-panel audit of where and how the GMM overrides Stage 2 heuristic labels.

		Panel layout (2 × 2):
				TL — Confusion matrix: heuristic_regime (rows) vs GMM regime (cols)
							 • Diagonal = agreement; off-diagonal = overrides
							 • Cell text: count + row-% (fraction of that heuristic class overridden)
							 • MISSING_MODALITY row always shows zeros (never GMM-routed)
				TR — Override rate bar chart per heuristic regime
							 • Bar height = % of samples in that heuristic class that were overridden
							 • Colour = REGIME_COLORS of the heuristic regime
							 • Annotated with absolute count and %
				BL — GMM confidence distribution (violin + strip) split by:
							 • override=True  (GMM disagreed with heuristic)
							 • override=False (GMM agreed)
							 • Horizontal dashed line at conf_threshold (default 0.60)
							 • Per-group median annotated
				BR — Sankey-style stacked bar: for each heuristic regime,
							 how the GMM redistributes samples across final regimes
							 • Each bar = 100% of that heuristic class
							 • Segments coloured by GMM-assigned regime
							 • Annotated with % when segment ≥ 5%

		Footer strip:
				Total samples | GMM-routed count | overall override rate |
				low-confidence count | feature_dim from gmm column

		Reads:
				stage4_parquet — auditable supervision matrix parquet
												 Required columns: heuristic_regime, regime, gmm (JSON str)
		"""
		# ── Load parquet ──────────────────────────────────────────────────────────
		try:
				df_raw = pd.read_parquet(stage4_parquet, engine="pyarrow")
		except Exception as e:
				print(f"[VIZ][V12][ERROR] Failed to read parquet: {e}")
				return

		required = {"heuristic_regime", "regime", "gmm"}
		missing  = required - set(df_raw.columns)
		if missing:
				print(f"[VIZ][V12][ERROR] Missing columns: {missing}. Available: {list(df_raw.columns)}")
				return

		# ── Parse gmm JSON column ─────────────────────────────────────────────────
		def _parse_gmm(x):
				if isinstance(x, dict):
						return x
				if isinstance(x, str):
						try:
								return json.loads(x)
						except Exception:
								return None
				return None  # None / NaN → MISSING_MODALITY or GMM-skipped

		df_raw["_gmm"] = df_raw["gmm"].apply(_parse_gmm)

		# Derived columns
		df_raw["_override"]  = df_raw["_gmm"].apply(
				lambda x: bool(x.get("regime_override", False)) if isinstance(x, dict) else False
		)
		df_raw["_conf"]      = df_raw["_gmm"].apply(
				lambda x: float(x["confidence"]) if isinstance(x, dict) and x.get("confidence") is not None else None
		)
		df_raw["_gmm_routed"] = df_raw["_gmm"].apply(lambda x: isinstance(x, dict))
		df_raw["_feat_dim"]   = df_raw["_gmm"].apply(
				lambda x: x.get("feature_dim") if isinstance(x, dict) else None
		)

		# Canonical regime order (consistent across all panels)
		REGIME_ORDER = ["AGREEMENT", "SOFT_CONFLICT", "HARD_CONFLICT", "MISSING_MODALITY"]
		all_regimes  = [r for r in REGIME_ORDER if r in df_raw["heuristic_regime"].unique()
										or r in df_raw["regime"].unique()]

		n_total      = len(df_raw)
		n_gmm_routed = int(df_raw["_gmm_routed"].sum())
		n_overrides  = int(df_raw["_override"].sum())
		override_pct = n_overrides / max(n_gmm_routed, 1) * 100
		n_low_conf   = int((df_raw["_conf"] < conf_threshold).sum()) if df_raw["_conf"].notna().any() else 0
		feat_dim_val = df_raw["_feat_dim"].dropna().mode()
		feat_dim_str = str(int(feat_dim_val.iloc[0])) if len(feat_dim_val) else "?"

		# ── Build confusion matrix ────────────────────────────────────────────────
		# Rows = heuristic_regime, Cols = GMM regime (final `regime` column)
		conf_mat = pd.crosstab(
				df_raw["heuristic_regime"],
				df_raw["regime"],
				rownames=["Heuristic"],
				colnames=["GMM"],
		).reindex(index=all_regimes, columns=all_regimes, fill_value=0)

		# Row-normalised (%) for annotation
		row_sums  = conf_mat.sum(axis=1).replace(0, 1)
		conf_pct  = conf_mat.div(row_sums, axis=0) * 100

		# ── Figure ────────────────────────────────────────────────────────────────
		fig = plt.figure(figsize=(16, 13), constrained_layout=True)
		fig.suptitle(
				f"V12 — Heuristic vs GMM Regime Confusion & Override Matrix  "
				f"(n={n_total:,}  |  GMM-routed={n_gmm_routed:,}  |  "
				f"feature_dim={feat_dim_str})",
				fontsize=12, fontweight="bold",
		)

		gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
		ax_cm    = fig.add_subplot(gs[0, 0])   # TL: confusion matrix
		ax_bar   = fig.add_subplot(gs[0, 1])   # TR: override rate bars
		ax_viol  = fig.add_subplot(gs[1, 0])   # BL: confidence violin
		ax_sank  = fig.add_subplot(gs[1, 1])   # BR: Sankey-style stacked bar

		# ── TL: Confusion matrix heatmap ─────────────────────────────────────────
		n_reg = len(all_regimes)
		cmap  = plt.cm.Blues

		im = ax_cm.imshow(conf_pct.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
		plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Row %")

		for i in range(n_reg):
				for j in range(n_reg):
						count = int(conf_mat.values[i, j])
						pct   = float(conf_pct.values[i, j])
						text_color = "white" if pct > 55 else "black"
						is_diag    = (i == j)
						weight     = "bold" if is_diag else "normal"
						ax_cm.text(
								j, i,
								f"{count:,}\n({pct:.1f}%)",
								ha="center", va="center",
								fontsize=8, color=text_color, fontweight=weight,
						)

		# Highlight diagonal (agreement) with a green border
		for i in range(n_reg):
				ax_cm.add_patch(plt.Rectangle(
						(i - 0.5, i - 0.5), 1, 1,
						fill=False, edgecolor="#2e7d32", linewidth=2.0, zorder=5,
				))

		short_labels = [r.replace("_", "\n") for r in all_regimes]
		ax_cm.set_xticks(range(n_reg))
		ax_cm.set_yticks(range(n_reg))
		ax_cm.set_xticklabels(short_labels, fontsize=8)
		ax_cm.set_yticklabels(short_labels, fontsize=8)
		ax_cm.set_xlabel("GMM-assigned regime  (final)", fontsize=9)
		ax_cm.set_ylabel("Heuristic regime  (Stage 2)", fontsize=9)
		ax_cm.set_title(
				"Confusion Matrix\n(row % | diagonal = agreement)",
				fontsize=9, fontweight="bold",
		)

		# ── TR: Override rate per heuristic regime ────────────────────────────────
		override_rates = {}
		override_counts = {}
		for reg in all_regimes:
				mask_reg = df_raw["heuristic_regime"] == reg
				n_reg_total    = int(mask_reg.sum())
				n_reg_override = int((mask_reg & df_raw["_override"]).sum())
				override_rates[reg]  = n_reg_override / max(n_reg_total, 1) * 100
				override_counts[reg] = (n_reg_override, n_reg_total)

		bar_colors = [REGIME_COLORS.get(r, "#607D8B") for r in all_regimes]
		bars = ax_bar.bar(
				range(n_reg),
				[override_rates[r] for r in all_regimes],
				color=bar_colors, alpha=0.82, edgecolor="white", linewidth=1.2,
				zorder=3,
		)

		# Annotate bars
		for idx, (bar, reg) in enumerate(zip(bars, all_regimes)):
				n_ov, n_tot = override_counts[reg]
				rate         = override_rates[reg]
				y_off        = max(rate * 0.03, 0.8)
				ax_bar.text(
						bar.get_x() + bar.get_width() / 2,
						rate + y_off,
						f"{n_ov:,} / {n_tot:,}\n({rate:.1f}%)",
						ha="center", va="bottom", fontsize=7.5, color="#333",
				)

		# Overall override rate reference line
		overall_rate = n_overrides / max(n_gmm_routed, 1) * 100
		ax_bar.axhline(
				overall_rate, color="#37474F", linewidth=1.3,
				linestyle="--", zorder=2,
				label=f"Overall {overall_rate:.1f}%",
		)
		ax_bar.set_xticks(range(n_reg))
		ax_bar.set_xticklabels(short_labels, fontsize=8)
		ax_bar.set_ylabel("Override rate  (%)", fontsize=9)
		ax_bar.set_ylim(0, min(100, max(override_rates.values()) * 1.30 + 5))
		ax_bar.set_title(
				"Override Rate per Heuristic Regime\n(% of class overridden by GMM)",
				fontsize=9, fontweight="bold",
		)
		ax_bar.legend(fontsize=8, frameon=False)
		ax_bar.spines[["top", "right"]].set_visible(False)

		# ── BL: GMM confidence violin split by override ───────────────────────────
		df_conf = df_raw[df_raw["_conf"].notna()].copy()
		df_conf["_override_label"] = df_conf["_override"].map(
				{True: "Override\n(GMM ≠ Heuristic)", False: "Agreement\n(GMM = Heuristic)"}
		)

		if len(df_conf) > 0:
				override_groups = ["Agreement\n(GMM = Heuristic)", "Override\n(GMM ≠ Heuristic)"]
				viol_colors     = ["#1565C0", "#c62828"]

				for gi, (grp_label, vcolor) in enumerate(zip(override_groups, viol_colors)):
						grp_data = df_conf[df_conf["_override_label"] == grp_label]["_conf"].values
						if len(grp_data) == 0:
								continue

						# Violin
						parts = ax_viol.violinplot(
								grp_data, positions=[gi],
								widths=0.55, showmedians=False, showextrema=False,
						)
						for pc in parts["bodies"]:
								pc.set_facecolor(vcolor)
								pc.set_alpha(0.45)

						# Strip (jittered scatter)
						jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=min(len(grp_data), 800))
						sample_idx = np.random.default_rng(42).choice(
								len(grp_data), size=min(len(grp_data), 800), replace=False
						)
						ax_viol.scatter(
								gi + jitter, grp_data[sample_idx],
								alpha=0.25, s=8, color=vcolor, edgecolors="none", zorder=3,
						)

						# Median line
						med = float(np.median(grp_data))
						ax_viol.hlines(
								med, gi - 0.25, gi + 0.25,
								colors=vcolor, linewidths=2.2, zorder=5,
						)
						ax_viol.text(
								gi + 0.28, med,
								f"med={med:.3f}",
								va="center", fontsize=8, color=vcolor, fontweight="bold",
						)

				# Confidence threshold line
				ax_viol.axhline(
						conf_threshold, color="#FF6F00", linewidth=1.4,
						linestyle="--", zorder=4,
						label=f"Threshold {conf_threshold}",
				)
				ax_viol.set_xticks([0, 1])
				ax_viol.set_xticklabels(override_groups, fontsize=8)
				ax_viol.set_ylabel("GMM confidence  (max prob)", fontsize=9)
				ax_viol.set_ylim(0, 1.05)
				ax_viol.set_title(
						"GMM Confidence Distribution\n(agreement vs override)",
						fontsize=9, fontweight="bold",
				)
				ax_viol.legend(fontsize=8, frameon=False)
				ax_viol.spines[["top", "right"]].set_visible(False)
		else:
				ax_viol.text(
						0.5, 0.5, "No GMM confidence data available",
						ha="center", va="center", transform=ax_viol.transAxes, fontsize=10,
				)

		# ── BR: Sankey-style stacked bar (heuristic → GMM redistribution) ─────────
		# For each heuristic regime: 100% stacked bar coloured by final GMM regime
		x_positions = np.arange(n_reg)
		bottoms     = np.zeros(n_reg)

		for gmm_reg in all_regimes:
				seg_heights = []
				for h_reg in all_regimes:
						n_h   = int((df_raw["heuristic_regime"] == h_reg).sum())
						n_hg  = int(
								((df_raw["heuristic_regime"] == h_reg) &
								 (df_raw["regime"] == gmm_reg)).sum()
						)
						seg_heights.append(n_hg / max(n_h, 1) * 100)

				color = REGIME_COLORS.get(gmm_reg, "#607D8B")
				bars_s = ax_sank.bar(
						x_positions, seg_heights,
						bottom=bottoms,
						color=color, alpha=0.82,
						edgecolor="white", linewidth=0.8,
						label=gmm_reg.replace("_", " "),
						zorder=3,
				)

				# Annotate segments ≥ 5%
				for xi, (height, bottom_val) in enumerate(zip(seg_heights, bottoms)):
						if height >= 5.0:
								ax_sank.text(
										xi, bottom_val + height / 2,
										f"{height:.0f}%",
										ha="center", va="center",
										fontsize=7.5, color="white", fontweight="bold",
								)
				bottoms += np.array(seg_heights)

		ax_sank.set_xticks(x_positions)
		ax_sank.set_xticklabels(short_labels, fontsize=8)
		ax_sank.set_ylabel("% of heuristic class", fontsize=9)
		ax_sank.set_ylim(0, 105)
		ax_sank.set_title(
				"GMM Redistribution per Heuristic Regime\n(100% stacked — coloured by final GMM regime)",
				fontsize=9, fontweight="bold",
		)
		ax_sank.legend(
				fontsize=7.5, frameon=False,
				loc="upper right", title="GMM regime", title_fontsize=8,
		)
		ax_sank.spines[["top", "right"]].set_visible(False)

		# ── Footer strip ──────────────────────────────────────────────────────────
		footer = (
				f"Total samples: {n_total:,}  |  "
				f"GMM-routed: {n_gmm_routed:,} ({n_gmm_routed / max(n_total, 1) * 100:.1f}%)  |  "
				f"Overrides: {n_overrides:,} ({override_pct:.1f}% of GMM-routed)  |  "
				f"Low-conf (< {conf_threshold}): {n_low_conf:,}  |  "
				f"feature_dim={feat_dim_str}"
		)
		fig.text(
				0.5, -0.01,
				footer,
				ha="center", va="top", fontsize=8,
				color="#444", style="italic",
				transform=fig.transFigure,
		)

		_save(fig, out_dir, "V12_heuristic_gmm_confusion")
		print(
				f"[VIZ][V12] Done. "
				f"n={n_total:,} | "
				f"GMM-routed={n_gmm_routed:,} | "
				f"overrides={n_overrides:,} ({override_pct:.1f}%) | "
				f"low-conf={n_low_conf:,} | "
				f"feature_dim={feat_dim_str}"
		)

# V13 — GMM Confidence Diagnostics (Calibration-Style)
def viz_gmm_confidence_diagnostics(
		stage4_parquet: str,
		conf_threshold: float = 0.60,
		n_bins:         int   = 10,
		out_dir:        Optional[str] = None,
):
		"""
		Six-panel calibration-style audit of GMM confidence scores.

		Panel layout (3 × 2):
				TL — Confidence histogram per regime (overlapping, density-normalised)
							 • One curve per final GMM regime
							 • Vertical dashed line at conf_threshold
							 • Median annotated per regime

				TC — Reliability / calibration diagram
							 • X-axis: mean confidence in bin  (10 equal-width bins over [0,1])
							 • Y-axis: empirical accuracy (fraction of bin where GMM agrees
												 with heuristic, i.e. regime_override=False)
							 • Perfect calibration diagonal (dashed)
							 • Bar chart of bin population in background (twin y-axis)
							 • Shaded over-/under-confidence zones

				TR — Confidence CDF per regime
							 • Cumulative distribution of confidence for each final regime
							 • Vertical line at conf_threshold
							 • Annotated: % of each regime below threshold (low-conf fraction)

				BL — Confidence vs supervision weight scatter (w_pos)
							 • X-axis: GMM confidence
							 • Y-axis: w_pos
							 • Colour = final regime
							 • Loess/rolling-mean trend line per regime
							 • Reveals whether low-confidence samples get down-weighted

				BC — Per-regime confidence box plot
							 • One box per final regime, coloured by REGIME_COLORS
							 • Overlaid strip (jittered) for n ≤ 2000 samples
							 • Annotated: median, IQR, low-conf count

				BR — Probability simplex heatmap (top-2 regime probabilities)
							 • X-axis: max probability (winning regime)
							 • Y-axis: 2nd-highest probability
							 • Colour = final regime
							 • Iso-margin lines at margin = max − 2nd = {0.1, 0.2, 0.3}
							 • Reveals ambiguous samples near decision boundaries

		Footer strip:
				n_total | n_gmm_routed | n_low_conf | mean_conf | median_conf |
				override_rate | feature_dim

		Reads:
				stage4_parquet — auditable supervision matrix parquet
												 Required columns: regime, heuristic_regime,
																					 gmm (JSON str), w_pos
		"""
		# ── Load & parse ──────────────────────────────────────────────────────────
		try:
				df_raw = pd.read_parquet(
						stage4_parquet,
						columns=["regime", "heuristic_regime", "gmm", "w_pos"],
						engine="pyarrow",
				)
		except Exception as e:
				print(f"[VIZ][V13][ERROR] Failed to read parquet: {e}")
				return

		def _parse_gmm(x):
				if isinstance(x, dict):  return x
				if isinstance(x, str):
						try:    return json.loads(x)
						except: return None
				return None

		df_raw["_gmm"]      = df_raw["gmm"].apply(_parse_gmm)
		df_raw["_routed"]   = df_raw["_gmm"].apply(lambda x: isinstance(x, dict))
		df_raw["_conf"]     = df_raw["_gmm"].apply(
				lambda x: float(x["confidence"]) if isinstance(x, dict) and x.get("confidence") is not None else None
		)
		df_raw["_override"] = df_raw["_gmm"].apply(
				lambda x: bool(x.get("regime_override", False)) if isinstance(x, dict) else False
		)
		df_raw["_probs"]    = df_raw["_gmm"].apply(
				lambda x: x.get("probabilities") if isinstance(x, dict) else None
		)
		df_raw["_feat_dim"] = df_raw["_gmm"].apply(
				lambda x: x.get("feature_dim") if isinstance(x, dict) else None
		)

		df = df_raw[df_raw["_routed"]].copy()   # GMM-routed only

		if df.empty:
				print("[VIZ][V13][WARN] No GMM-routed samples found. Nothing to plot.")
				return

		# ── Summary stats ─────────────────────────────────────────────────────────
		n_total      = len(df_raw)
		n_routed     = len(df)
		conf_vals    = df["_conf"].dropna()
		n_low_conf   = int((conf_vals < conf_threshold).sum())
		mean_conf    = float(conf_vals.mean())
		median_conf  = float(conf_vals.median())
		n_overrides  = int(df["_override"].sum())
		override_pct = n_overrides / max(n_routed, 1) * 100
		feat_dim_val = df["_feat_dim"].dropna().mode()
		feat_dim_str = str(int(feat_dim_val.iloc[0])) if len(feat_dim_val) else "?"

		REGIME_ORDER = ["AGREEMENT", "SOFT_CONFLICT", "HARD_CONFLICT", "MISSING_MODALITY"]
		regimes      = [r for r in REGIME_ORDER if r in df["regime"].unique()]

		fig = plt.figure(figsize=(20, 15), constrained_layout=True)
		fig.suptitle(
				f"GMM Confidence Diagnostics  "
				f"(n_routed={n_routed:,}  |  feature_dim={feat_dim_str}  |  "
				f"threshold={conf_threshold})",
				fontsize=12, fontweight="bold",
		)
		gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)
		ax_hist  = fig.add_subplot(gs[0, 0])   # TL
		ax_cal   = fig.add_subplot(gs[0, 1])   # TC  (reliability diagram)
		ax_cdf   = fig.add_subplot(gs[1, 0])   # TR  (CDF)
		ax_scat  = fig.add_subplot(gs[1, 1])   # BL  (conf vs w_pos)
		ax_box   = fig.add_subplot(gs[2, 0])   # BC  (box plot)
		ax_simp  = fig.add_subplot(gs[2, 1])   # BR  (simplex heatmap)

		# ── TL: Confidence histogram per regime ───────────────────────────────────
		bins_h = np.linspace(0, 1, 31)
		for regime in regimes:
				sub  = df[df["regime"] == regime]["_conf"].dropna()
				if sub.empty: continue
				color = REGIME_COLORS.get(regime, "#607D8B")
				ax_hist.hist(
						sub, bins=bins_h, alpha=0.50,
						color=color, density=True, edgecolor="none",
						label=f"{regime}  (n={len(sub):,})",
				)
				med = float(sub.median())
				ax_hist.axvline(
						med, color=color, linewidth=1.4,
						linestyle="--", alpha=0.85,
				)
				ax_hist.text(
						med + 0.01, ax_hist.get_ylim()[1] * 0.02,
						f"{med:.2f}", fontsize=7, color=color, rotation=90, va="bottom",
				)

		ax_hist.axvline(
				conf_threshold, color="#37474F", linewidth=1.6,
				linestyle=":", label=f"Threshold {conf_threshold}",
		)
		ax_hist.set_xlabel("GMM confidence  (max probability)", fontsize=9)
		ax_hist.set_ylabel("Density", fontsize=9)
		ax_hist.set_title("Confidence Distribution per Regime", fontsize=9, fontweight="bold")
		ax_hist.legend(fontsize=7.5, frameon=False)
		ax_hist.spines[["top", "right"]].set_visible(False)

		# ── TC: Reliability / calibration diagram ────────────────────────────────
		# "Accuracy" = fraction of bin where GMM agrees with heuristic (override=False)
		bin_edges  = np.linspace(0, 1, n_bins + 1)
		bin_mids   = (bin_edges[:-1] + bin_edges[1:]) / 2
		bin_acc    = []
		bin_counts = []

		df_cal = df[df["_conf"].notna()].copy()
		df_cal["_agree"] = (~df_cal["_override"]).astype(float)

		for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
				mask = (df_cal["_conf"] >= lo) & (df_cal["_conf"] < hi)
				sub  = df_cal[mask]
				bin_counts.append(len(sub))
				bin_acc.append(float(sub["_agree"].mean()) if len(sub) > 0 else np.nan)

		# Background bar (population)
		ax_cal_twin = ax_cal.twinx()
		ax_cal_twin.bar(
				bin_mids, bin_counts,
				width=(bin_edges[1] - bin_edges[0]) * 0.85,
				color="#B0BEC5", alpha=0.30, zorder=1,
		)
		ax_cal_twin.set_ylabel("Bin count", fontsize=8, color="#78909C")
		ax_cal_twin.tick_params(axis="y", labelcolor="#78909C", labelsize=7)
		ax_cal_twin.spines[["top"]].set_visible(False)

		# Perfect calibration diagonal
		ax_cal.plot([0, 1], [0, 1], color="#37474F", linewidth=1.2,
								linestyle="--", alpha=0.6, label="Perfect calibration", zorder=3)

		# Over/under-confidence shading
		ax_cal.fill_between([0, 1], [0, 1], [1, 1],
												alpha=0.05, color="#1565C0", label="Over-confident zone")
		ax_cal.fill_between([0, 1], [0, 0], [0, 1],
												alpha=0.05, color="#c62828", label="Under-confident zone")

		# Reliability curve
		valid = [(m, a) for m, a in zip(bin_mids, bin_acc) if not np.isnan(a)]
		if valid:
			xs, ys = zip(*valid)
			ax_cal.plot(
				xs, 
				ys, 
				color="#1565C0", 
				marker="o", 
				linewidth=2.0,
				markersize=6, 
				zorder=4, 
				label="Empirical accuracy"
			)
			# ECE annotation
			ece = float(np.nanmean([abs(a - m) * c for m, a, c in zip(bin_mids, bin_acc, bin_counts)]) / max(sum(bin_counts), 1))
			ax_cal.text(
					0.05, 0.92,
					f"ECE ≈ {ece:.4f}",
					transform=ax_cal.transAxes,
					fontsize=9, color="#1565C0", fontweight="bold",
			)

		ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
		ax_cal.set_xlabel("Mean confidence in bin", fontsize=9)
		ax_cal.set_ylabel("Fraction agreeing with heuristic", fontsize=9)
		ax_cal.set_title(
			"Reliability Diagram\n(GMM conf vs heuristic agreement rate)",
			fontsize=9, 
			fontweight="bold",
		)
		ax_cal.legend(fontsize=7.5, frameon=False, loc="lower right")
		ax_cal.spines[["top", "right"]].set_visible(False)

		# ── TR: Confidence CDF per regime ─────────────────────────────────────────
		for regime in regimes:
			sub = df[df["regime"] == regime]["_conf"].dropna().sort_values()
			if sub.empty: continue
			color = REGIME_COLORS.get(regime, "#607D8B")
			cdf = np.arange(1, len(sub) + 1) / len(sub)
			ax_cdf.plot(
				sub, cdf, 
				color=color, 
				linewidth=2.0,
				label=f"{regime} (n={len(sub):,})", 
				alpha=0.85
			)
			# Annotate % below threshold
			frac_low = float((sub < conf_threshold).mean()) * 100
			y_at_thr = float(cdf[sub.searchsorted(conf_threshold, side="right") - 1]) if sub.searchsorted(conf_threshold) > 0 else 0.0
			ax_cdf.annotate(
				f"{frac_low:.1f}% below {conf_threshold}",
				xy=(conf_threshold, y_at_thr),
				xytext=(conf_threshold - 0.22, y_at_thr + 0.06),
				fontsize=6.5, color=color,
				arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
			)

		ax_cdf.axvline(
			conf_threshold, 
			color="#37474F", 
			linewidth=1.1,
			linestyle=":", 
			label=f"Threshold {conf_threshold}"
		)
		ax_cdf.set_xlabel("GMM confidence", fontsize=9)
		ax_cdf.set_ylabel("Cumulative fraction", fontsize=9)
		ax_cdf.set_title("Confidence CDF per Regime\n(% below threshold annotated)", fontsize=9, fontweight="bold")
		ax_cdf.legend(fontsize=7.5, frameon=False)
		ax_cdf.spines[["top", "right"]].set_visible(False)

		# ── BL: Confidence vs w_pos scatter ───────────────────────────────────────
		df_scat = df[df["_conf"].notna() & df["w_pos"].notna()].copy()
		rng     = np.random.default_rng(42)

		for regime in regimes:
				sub   = df_scat[df_scat["regime"] == regime]
				if sub.empty: continue
				color = REGIME_COLORS.get(regime, "#607D8B")

				# Subsample for scatter readability
				idx = rng.choice(len(sub), size=min(len(sub), 1200), replace=False)
				sub_s = sub.iloc[idx]

				ax_scat.scatter(
						sub_s["_conf"], sub_s["w_pos"],
						c=color, alpha=0.20, s=10, edgecolors="none",
						label=f"{regime}  (n={len(sub):,})",
				)

				# Rolling mean trend line (sorted by conf)
				sub_sorted = sub.sort_values("_conf")
				window     = max(len(sub_sorted) // 20, 5)
				trend      = sub_sorted["w_pos"].rolling(window, center=True, min_periods=3).mean()
				ax_scat.plot(
					sub_sorted["_conf"], trend,
					color=color, 
					linewidth=2.0, 
					alpha=0.90, 
					zorder=4,
				)

		ax_scat.axvline(
			conf_threshold, 
			color="#37474F", 
			linewidth=1.4,
			linestyle=":", 
			label=f"Threshold {conf_threshold}"
		)
		ax_scat.set_xlabel("GMM confidence", fontsize=9)
		ax_scat.set_ylabel("w_pos  (positive supervision weight)", fontsize=9)
		ax_scat.set_title(
				"Confidence vs Supervision Weight (w_pos) (trend line = rolling mean per regime)",
				fontsize=9, 
				fontweight="bold",
		)
		ax_scat.legend(fontsize=7.5, frameon=False)
		ax_scat.spines[["top", "right"]].set_visible(False)

		# Per-regime confidence box plot
		box_data   = []
		box_labels = []
		box_colors = []

		for regime in regimes:
				sub = df[df["regime"] == regime]["_conf"].dropna()
				if sub.empty: continue
				box_data.append(sub.values)
				box_labels.append(regime.replace("_", "\n"))
				box_colors.append(REGIME_COLORS.get(regime, "#607D8B"))

		if box_data:
				bp = ax_box.boxplot(
						box_data,
						patch_artist=True,
						notch=False,
						widths=0.45,
						medianprops=dict(color="white", linewidth=2.0),
						whiskerprops=dict(linewidth=1.2),
						capprops=dict(linewidth=1.2),
						flierprops=dict(marker=".", markersize=3, alpha=0.3),
				)
				for patch, color in zip(bp["boxes"], box_colors):
						patch.set_facecolor(color)
						patch.set_alpha(0.70)

				# Jittered strip for small groups
				for xi, (data, color) in enumerate(zip(box_data, box_colors), start=1):
						if len(data) <= 2000:
								jitter = rng.uniform(-0.18, 0.18, size=len(data))
								ax_box.scatter(
										xi + jitter, data,
										alpha=0.18, s=7, color=color, edgecolors="none", zorder=3,
								)

				# Annotate: median + low-conf count
				for xi, (data, regime) in enumerate(
						zip(box_data, [r for r in regimes if not df[df["regime"] == r]["_conf"].dropna().empty]),
						start=1,
				):
						med      = float(np.median(data))
						n_low    = int((data < conf_threshold).sum())
						pct_low  = n_low / max(len(data), 1) * 100
						ax_box.text(
								xi, med + 0.02,
								f"med={med:.3f}\n{n_low:,} low\n({pct_low:.1f}%)",
								ha="center", 
								va="bottom", 
								fontsize=6.5, 
								color="#070606",
						)

				ax_box.axhline(conf_threshold, color="#37474F", linewidth=1.4,
											 linestyle=":", label=f"Threshold {conf_threshold}")
				ax_box.set_xticks(range(1, len(box_labels) + 1))
				ax_box.set_xticklabels(box_labels, fontsize=8)
				ax_box.set_ylabel("GMM confidence", fontsize=9)
				ax_box.set_ylim(0, 1.08)
				ax_box.set_title(
					"Confidence per Regime (median | low-conf count annotated)",
					fontsize=9, 
					fontweight="bold",
				)
				ax_box.legend(fontsize=8, frameon=False)
				ax_box.spines[["top", "right"]].set_visible(False)

		# ── BR: Probability simplex heatmap (max prob vs 2nd-highest prob) ────────
		simp_rows = []
		for _, row in df.iterrows():
				probs = row["_probs"]
				if not isinstance(probs, dict) or len(probs) < 2:
						continue
				sorted_probs = sorted(probs.values(), reverse=True)
				p1 = float(sorted_probs[0])
				p2 = float(sorted_probs[1])
				simp_rows.append({
						"p1":     p1,
						"p2":     p2,
						"margin": p1 - p2,
						"regime": row["regime"],
				})

		df_simp = pd.DataFrame(simp_rows)

		if not df_simp.empty:
				for regime in regimes:
						sub   = df_simp[df_simp["regime"] == regime]
						if sub.empty: continue
						color = REGIME_COLORS.get(regime, "#607D8B")
						idx   = rng.choice(len(sub), size=min(len(sub), 1500), replace=False)
						sub_s = sub.iloc[idx]
						ax_simp.scatter(
								sub_s["p1"], sub_s["p2"],
								c=color, alpha=0.22, s=10, edgecolors="none",
								label=f"{regime}  (n={len(sub):,})",
						)

				# Iso-margin lines: p1 − p2 = margin_val  →  p2 = p1 − margin_val
				x_line = np.linspace(0, 1, 200)
				for margin_val, ls in [(0.10, ":"), (0.20, "--"), (0.30, "-.")]:
						y_line = np.clip(x_line - margin_val, 0, 1)
						ax_simp.plot(
								x_line, y_line,
								color="#37474F", 
								linewidth=1.0, 
								linestyle=ls, 
								alpha=0.60,
								label=f"margin={margin_val}",
						)

				# Feasibility boundary: p1 + p2 ≤ 1  →  p2 = 1 − p1
				ax_simp.fill_between(
						x_line, np.clip(1 - x_line, 0, 1), 1,
						alpha=0.06, color="#37474F", label="Infeasible (p1+p2>1)",
				)

				ax_simp.set_xlim(0, 1); ax_simp.set_ylim(0, 1)
				ax_simp.set_xlabel("Max probability  (winning regime)", fontsize=9)
				ax_simp.set_ylabel("2nd-highest probability", fontsize=9)
				ax_simp.set_title(
					"Probability Simplex (max vs 2nd-highest)",
					fontsize=9, 
					fontweight="bold",
				)
				ax_simp.legend(fontsize=7, frameon=False, ncol=3, loc="best")
				ax_simp.spines[["top", "right"]].set_visible(False)
		else:
				ax_simp.text(
						0.5, 0.5, "No probability data available\n(probabilities key missing in gmm column)",
						ha="center", va="center", transform=ax_simp.transAxes, fontsize=9,
				)

		footer = (
				f"Total samples: {n_total:,}  |  "
				f"GMM-routed: {n_routed:,} ({n_routed / max(n_total, 1) * 100:.1f}%)  |  "
				f"Low-conf (< {conf_threshold}): {n_low_conf:,} ({n_low_conf / max(n_routed, 1) * 100:.1f}%)  |  "
				f"Mean conf: {mean_conf:.4f}  |  Median conf: {median_conf:.4f}  |  "
				f"Override rate: {override_pct:.1f}%  |  feature_dim={feat_dim_str}"
		)
		fig.text(
				0.5, -0.01,
				footer,
				ha="center", va="top", fontsize=8,
				color="#444", style="italic",
				transform=fig.transFigure,
		)

		_save(fig, out_dir, "V13_gmm_confidence_diagnostics")
		print(
				f"[VIZ][V13] Done. "
				f"n_routed={n_routed:,} | "
				f"mean_conf={mean_conf:.4f} | "
				f"median_conf={median_conf:.4f} | "
				f"low_conf={n_low_conf:,} ({n_low_conf / max(n_routed, 1) * 100:.1f}%) | "
				f"override_rate={override_pct:.1f}%"
		)

# V14 — Probability-Simplex View (Ternary) of GMM Outputs
def viz_gmm_probability_simplex(
		stage4_parquet: str,
		conf_threshold: float = 0.60,
		max_scatter:    int   = 4000,   # max points per sub-panel (random subsample)
		out_dir:        Optional[str] = None,
):
		"""
		Ternary-plot audit of the GMM's full probability distribution over the
		three routable regimes (AGREEMENT, SOFT_CONFLICT, HARD_CONFLICT).
		MISSING_MODALITY is never GMM-routed and is excluded from the simplex.

		The ternary coordinate system maps each sample's (p_A, p_S, p_H) triplet
		to a 2-D point inside an equilateral triangle:
				• Bottom-left  vertex = pure AGREEMENT       (1, 0, 0)
				• Bottom-right vertex = pure HARD_CONFLICT   (0, 0, 1)
				• Top          vertex = pure SOFT_CONFLICT   (0, 1, 0)

		Panel layout (2 × 3):
				TL — Full simplex: all GMM-routed samples
							 • Colour = final GMM regime (REGIME_COLORS)
							 • Marker size ∝ GMM confidence
							 • Iso-confidence contours at 0.50, 0.70, 0.90
								 (regions where max-prob ≥ threshold)
							 • Triangle vertices and edge labels annotated

				TC — Override samples only (regime_override = True)
							 • Colour = FINAL regime (where GMM moved the sample TO)
							 • Marker shape = heuristic regime (where it came FROM):
									 AGREEMENT      → circle  "o"
									 SOFT_CONFLICT  → triangle "^"
									 HARD_CONFLICT  → square  "s"
							 • Reveals the geometric location of overridden samples

				TR — Low-confidence samples (conf < conf_threshold)
							 • Colour = final regime
							 • Annotated with density contour (KDE)
							 • Reveals whether low-conf samples cluster near edges or centre

				BL — Density heatmap (2-D histogram in barycentric coords)
							 • Hexbin of all GMM-routed samples
							 • Colour = log-count
							 • Overlaid regime boundary lines (equal-probability edges)

				BC — Per-heuristic-regime simplex (small multiples, 3 sub-triangles)
							 • One mini-ternary per heuristic regime
							 • Colour = final GMM regime
							 • Shows how each heuristic class spreads across the simplex

				BR — Margin histogram (p_max − p_2nd)
							 • Histogram of decision margin per final regime
							 • Vertical lines at margin = 0.10, 0.20, 0.30
							 • Annotated: % of samples with margin < 0.20 (ambiguous zone)

		Footer strip:
				n_total | n_routed | n_overrides | n_low_conf | mean_margin | feature_dim

		Reads:
				stage4_parquet — auditable supervision matrix parquet
												 Required columns: regime, heuristic_regime, gmm (JSON str)
		"""
		import matplotlib.tri as mtri
		from matplotlib.patches import Polygon as MplPolygon
		from matplotlib.colors import Normalize
		from scipy.stats import gaussian_kde

		# ── Load & parse ──────────────────────────────────────────────────────────
		try:
				df_raw = pd.read_parquet(
						stage4_parquet,
						columns=["regime", "heuristic_regime", "gmm"],
						engine="pyarrow",
				)
		except Exception as e:
				print(f"[VIZ][V14][ERROR] Failed to read parquet: {e}")
				return

		def _parse_gmm(x):
				if isinstance(x, dict):  return x
				if isinstance(x, str):
						try:    return json.loads(x)
						except: return None
				return None

		df_raw["_gmm"]      = df_raw["gmm"].apply(_parse_gmm)
		df_raw["_routed"]   = df_raw["_gmm"].apply(lambda x: isinstance(x, dict))
		df_raw["_conf"]     = df_raw["_gmm"].apply(
				lambda x: float(x["confidence"])
				if isinstance(x, dict) and x.get("confidence") is not None else None
		)
		df_raw["_override"] = df_raw["_gmm"].apply(
				lambda x: bool(x.get("regime_override", False))
				if isinstance(x, dict) else False
		)
		df_raw["_probs"]    = df_raw["_gmm"].apply(
				lambda x: x.get("probabilities") if isinstance(x, dict) else None
		)
		df_raw["_feat_dim"] = df_raw["_gmm"].apply(
				lambda x: x.get("feature_dim") if isinstance(x, dict) else None
		)

		df = df_raw[df_raw["_routed"] & df_raw["_probs"].notna()].copy()

		if df.empty:
				print("[VIZ][V14][WARN] No GMM-routed samples with probability data found.")
				return

		# ── Extract (p_A, p_S, p_H) triplets ─────────────────────────────────────
		TERNARY_REGIMES = ["AGREEMENT", "SOFT_CONFLICT", "HARD_CONFLICT"]

		def _extract_triplet(probs_dict):
				if not isinstance(probs_dict, dict):
						return None
				vals = [probs_dict.get(r, 0.0) for r in TERNARY_REGIMES]
				s    = sum(vals)
				if s <= 0:
						return None
				return [v / s for v in vals]   # re-normalise to sum=1 over 3 regimes

		df["_triplet"] = df["_probs"].apply(_extract_triplet)
		df = df[df["_triplet"].notna()].copy()

		if df.empty:
				print("[VIZ][V14][WARN] No valid (p_A, p_S, p_H) triplets found in probabilities.")
				return

		triplets = np.array(df["_triplet"].tolist())   # shape (N, 3)
		p_A = triplets[:, 0]
		p_S = triplets[:, 1]
		p_H = triplets[:, 2]

		# ── Barycentric → 2-D Cartesian conversion ────────────────────────────────
		# Equilateral triangle:
		#   vertex A (AGREEMENT)      = (0,   0)
		#   vertex H (HARD_CONFLICT)  = (1,   0)
		#   vertex S (SOFT_CONFLICT)  = (0.5, √3/2)
		SQRT3_2 = np.sqrt(3) / 2

		def bary_to_cart(pa, ps, ph):
				"""Convert barycentric (p_A, p_S, p_H) → Cartesian (x, y)."""
				x = ph * 1.0 + ps * 0.5   # H at (1,0), S at (0.5, √3/2), A at (0,0)
				y = ps * SQRT3_2
				return x, y

		def _cart_all(pa, ps, ph):
				return bary_to_cart(pa, ps, ph)

		x_all, y_all = _cart_all(p_A, p_S, p_H)
		df["_x"] = x_all
		df["_y"] = y_all

		# Confidence and margin
		conf_all   = df["_conf"].values
		sorted_p   = np.sort(triplets, axis=1)[:, ::-1]
		margin_all = sorted_p[:, 0] - sorted_p[:, 1]
		df["_margin"] = margin_all

		# ── Summary stats ─────────────────────────────────────────────────────────
		n_total     = len(df_raw)
		n_routed    = len(df)
		n_overrides = int(df["_override"].sum())
		n_low_conf  = int((df["_conf"] < conf_threshold).sum())
		mean_margin = float(np.nanmean(margin_all))
		feat_dim_val = df["_feat_dim"].dropna().mode()
		feat_dim_str = str(int(feat_dim_val.iloc[0])) if len(feat_dim_val) else "?"

		rng = np.random.default_rng(42)

		# ── Triangle drawing helper ───────────────────────────────────────────────
		VERTICES = np.array([[0, 0], [1, 0], [0.5, SQRT3_2]])   # A, H, S
		VERTEX_LABELS = [
				("AGREEMENT",     -0.10, -0.06),
				("HARD\nCONFLICT", 1.05, -0.06),
				("SOFT\nCONFLICT", 0.50,  SQRT3_2 + 0.05),
		]
		EDGE_LABELS = [
				# midpoint of each edge, label, rotation
				(0.25, SQRT3_2 / 2 + 0.02, "A ↔ S",  60),
				(0.75, SQRT3_2 / 2 + 0.02, "S ↔ H", -60),
				(0.50, -0.06,               "A ↔ H",   0),
		]

		def _draw_triangle(ax, alpha=0.85, lw=1.4):
				tri = MplPolygon(
						VERTICES, closed=True,
						fill=False, edgecolor="#37474F",
						linewidth=lw, zorder=10,
				)
				ax.add_patch(tri)
				for label, vx, vy in VERTEX_LABELS:
						ax.text(vx, vy, label, ha="center", va="center",
										fontsize=7.5, fontweight="bold", color="#37474F", zorder=11)
				ax.set_xlim(-0.18, 1.18)
				ax.set_ylim(-0.14, SQRT3_2 + 0.14)
				ax.set_aspect("equal")
				ax.axis("off")

		def _draw_iso_confidence(ax, thresholds=(0.50, 0.70, 0.90)):
				"""
				Draw iso-confidence contours: locus of points where max(p_A,p_S,p_H)=t.
				Each contour is a hexagon inscribed in the triangle.
				"""
				for t in thresholds:
						# The region max(p) >= t is a triangle near each vertex.
						# The boundary max(p) = t is a line segment parallel to the opposite edge.
						# For vertex A (p_A = t): p_A = t, p_S + p_H = 1-t
						# Segment from (t, (1-t), 0) to (t, 0, (1-t)) in barycentric
						pts = []
						for vi in range(3):
								# vertex vi has p_vi = t, the other two split (1-t) equally at endpoints
								for vj in range(3):
										if vj == vi: continue
										p = [0.0, 0.0, 0.0]
										p[vi] = t
										p[vj] = 1.0 - t
										pts.append(bary_to_cart(p[0], p[1], p[2]))
						# Sort by angle around centroid
						cx = np.mean([p[0] for p in pts])
						cy = np.mean([p[1] for p in pts])
						pts_sorted = sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
						xs = [p[0] for p in pts_sorted] + [pts_sorted[0][0]]
						ys = [p[1] for p in pts_sorted] + [pts_sorted[0][1]]
						ax.plot(xs, ys, color="#37474F", linewidth=0.8,
										linestyle="--", alpha=0.45, zorder=9)
						ax.text(cx, cy, f"{t:.0%}", ha="center", va="center",
										fontsize=6, color="#37474F", alpha=0.65, zorder=9)

		def _draw_equal_prob_lines(ax):
				"""
				Draw the 3 equal-probability lines (p_i = p_j) that divide the simplex
				into 3 Voronoi regions (one per regime).
				Each line passes through the centroid (1/3, 1/3, 1/3) and a vertex midpoint.
				"""
				centroid = bary_to_cart(1/3, 1/3, 1/3)
				# Midpoints of each edge
				edge_mids_bary = [
						(0.5, 0.5, 0.0),   # midpoint A-S edge
						(0.0, 0.5, 0.5),   # midpoint S-H edge
						(0.5, 0.0, 0.5),   # midpoint A-H edge
				]
				for bary in edge_mids_bary:
						ep = bary_to_cart(*bary)
						ax.plot(
								[centroid[0], ep[0]], [centroid[1], ep[1]],
								color="#78909C", linewidth=0.9, linestyle=":",
								alpha=0.55, zorder=8,
						)

		# ── Figure ────────────────────────────────────────────────────────────────
		fig = plt.figure(figsize=(18, 12), constrained_layout=True)
		fig.suptitle(
				f"V14 — GMM Probability Simplex (Ternary)  "
				f"(n_routed={n_routed:,}  |  feature_dim={feat_dim_str}  |  "
				f"threshold={conf_threshold})",
				fontsize=12, fontweight="bold",
		)
		gs = fig.add_gridspec(2, 3, hspace=0.10, wspace=0.05)
		ax_full   = fig.add_subplot(gs[0, 0])   # TL
		ax_over   = fig.add_subplot(gs[0, 1])   # TC
		ax_lowc   = fig.add_subplot(gs[0, 2])   # TR
		ax_hex    = fig.add_subplot(gs[1, 0])   # BL
		ax_small  = fig.add_subplot(gs[1, 1])   # BC  (small multiples)
		ax_margin = fig.add_subplot(gs[1, 2])   # BR  (margin histogram — Cartesian)

		# ── TL: Full simplex ──────────────────────────────────────────────────────
		_draw_triangle(ax_full)
		_draw_iso_confidence(ax_full, thresholds=(0.50, 0.70, 0.90))
		_draw_equal_prob_lines(ax_full)

		for regime in TERNARY_REGIMES:
				mask  = df["regime"] == regime
				sub   = df[mask]
				if sub.empty: continue
				color = REGIME_COLORS.get(regime, "#607D8B")
				idx   = rng.choice(len(sub), size=min(len(sub), max_scatter), replace=False)
				sub_s = sub.iloc[idx]
				# Size ∝ confidence
				sizes = np.clip(sub_s["_conf"].fillna(0.5).values * 30, 4, 40)
				ax_full.scatter(
						sub_s["_x"], sub_s["_y"],
						c=color, s=sizes, alpha=0.30, edgecolors="none",
						label=f"{regime}  (n={len(sub):,})",
						zorder=5,
				)

		ax_full.set_title(
				"All GMM-routed samples\n(size ∝ confidence | iso-conf contours)",
				fontsize=8.5, fontweight="bold",
		)
		ax_full.legend(fontsize=7, frameon=False, loc="upper right",
									 bbox_to_anchor=(1.02, 1.0))

		# ── TC: Override samples ──────────────────────────────────────────────────
		_draw_triangle(ax_over)
		_draw_equal_prob_lines(ax_over)

		df_over = df[df["_override"]].copy()
		HEURISTIC_MARKERS = {
				"AGREEMENT":     "o",
				"SOFT_CONFLICT": "^",
				"HARD_CONFLICT": "s",
		}

		if df_over.empty:
				ax_over.text(0.5, SQRT3_2 / 2, "No overrides found",
										 ha="center", va="center", fontsize=9, color="#555")
		else:
				for h_regime, marker in HEURISTIC_MARKERS.items():
						sub_h = df_over[df_over["heuristic_regime"] == h_regime]
						if sub_h.empty: continue
						for f_regime in TERNARY_REGIMES:
								sub_hf = sub_h[sub_h["regime"] == f_regime]
								if sub_hf.empty: continue
								color = REGIME_COLORS.get(f_regime, "#607D8B")
								idx   = rng.choice(len(sub_hf),
																	 size=min(len(sub_hf), max_scatter // 3),
																	 replace=False)
								sub_s = sub_hf.iloc[idx]
								ax_over.scatter(
										sub_s["_x"], sub_s["_y"],
										c=color, marker=marker, s=28,
										alpha=0.55, edgecolors="white", linewidths=0.4,
										label=f"{h_regime[:3]}→{f_regime[:3]}  (n={len(sub_hf):,})",
										zorder=5,
								)

		ax_over.set_title(
				f"Override samples only  (n={len(df_over):,})\n"
				"colour=final regime | shape=heuristic regime",
				fontsize=8.5, fontweight="bold",
		)
		ax_over.legend(fontsize=6.5, frameon=False, loc="upper right",
									 bbox_to_anchor=(1.02, 1.0))

		# ── TR: Low-confidence samples + KDE contour ──────────────────────────────
		_draw_triangle(ax_lowc)
		_draw_equal_prob_lines(ax_lowc)

		df_low = df[df["_conf"] < conf_threshold].copy()

		if df_low.empty:
				ax_lowc.text(0.5, SQRT3_2 / 2,
										 f"No samples below\nthreshold {conf_threshold}",
										 ha="center", va="center", fontsize=9, color="#555")
		else:
				for regime in TERNARY_REGIMES:
						sub   = df_low[df_low["regime"] == regime]
						if sub.empty: continue
						color = REGIME_COLORS.get(regime, "#607D8B")
						idx   = rng.choice(len(sub), size=min(len(sub), max_scatter), replace=False)
						sub_s = sub.iloc[idx]
						ax_lowc.scatter(
								sub_s["_x"], sub_s["_y"],
								c=color, s=14, alpha=0.40, edgecolors="none",
								label=f"{regime}  (n={len(sub):,})",
								zorder=5,
						)

				# KDE contour over all low-conf points
				if len(df_low) >= 10:
						try:
								kde = gaussian_kde(
										np.vstack([df_low["_x"].values, df_low["_y"].values]),
										bw_method="scott",
								)
								gx = np.linspace(-0.05, 1.05, 120)
								gy = np.linspace(-0.05, SQRT3_2 + 0.05, 120)
								GX, GY = np.meshgrid(gx, gy)
								Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
								ax_lowc.contour(
										GX, GY, Z,
										levels=5, cmap="Reds", alpha=0.55,
										linewidths=1.0, zorder=6,
								)
						except Exception:
								pass   # KDE can fail on degenerate data

		ax_lowc.set_title(
				f"Low-confidence samples  (conf < {conf_threshold}, n={len(df_low):,})\n"
				"colour=final regime | KDE contour overlay",
				fontsize=8.5, fontweight="bold",
		)
		ax_lowc.legend(fontsize=7, frameon=False, loc="upper right",
									 bbox_to_anchor=(1.02, 1.0))

		# ── BL: Density hexbin ────────────────────────────────────────────────────
		_draw_triangle(ax_hex)
		_draw_equal_prob_lines(ax_hex)

		# Mask points outside the triangle (clip to valid barycentric region)
		hb = ax_hex.hexbin(
				df["_x"].values, df["_y"].values,
				gridsize=35, cmap="YlOrRd",
				bins="log", mincnt=1,
				alpha=0.75, zorder=4,
				extent=(-0.05, 1.05, -0.05, SQRT3_2 + 0.05),
		)
		plt.colorbar(hb, ax=ax_hex, fraction=0.035, pad=0.02,
								 label="log(count)", shrink=0.75)
		ax_hex.set_title(
				"Density heatmap (log-count hexbin)\nall GMM-routed samples",
				fontsize=8.5, fontweight="bold",
		)

		# ── BC: Small multiples — one mini-ternary per heuristic regime ───────────
		ax_small.axis("off")
		HEURISTIC_REGIMES = ["AGREEMENT", "SOFT_CONFLICT", "HARD_CONFLICT"]
		n_hr = len(HEURISTIC_REGIMES)

		# Manually place 3 sub-axes inside ax_small's bounding box
		pos = ax_small.get_position()   # in figure coords
		sub_w = pos.width  / n_hr
		sub_h = pos.height

		sub_axes = []
		for i in range(n_hr):
				sub_ax = fig.add_axes([
						pos.x0 + i * sub_w + sub_w * 0.05,
						pos.y0,
						sub_w * 0.90,
						sub_h,
				])
				sub_axes.append(sub_ax)

		for sub_ax, h_regime in zip(sub_axes, HEURISTIC_REGIMES):
				_draw_triangle(sub_ax)
				_draw_equal_prob_lines(sub_ax)

				sub_df = df[df["heuristic_regime"] == h_regime]
				n_sub  = len(sub_df)

				for f_regime in TERNARY_REGIMES:
						sub_f = sub_df[sub_df["regime"] == f_regime]
						if sub_f.empty: continue
						color = REGIME_COLORS.get(f_regime, "#607D8B")
						idx   = rng.choice(len(sub_f),
															 size=min(len(sub_f), max_scatter // n_hr),
															 replace=False)
						sub_s = sub_f.iloc[idx]
						sub_ax.scatter(
								sub_s["_x"], sub_s["_y"],
								c=color, s=8, alpha=0.35, edgecolors="none",
								zorder=5,
						)

				short = h_regime.replace("_", "\n")
				sub_ax.set_title(
						f"Heuristic:\n{short}\n(n={n_sub:,})",
						fontsize=7, fontweight="bold", pad=2,
				)

		# Shared legend for small multiples
		legend_handles = [
				plt.Line2D([0], [0], marker="o", color="w",
									 markerfacecolor=REGIME_COLORS.get(r, "#607D8B"),
									 markersize=7, label=r.replace("_", " "))
				for r in TERNARY_REGIMES
		]
		sub_axes[-1].legend(
				handles=legend_handles,
				fontsize=6.5, frameon=False,
				loc="lower right",
				title="GMM regime", title_fontsize=6.5,
				bbox_to_anchor=(1.05, -0.02),
		)

		# ── BR: Margin histogram ──────────────────────────────────────────────────
		# ax_margin is a normal Cartesian axis
		bins_m = np.linspace(0, 1, 31)
		for regime in TERNARY_REGIMES:
				sub   = df[df["regime"] == regime]
				if sub.empty: continue
				color = REGIME_COLORS.get(regime, "#607D8B")
				ax_margin.hist(
						sub["_margin"].values, bins=bins_m,
						color=color, alpha=0.50, density=True, edgecolor="none",
						label=f"{regime}  (n={len(sub):,})",
				)
				med = float(sub["_margin"].median())
				ax_margin.axvline(med, color=color, linewidth=1.3,
													linestyle="--", alpha=0.80)

		# Ambiguity threshold lines
		for mv, ls in [(0.10, ":"), (0.20, "--"), (0.30, "-.")]:
				ax_margin.axvline(mv, color="#37474F", linewidth=1.0,
													linestyle=ls, alpha=0.60, label=f"margin={mv}")

		# Annotate % ambiguous (margin < 0.20)
		n_ambig = int((df["_margin"] < 0.20).sum())
		pct_amb = n_ambig / max(n_routed, 1) * 100
		ax_margin.text(
				0.10, 0.92,
				f"{n_ambig:,} samples\n({pct_amb:.1f}%) margin < 0.20",
				transform=ax_margin.transAxes,
				fontsize=8, color="#c62828", fontweight="bold",
				va="top", ha="left",
		)

		ax_margin.set_xlabel("Decision margin  (p_max − p_2nd)", fontsize=9)
		ax_margin.set_ylabel("Density", fontsize=9)
		ax_margin.set_title(
				"Decision Margin Distribution\n(p_max − p_2nd per regime)",
				fontsize=8.5, fontweight="bold",
		)
		ax_margin.legend(fontsize=7, frameon=False, ncol=2)
		ax_margin.spines[["top", "right"]].set_visible(False)

		# ── Footer ────────────────────────────────────────────────────────────────
		footer = (
				f"Total samples: {n_total:,}  |  "
				f"GMM-routed: {n_routed:,} ({n_routed / max(n_total, 1) * 100:.1f}%)  |  "
				f"Overrides: {n_overrides:,} ({n_overrides / max(n_routed, 1) * 100:.1f}%)  |  "
				f"Low-conf (< {conf_threshold}): {n_low_conf:,} ({n_low_conf / max(n_routed, 1) * 100:.1f}%)  |  "
				f"Mean margin: {mean_margin:.4f}  |  "
				f"feature_dim={feat_dim_str}"
		)
		fig.text(
				0.5, -0.01,
				footer,
				ha="center", va="top", fontsize=8,
				color="#444", style="italic",
				transform=fig.transFigure,
		)

		_save(fig, out_dir, "V14_gmm_probability_simplex")
		print(
				f"[VIZ][V14] Done. "
				f"n_routed={n_routed:,} | "
				f"n_overrides={n_overrides:,} | "
				f"n_low_conf={n_low_conf:,} | "
				f"mean_margin={mean_margin:.4f} | "
				f"feature_dim={feat_dim_str}"
		)

def set_seeds(seed: int = 42):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
				torch.cuda.manual_seed_all(seed)

def main():
	parser = argparse.ArgumentParser(
		description="Regime-Aware Consolidation Visualization Suite",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--audit_jsonl", "-jsonl",
		type=str,
		required=True,
		help="Path to Stage 2 Evidence Receipt JSONL (e.g. test_mlm_cot_modality_conflict_audit.jsonl)",
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
		default=15,
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
	all_viz = {"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12"}
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

	gmm_pkl = os.path.join(OUTPUT_DIR, AUDIT_FILE.replace(".jsonl", "_conflict_gmm.pkl"))
	if not os.path.exists(gmm_pkl) and "V10" in run_set:
		print(f"[VIZ][SKIP] gmm_pkl not found at {gmm_pkl} → skipping V10.")
		run_set.discard("V10")

	stage4_parquet = os.path.join(OUTPUT_DIR, AUDIT_FILE.replace(".jsonl", "_auditable_matrix.parquet"))
	if not os.path.exists(stage4_parquet):
		for v in ("V6", "V7", "V8"):
			if v in run_set:
				print(f"[VIZ][SKIP] --stage4_parquet: {stage4_parquet} not provided → skipping {v}.")
				run_set.discard(v)

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
			audit_jsonl					= args.audit_jsonl,
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

	if "V9" in run_set:
		viz_semantic_asymmetry(args.audit_jsonl, out_dir=VIZ_DIR,)

	if os.path.exists(gmm_pkl):
		viz_gmm_feature_space(
			audit_jsonl=args.audit_jsonl,
			gmm_pkl=gmm_pkl,
			out_dir=VIZ_DIR,
		)
		viz_bic_aic(gmm_pkl=gmm_pkl, out_dir=VIZ_DIR,)

	stage4_parquet = os.path.join(OUTPUT_DIR, AUDIT_FILE.replace(".jsonl", "_auditable_matrix.parquet"))
	if os.path.exists(stage4_parquet):
		viz_heuristic_gmm_confusion(stage4_parquet=stage4_parquet, out_dir=VIZ_DIR,)
		viz_gmm_confidence_diagnostics(stage4_parquet=stage4_parquet, out_dir=VIZ_DIR,)
		viz_gmm_probability_simplex(stage4_parquet=stage4_parquet, out_dir=VIZ_DIR,)

if __name__ == "__main__":
	main()