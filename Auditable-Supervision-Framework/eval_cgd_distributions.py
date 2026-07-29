# Micro-CGD Distribution & Vocabulary-Compression Diagnostics
# (Proving Claim A / Pillar 2: "Supervision Provenance is Real")

# Consumes the Stage 2 modality-conflict audit JSONL and produces the
# empirical receipts that the Micro-CGD audit (Coherence / Grounding /
# Density) carries genuine, regime-separating signal — and that the
# Stage 3→4 consolidation compresses the raw extracted vocabulary into a
# tighter canonical vocabulary V.
#
# Three deliverables (one per sub-claim of Pillar 2)
# ────────────────────────────────────────────────────────────────────────
#   (i)   Violin plots of C / G / D per regime           → the audit SEPARATES
#   (ii)  Vocabulary compression ratio Φ                 → the audit PRUNES
#   (iii) Pareto sweep (Φ vs mean-Grounding)             → the audit is TUNABLE
#
# The three CGD axes (from the Stage 2 Evidence_Receipt / metrics dict):
#   • C — Coherence : internal semantic consistency of the extracted concept set
#   • G — Grounding : cross-modal support (is the text concept visible in image?)
#   • D — Density   : concept saturation / redundancy of the raw extraction
#
# Vocabulary-Compression ratio:
#             |V_canonical|              # concepts surviving Stage 3→4 consolidation
#   Φ = 1  −  ──────────────      Φ ∈ [0, 1],  higher Φ = more aggressive pruning
#             |V_raw|                    # concepts emitted by the Stage 1 VLM
#
# Pareto sweep:
#   Sweep a Grounding threshold τ_G over [0, 1]; for each τ_G keep only
#   concepts with G ≥ τ_G, recompute Φ(τ_G) and the mean retained Grounding.
#   The knee of the (Φ, mean-G) frontier is the audit's operating point.
#
# Outputs (all under <ddir>/outputs/)
# ────────────────────────────────────────────────────────────────────────
#   cgd_violin_per_regime.png        — C/G/D violins split by regime
#   cgd_distributions_kde.png        — marginal KDEs of C/G/D (all samples)
#   cgd_vocab_compression.png        — |V_raw| vs |V_canon| bar + Φ annotation
#   cgd_pareto_sweep.png             — Φ(τ_G) vs mean-Grounding frontier
#   cgd_distributions_results.json   — per-regime moments, Φ, Pareto table
#   cgd_summary.tex                  — LaTeX table: per-regime C/G/D means + Φ
#
# Design contract
# ────────────────────────────────────────────────────────────────────────
# • Mirrors eval_gmm_diagnostics.py exactly: same sys.path injection,
#   `from utils import *`, JSONL-first loader that explodes the nested
#   `metrics` dict, alias-probing column resolver, tree-style logging,
#   `-csv` / `-v` short flags, and path derivation from --metadata.
# • The script NEVER assumes a literal field name: C/G/D and the vocab
#   counts are resolved via alias probes and fail loudly (KeyError listing
#   available columns) if absent — so you learn immediately whether Stage 2
#   needs to persist a field.

# how to run:
# python eval_cgd_distributions.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv -v

import os
import sys

HOME = os.getenv("HOME", "")
USER = os.getenv("USER", "")
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

for _d in [
	os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip"),
	os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc"),
	os.path.join(IMACCESS_PROJECT_WORKSPACE, "historyCLIP"),
	os.path.join(IMACCESS_PROJECT_WORKSPACE, "Auditable-Supervision-Framework"),
]:
	if _d not in sys.path:
		sys.path.insert(0, _d)

from utils import *

# ── The three Micro-CGD axes we expect in the Stage 2 receipt ──────────────
CGD_AXES = ["coherence", "grounding", "density"]

# Alternative column names to probe (Stage 2 may serialise them differently)
CGD_ALIASES = {
	"coherence": ["coherence", "C", "c_score", "coherence_score", "micro_C", "cgd_C"],
	"grounding": ["grounding", "G", "g_score", "grounding_score", "micro_G", "cgd_G"],
	"density":   ["density",   "D", "d_score", "density_score",   "micro_D", "cgd_D"],
}

CGD_DISPLAY = {
	"coherence": "Coherence (C)",
	"grounding": "Grounding (G)",
	"density":   "Density (D)",
}
CGD_COLORS = {
	"coherence": "#1f77b4",   # blue
	"grounding": "#2ca02c",   # green
	"density":   "#9467bd",   # purple
}

# ── Vocabulary count aliases (raw extraction vs canonical consolidated) ────
# Raw   : concepts the Stage 1 VLM emitted for the sample (pre-consolidation)
# Canon : concepts that survived Stage 3→4 (== positive_targets on the matrix)
VOCAB_RAW_ALIASES = [
	"n_raw_concepts", "raw_vocab_size", "n_extracted", "n_raw_labels",
	"raw_concepts", "extracted_concepts", "n_concepts_raw",
]
VOCAB_CANON_ALIASES = [
	"n_canonical_concepts", "canon_vocab_size", "n_positive_targets",
	"n_canonical", "positive_targets", "canonical_concepts", "n_concepts_canon",
]
# Per-concept grounding list — needed for the Pareto sweep. If Stage 2 stored
# a per-concept grounding vector we can prune concept-by-concept; otherwise we
# fall back to the sample-level scalar G for a coarser sweep.
CONCEPT_GROUNDING_ALIASES = [
	"concept_grounding", "per_concept_grounding", "grounding_per_concept",
	"g_per_concept", "concept_g_scores",
]

VALID_REGIMES = ["AGREEMENT", "SOFT_CONFLICT", "HARD_CONFLICT"]
SKIP_REGIMES  = {"MISSING_MODALITY", "INVALID_JSON"}

REGIME_COLORS = {
	"AGREEMENT":     "#2ca02c",
	"SOFT_CONFLICT": "#ff7f0e",
	"HARD_CONFLICT": "#d62728",
}
REGIME_DISPLAY = {
	"AGREEMENT":     "Agreement",
	"SOFT_CONFLICT": "Soft Conflict",
	"HARD_CONFLICT": "Hard Conflict",
}

PARETO_TAUS = np.linspace(0.0, 1.0, 41)   # Grounding-threshold sweep grid
SUBSAMPLE_SEED = 42

# ────────────────────────────────────────────────────────────────────────
# Loading  (mirrors eval_gmm_diagnostics.py _load_df_from_source)
# ────────────────────────────────────────────────────────────────────────
def _load_df_from_source(source_path: str, verbose: bool = True) -> pd.DataFrame:
	"""
	Load a DataFrame from a Stage 2 JSONL audit file (preferred) or parquet.
	For JSONL, the nested `metrics` sub-dict is exploded into flat columns so
	that coherence / grounding / density are directly accessible.
	"""
	ext = os.path.splitext(source_path)[1].lower()

	if ext == ".jsonl":
		if verbose:
			print(f"\n[load_df_from_source] Reading JSONL: {source_path}")
		records = []
		with open(source_path, "r", encoding="utf-8") as fh:
			for line in fh:
				line = line.strip()
				if not line:
					continue
				try:
					records.append(json.loads(line))
				except json.JSONDecodeError:
					continue
		df = pd.DataFrame(records)

		if "metrics" in df.columns:
			metrics_flat = pd.json_normalize(
				df["metrics"].apply(lambda x: x if isinstance(x, dict) else {})
			)
			new_cols = [c for c in metrics_flat.columns if c not in df.columns]
			df = pd.concat([df.drop(columns=["metrics"]), metrics_flat[new_cols]], axis=1)

		if verbose:
			print(f"  ├─ records : {len(df):,}")
			print(f"  └─ columns : {list(df.columns)}")
		return df

	else:
		if verbose:
			print(f"\n[load_df_from_source] Reading parquet: {source_path}")
		df = pd.read_parquet(source_path)
		if verbose:
			print(f"  ├─ df      : {df.shape}")
			print(f"  └─ Columns : {list(df.columns)}")
		return df

def _resolve_columns(df: pd.DataFrame, alias_map: Dict[str, List[str]], verbose: bool = True, tag: str = "") -> Dict[str, str]:
	"""Map each canonical name → the first alias present in df.columns."""
	resolved: Dict[str, str] = {}
	for canon, aliases in alias_map.items():
		for cand in aliases:
			if cand in df.columns:
				resolved[canon] = cand
				break
	if verbose:
		print(f"\n[resolve_columns] {tag} resolution:")
		for canon in alias_map:
			print(f"  ├─ {canon:<16s} → {resolved.get(canon, '(not found)')}")
	return resolved

def _resolve_single(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
	for cand in aliases:
		if cand in df.columns:
			return cand
	return None

def _to_len(cell) -> float:
	"""Coerce a list / str-list / scalar count into a float length."""
	if isinstance(cell, (list, np.ndarray)):
		return float(len(cell))
	if isinstance(cell, (int, float, np.integer, np.floating)) and not (isinstance(cell, float) and np.isnan(cell)):
		return float(cell)
	if isinstance(cell, str):
		try:
			parsed = ast.literal_eval(cell)
			return float(len(parsed)) if isinstance(parsed, (list, tuple)) else float(parsed)
		except (ValueError, SyntaxError):
			return np.nan
	return np.nan

def load_cgd_frame(audit_source: str, verbose: bool = True) -> pd.DataFrame:
	"""
	Load the Stage 2 audit file and return a tidy DataFrame with columns:
	  coherence, grounding, density, regime,
	  [n_raw, n_canon]        — present only if the vocab counts were persisted
	  [concept_grounding]     — present only if a per-concept G vector was stored
	Skip-regimes and NaN-CGD rows are removed.
	"""
	assert os.path.isfile(audit_source), f"[load_cgd_frame] Not found: {audit_source}"
	df = _load_df_from_source(audit_source, verbose=verbose)

	# Resolve the three CGD axes
	resolved = _resolve_columns(df, CGD_ALIASES, verbose=verbose, tag="Micro-CGD axes")
	missing = [a for a in CGD_AXES if a not in resolved]
	if missing:
		raise KeyError(
			f"[load_cgd_frame] Could not locate CGD axes {missing} in {audit_source}.\n"
			f"Available columns: {list(df.columns)}\n"
			f"→ Ensure Stage 2 persisted coherence / grounding / density "
			f"(flat or inside the 'metrics' key of the JSONL)."
		)

	out = pd.DataFrame({
		"coherence": pd.to_numeric(df[resolved["coherence"]], errors="coerce"),
		"grounding": pd.to_numeric(df[resolved["grounding"]], errors="coerce"),
		"density":   pd.to_numeric(df[resolved["density"]],   errors="coerce"),
	})

	# Regime — prefer heuristic_regime (Stage 2 authority) over regime (Stage 4)
	regime_col = next(
		(c for c in ("heuristic_regime", "regime", "Regime", "regime_label") if c in df.columns),
		None,
	)
	if regime_col is None:
		raise KeyError(f"[load_cgd_frame] No regime column found in {audit_source}")
	out["regime"] = df[regime_col].astype(str).str.upper().str.replace(" ", "_")

	# Vocab counts (optional — enables Φ)
	raw_col   = _resolve_single(df, VOCAB_RAW_ALIASES)
	canon_col = _resolve_single(df, VOCAB_CANON_ALIASES)
	if raw_col is not None:
		out["n_raw"] = df[raw_col].apply(_to_len)
	if canon_col is not None:
		out["n_canon"] = df[canon_col].apply(_to_len)

	# Per-concept grounding (optional — enables fine-grained Pareto sweep)
	cg_col = _resolve_single(df, CONCEPT_GROUNDING_ALIASES)
	if cg_col is not None:
		def _as_list(cell):
			if isinstance(cell, (list, np.ndarray)):
				return list(cell)
			if isinstance(cell, str):
				try:
					v = ast.literal_eval(cell)
					return list(v) if isinstance(v, (list, tuple)) else []
				except (ValueError, SyntaxError):
					return []
			return []
		out["concept_grounding"] = df[cg_col].apply(_as_list)

	if verbose:
		print(f"\n[load_cgd_frame] Optional-column availability:")
		print(f"  ├─ n_raw            : {'yes' if raw_col   else 'NO — Φ from |V| union fallback'} ({raw_col})")
		print(f"  ├─ n_canon          : {'yes' if canon_col else 'NO'} ({canon_col})")
		print(f"  └─ concept_grounding: {'yes' if cg_col    else 'NO — Pareto on sample-level G'} ({cg_col})")

	# Filter skip-regimes + NaN CGD rows
	n_before = len(out)
	out = out[~out["regime"].isin(SKIP_REGIMES)]
	out = out.dropna(subset=CGD_AXES)
	n_after = len(out)

	if verbose:
		print(f"\n[load_cgd_frame] Filtering")
		print(f"  ├─ Before        : {n_before:,}")
		print(f"  ├─ After (valid) : {n_after:,}")
		print(f"  └─ Dropped       : {n_before - n_after:,} (skip-regime / NaN CGD)")
		print(f"\n  Regime distribution:")
		for r, c in out["regime"].value_counts().items():
			print(f"    ├─ {r:<16s}: {c:>7,} ({c/max(n_after,1)*100:5.1f}%)")

	return out.reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────
# (i)  Per-regime C/G/D moments
# ────────────────────────────────────────────────────────────────────────
def compute_cgd_moments(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
	"""Per-regime mean / std / median for each CGD axis."""
	moments: Dict[str, Any] = {}
	for reg in VALID_REGIMES:
		sub = df[df["regime"] == reg]
		if len(sub) == 0:
			continue
		moments[reg] = {"n": int(len(sub))}
		for axis in CGD_AXES:
			moments[reg][axis] = {
				"mean":   float(sub[axis].mean()),
				"std":    float(sub[axis].std()),
				"median": float(sub[axis].median()),
			}
	if verbose:
		print(f"\n[compute_cgd_moments] Per-regime CGD means")
		print(f"  {'Regime':<16s} {'n':>8s} {'C':>10s} {'G':>10s} {'D':>10s}")
		for reg in VALID_REGIMES:
			if reg not in moments:
				continue
			m = moments[reg]
			print(
				f"  {REGIME_DISPLAY.get(reg, reg):<16s} {m['n']:>8,} "
				f"{m['coherence']['mean']:>10.3f} "
				f"{m['grounding']['mean']:>10.3f} "
				f"{m['density']['mean']:>10.3f}"
			)
	return moments

# ────────────────────────────────────────────────────────────────────────
# (ii)  Vocabulary compression ratio Φ
# ────────────────────────────────────────────────────────────────────────
def compute_vocab_compression(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
	"""
	Φ = 1 − |V_canon| / |V_raw|, computed both:
	  • micro  (summed over all samples: Σn_canon / Σn_raw)
	  • macro  (mean of per-sample 1 − n_canon/n_raw)
	Falls back to a sentinel if the vocab counts were not persisted.
	"""
	if "n_raw" not in df.columns or "n_canon" not in df.columns:
		if verbose:
			print(
				f"\n[compute_vocab_compression] ⚠ vocab counts absent — "
				f"cannot compute Φ. Persist n_raw_concepts / n_canonical_concepts "
				f"in Stage 2 to enable this receipt."
			)
		return {"available": False}

	valid = df.dropna(subset=["n_raw", "n_canon"])
	valid = valid[valid["n_raw"] > 0]

	sum_raw   = float(valid["n_raw"].sum())
	sum_canon = float(valid["n_canon"].sum())
	phi_micro = 1.0 - (sum_canon / sum_raw) if sum_raw > 0 else float("nan")

	per_sample_phi = 1.0 - (valid["n_canon"] / valid["n_raw"])
	phi_macro = float(per_sample_phi.mean())

	# Per-regime Φ (micro)
	phi_by_regime: Dict[str, float] = {}
	for reg in VALID_REGIMES:
		sub = valid[valid["regime"] == reg]
		if len(sub) == 0:
			continue
		sr = float(sub["n_raw"].sum())
		sc = float(sub["n_canon"].sum())
		phi_by_regime[reg] = 1.0 - (sc / sr) if sr > 0 else float("nan")

	result = {
		"available":      True,
		"phi_micro":      phi_micro,
		"phi_macro":      phi_macro,
		"sum_raw":        sum_raw,
		"sum_canon":      sum_canon,
		"mean_raw":       float(valid["n_raw"].mean()),
		"mean_canon":     float(valid["n_canon"].mean()),
		"phi_by_regime":  phi_by_regime,
		"per_sample_phi": per_sample_phi.to_numpy(),
	}

	if verbose:
		print(f"\n[compute_vocab_compression] Vocabulary compression")
		print(f"  ├─ ΣV_raw          : {sum_raw:,.0f}")
		print(f"  ├─ ΣV_canon        : {sum_canon:,.0f}")
		print(f"  ├─ Φ (micro)       : {phi_micro:.4f}")
		print(f"  ├─ Φ (macro)       : {phi_macro:.4f}")
		print(f"  ├─ mean |V_raw|    : {result['mean_raw']:.2f} concepts/sample")
		print(f"  ├─ mean |V_canon|  : {result['mean_canon']:.2f} concepts/sample")
		print(f"  └─ Φ by regime     :")
		for reg, phi in phi_by_regime.items():
			print(f"       ├─ {REGIME_DISPLAY.get(reg, reg):<16s}: {phi:.4f}")
	return result

# ────────────────────────────────────────────────────────────────────────
# (iii)  Pareto sweep — Φ(τ_G) vs mean retained Grounding
# ────────────────────────────────────────────────────────────────────────
def compute_pareto_sweep(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
	"""
	Sweep a Grounding threshold τ_G. For each τ_G:
	  • prune concepts with G < τ_G
	  • record retained fraction (→ implied Φ_sweep = 1 − retained_fraction)
	  • record mean Grounding of the retained concepts

	Two granularities:
	  • FINE  : if per-concept grounding vectors are present, prune concept-wise
	  • COARSE: else, treat each sample's scalar G as one "concept" and prune
	            whole samples (still a valid quality/coverage frontier)
	"""
	fine = "concept_grounding" in df.columns and df["concept_grounding"].apply(len).sum() > 0

	if fine:
		# Flatten all per-concept grounding scores into one vector
		all_g = np.concatenate([
			np.asarray(v, dtype=np.float64)
			for v in df["concept_grounding"] if len(v) > 0
		])
		mode = "FINE (per-concept)"
	else:
		all_g = df["grounding"].to_numpy(dtype=np.float64)
		mode = "COARSE (sample-level G)"

	all_g = all_g[~np.isnan(all_g)]
	total = len(all_g)

	taus, retained_frac, mean_g, phi_sweep = [], [], [], []
	for tau in PARETO_TAUS:
		keep = all_g >= tau
		n_keep = int(keep.sum())
		frac = n_keep / total if total > 0 else 0.0
		taus.append(float(tau))
		retained_frac.append(frac)
		phi_sweep.append(1.0 - frac)                       # implied compression
		mean_g.append(float(all_g[keep].mean()) if n_keep > 0 else float("nan"))

	# Knee: max distance from the chord joining first & last points on the
	# (Φ_sweep, mean_G) frontier — the classic Kneedle-style geometric knee.
	P = np.array(phi_sweep)
	M = np.array(mean_g)
	valid = ~np.isnan(M)
	knee_tau = None
	if valid.sum() >= 3:
		Pv, Mv, Tv = P[valid], M[valid], np.array(taus)[valid]
		p0, p1 = np.array([Pv[0], Mv[0]]), np.array([Pv[-1], Mv[-1]])
		chord = p1 - p0
		chord_norm = np.linalg.norm(chord) + 1e-12
		dists = []
		for k in range(len(Pv)):
			pt = np.array([Pv[k], Mv[k]])
			d = np.abs(np.cross(chord, pt - p0)) / chord_norm
			dists.append(d)
		knee_idx = int(np.argmax(dists))
		knee_tau = float(Tv[knee_idx])

	result = {
		"mode":           mode,
		"fine":           bool(fine),
		"total_units":    int(total),
		"taus":           taus,
		"retained_frac":  retained_frac,
		"phi_sweep":      phi_sweep,
		"mean_grounding": mean_g,
		"knee_tau":       knee_tau,
	}

	if verbose:
		print(f"\n[compute_pareto_sweep] Grounding-threshold sweep — {mode}")
		print(f"  ├─ total units     : {total:,}")
		print(f"  ├─ τ grid          : [{PARETO_TAUS[0]:.2f} .. {PARETO_TAUS[-1]:.2f}] ({len(PARETO_TAUS)} pts)")
		print(f"  └─ knee τ_G        : {knee_tau if knee_tau is not None else '—'}")
	return result

# ────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────
def plot_cgd_violins(df: pd.DataFrame, output_path: str) -> None:
	"""One subplot per CGD axis; violin split by regime."""
	fig, axes = plt.subplots(1, 3, figsize=(16, 5))
	present = [r for r in VALID_REGIMES if (df["regime"] == r).any()]
	positions = np.arange(len(present))

	for ax, axis in zip(axes, CGD_AXES):
		data = [df.loc[df["regime"] == r, axis].to_numpy() for r in present]
		parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True, widths=0.8)
		for pc, reg in zip(parts["bodies"], present):
			pc.set_facecolor(REGIME_COLORS.get(reg, "#888888"))
			pc.set_alpha(0.65)
			pc.set_edgecolor("black")
		for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
			if key in parts:
				parts[key].set_color("black")
				parts[key].set_linewidth(1.0)
		ax.set_xticks(positions)
		ax.set_xticklabels([REGIME_DISPLAY.get(r, r) for r in present], rotation=15)
		ax.set_title(CGD_DISPLAY[axis])
		ax.set_ylabel("score")
		ax.grid(axis="y", alpha=0.3)

	fig.suptitle("Micro-CGD Distributions per Modality-Conflict Regime", fontsize=13)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_cgd_violins] {output_path}")

def plot_cgd_kde(df: pd.DataFrame, output_path: str) -> None:
	"""Marginal histograms (density-normalised) of C/G/D over all samples."""
	fig, ax = plt.subplots(figsize=(8, 5))
	for axis in CGD_AXES:
		vals = df[axis].to_numpy()
		vals = vals[~np.isnan(vals)]
		ax.hist(
			vals, bins=50, density=True, histtype="step", linewidth=2,
			color=CGD_COLORS[axis], label=CGD_DISPLAY[axis],
		)
	ax.set_xlabel("score")
	ax.set_ylabel("density")
	ax.set_title("Marginal Distributions of Micro-CGD Axes")
	ax.legend(loc="best")
	ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_cgd_kde] {output_path}")

def plot_vocab_compression(vc: Dict[str, Any], output_path: str) -> None:
	if not vc.get("available", False):
		print(f"[plot_vocab_compression] skipped — vocab counts absent.")
		return
	fig, ax = plt.subplots(figsize=(7, 5))
	labels = ["Raw\n$|V_{raw}|$", "Canonical\n$|V_{canon}|$"]
	vals   = [vc["mean_raw"], vc["mean_canon"]]
	bars = ax.bar(labels, vals, color=["#9467bd", "#2ca02c"], alpha=0.8, edgecolor="black")
	for b, v in zip(bars, vals):
		ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=11)
	ax.set_ylabel("mean concepts / sample")
	ax.set_title(
		f"Vocabulary Compression   "
		f"$\\Phi_{{micro}}$ = {vc['phi_micro']:.3f}   |   "
		f"$\\Phi_{{macro}}$ = {vc['phi_macro']:.3f}"
	)
	ax.grid(axis="y", alpha=0.3)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_vocab_compression] {output_path}")

def plot_pareto_sweep(ps: Dict[str, Any], output_path: str) -> None:
	fig, ax = plt.subplots(figsize=(8, 5))
	phi = np.array(ps["phi_sweep"])
	mg  = np.array(ps["mean_grounding"])
	taus = np.array(ps["taus"])

	sc = ax.scatter(phi, mg, c=taus, cmap="viridis", s=35, zorder=3)
	ax.plot(phi, mg, "-", color="#888888", alpha=0.5, zorder=2)
	cbar = fig.colorbar(sc, ax=ax)
	cbar.set_label(r"grounding threshold $\tau_G$")

	if ps.get("knee_tau") is not None:
		k = int(np.argmin(np.abs(taus - ps["knee_tau"])))
		ax.scatter(
			phi[k], mg[k], s=240, marker="X", edgecolors="black",
			linewidths=1.8, c="#d62728", zorder=5,
			label=fr"knee  $\tau_G$={ps['knee_tau']:.2f}",
		)
		ax.legend(loc="best")

	ax.set_xlabel(r"compression $\Phi_{sweep} = 1 - $ retained fraction")
	ax.set_ylabel("mean retained Grounding")
	ax.set_title(f"Pareto Frontier: Compression vs Grounding  [{ps['mode']}]")
	ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_pareto_sweep] {output_path}")

# ────────────────────────────────────────────────────────────────────────
# Serialisation
# ────────────────────────────────────────────────────────────────────────
def save_json(results: Dict[str, Any], output_path: str) -> None:
	def _clean(obj):
		if isinstance(obj, float) and np.isnan(obj):
			return None
		if isinstance(obj, (np.floating,)):
			return float(obj)
		if isinstance(obj, (np.integer,)):
			return int(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		if isinstance(obj, dict):
			return {k: _clean(v) for k, v in obj.items()}
		if isinstance(obj, list):
			return [_clean(v) for v in obj]
		return obj
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(_clean(results), f, indent=2, ensure_ascii=False)
	print(f"[save_json] {output_path}")

def save_summary_latex(
	moments: Dict[str, Any],
	vc: Dict[str, Any],
	ps: Dict[str, Any],
	output_path: str,
) -> None:
	phi_str = f"{vc['phi_micro']:.3f}" if vc.get("available") else "n/a"
	knee_str = f"{ps['knee_tau']:.2f}" if ps.get("knee_tau") is not None else "n/a"
	lines = [
		r"\begin{table}[t]",
		r"\centering",
		r"\caption{Per-regime Micro-CGD means and vocabulary compression. "
		rf"Global $\Phi_{{micro}}={phi_str}$; Pareto knee $\tau_G={knee_str}$.}}",
		r"\label{tab:cgd_summary}",
		r"\begin{tabular}{lrrrrr}",
		r"\toprule",
		r"Regime & $n$ & Coherence & Grounding & Density & $\Phi$ \\",
		r"\midrule",
	]
	phi_by = vc.get("phi_by_regime", {}) if vc.get("available") else {}
	for reg in VALID_REGIMES:
		if reg not in moments:
			continue
		m = moments[reg]
		phi_r = f"{phi_by[reg]:.3f}" if reg in phi_by else "--"
		lines.append(
			f"{REGIME_DISPLAY.get(reg, reg)} & "
			f"{m['n']:,} & "
			f"{m['coherence']['mean']:.3f} & "
			f"{m['grounding']['mean']:.3f} & "
			f"{m['density']['mean']:.3f} & "
			f"{phi_r} \\\\"
		)
	lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
	with open(output_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"[save_summary_latex] {output_path}")

# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Micro-CGD Distribution & Vocabulary-Compression Diagnostics (Pillar 2)",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument("--metadata", "-csv", required=True, help="Path to dataset.csv (audit path derived from this)")
	p.add_argument("--verbose", "-v", action="store_true")
	return p.parse_args()


def main():
	args = parse_args()
	if args.verbose:
		print(args)

	ddir        = os.path.dirname(args.metadata)
	outputs_dir = os.path.join(ddir, "outputs")
	os.makedirs(outputs_dir, exist_ok=True)

	metadata_file = os.path.basename(args.metadata)
	stem = metadata_file.replace(".csv", "_mlm_cot_modality_conflict_audit")

	# Priority 1 — Stage 2 JSONL (metrics nested under 'metrics' key)
	audit_jsonl   = os.path.join(ddir, f"{stem}.jsonl")
	# Priority 2 — Stage 2 flat parquet
	audit_parquet = os.path.join(outputs_dir, f"{stem}.parquet")
	# Priority 3 — Stage 4 supervision matrix (last resort)
	audit_matrix  = os.path.join(
		outputs_dir,
		metadata_file.replace(".csv", "_mlm_cot_modality_conflict_audit_auditable_supervision_matrix.parquet"),
	)

	if os.path.isfile(audit_jsonl):
		audit_source = audit_jsonl
	elif os.path.isfile(audit_parquet):
		print(f"[main] Stage 2 JSONL not found; using flat audit parquet:\n  {audit_parquet}")
		audit_source = audit_parquet
	else:
		print(f"[main] Stage 2 JSONL/parquet not found; falling back to supervision matrix:\n  {audit_matrix}")
		audit_source = audit_matrix

	print(f"\n{'='*80}")
	print(f"[eval_cgd_distributions] Micro-CGD Distribution & Compression Diagnostics")
	print(f"  ├─ Metadata     : {args.metadata}")
	print(f"  ├─ Audit source : {audit_source}")
	print(f"  └─ Output dir   : {outputs_dir}")
	print(f"{'='*80}")

	# 1. Load tidy CGD frame
	df = load_cgd_frame(audit_source, verbose=args.verbose)

	# 2. Deliverable (i) — per-regime moments + violins
	moments = compute_cgd_moments(df, verbose=args.verbose)
	plot_cgd_violins(df, os.path.join(outputs_dir, "cgd_violin_per_regime.png"))
	plot_cgd_kde(df,     os.path.join(outputs_dir, "cgd_distributions_kde.png"))

	# 3. Deliverable (ii) — vocabulary compression Φ
	vc = compute_vocab_compression(df, verbose=args.verbose)
	plot_vocab_compression(vc, os.path.join(outputs_dir, "cgd_vocab_compression.png"))

	# 4. Deliverable (iii) — Pareto sweep
	ps = compute_pareto_sweep(df, verbose=args.verbose)
	plot_pareto_sweep(ps, os.path.join(outputs_dir, "cgd_pareto_sweep.png"))

	# 5. Serialise
	results = {
		"audit_source":  audit_source,
		"n_samples":     int(len(df)),
		"cgd_axes":      CGD_AXES,
		"moments":       moments,
		"vocab_compression": {k: v for k, v in vc.items() if k != "per_sample_phi"},
		"pareto":        ps,
		"regime_dist":   df["regime"].value_counts().to_dict(),
	}
	save_json(results, os.path.join(outputs_dir, "cgd_distributions_results.json"))
	save_summary_latex(moments, vc, ps, os.path.join(outputs_dir, "cgd_summary.tex"))

	# 6. Headline verdict
	print(f"\n{'='*80}")
	print(f"[VERDICT]  Pillar 2 — Supervision Provenance is Real")
	if "HARD_CONFLICT" in moments and "AGREEMENT" in moments:
		g_agree = moments["AGREEMENT"]["grounding"]["mean"]
		g_hard  = moments["HARD_CONFLICT"]["grounding"]["mean"]
		sep = g_agree - g_hard
		arrow = "✓" if sep > 0 else "⚠"
		print(f"  {arrow}  Grounding separates regimes: "
					f"G(Agreement)={g_agree:.3f}  >  G(Hard)={g_hard:.3f}   (Δ={sep:+.3f})")
	if vc.get("available"):
		print(f"  ✓  Vocabulary compression Φ_micro = {vc['phi_micro']:.3f} "
					f"(mean {vc['mean_raw']:.1f} → {vc['mean_canon']:.1f} concepts/sample)")
	else:
		print(f"  ⚠  Φ unavailable — persist n_raw_concepts / n_canonical_concepts in Stage 2.")
	if ps.get("knee_tau") is not None:
		print(f"  ✓  Pareto knee at τ_G = {ps['knee_tau']:.2f}  [{ps['mode']}]")
	print(f"{'='*80}\n")


if __name__ == "__main__":
	main()