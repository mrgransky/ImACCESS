# Micro-CGD Distribution & Compression Diagnostics
# (Proving Claim A / Pillar 2: "Supervision Provenance is Real")
#
# This script proves that the Stage 3 Micro-CGD audit produces a genuine,
# regime-separating signal and that the Stage 3->4 consolidation measurably
# compresses the raw VLM vocabulary into a canonical target vocabulary.
#
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL DATA-FLOW NOTE (why the previous version failed):
#
#   Stage 2 (stage2_modality_conflict.py) emits ONLY the router metrics
#   {set_similarity, orphan_ratio, asymmetry_gap, entail_*, denser_modality}
#   into  <stem>_mlm_cot_modality_conflict_audit.jsonl.
#   It NEVER computes Coherence / Grounding / Density.
#
#   The Micro-CGD scores are BORN in Stage 3/4 (stage3_4_cgd_consolidation.py),
#   method audit_concept_CGD(), which returns {"C","G","D"} PER CONCEPT.
#   They are persisted to:
#
#     (1) Stage 3 CGD JSONL   [PRIMARY]
#         outputs/<stem>_auditable_supervision_cgd.jsonl
#         record = {doc_url, regime, cgd_scores}
#         cgd_scores[raw_concept] = {C, G, D, canonical, source_modality}
#
#     (2) Stage 4 matrix      [FALLBACK]
#         outputs/<stem>_auditable_supervision_matrix.parquet
#         column `audited_concepts` (JSON string) = same per-concept dict
#         column `positive_targets` = canonical vocabulary retained per sample
#
#   Therefore C/G/D are UPPERCASE, NESTED per-concept, regime-tagged, and live
#   in the Stage 3 file — not the Stage 2 audit this script used to read.
# ─────────────────────────────────────────────────────────────────────────────
#
# Deliverables (Pillar 2)
# ───────────────────────
#   (i)   Violin plots of C / G / D per regime           -> cgd_violin_per_regime.png
#   (ii)  Vocabulary compression ratio  Phi = 1 - |Vc|/|Vr|
#   (iii) Pareto sweep of grounding threshold tau_G       -> cgd_pareto_sweep.png
#
# Outputs (all under <ddir>/outputs/)
# ───────────────────────────────────
#   cgd_violin_per_regime.png     — C/G/D violins split by regime (headline)
#   cgd_distributions_kde.png     — marginal CGD densities
#   cgd_vocab_compression.png     — |V_raw| -> |V_canon| with Phi annotation
#   cgd_pareto_sweep.png          — Phi(tau_G) vs retained-grounding frontier + knee
#   cgd_distributions_results.json— moments, Phi, full Pareto table
#   cgd_summary.tex               — per-regime C/G/D means + Phi LaTeX table
#
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

# ── Micro-CGD axes: literally C / G / D as returned by audit_concept_CGD ─────
CGD_AXES = ["C", "G", "D"]

# Per-concept dicts use uppercase keys; probe a few defensive aliases anyway
# in case a future Stage 3 revision renames them.
CGD_ALIASES = {
	"C": ["C", "coherence", "coverage", "c_score", "Coverage"],
	"G": ["G", "grounding", "g_score", "Grounding"],
	"D": ["D", "density", "d_score", "Density"],
}

CGD_DISPLAY = {
	"C": "Coverage $C$",
	"G": "Grounding $G$",
	"D": "Density $D$",
}

VALID_REGIMES = ["AGREEMENT", "SOFT_CONFLICT", "HARD_CONFLICT"]
SKIP_REGIMES  = {"MISSING_MODALITY", "INVALID_JSON"}

REGIME_COLORS = {
	"AGREEMENT":     "#2ca02c",   # green
	"SOFT_CONFLICT": "#ff7f0e",   # orange
	"HARD_CONFLICT": "#d62728",   # red
}
REGIME_DISPLAY = {
	"AGREEMENT":     "Agreement",
	"SOFT_CONFLICT": "Soft Conflict",
	"HARD_CONFLICT": "Hard Conflict",
}

PARETO_N_STEPS = 41          # tau_G grid resolution over [0, 1]
PLOT_SEED      = 42

# ─────────────────────────────────────────────────────────────────────────────
# Source resolution: Stage 3 CGD JSONL (primary) -> Stage 4 matrix (fallback)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_cgd_source(metadata_path: str, verbose: bool = True) -> str:
	"""
	Derive the Micro-CGD source path from --metadata, mirroring the exact
	naming convention in stage3_4_cgd_consolidation.py.

	Stage 2 audit stem : <meta>_mlm_cot_modality_conflict_audit
	Stage 3 CGD JSONL  : outputs/<stem>_auditable_supervision_cgd.jsonl   [PRIMARY]
	Stage 4 matrix     : outputs/<stem>_auditable_supervision_matrix.parquet [FALLBACK]
	"""
	ddir        = os.path.dirname(metadata_path)
	outputs_dir = os.path.join(ddir, "outputs")
	meta        = os.path.basename(metadata_path)
	stem        = meta.replace(".csv", "_mlm_cot_modality_conflict_audit")

	stage3_jsonl  = os.path.join(outputs_dir, f"{stem}_auditable_supervision_cgd.jsonl")
	stage4_parquet = os.path.join(outputs_dir, f"{stem}_auditable_supervision_matrix.parquet")

	if os.path.isfile(stage3_jsonl):
		return stage3_jsonl
	if os.path.isfile(stage4_parquet):
		if verbose:
			print(f"[resolve_cgd_source] Stage 3 CGD JSONL not found; "
				  f"falling back to Stage 4 matrix:\n  {stage4_parquet}")
		return stage4_parquet

	raise FileNotFoundError(
		f"[resolve_cgd_source] Neither the Stage 3 CGD JSONL nor the Stage 4 "
		f"supervision matrix was found.\n"
		f"  Expected primary  : {stage3_jsonl}\n"
		f"  Expected fallback : {stage4_parquet}\n"
		f"-> Run stage3_4_cgd_consolidation.py first."
	)

# ─────────────────────────────────────────────────────────────────────────────
# Loading & exploding the per-concept CGD records
# ─────────────────────────────────────────────────────────────────────────────
def _coerce_scores_dict(cell) -> Dict[str, Any]:
	"""Return a {concept: {C,G,D,...}} dict from a raw cell (dict or JSON str)."""
	if isinstance(cell, dict):
		return cell
	if isinstance(cell, str):
		try:
			return json.loads(cell)
		except (json.JSONDecodeError, ValueError):
			try:
				return ast.literal_eval(cell)
			except (ValueError, SyntaxError):
				return {}
	return {}

def _resolve_axis_keys(sample_scores: Dict[str, Any], verbose: bool = True) -> Dict[str, str]:
	"""
	Inspect one non-empty per-concept score dict to resolve which literal key
	backs each canonical axis (C/G/D). Fails loudly with the available keys.
	"""
	# grab the first concept's inner dict
	inner = {}
	for _c, _d in sample_scores.items():
		if isinstance(_d, dict) and _d:
			inner = _d
			break

	resolved: Dict[str, str] = {}
	for axis, aliases in CGD_ALIASES.items():
		for cand in aliases:
			if cand in inner:
				resolved[axis] = cand
				break

	if verbose:
		print(f"\n[resolve_axis_keys] Micro-CGD per-concept key resolution:")
		for axis in CGD_AXES:
			print(f"  ├─ {axis} ({CGD_ALIASES[axis][1]:<10s}) → {resolved.get(axis, '(not found)')}")

	missing = [a for a in CGD_AXES if a not in resolved]
	if missing:
		raise KeyError(
			f"[resolve_axis_keys] Could not locate CGD axes {missing} inside the "
			f"per-concept score dict.\nInner keys available: {list(inner.keys())}\n"
			f"-> Stage 3 audit_concept_CGD() must return C / G / D per concept."
		)
	return resolved

def load_cgd_records(source_path: str, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Load the Stage 3 CGD JSONL (or Stage 4 matrix) and build TWO long frames:

	  concept_df  — one row PER (sample, concept):
	                {doc_url, regime, concept, canonical, source_modality, C, G, D}
	  sample_df   — one row PER sample:
	                {doc_url, regime, n_raw, n_canonical, C_mean, G_mean, D_mean}

	These back the violins (pooled per-concept), the Phi compression metric
	(per-sample raw vs canonical counts), and the Pareto sweep (per-concept G).
	"""
	assert os.path.isfile(source_path), f"[load_cgd_records] Not found: {source_path}"
	ext = os.path.splitext(source_path)[1].lower()

	# ── Read raw records into a frame with {doc_url, regime, scores_dict} ─────
	rows: List[Dict[str, Any]] = []
	if ext == ".jsonl":
		if verbose:
			print(f"\n[load_cgd_records] Reading Stage 3 CGD JSONL: {source_path}")
		with open(source_path, "r", encoding="utf-8") as fh:
			for line in fh:
				line = line.strip()
				if not line:
					continue
				try:
					rec = json.loads(line)
				except json.JSONDecodeError:
					continue
				rows.append({
					"doc_url": rec.get("doc_url"),
					"regime":  rec.get("regime"),
					"scores":  _coerce_scores_dict(rec.get("cgd_scores", {})),
				})
	else:
		if verbose:
			print(f"\n[load_cgd_records] Reading Stage 4 matrix parquet: {source_path}")
		pf = pd.read_parquet(source_path)
		# Stage 4 stores the per-concept dict in `audited_concepts` (JSON string)
		aud_col = next(
			(c for c in ("audited_concepts", "cgd_scores") if c in pf.columns), None
		)
		if aud_col is None:
			raise KeyError(
				f"[load_cgd_records] Stage 4 matrix lacks an 'audited_concepts' "
				f"column.\nAvailable: {list(pf.columns)}"
			)
		regime_col = "regime" if "regime" in pf.columns else "heuristic_regime"
		for _, r in pf.iterrows():
			rows.append({
				"doc_url": r.get("doc_url"),
				"regime":  r.get(regime_col),
				"scores":  _coerce_scores_dict(r.get(aud_col, {})),
			})

	if verbose:
		print(f"  ├─ records : {len(rows):,}")

	# ── Resolve the literal C/G/D keys from the first non-empty record ────────
	first_scores = next((r["scores"] for r in rows if r["scores"]), {})
	if not first_scores:
		raise ValueError(
			f"[load_cgd_records] No non-empty cgd_scores found in {source_path}. "
			f"Did Stage 3 write per-concept audits?"
		)
	axis_keys = _resolve_axis_keys(first_scores, verbose=verbose)
	kC, kG, kD = axis_keys["C"], axis_keys["G"], axis_keys["D"]

	# ── Explode to concept-level and aggregate to sample-level ────────────────
	concept_rows: List[Dict[str, Any]] = []
	sample_rows:  List[Dict[str, Any]] = []

	for r in rows:
		regime = str(r["regime"]).upper().replace(" ", "_") if r["regime"] else "UNKNOWN"
		if regime in SKIP_REGIMES:
			continue
		scores = r["scores"]
		if not scores:
			continue

		canon_set: set = set()
		cvals, gvals, dvals = [], [], []
		n_raw = 0
		for concept, sd in scores.items():
			if not isinstance(sd, dict):
				continue
			n_raw += 1
			c = sd.get(kC, np.nan)
			g = sd.get(kG, np.nan)
			d = sd.get(kD, np.nan)
			canon = sd.get("canonical")
			if canon is not None:
				canon_set.add(canon)
			concept_rows.append({
				"doc_url":         r["doc_url"],
				"regime":          regime,
				"concept":         concept,
				"canonical":       canon,
				"source_modality": sd.get("source_modality"),
				"C": c, "G": g, "D": d,
			})
			cvals.append(c); gvals.append(g); dvals.append(d)

		if n_raw == 0:
			continue
		sample_rows.append({
			"doc_url":     r["doc_url"],
			"regime":      regime,
			"n_raw":       n_raw,
			"n_canonical": len(canon_set),
			"C_mean":      float(np.nanmean(cvals)) if cvals else np.nan,
			"G_mean":      float(np.nanmean(gvals)) if gvals else np.nan,
			"D_mean":      float(np.nanmean(dvals)) if dvals else np.nan,
		})

	concept_df = pd.DataFrame(concept_rows)
	sample_df  = pd.DataFrame(sample_rows)

	# Coerce numeric
	for ax in CGD_AXES:
		concept_df[ax] = pd.to_numeric(concept_df[ax], errors="coerce")

	if verbose:
		print(f"  ├─ concept-level rows : {len(concept_df):,}")
		print(f"  ├─ sample-level rows  : {len(sample_df):,}")
		print(f"\n  Regime distribution (samples):")
		for rr, cc in sample_df["regime"].value_counts().items():
			print(f"    ├─ {rr:<16s}: {cc:>7,} ({cc/max(len(sample_df),1)*100:5.1f}%)")
		print(f"\n  Regime distribution (concepts):")
		for rr, cc in concept_df["regime"].value_counts().items():
			print(f"    ├─ {rr:<16s}: {cc:>7,} ({cc/max(len(concept_df),1)*100:5.1f}%)")

	return concept_df, sample_df

# ─────────────────────────────────────────────────────────────────────────────
# (i) Per-regime CGD moments + violin plots
# ─────────────────────────────────────────────────────────────────────────────
def compute_cgd_moments(concept_df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
	"""Per-regime mean/std/median for each CGD axis (pooled over concepts)."""
	moments: Dict[str, Any] = {}
	for reg in VALID_REGIMES:
		sub = concept_df[concept_df["regime"] == reg]
		if len(sub) == 0:
			continue
		moments[reg] = {}
		for ax in CGD_AXES:
			vals = sub[ax].dropna().to_numpy()
			moments[reg][ax] = {
				"mean":   float(np.mean(vals)) if len(vals) else float("nan"),
				"std":    float(np.std(vals))  if len(vals) else float("nan"),
				"median": float(np.median(vals)) if len(vals) else float("nan"),
				"n":      int(len(vals)),
			}
	if verbose:
		print(f"\n[compute_cgd_moments] Per-regime CGD means (pooled over concepts):")
		hdr = "  " + " " * 16 + "".join(f"{CGD_DISPLAY[a].split('$')[1]:>14s}" for a in CGD_AXES)
		print(hdr)
		for reg in VALID_REGIMES:
			if reg not in moments:
				continue
			row = "  " + f"{REGIME_DISPLAY[reg]:<16s}"
			for ax in CGD_AXES:
				row += f"{moments[reg][ax]['mean']:>14.4f}"
			print(row)
	return moments

def plot_cgd_violins(concept_df: pd.DataFrame, output_path: str) -> None:
	"""One panel per CGD axis; violins split by regime, with mean + median."""
	fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
	regimes_present = [r for r in VALID_REGIMES if (concept_df["regime"] == r).any()]

	for ax_i, axis in enumerate(CGD_AXES):
		ax = axes[ax_i]
		data, colors = [], []
		for reg in regimes_present:
			vals = concept_df.loc[concept_df["regime"] == reg, axis].dropna().to_numpy()
			data.append(vals if len(vals) else np.array([np.nan]))
			colors.append(REGIME_COLORS[reg])

		parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
		for pc, col in zip(parts["bodies"], colors):
			pc.set_facecolor(col)
			pc.set_edgecolor("black")
			pc.set_alpha(0.55)

		# Overlay mean (diamond) + median (bar)
		for i, vals in enumerate(data, start=1):
			v = vals[~np.isnan(vals)]
			if len(v) == 0:
				continue
			ax.scatter(i, np.mean(v),  marker="D", s=45, color="black", zorder=5)
			ax.hlines(np.median(v), i - 0.18, i + 0.18, color="white", linewidth=2, zorder=6)

		ax.set_xticks(range(1, len(regimes_present) + 1))
		ax.set_xticklabels([REGIME_DISPLAY[r] for r in regimes_present], rotation=12)
		ax.set_title(CGD_DISPLAY[axis])
		ax.set_ylabel("score")
		ax.grid(axis="y", alpha=0.25)

	fig.suptitle("Micro-CGD Score Distributions by Conflict Regime "
				 "(◆ mean, — median)", fontsize=13)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_cgd_violins] {output_path}")

def plot_cgd_kde(concept_df: pd.DataFrame, output_path: str) -> None:
	"""Marginal KDE-style densities of each axis, split by regime."""
	from scipy.stats import gaussian_kde
	fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
	grid = np.linspace(0, 1, 200)
	for ax_i, axis in enumerate(CGD_AXES):
		ax = axes[ax_i]
		for reg in VALID_REGIMES:
			vals = concept_df.loc[concept_df["regime"] == reg, axis].dropna().to_numpy()
			if len(vals) < 5 or np.std(vals) < 1e-6:
				continue
			try:
				kde = gaussian_kde(vals)
				ax.plot(grid, kde(grid), color=REGIME_COLORS[reg],
						label=REGIME_DISPLAY[reg], linewidth=2)
				ax.fill_between(grid, kde(grid), color=REGIME_COLORS[reg], alpha=0.12)
			except np.linalg.LinAlgError:
				continue
		ax.set_title(CGD_DISPLAY[axis])
		ax.set_xlabel("score")
		ax.set_ylabel("density")
		ax.grid(alpha=0.25)
	axes[0].legend(loc="best")
	fig.suptitle("Marginal Micro-CGD Densities by Regime", fontsize=13)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_cgd_kde] {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# (ii) Vocabulary compression ratio  Phi = 1 - |V_canon| / |V_raw|
# ─────────────────────────────────────────────────────────────────────────────
def compute_vocab_compression(
	concept_df: pd.DataFrame,
	sample_df: pd.DataFrame,
	verbose: bool = True,
) -> Dict[str, Any]:
	"""
	Two complementary views of the Stage 3->4 vocabulary compression:

	  MICRO (corpus-level) : Phi = 1 - |V_canon_global| / |V_raw_global|
	      |V_raw_global|   = number of distinct raw VLM concepts audited
	      |V_canon_global| = number of distinct canonical concepts retained
	  MACRO (per-sample)   : mean over samples of (1 - n_canonical / n_raw)

	Also a per-regime micro Phi breakdown.
	"""
	# Global distinct counts
	raw_global   = concept_df["concept"].nunique()
	canon_global = concept_df["canonical"].dropna().nunique()
	phi_micro    = 1.0 - (canon_global / raw_global) if raw_global else float("nan")

	# Macro: mean per-sample compression (guard n_raw==0)
	valid = sample_df[sample_df["n_raw"] > 0].copy()
	valid["phi_sample"] = 1.0 - (valid["n_canonical"] / valid["n_raw"])
	phi_macro = float(valid["phi_sample"].mean()) if len(valid) else float("nan")

	# Per-regime micro Phi
	per_regime: Dict[str, Any] = {}
	for reg in VALID_REGIMES:
		sub = concept_df[concept_df["regime"] == reg]
		if len(sub) == 0:
			continue
		r_raw   = sub["concept"].nunique()
		r_canon = sub["canonical"].dropna().nunique()
		per_regime[reg] = {
			"raw":   int(r_raw),
			"canon": int(r_canon),
			"phi":   1.0 - (r_canon / r_raw) if r_raw else float("nan"),
		}

	result = {
		"V_raw_global":   int(raw_global),
		"V_canon_global": int(canon_global),
		"phi_micro":      phi_micro,
		"phi_macro":      phi_macro,
		"per_regime":     per_regime,
	}

	if verbose:
		print(f"\n[compute_vocab_compression] Stage 3->4 vocabulary compression")
		print(f"  ├─ |V_raw|   (distinct raw concepts)      : {raw_global:,}")
		print(f"  ├─ |V_canon| (distinct canonical concepts): {canon_global:,}")
		print(f"  ├─ Phi_micro = 1 - |Vc|/|Vr|              : {phi_micro:.4f} "
			  f"({phi_micro*100:.1f}% compression)")
		print(f"  ├─ Phi_macro (mean per-sample)            : {phi_macro:.4f}")
		print(f"  └─ Per-regime Phi:")
		for reg, v in per_regime.items():
			print(f"       {REGIME_DISPLAY[reg]:<16s}: "
				  f"raw={v['raw']:>6,} canon={v['canon']:>6,} Phi={v['phi']:.4f}")

	return result

def plot_vocab_compression(vocab: Dict[str, Any], output_path: str) -> None:
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

	# Left: global raw -> canonical bar
	ax1.bar(["$|V_{raw}|$", "$|V_{canon}|$"],
			[vocab["V_raw_global"], vocab["V_canon_global"]],
			color=["#7f7f7f", "#1f77b4"], edgecolor="black", alpha=0.85)
	ax1.set_ylabel("distinct concepts")
	ax1.set_title(f"Corpus Vocabulary Compression\n"
				  f"$\\Phi_{{micro}} = {vocab['phi_micro']:.3f}$ "
				  f"({vocab['phi_micro']*100:.1f}% reduction)")
	for i, val in enumerate([vocab["V_raw_global"], vocab["V_canon_global"]]):
		ax1.text(i, val, f"{val:,}", ha="center", va="bottom", fontweight="bold")
	ax1.grid(axis="y", alpha=0.25)

	# Right: per-regime Phi
	pr = vocab["per_regime"]
	regs = [r for r in VALID_REGIMES if r in pr]
	ax2.bar([REGIME_DISPLAY[r] for r in regs],
			[pr[r]["phi"] for r in regs],
			color=[REGIME_COLORS[r] for r in regs], edgecolor="black", alpha=0.85)
	ax2.set_ylabel("$\\Phi$ (per-regime)")
	ax2.set_title("Compression Ratio by Regime")
	ax2.set_ylim(0, 1)
	for i, r in enumerate(regs):
		ax2.text(i, pr[r]["phi"], f"{pr[r]['phi']:.3f}",
				 ha="center", va="bottom", fontweight="bold")
	ax2.grid(axis="y", alpha=0.25)

	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_vocab_compression] {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# (iii) Pareto sweep of grounding threshold tau_G
# ─────────────────────────────────────────────────────────────────────────────
def compute_pareto_sweep(
    concept_df: pd.DataFrame,
    n_steps: int = PARETO_N_STEPS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fine-grained (per-concept) sweep of the grounding threshold tau_G.
    ...
    """
    g = concept_df["G"].dropna().to_numpy()
    if len(g) == 0:
        raise ValueError("[compute_pareto_sweep] No grounding (G) values available.")

    canon_all = concept_df.dropna(subset=["canonical"])
    total_canon = canon_all["canonical"].nunique()
    n_total = len(g)

    taus = np.linspace(0.0, 1.0, n_steps)
    retained_frac, compression, mean_g_kept, canon_frac = [], [], [], []

    for t in taus:
        keep = g >= t
        rf = keep.mean()
        retained_frac.append(float(rf))
        compression.append(float(1.0 - rf))
        mean_g_kept.append(float(g[keep].mean()) if keep.any() else float("nan"))
        if total_canon:
            kept_canon = canon_all.loc[canon_all["G"].to_numpy() >= t, "canonical"].nunique()
            canon_frac.append(float(kept_canon / total_canon))
        else:
            canon_frac.append(float("nan"))

    # ── Kneedle knee on (compression, mean_grounding_kept) frontier ──────────
    x = np.array(compression, dtype=np.float64)
    y = np.array(mean_g_kept, dtype=np.float64)
    valid = ~np.isnan(y)
    knee_tau, knee_idx = float("nan"), None
    if valid.sum() >= 3:
        xv, yv, tv = x[valid], y[valid], taus[valid]
        # normalise both axes to [0,1] before measuring distance from chord
        # FIX: Replace deprecated .ptp() with np.ptp() or manual calculation
        xn = (xv - xv.min()) / (np.ptp(xv) + 1e-12)
        yn = (yv - yv.min()) / (np.ptp(yv) + 1e-12)
        # Alternative manual calculation if np.ptp is also unavailable:
        # xn = (xv - xv.min()) / ((xv.max() - xv.min()) + 1e-12)
        # yn = (yv - yv.min()) / ((yv.max() - yv.min()) + 1e-12)
        x0, y0, x1, y1 = xn[0], yn[0], xn[-1], yn[-1]
        denom = np.hypot(x1 - x0, y1 - y0) + 1e-12
        dist = np.abs((y1 - y0) * xn - (x1 - x0) * yn + x1 * y0 - y1 * x0) / denom
        k = int(np.argmax(dist))
        knee_idx = int(np.where(valid)[0][k])
        knee_tau = float(tv[k])

    result = {
        "mode":              "fine_per_concept",
        "n_concepts":        int(n_total),
        "total_canonical":   int(total_canon),
        "tau_grid":          taus.tolist(),
        "retained_frac":     retained_frac,
        "compression":       compression,
        "mean_grounding_kept": mean_g_kept,
        "canon_retained_frac": canon_frac,
        "knee_tau":          knee_tau,
        "knee_idx":          knee_idx,
    }

    if verbose:
        print(f"\n[compute_pareto_sweep] tau_G sweep (fine / per-concept)")
        print(f"  ├─ concepts swept : {n_total:,}")
        print(f"  ├─ grid points    : {n_steps}")
        if knee_idx is not None:
            print(f"  └─ knee at tau_G  = {knee_tau:.3f}  "
                  f"(compression={compression[knee_idx]:.3f}, "
                  f"mean_G_kept={mean_g_kept[knee_idx]:.3f})")
        else:
            print(f"  └─ knee           : (insufficient points)")

    return result

def plot_pareto_sweep(pareto: Dict[str, Any], output_path: str) -> None:
	taus = np.array(pareto["tau_grid"])
	fig, ax1 = plt.subplots(figsize=(9, 5.5))

	# Left axis: compression (pruned fraction)
	ax1.plot(taus, pareto["compression"], "o-", color="#1f77b4",
			 label="Compression $1-r(\\tau_G)$", linewidth=2, markersize=3)
	ax1.plot(taus, pareto["canon_retained_frac"], "s--", color="#9467bd",
			 label="Canonical retained frac", linewidth=1.6, markersize=3, alpha=0.8)
	ax1.set_xlabel("Grounding threshold $\\tau_G$")
	ax1.set_ylabel("fraction", color="#1f77b4")
	ax1.tick_params(axis="y", labelcolor="#1f77b4")
	ax1.set_ylim(0, 1.02)

	# Right axis: mean grounding retained (quality)
	ax2 = ax1.twinx()
	ax2.plot(taus, pareto["mean_grounding_kept"], "^-", color="#d62728",
			 label="Mean $G$ of kept concepts", linewidth=2, markersize=3)
	ax2.set_ylabel("mean grounding of kept", color="#d62728")
	ax2.tick_params(axis="y", labelcolor="#d62728")

	# Knee marker
	if pareto["knee_idx"] is not None:
		kt = pareto["knee_tau"]
		ax1.axvline(kt, color="black", linestyle=":", alpha=0.7)
		ax1.annotate(f"knee $\\tau_G={kt:.3f}$",
					 xy=(kt, 0.5), xytext=(kt + 0.05, 0.6),
					 arrowprops=dict(arrowstyle="->", color="black"),
					 fontsize=10, fontweight="bold")

	lines1, lbl1 = ax1.get_legend_handles_labels()
	lines2, lbl2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="center right")
	ax1.set_title("Pareto Frontier: Grounding Threshold $\\tau_G$ "
				  "vs Compression / Retained Quality")
	ax1.grid(alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_pareto_sweep] {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────
def save_json(results: Dict[str, Any], output_path: str) -> None:
	def _clean(obj):
		if isinstance(obj, float) and np.isnan(obj):
			return None
		if isinstance(obj, (np.floating,)):
			return None if np.isnan(obj) else float(obj)
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
	vocab: Dict[str, Any],
	pareto: Dict[str, Any],
	output_path: str,
) -> None:
	lines = [
		r"\begin{table}[t]",
		r"\centering",
		r"\caption{Micro-CGD per-regime means and Stage 3$\rightarrow$4 "
		rf"vocabulary compression. Global $\Phi_{{micro}}={vocab['phi_micro']:.3f}$ "
		rf"($|V_{{raw}}|={vocab['V_raw_global']:,}\rightarrow "
		rf"|V_{{canon}}|={vocab['V_canon_global']:,}$); "
		rf"Pareto knee at $\tau_G={pareto['knee_tau']:.3f}$.}}",
		r"\label{tab:cgd_summary}",
		r"\begin{tabular}{lrrrr}",
		r"\toprule",
		r"Regime & $C$ & $G$ & $D$ & $\Phi$ \\",
		r"\midrule",
	]
	for reg in VALID_REGIMES:
		if reg not in moments:
			continue
		mC = moments[reg]["C"]["mean"]
		mG = moments[reg]["G"]["mean"]
		mD = moments[reg]["D"]["mean"]
		phi = vocab["per_regime"].get(reg, {}).get("phi", float("nan"))
		lines.append(
			f"{REGIME_DISPLAY[reg]} & {mC:.3f} & {mG:.3f} & {mD:.3f} & {phi:.3f} \\\\"
		)
	lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
	with open(output_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"[save_summary_latex] {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Micro-CGD Distribution & Compression Diagnostics (Pillar 2)",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument("--metadata", "-csv", required=True, help="Path to dataset.csv (CGD source path derived from this)")
	p.add_argument("--pareto_steps", "-ps", type=int, default=PARETO_N_STEPS, help="Number of tau_G grid points in the Pareto sweep")
	p.add_argument("--verbose", "-v", action="store_true")
	return p.parse_args()


def main():
	args = parse_args()
	if args.verbose:
		print(args)

	ddir = os.path.dirname(args.metadata)

	outputs_dir  = os.path.join(ddir, "outputs")
	os.makedirs(outputs_dir, exist_ok=True)

	viz_dir = os.path.join(outputs_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)

	cgd_source = resolve_cgd_source(args.metadata, verbose=args.verbose)

	print(f"\n{'='*80}")
	print(f"[eval_cgd_distributions] Micro-CGD Distribution & Compression Diagnostics")
	print(f"  ├─ Metadata     : {args.metadata}")
	print(f"  ├─ CGD source   : {cgd_source}")
	print(f"  └─ Output dir   : {outputs_dir}")
	print(f"{'='*80}")

	# 1. Load & explode per-concept CGD records
	concept_df, sample_df = load_cgd_records(cgd_source, verbose=args.verbose)

	# 2. (i) Per-regime moments + violins
	moments = compute_cgd_moments(concept_df, verbose=args.verbose)
	plot_cgd_violins(
		concept_df, 
		os.path.join(viz_dir, "cgd_violin_per_regime.png")
	)
	plot_cgd_kde(
		concept_df, 
		os.path.join(viz_dir, "cgd_distributions_kde.png")
	)

	# 3. (ii) Vocabulary compression Phi
	vocab = compute_vocab_compression(concept_df, sample_df, verbose=args.verbose)
	plot_vocab_compression(
		vocab, 
		os.path.join(viz_dir, "cgd_vocab_compression.png")
	)

	# 4. (iii) Pareto sweep of tau_G
	pareto = compute_pareto_sweep(concept_df, n_steps=args.pareto_steps, verbose=args.verbose)
	plot_pareto_sweep(
		pareto, 
		os.path.join(viz_dir, "cgd_pareto_sweep.png")
	)

	# 5. Serialise
	results = {
		"cgd_source":   cgd_source,
		"n_samples":    int(len(sample_df)),
		"n_concepts":   int(len(concept_df)),
		"cgd_axes":     CGD_AXES,
		"moments":      moments,
		"vocab_compression": vocab,
		"pareto":       pareto,
	}
	save_json(results, os.path.join(outputs_dir, "cgd_distributions_results.json"))
	save_summary_latex(
		moments, 
		vocab, 
		pareto,
		os.path.join(outputs_dir, "cgd_summary.tex")
	)

	# 6. Headline verdict
	print(f"\n{'='*80}")
	print(f"[VERDICT]  Pillar 2 — Supervision Provenance is Real")

	# regime separation sanity: Hard-Conflict grounding should be the lowest
	g_by_reg = {r: moments[r]["G"]["mean"] for r in VALID_REGIMES if r in moments}
	if g_by_reg:
		ordered = sorted(g_by_reg.items(), key=lambda kv: kv[1])
		print(f"  ├─ Grounding G ranking (low→high): " +
			  ", ".join(f"{REGIME_DISPLAY[r]}={v:.3f}" for r, v in ordered))
		if ordered[0][0] == "HARD_CONFLICT":
			print(f"  │   ✓ Hard Conflict has the LOWEST grounding — CGD separates regimes.")
		else:
			print(f"  │   ⚠ Hard Conflict is not the lowest-grounded — discuss in paper.")

	print(
		f"  ├─ Vocabulary compression Phi_micro : {vocab['phi_micro']:.4f} "
		f"({vocab['phi_micro']*100:.1f}%)"
	)
	print(f"  └─ Pareto knee tau_G                : {pareto['knee_tau']:.4f}")
	print(f"{'='*80}\n")

if __name__ == "__main__":
	main()