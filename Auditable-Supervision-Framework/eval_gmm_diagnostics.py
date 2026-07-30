# GMM Model-Selection Diagnostics (Proving Claim B: "Conflict is a Dataset Property")

# Fits a Gaussian Mixture Model over the three continuous modality-conflict
# axes and proves — via BIC and Silhouette — that k=3 (Agreement / Soft /
# Hard) is the DATA-DRIVEN optimum, not an assumed hyper-parameter.
#
# The three axes (from the Stage 2 Evidence_Receipt):
#   • set_similarity  — symmetric cosine overlap of C_text and C_vis
#   • orphan_ratio    — asymmetric coverage (fraction of text-only concepts)
#   • asymmetry_gap   — directional NLI entailment gap (text→img vs img→text)
#
# Outputs (all under <ddir>/outputs/)
# ───────────────────────────────────
#   gmm_diagnostics_bic_silhouette.png   — model-selection curves (k=1..K_MAX)
#   gmm_diagnostics_3d_scatter.png       — 3D feature cloud coloured by regime
#   gmm_diagnostics_pairplot.png         — 2D marginal projections
#   gmm_diagnostics_results.json         — BIC/AIC/Silhouette per k + centroids
#   gmm_diagnostics_centroids.tex        — LaTeX centroid-means table
#
# Design contract
# ───────────────
# • The GMM is fit on the SAME continuous axes the Stage 2 router thresholds,
#   so the k=3 argmin(BIC) directly justifies the three-regime taxonomy.
# • The heuristic `regime` column (from the parquet) is treated as the
#   reference labelling; we report the GMM-vs-heuristic override rate to
#   quantify how often the soft probabilistic model disagrees with the
#   hard thresholds.

# how to run:
# python eval_gmm_diagnostics.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv -v

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

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# ── The three continuous conflict axes we expect in the Stage 2 receipt ──────
FEATURE_AXES = ["set_similarity", "orphan_ratio", "asymmetry_gap"]

# Alternative column names to probe (Stage 2 may serialise them differently)
FEATURE_ALIASES = {
	"set_similarity": ["set_similarity", "cosine_similarity", "sym_similarity", "set_sim"],
	"orphan_ratio":   ["orphan_ratio", "text_orphan_ratio", "coverage_gap", "orphan_frac"],
	"asymmetry_gap":  ["asymmetry_gap", "nli_gap", "entailment_gap", "asym_gap"],
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

K_MAX      = 6      # sweep k = 1 .. K_MAX
GMM_SEED   = 42
GMM_N_INIT = 10     # restarts per k for stable EM

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_feature_columns(df: pd.DataFrame, verbose: bool = True) -> Dict[str, str]:
	"""
	Map each canonical axis name to the actual column present in df.
	Probes FEATURE_ALIASES; also digs into an `Evidence_Receipt` JSON column
	if the flat columns are absent.
	Returns {canonical_axis: actual_column_name}.
	"""
	resolved: Dict[str, str] = {}
	for axis, aliases in FEATURE_ALIASES.items():
		for cand in aliases:
			if cand in df.columns:
				resolved[axis] = cand
				break
	if verbose:
		print(f"\n[resolve_features] Flat-column resolution:")
		for axis in FEATURE_AXES:
			print(f"  ├─ {axis:<16s} → {resolved.get(axis, '(not found)')}")
	return resolved

def _explode_evidence_receipt(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
	"""
	If the three axes are nested inside an `Evidence_Receipt` JSON column,
	explode them into flat columns. No-op if the flat columns already exist.
	"""
	receipt_col = next(
		(c for c in ("Evidence_Receipt", "evidence_receipt", "receipt") if c in df.columns),
		None,
	)
	if receipt_col is None:
		return df

	if verbose:
		print(f"\n[explode_receipt] Parsing nested '{receipt_col}' JSON column ...")

	def _parse(cell):
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

	parsed = df[receipt_col].apply(_parse)

	for axis, aliases in FEATURE_ALIASES.items():
		if axis in df.columns:
			continue
		def _extract(d, aliases=aliases):
			for key in aliases:
				if isinstance(d, dict) and key in d:
					return d[key]
			return np.nan
		df[axis] = parsed.apply(_extract)

	return df

def _load_df_from_source(source_path: str, verbose: bool = True) -> pd.DataFrame:
	"""
	Load a DataFrame from either a Stage 2 JSONL audit file or a parquet file.
	For JSONL, the `metrics` sub-dict is exploded into flat columns so that
	set_similarity / orphan_ratio / asymmetry_gap are directly accessible.
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

		# Explode the nested `metrics` dict into flat columns
		if "metrics" in df.columns:
			metrics_flat = pd.json_normalize(df["metrics"].apply(
				lambda x: x if isinstance(x, dict) else {}
			))
			# Drop any metrics columns that already exist at the top level
			new_cols = [c for c in metrics_flat.columns if c not in df.columns]
			df = pd.concat([df.drop(columns=["metrics"]), metrics_flat[new_cols]], axis=1)

		if verbose:
			print(f"  ├─ records : {len(df):,}")
			print(f"  └─ columns : {list(df.columns)}")
		return df

	else:
		# Parquet (or any other format pandas can read)
		if verbose:
			print(f"\n[load_df_from_source] Reading parquet: {source_path}")
		df = pd.read_parquet(source_path)
		if verbose:
			print(f"  ├─ df      : {df.shape}")
			print(f"  └─ Columns : {list(df.columns)}")
		return df

def load_conflict_features(
	audit_source: str,
	verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
	"""
	Load the Stage 2 audit file (JSONL **or** parquet) and extract the
	(N, 3) conflict-feature matrix plus the heuristic regime labels.

	Accepts:
	  • *_modality_conflict_audit.jsonl   — preferred; metrics are nested
	    under the ``metrics`` key and are exploded automatically.
	  • *_modality_conflict_audit.parquet — flat parquet produced by Stage 2
	    if the export block is present.
	  • *_auditable_supervision_matrix.parquet — last-resort fallback (will
	    fail if the axes were not carried forward into Stage 4).

	Returns
	-------
	X          : (N, 3) float array  — [set_similarity, orphan_ratio, asymmetry_gap]
	regimes    : (N,)   str array    — heuristic regime label per row
	df_valid   : the filtered DataFrame (skip regimes + NaN rows removed)
	"""
	assert os.path.isfile(audit_source), f"[load_conflict_features] Not found: {audit_source}"

	df = _load_df_from_source(audit_source, verbose=verbose)

	# Try nested receipt explosion first (parquet path), then resolve flat columns
	df = _explode_evidence_receipt(df, verbose=verbose)
	resolved = _resolve_feature_columns(df, verbose=verbose)

	missing = [a for a in FEATURE_AXES if a not in resolved and a not in df.columns]
	if missing:
		raise KeyError(
			f"[load_conflict_features] Could not locate conflict axes {missing} "
			f"in {audit_source}.\nAvailable columns: {list(df.columns)}\n"
			f"→ Ensure Stage 2 persisted set_similarity / orphan_ratio / "
			f"asymmetry_gap (flat or inside the 'metrics' key of the JSONL)."
		)

	# Build the feature frame using resolved names (fall back to canonical)
	col_map = {axis: resolved.get(axis, axis) for axis in FEATURE_AXES}
	feat_df = df[[col_map[a] for a in FEATURE_AXES]].copy()
	feat_df.columns = FEATURE_AXES

	# Regime column — prefer heuristic_regime (Stage 2 authority) over regime (Stage 4)
	regime_col = next(
		(c for c in ("heuristic_regime", "regime", "Regime", "regime_label") if c in df.columns),
		None,
	)
	if regime_col is None:
		raise KeyError(f"[load_conflict_features] No regime column found in {audit_source}")
	feat_df["regime"] = df[regime_col].astype(str).str.upper().str.replace(" ", "_")

	# Drop skip-regimes and NaN feature rows
	n_before = len(feat_df)
	feat_df = feat_df[~feat_df["regime"].isin(SKIP_REGIMES)]
	feat_df = feat_df.dropna(subset=FEATURE_AXES)
	n_after = len(feat_df)

	if verbose:
		print(f"\n[load_conflict_features] Filtering")
		print(f"  ├─ Before        : {n_before:,}")
		print(f"  ├─ After (valid) : {n_after:,}")
		print(f"  └─ Dropped       : {n_before - n_after:,} (skip-regime / NaN asymmetry_gap)")
		print(f"\n  Heuristic regime distribution:")
		for r, c in feat_df["regime"].value_counts().items():
			print(f"    ├─ {r:<16s}: {c:>7,} ({c/n_after*100:5.1f}%)")

	X       = feat_df[FEATURE_AXES].to_numpy(dtype=np.float64)
	regimes = feat_df["regime"].to_numpy()
	return X, regimes, feat_df

# ─────────────────────────────────────────────────────────────────────────────
# GMM model selection
# ─────────────────────────────────────────────────────────────────────────────
def fit_gmm_sweep(
	X: np.ndarray,
	k_max: int = K_MAX,
	verbose: bool = True,
) -> Dict[str, Any]:
	"""
	Fit GMMs for k = 1 .. k_max on standardised features.
	Returns per-k BIC, AIC, log-likelihood, and Silhouette (k>=2).
	"""
	scaler = StandardScaler()
	Xz = scaler.fit_transform(X)

	ks          = list(range(1, k_max + 1))
	bic, aic    = [], []
	loglik      = []
	silhouette  = []
	models: Dict[int, GaussianMixture] = {}

	if verbose:
		print(f"\n[fit_gmm_sweep] Sweeping k = 1 .. {k_max}  (n_init={GMM_N_INIT})")
		print(f"  {'k':>3s} {'BIC':>14s} {'AIC':>14s} {'logL':>14s} {'silhouette':>12s}")

	for k in ks:
		gmm = GaussianMixture(
			n_components=k,
			covariance_type="full",
			n_init=GMM_N_INIT,
			max_iter=300,
			random_state=GMM_SEED,
			reg_covar=1e-6,
		)
		gmm.fit(Xz)
		models[k] = gmm

		b = gmm.bic(Xz)
		a = gmm.aic(Xz)
		ll = gmm.score(Xz) * len(Xz)
		bic.append(b)
		aic.append(a)
		loglik.append(ll)

		if k >= 2:
			labels = gmm.predict(Xz)
			# Silhouette can be costly on huge N → subsample for stability
			if len(Xz) > 20000:
				rng = np.random.default_rng(GMM_SEED)
				idx = rng.choice(len(Xz), 20000, replace=False)
				sil = silhouette_score(Xz[idx], labels[idx])
			else:
				sil = silhouette_score(Xz, labels)
			silhouette.append(sil)
		else:
			silhouette.append(np.nan)

		if verbose:
			sil_str = f"{silhouette[-1]:.4f}" if not np.isnan(silhouette[-1]) else "—"
			print(f"  {k:>3d} {b:>14.1f} {a:>14.1f} {ll:>14.1f} {sil_str:>12s}")

	best_k_bic = ks[int(np.argmin(bic))]
	# Silhouette argmax ignores k=1 (NaN)
	sil_arr = np.array(silhouette, dtype=np.float64)
	best_k_sil = ks[int(np.nanargmax(sil_arr))] if not np.all(np.isnan(sil_arr)) else None

	if verbose:
		print(f"\n  ├─ argmin(BIC)        → k = {best_k_bic}")
		print(f"  └─ argmax(Silhouette) → k = {best_k_sil}")

	return {
		"ks":          ks,
		"bic":         bic,
		"aic":         aic,
		"loglik":      loglik,
		"silhouette":  silhouette,
		"best_k_bic":  best_k_bic,
		"best_k_sil":  best_k_sil,
		"models":      models,
		"scaler":      scaler,
		"Xz":          Xz,
	}

def align_gmm_to_regimes(
	gmm: GaussianMixture,
	scaler: StandardScaler,
	Xz: np.ndarray,
	regimes: np.ndarray,
	verbose: bool = True,
) -> Dict[str, Any]:
	"""
	Match GMM components to the three heuristic regimes by majority vote,
	then compute the GMM-vs-heuristic override rate and per-regime centroids
	in the ORIGINAL (un-standardised) feature space.
	"""
	comp = gmm.predict(Xz)                       # (N,) component id
	k    = gmm.n_components

	# Majority-vote label for each component
	comp_to_regime: Dict[int, str] = {}
	for c in range(k):
		mask = comp == c
		if mask.sum() == 0:
			comp_to_regime[c] = f"COMP_{c}"
			continue
		vals, counts = np.unique(regimes[mask], return_counts=True)
		comp_to_regime[c] = vals[int(np.argmax(counts))]

	gmm_regime = np.array([comp_to_regime[c] for c in comp])
	override_rate = float(np.mean(gmm_regime != regimes))

	# Centroids in original space
	centroids_z = gmm.means_                      # (k, 3) standardised
	centroids   = scaler.inverse_transform(centroids_z)

	centroid_table = {}
	for c in range(k):
		centroid_table[comp_to_regime[c]] = {
			axis: float(centroids[c, j]) for j, axis in enumerate(FEATURE_AXES)
		}
		centroid_table[comp_to_regime[c]]["weight"] = float(gmm.weights_[c])

	if verbose:
		print(f"\n[align_gmm_to_regimes] Component → regime (majority vote)")
		for c in range(k):
			print(f"  ├─ Component {c} → {comp_to_regime[c]}  (π={gmm.weights_[c]:.3f})")
		print(f"  └─ GMM-vs-heuristic override rate : {override_rate*100:.2f}%")
		print(f"\n  Centroid means (original feature space):")
		for reg, vals in centroid_table.items():
			print(
				f"    ├─ {reg:<16s}: "
				f"sim={vals['set_similarity']:.3f}  "
				f"orphan={vals['orphan_ratio']:.3f}  "
				f"asym={vals['asymmetry_gap']:.3f}"
			)

	return {
		"comp_to_regime": comp_to_regime,
		"gmm_regime":     gmm_regime,
		"override_rate":  override_rate,
		"centroids":      centroid_table,
	}

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_bic_silhouette(
	sweep: Dict[str, Any], 
	output_path: str
):
	ks  = sweep["ks"]
	fig, ax1 = plt.subplots(figsize=(8, 5))

	ax1.plot(ks, sweep["bic"], "o-", color="#1f77b4", label="BIC", linewidth=2)
	ax1.plot(ks, sweep["aic"], "s--", color="#7f7f7f", label="AIC", linewidth=1.5, alpha=0.7)
	ax1.set_xlabel("Number of components $k$")
	ax1.set_ylabel("BIC / AIC  (lower = better)", color="#1f77b4")
	ax1.tick_params(axis="y", labelcolor="#1f77b4")
	ax1.axvline(sweep["best_k_bic"], color="#1f77b4", linestyle=":", alpha=0.6)

	ax2 = ax1.twinx()
	ax2.plot(ks, sweep["silhouette"], "^-", color="#d62728", label="Silhouette", linewidth=2)
	ax2.set_ylabel("Silhouette  (higher = better)", color="#d62728")
	ax2.tick_params(axis="y", labelcolor="#d62728")

	ax1.set_title(
		f"GMM Model Selection  —  argmin(BIC)=k={sweep['best_k_bic']},  "
		f"argmax(Sil)=k={sweep['best_k_sil']}"
	)
	ax1.set_xticks(ks)
	lines1, lbl1 = ax1.get_legend_handles_labels()
	lines2, lbl2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="best")
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_bic_silhouette] {output_path}")

def plot_3d_scatter(
	X: np.ndarray,
	regimes: np.ndarray,
	centroids: Dict[str, Any],
	output_path: str,
	max_points: int = 8000,
) -> None:
	fig = plt.figure(figsize=(9, 7))
	ax  = fig.add_subplot(111, projection="3d")

	# Subsample for legibility
	if len(X) > max_points:
		rng = np.random.default_rng(GMM_SEED)
		idx = rng.choice(len(X), max_points, replace=False)
		Xp, rp = X[idx], regimes[idx]
	else:
		Xp, rp = X, regimes

	for reg in VALID_REGIMES:
		m = rp == reg
		if m.sum() == 0:
			continue
		ax.scatter(
			Xp[m, 0], Xp[m, 1], Xp[m, 2],
			s=6, alpha=0.35,
			c=REGIME_COLORS.get(reg, "#888888"),
			label=REGIME_DISPLAY.get(reg, reg),
		)

	# Overlay centroids
	for reg, vals in centroids.items():
		if reg not in REGIME_COLORS:
			continue
		ax.scatter(
			vals["set_similarity"], vals["orphan_ratio"], vals["asymmetry_gap"],
			s=260, marker="X", edgecolors="black", linewidths=1.5,
			c=REGIME_COLORS[reg], zorder=10,
		)

	ax.set_xlabel("set_similarity")
	ax.set_ylabel("orphan_ratio")
	ax.set_zlabel("asymmetry_gap")
	ax.set_title("Modality-Conflict Feature Space (coloured by regime)")
	ax.legend(loc="upper left", markerscale=2)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_3d_scatter] {output_path}")

def plot_pairwise(X: np.ndarray, regimes: np.ndarray, output_path: str, max_points: int = 8000) -> None:
	"""2D marginal projections of the three axes."""
	if len(X) > max_points:
		rng = np.random.default_rng(GMM_SEED)
		idx = rng.choice(len(X), max_points, replace=False)
		Xp, rp = X[idx], regimes[idx]
	else:
		Xp, rp = X, regimes

	pairs = [(0, 1), (0, 2), (1, 2)]
	fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
	for ax, (i, j) in zip(axes, pairs):
		for reg in VALID_REGIMES:
			m = rp == reg
			if m.sum() == 0:
				continue
			ax.scatter(
				Xp[m, i], Xp[m, j],
				s=6, alpha=0.35,
				c=REGIME_COLORS.get(reg, "#888888"),
				label=REGIME_DISPLAY.get(reg, reg),
			)
		ax.set_xlabel(FEATURE_AXES[i])
		ax.set_ylabel(FEATURE_AXES[j])
	axes[0].legend(loc="best", markerscale=2)
	fig.suptitle("Pairwise Marginal Projections of Conflict Axes")
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"[plot_pairwise] {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────
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

def save_centroid_latex(centroids: Dict[str, Any], override_rate: float, best_k: int, output_path: str) -> None:
	lines = [
		r"\begin{table}[t]",
		r"\centering",
		r"\caption{GMM Centroid Means over Modality-Conflict Axes "
		rf"($k={best_k}$, argmin BIC). GMM-vs-heuristic override "
		rf"rate: {override_rate*100:.2f}\%.}}",
		r"\label{tab:gmm_centroids}",
		r"\begin{tabular}{lrrrr}",
		r"\toprule",
		r"Regime & $\pi$ & set\_similarity & orphan\_ratio & asymmetry\_gap \\",
		r"\midrule",
	]
	for reg in VALID_REGIMES:
		if reg not in centroids:
			continue
		v = centroids[reg]
		lines.append(
			f"{REGIME_DISPLAY.get(reg, reg)} & "
			f"{v.get('weight', float('nan')):.3f} & "
			f"{v['set_similarity']:.3f} & "
			f"{v['orphan_ratio']:.3f} & "
			f"{v['asymmetry_gap']:.3f} \\\\"
		)
	lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
	with open(output_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"[save_centroid_latex] {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="GMM Model-Selection Diagnostics for Modality-Conflict Regimes",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument("--metadata", "-csv", required=True, help="Path to dataset.csv (parquet path derived from this)")
	p.add_argument("--k_max", "-k", type=int, default=K_MAX, help="Max components to sweep")
	p.add_argument("--verbose", "-v", action="store_true")
	return p.parse_args()

def main():
	args = parse_args()
	if args.verbose:
		print(args)

	ddir = os.path.dirname(args.metadata)

	outputs_dir = os.path.join(ddir, "outputs")
	os.makedirs(outputs_dir, exist_ok=True)

	viz_dir = os.path.join(outputs_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)

	metadata_file = os.path.basename(args.metadata)

	stem = metadata_file.replace(".csv", "_mlm_cot_modality_conflict_audit")

	# Priority 1 — Stage 2 JSONL (metrics nested under 'metrics' key, always present)
	audit_jsonl   = os.path.join(ddir, f"{stem}.jsonl")
	# Priority 2 — Stage 2 flat parquet (written only if the export block is present)
	audit_parquet = os.path.join(outputs_dir, f"{stem}.parquet")
	# Priority 3 — Stage 4 supervision matrix (last resort; axes usually absent)
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
	print(f"[eval_gmm_diagnostics] GMM Model-Selection Diagnostics")
	print(f"  ├─ Metadata     : {args.metadata}")
	print(f"  ├─ Audit source : {audit_source}")
	print(f"  ├─ Output dir   : {outputs_dir}")
	print(f"  └─ k sweep      : 1 .. {args.k_max}")
	print(f"{'='*80}")

	# 1. Load conflict features + heuristic regimes
	X, regimes, feat_df = load_conflict_features(audit_source, verbose=args.verbose)

	# 2. GMM sweep + model selection
	sweep = fit_gmm_sweep(X, k_max=args.k_max, verbose=args.verbose)

	# 3. Align the best-k GMM to heuristic regimes (report override rate)
	best_k   = sweep["best_k_bic"]
	best_gmm = sweep["models"][best_k]
	align = align_gmm_to_regimes(
		gmm=best_gmm,
		scaler=sweep["scaler"],
		Xz=sweep["Xz"],
		regimes=regimes,
		verbose=args.verbose,
	)

	# 4. Plots
	plot_bic_silhouette(
		sweep, 
		os.path.join(viz_dir, "gmm_diagnostics_bic_silhouette.png")
	)
	plot_3d_scatter(
		X, 
		regimes, 
		align["centroids"], 
		os.path.join(viz_dir, "gmm_diagnostics_3d_scatter.png")
	)
	plot_pairwise(
		X, 
		regimes, 
		os.path.join(viz_dir, "gmm_diagnostics_pairplot.png")
	)

	# 5. Serialise results
	results = {
		"audit_source":    audit_source,
		"n_samples":       int(len(X)),
		"feature_axes":    FEATURE_AXES,
		"ks":              sweep["ks"],
		"bic":             sweep["bic"],
		"aic":             sweep["aic"],
		"loglik":          sweep["loglik"],
		"silhouette":      sweep["silhouette"],
		"best_k_bic":      sweep["best_k_bic"],
		"best_k_sil":      sweep["best_k_sil"],
		"override_rate":   align["override_rate"],
		"comp_to_regime":  align["comp_to_regime"],
		"centroids":       align["centroids"],
		"heuristic_dist":  feat_df["regime"].value_counts().to_dict(),
	}
	save_json(results, os.path.join(outputs_dir, "gmm_diagnostics_results.json"))
	save_centroid_latex(
		align["centroids"], align["override_rate"], best_k,
		os.path.join(outputs_dir, "gmm_diagnostics_centroids.tex"),
	)

	# 6. Headline verdict
	print(f"\n{'='*80}")
	print(f"[VERDICT]  argmin(BIC) → k = {sweep['best_k_bic']}   "
				f"|   argmax(Silhouette) → k = {sweep['best_k_sil']}")
	if sweep["best_k_bic"] == 3:
		print(f"  ✓  Data-driven optimum CONFIRMS the 3-regime taxonomy (Claim B).")
	else:
		print(f"  ⚠  argmin(BIC)={sweep['best_k_bic']} ≠ 3 — discuss this honestly in the paper.")
	print(f"  └─ GMM-vs-heuristic override rate: {align['override_rate']*100:.2f}%")
	print(f"{'='*80}\n")

if __name__ == "__main__":
	main()