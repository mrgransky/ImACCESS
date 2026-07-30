# Regime-Stratified Evaluation
# ────
# Loads a trained Stage 5 checkpoint and evaluates retrieval metrics
# stratified by modality-conflict regime:
# AGREEMENT / SOFT_CONFLICT / HARD_CONFLICT
#
# Optionally compares RACL against Vanilla CLIP (B0) zero-shot baseline
# using --compare_baseline.  The Δ column (RACL − B0) is printed per regime
# and included in all output artefacts.
#
# Outputs
# ────
#   <output_dir>/regime_stratified_results.json   — full metrics dict
#   <output_dir>/regime_stratified_table.tex      — LaTeX-ready table
#   <output_dir>/regime_stratified_results.csv    — flat CSV for plotting
#
#
# Design contract
# ────
# • Regime buckets are determined by the parquet's `regime` column — the
#   same authority used during training.  No re-routing is performed here.
# • Samples with MISSING_MODALITY / INVALID_JSON regimes are excluded from
#   all per-regime tables but counted in a separate "skipped" row.
# • Gap_rel = (mAP_rare - mAP_head) / (mAP_head + ε) is reported per bucket
#   and globally.  A less-negative Gap_rel indicates better tail recovery.
# • --compare_baseline loads vanilla CLIP weights (no checkpoint) and runs
#   the same evaluation pass.  No extra data loading is required.
# ────

# how to run (RACL only):
# python eval_regime_stratified.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv -v
#
# how to run (RACL + B0 baseline comparison):
# python eval_regime_stratified.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv --compare_baseline -v

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

import clip
from stage5_dataset_loader import (
	get_stage5_dataloaders,
	load_supervision_matrix,
	RegimeAwareDataset,
	customized_collate_fn,
	FALLBACK_REGIME,
)
from stage5_regime_conditioned_training import (
	build_class_embeddings,
	load_checkpoint,
	setup_peft,
	_compute_retrieval_metrics,
	_mean_average_precision,
	_precision_at_k,
	_ndcg_at_k,
)
from loss import compute_loss_masks

VALID_REGIMES  = ["AGREEMENT", "SOFT_CONFLICT", "HARD_CONFLICT"]
SKIP_REGIMES   = {"MISSING_MODALITY", "INVALID_JSON", FALLBACK_REGIME}
GAP_REL_EPS    = 1e-8   # denominator guard for Gap_rel

REGIME_DISPLAY = {
	"AGREEMENT":    "Agreement",
	"SOFT_CONFLICT": "Soft Conflict",
	"HARD_CONFLICT": "Hard Conflict",
	"ALL":           "All (Global)",
	"SKIPPED":       "Skipped",
}

@torch.no_grad()
def evaluate_regime_stratified(
	model:            torch.nn.Module,
	val_loader:       DataLoader,
	all_class_embeds: torch.Tensor,       # [C, D] — normalised, on device
	active_mask:      torch.Tensor,       # [C] bool
	head_mask:        torch.Tensor,       # [C] bool
	rare_mask:        torch.Tensor,       # [C] bool
	device:           torch.device,
	verbose:          bool = True,
) -> Dict[str, Any]:
	"""
	Single forward pass over val_loader.
	Accumulates scores and targets per regime bucket.
	Returns
	----
	results : dict with keys
		"ALL"          → global metrics dict
		"AGREEMENT"    → per-regime metrics dict
		"SOFT_CONFLICT"→ per-regime metrics dict
		"HARD_CONFLICT"→ per-regime metrics dict
		"SKIPPED"      → {"count": int}
		"regime_counts"→ {regime: int}
		"n_total"      → int
	"""
	model.eval()
	class_embeds = torch.nn.functional.normalize(all_class_embeds, dim=-1).to(device)

	# Accumulators: regime → list of (scores [1,C], targets [1,C])
	bucket_scores: Dict[str, List[torch.Tensor]] = defaultdict(list)
	bucket_targets: Dict[str, List[torch.Tensor]] = defaultdict(list)
	regime_counts: Dict[str, int] = defaultdict(int)
	n_total = 0

	if verbose:
		print(f"\n[EVAL] {len(val_loader.dataset):,} (VAL) samples [takes a while...]")

	for batch_idx, batch in enumerate(val_loader):
		if not batch:
			continue

		images = batch["image"].to(device, non_blocking=True)
		label_vec = batch["label_vec"].to(device, non_blocking=True)
		regimes = batch["regime"]   # List[str], length B

		# Forward: image embeddings → cosine similarities
		image_embeds = torch.nn.functional.normalize(model.encode_image(images), dim=-1).float() # [B, D]
		scores = torch.matmul(image_embeds, class_embeds.T) # [B, C]

		# Route each sample to its regime bucket
		for i, regime in enumerate(regimes):
			n_total += 1
			regime_counts[regime] += 1
			s = scores[i].unsqueeze(0).cpu()       # [1, C]
			t = label_vec[i].unsqueeze(0).cpu()    # [1, C]

			if regime in SKIP_REGIMES:
				bucket_scores["SKIPPED"].append(s)
				bucket_targets["SKIPPED"].append(t)
			else:
				# Normalise to canonical name (upper-case, underscore)
				canonical = regime.upper().replace(" ", "_")
				bucket_scores[canonical].append(s)
				bucket_targets[canonical].append(t)

				# Also accumulate into ALL
				bucket_scores["ALL"].append(s)
				bucket_targets["ALL"].append(t)

	if verbose:
		print(f"\nRegime distribution:")

		for r, cnt in sorted(regime_counts.items()):
			pct = cnt / max(n_total, 1) * 100
			print(f"  ├─ {r:<22s}: {cnt:>6,} ({pct:5.1f}%)")

		print(f"  └─ Total: {n_total:,}")

	# Compute metrics per bucket
	results: Dict[str, Any] = {"regime_counts": dict(regime_counts), "n_total": n_total}

	for bucket in VALID_REGIMES + ["ALL"]:
		if bucket not in bucket_scores or len(bucket_scores[bucket]) == 0:
			results[bucket] = _empty_metrics(bucket)
			continue

		s_cat = torch.cat(bucket_scores[bucket],  dim=0)  # [N_bucket, C]
		t_cat = torch.cat(bucket_targets[bucket], dim=0)  # [N_bucket, C]

		metrics = _compute_retrieval_metrics(
			scores=s_cat,
			targets=t_cat,
			active_mask=active_mask.cpu(),
			head_mask=head_mask.cpu(),
			rare_mask=rare_mask.cpu(),
		)

		# Rename val_* keys → bucket-prefixed keys for clarity
		renamed = {k.replace("val_", ""): v for k, v in metrics.items()}

		# Add Gap_rel
		renamed["gap_rel"] = _compute_gap_rel(
			renamed.get("map_head", float("nan")),
			renamed.get("map_rare", float("nan")),
		)
		# Add sample count for this bucket
		renamed["n_samples"] = len(bucket_scores[bucket])
		results[bucket] = renamed

	# Skipped count
	results["SKIPPED"] = {
		"count": regime_counts.get("MISSING_MODALITY", 0)
		+ regime_counts.get("INVALID_JSON", 0)
		+ regime_counts.get(FALLBACK_REGIME, 0)
	}

	return results

def _compute_gap_rel(map_head: float, map_rare: float) -> float:
	"""
	Gap_rel = (mAP_rare - mAP_head) / (mAP_head + ε)
	Negative → head dominates; 
	closer to 0 → better tail recovery.
	"""
	if np.isnan(map_head) or np.isnan(map_rare):
		return float("nan")

	return (map_rare - map_head) / (map_head + GAP_REL_EPS)

def _empty_metrics(bucket: str) -> Dict[str, Any]:
		return {
				"n_samples": 0,
				"map_all":   float("nan"),
				"map_head":  float("nan"),
				"map_rare":  float("nan"),
				"p@1":       float("nan"),
				"p@5":       float("nan"),
				"ndcg@5":    float("nan"),
				"gap_rel":   float("nan"),
				"note":      f"No samples in bucket '{bucket}'",
		}

def _fmt(v: Any, decimals: int = 4) -> str:
		"""Format a float for display; return '—' for NaN."""
		if isinstance(v, float) and np.isnan(v):
				return "—"
		if isinstance(v, float):
				return f"{v:.{decimals}f}"
		return str(v)

def _delta_str(racl_val: float, b0_val: float) -> str:
	"""Format Δ = RACL − B0 with sign; return '—' if either is NaN."""
	if np.isnan(racl_val) or np.isnan(b0_val):
		return "—"
	delta = racl_val - b0_val
	return f"{delta:+.4f}"

def print_results_table(
	results:    Dict[str, Any],
	b0_results: Optional[Dict[str, Any]] = None,
) -> None:
	"""
	Pretty-print a regime-stratified results table to stdout.
	If b0_results is provided, a Δ(mAP-all) column is appended showing
	RACL − B0 per regime bucket.
	"""
	has_b0 = b0_results is not None
	width  = 120 if has_b0 else 100

	header_cols = (
		f"{'Regime':<22s} {'N':>7s} "
		f"{'mAP-all':>9s} {'mAP-head':>9s} {'mAP-rare':>9s} "
		f"{'Gap_rel':>9s} {'P@1':>7s} {'P@5':>7s} {'nDCG@5':>8s}"
	)
	if has_b0:
		header_cols += f"  {'B0 mAP-all':>10s} {'Δ mAP-all':>10s}"

	print(f"\n{'─'*width}\n{header_cols}\n{'─'*width}")

	for bucket in VALID_REGIMES + ["ALL"]:
		m = results.get(bucket, {})
		if not m or m.get("n_samples", 0) == 0:
			print(f"  {REGIME_DISPLAY.get(bucket, bucket):<20s}  (no samples)")
			continue

		row = (
			f"  {REGIME_DISPLAY.get(bucket, bucket):<20s} "
			f"{m['n_samples']:>7,} "
			f"{_fmt(m.get('map_all')):>9s} "
			f"{_fmt(m.get('map_head')):>9s} "
			f"{_fmt(m.get('map_rare')):>9s} "
			f"{_fmt(m.get('gap_rel')):>9s} "
			f"{_fmt(m.get('p@1')):>7s} "
			f"{_fmt(m.get('p@5')):>7s} "
			f"{_fmt(m.get('ndcg@5')):>8s}"
		)
		if has_b0:
			b0_m    = b0_results.get(bucket, {})
			b0_map  = b0_m.get("map_all", float("nan"))
			row += f"  {_fmt(b0_map):>10s} {_delta_str(m.get('map_all', float('nan')), b0_map):>10s}"
		print(row)

	skipped = results.get("SKIPPED", {}).get("count", 0)
	print(f"{'─'*width}")
	print(f"  {'Skipped (MISSING/INVALID)':<20s} {skipped:>7,}")
	print(f"  {'Total':<20s} {results.get('n_total', 0):>7,}")
	print(f"{'─'*width}\n")

	# Verdict block when baseline is present
	if has_b0:
		print(f"{'='*width}")
		print(f"[VERDICT]  RACL vs. Vanilla CLIP (B0) — Regime-Stratified")
		for bucket in VALID_REGIMES + ["ALL"]:
			m    = results.get(bucket, {})
			b0_m = b0_results.get(bucket, {})
			racl_map = m.get("map_all", float("nan"))
			b0_map   = b0_m.get("map_all", float("nan"))
			if np.isnan(racl_map) or np.isnan(b0_map):
				continue
			delta  = racl_map - b0_map
			symbol = "✓" if delta > 0 else "⚠"
			label  = REGIME_DISPLAY.get(bucket, bucket)
			print(f"  {symbol}  [{label:<16s}]  RACL={racl_map:.4f}  B0={b0_map:.4f}  Δ={delta:+.4f}")
		print(f"{'='*width}\n")

def save_latex_table(
	results: Dict[str, Any],
	output_path: str,
	b0_results: Optional[Dict[str, Any]] = None,
) -> None:
	has_b0 = b0_results is not None

	if has_b0:
			col_spec = r"lrrrrrrrrrr" # 1 + 10 = 11 cols
			col_header = (
					r"Regime & $N$ & \multicolumn{3}{c}{RACL (Ours)} & "
					r"$\text{Gap}_{\text{rel}}$ & P@1 & P@5 & nDCG@5 & "
					r"B0 mAP & $\Delta$mAP \\"
			)
			# SPLIT into two lines
			sub_header = [
					r"\cmidrule(lr){3-5}",
					r"& & mAP-all & mAP-head & mAP-rare & & & & & & \\"
			]
	else:
			col_spec = r"lrrrrrrrr" # 1 + 8 = 9 cols - FIXED
			col_header = (
					r"Regime & $N$ & mAP-all & mAP-head & mAP-rare & "
					r"$\text{Gap}_{\text{rel}}$ & P@1 & P@5 & nDCG@5 \\"
			)
			sub_header = None

	caption = (
		r"Regime-Stratified Retrieval Metrics on HISTORY-X4 (val split). "
		+ (r"$\Delta$mAP = RACL $-$ Vanilla CLIP (B0)." if has_b0 else "")
	)
	lines = [
		r"\begin{table}[t]",
		r"\centering",
		rf"\caption{{{caption}}}",
		r"\label{tab:regime_stratified}",
		rf"\begin{{tabular}}{{{col_spec}}}",
		r"\toprule",
		col_header,
	]

	if sub_header:
		lines.extend(sub_header)
	lines.append(r"\midrule")
	for bucket in VALID_REGIMES:
			m = results.get(bucket, {})
			if not m or m.get("n_samples", 0) == 0:
					continue
			row = (
					f"{REGIME_DISPLAY.get(bucket, bucket)} & "
					f"{m['n_samples']:,} & "
					f"{_fmt(m.get('map_all'))} & "
					f"{_fmt(m.get('map_head'))} & "
					f"{_fmt(m.get('map_rare'))} & "
					f"{_fmt(m.get('gap_rel'))} & "
					f"{_fmt(m.get('p@1'))} & "
					f"{_fmt(m.get('p@5'))} & "
					f"{_fmt(m.get('ndcg@5'))}"
			)
			if has_b0:
					b0_m = b0_results.get(bucket, {})
					b0_map = b0_m.get("map_all", float("nan"))
					row += f" & {_fmt(b0_map)} & {_delta_str(m.get('map_all', float('nan')), b0_map)}"
			row += r" \\"
			lines.append(row)
	lines.append(r"\midrule")
	m_all = results.get("ALL", {})
	if m_all and m_all.get("n_samples", 0) > 0:
		def _b(v): return rf"\textbf{{{v}}}"
		n_str = f"{m_all['n_samples']:,}"
		row = (
				f"{_b('All (Global)')} & "
				f"{_b(n_str)} & "
				f"{_b(_fmt(m_all.get('map_all')))} & "
				f"{_b(_fmt(m_all.get('map_head')))} & "
				f"{_b(_fmt(m_all.get('map_rare')))} & "
				f"{_b(_fmt(m_all.get('gap_rel')))} & "
				f"{_b(_fmt(m_all.get('p@1')))} & "
				f"{_b(_fmt(m_all.get('p@5')))} & "
				f"{_b(_fmt(m_all.get('ndcg@5')))}"
		)
		if has_b0:
			b0_all = b0_results.get("ALL", {})
			b0_map = b0_all.get("map_all", float("nan"))
			row += f" & {_b(_fmt(b0_map))} & {_b(_delta_str(m_all.get('map_all', float('nan')), b0_map))}"

		row += r" \\"
		lines.append(row)

	lines += [
		r"\bottomrule",
		r"\end{tabular}",
		r"\end{table}",
	]

	with open(output_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")

def save_csv(
	results:    Dict[str, Any],
	output_path: str,
	b0_results: Optional[Dict[str, Any]] = None,
) -> None:
	"""
	Write a flat CSV for downstream plotting (e.g., matplotlib / seaborn).
	One row per regime bucket.  When b0_results is provided, B0 metrics and
	Δ columns are appended.
	"""
	has_b0 = b0_results is not None
	fieldnames = [
		"regime", "n_samples",
		"map_all", "map_head", "map_rare", "gap_rel",
		"p@1", "p@5", "ndcg@5",
	]
	if has_b0:
		fieldnames += ["b0_map_all", "b0_map_head", "b0_map_rare", "b0_gap_rel", "delta_map_all"]

	rows = []
	for bucket in VALID_REGIMES + ["ALL"]:
		m = results.get(bucket, {})
		if not m:
			continue
		row = {
			"regime":    bucket,
			"n_samples": m.get("n_samples", 0),
			"map_all":   m.get("map_all",  float("nan")),
			"map_head":  m.get("map_head", float("nan")),
			"map_rare":  m.get("map_rare", float("nan")),
			"gap_rel":   m.get("gap_rel",  float("nan")),
			"p@1":       m.get("p@1",      float("nan")),
			"p@5":       m.get("p@5",      float("nan")),
			"ndcg@5":    m.get("ndcg@5",   float("nan")),
		}
		if has_b0:
			b0_m = b0_results.get(bucket, {})
			b0_map = b0_m.get("map_all", float("nan"))
			racl_map = m.get("map_all", float("nan"))
			row["b0_map_all"]   = b0_map
			row["b0_map_head"]  = b0_m.get("map_head", float("nan"))
			row["b0_map_rare"]  = b0_m.get("map_rare", float("nan"))
			row["b0_gap_rel"]   = b0_m.get("gap_rel",  float("nan"))
			row["delta_map_all"] = (
				racl_map - b0_map
				if not (np.isnan(racl_map) or np.isnan(b0_map))
				else float("nan")
			)
		rows.append(row)

	with open(output_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"[save_csv] {output_path}")

def save_json(results: Dict[str, Any], output_path: str) -> None:
	"""Serialise the full results dict to JSON (NaN → null)."""
	def _nan_to_none(obj):
		if isinstance(obj, float) and np.isnan(obj):
			return None

		if isinstance(obj, dict):
			return {k: _nan_to_none(v) for k, v in obj.items()}

		if isinstance(obj, list):
			return [_nan_to_none(v) for v in obj]

		return obj

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(_nan_to_none(results), f, indent=2, ensure_ascii=False)

	print(f"[save_json] {output_path}")

def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Regime-Stratified Evaluation for Stage 5 RACL Model",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	# Required
	p.add_argument("--metadata", "-csv", required=True, help="Path to dataset.csv (train/val splits inferred from this)")

	# Model
	p.add_argument("--clip_model", '-cm', default="ViT-B/32", help="CLIP backbone")
	p.add_argument(
		"--peft_method", 
		'-peft', 
		default="lora", 
		choices=[
			"lora", "lora_plus", "dora", "rslora", "ia3", "vera",
			"tip_adapter", "tip_adapter_f",
			"clip_adapter_v", "clip_adapter_t", "clip_adapter_vt",
			"probe", 
			"full"
		],
		help="PEFT method — must match training config"
	)

	# Data
	p.add_argument("--batch_size", '-bs',  type=int, default=256)
	p.add_argument("--num_workers", '-nw', type=int, default=8)
	p.add_argument("--id_col",      default="doc_url")
	p.add_argument("--text_col",    default="multimodal_labels")

	# Loss masks (must match training config for consistent head/rare split)
	p.add_argument("--pw_mode", default="sqrt", choices=["log", "sqrt", "linear"], help="pos_weight mode — must match training config")
	p.add_argument("--pw_max_cap", type=float, default=50.0, help="pos_weight cap — must match training config")

	# Baseline comparison
	p.add_argument(
		"--compare_baseline", "-b0",
		action="store_true",
		help=(
			"Also evaluate Vanilla CLIP (B0) zero-shot on the same val split "
			"and report Δ = RACL − B0 per regime bucket. "
			"No extra checkpoint is needed — B0 uses the original CLIP weights."
		),
	)

	# Misc
	p.add_argument("--verbose", "-v", action="store_true")

	return p.parse_args()

def main():
	args = parse_args()
	if args.verbose:
		print(args)

	ddir = os.path.dirname(args.metadata)

	outputs_dir = os.path.join(ddir, "outputs")
	os.makedirs(outputs_dir, exist_ok=True)

	checkpoint_dir = Path(os.path.join(outputs_dir, "checkpoints"))

	# Find all files ending with .pt in that directory
	pt_files = list(checkpoint_dir.glob("*.pt"))

	if not pt_files:
		raise FileNotFoundError(f"No .pt files found in {checkpoint_dir}")

	# If you want the full path as a string instead, use:
	checkpoint_fpath = str(pt_files[0])

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# 1. Model + PEFT
	model, _ = clip.load(
		name=args.clip_model,
		device=device,
		# jit=False, # training or finetuning => jit=False
		# random_weights=False, # finetuning => random_weights=False
		# dropout=0.0,
		download_root=get_model_directory(path=ddir),
	)
	model.name = args.clip_model # Custom attribute to store model name
	model_name = model.__class__.__name__
	model_arch = re.sub(r'[/@]', '_', model.name) if hasattr(model, 'name') else 'unknown_arch'
	input_resolution = getattr(model.visual, "input_resolution", None)

	print(f"\n{'='*80}")
	print(f"[eval_regime_stratified] Regime-Stratified Evaluation")
	print(f"  ├─ Metadata    : {args.metadata}")
	print(f"  ├─ Output dir  : {outputs_dir}")
	print(f"  ├─ Checkpoint  : {checkpoint_fpath}")
	print(f"  ├─ CLIP model  : {args.clip_model}")
	print(f"  ├─ resolution  : {input_resolution}")
	print(f"  ├─ PEFT method : {args.peft_method}")
	print(f"  ├─ B0 baseline : {'yes' if args.compare_baseline else 'no'}")
	print(f"  └─ Device      : {device}")
	print(f"{'='*80}")

	model, _ = setup_peft(
		model=model,
		peft_method=args.peft_method,
		verbose=False,
	)
	model = model.to(device)

	# 2. DataLoaders 
	# We only need the val_loader for evaluation.
	# train_loader is used solely to compute loss masks (head/rare split).
	train_loader, val_loader = get_stage5_dataloaders(
		metadata_fpth=args.metadata,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		input_resolution=input_resolution,
		id_col=args.id_col,
		text_col=args.text_col,
		verbose=args.verbose,
	)
	label_dict  = train_loader.dataset.label_dict
	num_classes = len(label_dict)

	# 3. Loss masks (head / rare split)
	# Must use the same pw_mode and pw_max_cap as training to ensure
	# head_mask / rare_mask are identical to those used during training.
	loss_masks = compute_loss_masks(
		loader=train_loader,
		num_classes=num_classes,
		device=device,
		pw_mode=args.pw_mode,
		pw_max_cap=args.pw_max_cap,
		verbose=args.verbose,
	)
	active_mask = loss_masks["active_mask"]
	head_mask   = loss_masks["head_mask"]
	rare_mask   = loss_masks["rare_mask"]

	# 4. Load checkpoint
	epoch, ckpt_metrics = load_checkpoint(
		ckpt_path=checkpoint_fpath,
		model=model,
		device=device,
		verbose=args.verbose,
	)

	print(f"[eval] Evaluating checkpoint from epoch {epoch}")
	if ckpt_metrics:
		print(f"  ├─ Checkpoint val_loss : {ckpt_metrics.get('val_loss', float('nan')):.6f}")
		print(f"  └─ Checkpoint mAP-all  : {ckpt_metrics.get('val_map_all', float('nan')):.4f}")

	# 5. Build class embeddings
	all_class_embeds = build_class_embeddings(
		model=model,
		label_dict=label_dict,
		device=device,
		verbose=args.verbose,
	).to(device)

	# 6. Regime-stratified evaluation
	results = evaluate_regime_stratified(
		model=model,
		val_loader=val_loader,
		all_class_embeds=all_class_embeds,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		device=device,
		verbose=args.verbose,
	)

	# 6b. Optional: Vanilla CLIP B0 baseline evaluation
	b0_results = None
	if args.compare_baseline:
		print(f"\n[eval] Evaluating Vanilla CLIP {args.clip_model} (B0) zero-shot baseline")
		model_b0, _ = clip.load(
			name=args.clip_model,
			device=device,
			download_root=get_model_directory(path=ddir),
		)
		model_b0.name = args.clip_model
		model_b0, _ = setup_peft(
			model=model_b0, 
			peft_method=args.peft_method, 
			verbose=False
		)
		model_b0 = model_b0.to(device)

		b0_class_embeds = build_class_embeddings(
			model=model_b0,
			label_dict=label_dict,
			device=device,
			verbose=args.verbose,
		).to(device)

		b0_results = evaluate_regime_stratified(
			model=model_b0,
			val_loader=val_loader,
			all_class_embeds=b0_class_embeds,
			active_mask=active_mask,
			head_mask=head_mask,
			rare_mask=rare_mask,
			device=device,
			verbose=args.verbose,
		)
		del model_b0
		if args.verbose:
			print(f"[eval] B0 global mAP-all: {b0_results.get('ALL', {}).get('map_all', float('nan')):.4f}")

	# Attach checkpoint provenance to results
	results["checkpoint"] = checkpoint_fpath
	results["checkpoint_epoch"] = epoch
	results["clip_model"] = args.clip_model
	results["peft_method"] = args.peft_method

	print_results_table(results, b0_results=b0_results)

	json_path = os.path.join(outputs_dir, "regime_stratified_results.json")
	latex_path = os.path.join(outputs_dir, "regime_stratified_table.tex")
	csv_path = os.path.join(outputs_dir, "regime_stratified_results.csv")

	save_json({"racl": results, "b0": b0_results} if b0_results else results, json_path)
	save_latex_table(results, latex_path, b0_results=b0_results)
	save_csv(results, csv_path, b0_results=b0_results)

if __name__ == "__main__":
	main()