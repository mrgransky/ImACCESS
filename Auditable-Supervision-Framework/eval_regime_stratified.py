# eval_regime_stratified.py
#
# Regime-Stratified Evaluation Script
# ─────────────────────────────────────────────────────────────────────────────
# Loads a trained Stage 5 checkpoint and evaluates retrieval metrics
# stratified by modality-conflict regime (AGREEMENT / SOFT_CONFLICT /
# HARD_CONFLICT).
#
# Outputs
# ───────
#   <output_dir>/regime_stratified_results.json   — full metrics dict
#   <output_dir>/regime_stratified_table.tex      — LaTeX-ready table
#   <output_dir>/regime_stratified_results.csv    — flat CSV for plotting
#
# Usage
# ─────
#   python eval_regime_stratified.py \
#       --checkpoint  /path/to/stage5_best_model.pt \
#       --metadata    /path/to/dataset.csv \
#       --supervision /path/to/auditable_supervision_matrix.parquet \
#       --clip_model  ViT-L/14 \
#       --peft_method lora \
#       --batch_size  256 \
#       --output_dir  ./eval_outputs \
#       --verbose
#
# Design contract
# ───────────────
# • Regime buckets are determined by the parquet's `regime` column — the
#   same authority used during training.  No re-routing is performed here.
# • Samples with MISSING_MODALITY / INVALID_JSON regimes are excluded from
#   all per-regime tables but counted in a separate "skipped" row.
# • Gap_rel = (mAP_rare - mAP_head) / (mAP_head + ε) is reported per bucket
#   and globally.  A less-negative Gap_rel indicates better tail recovery.
# • The script is read-only with respect to the checkpoint and parquet.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# 1. Regime-Bucketed Evaluation
# ─────────────────────────────────────────────────────────────────────────────

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
		-------
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
		bucket_scores:  Dict[str, List[torch.Tensor]] = defaultdict(list)
		bucket_targets: Dict[str, List[torch.Tensor]] = defaultdict(list)
		regime_counts:  Dict[str, int]                = defaultdict(int)
		n_total = 0

		if verbose:
				print(f"\n[eval_regime_stratified] Running inference on {len(val_loader.dataset):,} samples …")

		for batch_idx, batch in enumerate(val_loader):
				if not batch:
						continue

				images    = batch["image"].to(device, non_blocking=True)
				label_vec = batch["label_vec"].to(device, non_blocking=True)
				regimes   = batch["regime"]   # List[str], length B

				# Forward: image embeddings → cosine similarities
				image_embeds = torch.nn.functional.normalize(
						model.encode_image(images), dim=-1
				).float()                                          # [B, D]
				scores = torch.matmul(image_embeds, class_embeds.T)  # [B, C]

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

				if verbose and (batch_idx % max(1, len(val_loader) // 10) == 0):
						print(f"  [{batch_idx:04d}/{len(val_loader):04d}] processed {n_total:,} samples")

		if verbose:
				print(f"\n[eval_regime_stratified] Regime distribution:")
				for r, cnt in sorted(regime_counts.items()):
						pct = cnt / max(n_total, 1) * 100
						print(f"  ├─ {r:<22s}: {cnt:>6,} ({pct:5.1f}%)")
				print(f"  └─ Total: {n_total:,}")

		# ── Compute metrics per bucket ────────────────────────────────────────────
		results: Dict[str, Any] = {
				"regime_counts": dict(regime_counts),
				"n_total":       n_total,
		}

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
		results["SKIPPED"] = {"count": regime_counts.get("MISSING_MODALITY", 0)
																	+ regime_counts.get("INVALID_JSON", 0)
																	+ regime_counts.get(FALLBACK_REGIME, 0)}

		return results


def _compute_gap_rel(map_head: float, map_rare: float) -> float:
		"""
		Gap_rel = (mAP_rare - mAP_head) / (mAP_head + ε)
		Negative → head dominates; closer to 0 → better tail recovery.
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Output Formatters
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: Any, decimals: int = 4) -> str:
		"""Format a float for display; return '—' for NaN."""
		if isinstance(v, float) and np.isnan(v):
				return "—"
		if isinstance(v, float):
				return f"{v:.{decimals}f}"
		return str(v)

def print_results_table(results: Dict[str, Any]) -> None:
		"""Pretty-print a regime-stratified results table to stdout."""
		header = (
				f"\n{'─'*100}\n"
				f"{'Regime':<22s} {'N':>7s} "
				f"{'mAP-all':>9s} {'mAP-head':>9s} {'mAP-rare':>9s} "
				f"{'Gap_rel':>9s} {'P@1':>7s} {'P@5':>7s} {'nDCG@5':>8s}\n"
				f"{'─'*100}"
		)
		print(header)

		for bucket in VALID_REGIMES + ["ALL"]:
				m = results.get(bucket, {})
				if not m or m.get("n_samples", 0) == 0:
						print(f"  {REGIME_DISPLAY.get(bucket, bucket):<20s}  (no samples)")
						continue
				print(
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

		skipped = results.get("SKIPPED", {}).get("count", 0)
		print(f"{'─'*100}")
		print(f"  {'Skipped (MISSING/INVALID)':<20s} {skipped:>7,}")
		print(f"  {'Total':<20s} {results.get('n_total', 0):>7,}")
		print(f"{'─'*100}\n")

def save_latex_table(results: Dict[str, Any], output_path: str) -> None:
		"""
		Write a LaTeX booktabs table to output_path.
		Suitable for direct inclusion in the paper's experiments section.
		"""
		lines = [
				r"\begin{table}[t]",
				r"\centering",
				r"\caption{Regime-Stratified Retrieval Metrics on HISTORY-X4 Test Set.}",
				r"\label{tab:regime_stratified}",
				r"\begin{tabular}{lrrrrrrrr}",
				r"\toprule",
				r"Regime & $N$ & mAP-all & mAP-head & mAP-rare & $\text{Gap}_{\text{rel}}$ "
				r"& P@1 & P@5 & nDCG@5 \\",
				r"\midrule",
		]

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
						f"{_fmt(m.get('ndcg@5'))} \\\\"
				)
				lines.append(row)

		lines.append(r"\midrule")

		# Global row
		m_all = results.get("ALL", {})
		if m_all and m_all.get("n_samples", 0) > 0:
				row = (
						r"\textbf{All (Global)} & "
						f"\\textbf{{{m_all['n_samples']:,}}} & "
						f"\\textbf{{{_fmt(m_all.get('map_all'))}}} & "
						f"\\textbf{{{_fmt(m_all.get('map_head'))}}} & "
						f"\\textbf{{{_fmt(m_all.get('map_rare'))}}} & "
						f"\\textbf{{{_fmt(m_all.get('gap_rel'))}}} & "
						f"\\textbf{{{_fmt(m_all.get('p@1'))}}} & "
						f"\\textbf{{{_fmt(m_all.get('p@5'))}}} & "
						f"\\textbf{{{_fmt(m_all.get('ndcg@5'))}}} \\\\"
				)
				lines.append(row)

		lines += [
				r"\bottomrule",
				r"\end{tabular}",
				r"\end{table}",
		]

		with open(output_path, "w", encoding="utf-8") as f:
				f.write("\n".join(lines) + "\n")

		print(f"[save_latex_table] Written → {output_path}")

def save_csv(results: Dict[str, Any], output_path: str) -> None:
		"""
		Write a flat CSV for downstream plotting (e.g., matplotlib / seaborn).
		One row per regime bucket.
		"""
		fieldnames = [
				"regime", "n_samples",
				"map_all", "map_head", "map_rare", "gap_rel",
				"p@1", "p@5", "ndcg@5",
		]
		rows = []
		for bucket in VALID_REGIMES + ["ALL"]:
				m = results.get(bucket, {})
				if not m:
						continue
				rows.append({
						"regime":    bucket,
						"n_samples": m.get("n_samples", 0),
						"map_all":   m.get("map_all",  float("nan")),
						"map_head":  m.get("map_head", float("nan")),
						"map_rare":  m.get("map_rare", float("nan")),
						"gap_rel":   m.get("gap_rel",  float("nan")),
						"p@1":       m.get("p@1",      float("nan")),
						"p@5":       m.get("p@5",      float("nan")),
						"ndcg@5":    m.get("ndcg@5",   float("nan")),
				})

		with open(output_path, "w", newline="", encoding="utf-8") as f:
				writer = csv.DictWriter(f, fieldnames=fieldnames)
				writer.writeheader()
				writer.writerows(rows)

		print(f"[save_csv] Written → {output_path}")

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

		print(f"[save_json] Written → {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Regime-Stratified Evaluation for Stage 5 RACL Model",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	# Required
	p.add_argument("--checkpoint", "-ckpt", required=True, help="Path to stage5_best_model.pt")
	p.add_argument("--metadata", "-csv", required=True, help="Path to dataset.csv (train/val splits inferred from this)")
	p.add_argument("--supervision", "-sup",  required=True, help="Path to auditable_supervision_matrix.parquet")

	# Model
	p.add_argument("--clip_model",  default="ViT-B/32",help="CLIP backbone used during training")
	p.add_argument(
		"--peft_method", 
		default="lora", 
		choices=[
			"lora", "lora_plus", "dora", "rslora", "ia3", "vera",
			"tip_adapter", "tip_adapter_f",
			"clip_adapter_v", "clip_adapter_t", "clip_adapter_vt",
			"probe", "full"
		],
		help="PEFT method used during training"
	)
	# Data
	p.add_argument("--batch_size",  type=int, default=256)
	p.add_argument("--num_workers", type=int, default=4)
	p.add_argument("--resolution",  type=int, default=224)
	p.add_argument("--id_col",      default="doc_url")
	p.add_argument("--text_col",    default="multimodal_labels")

	# Loss masks (must match training config for consistent head/rare split)
	p.add_argument("--pw_mode", default="sqrt", choices=["log", "sqrt", "linear"], help="pos_weight mode — must match training config")
	p.add_argument("--pw_max_cap", type=float, default=50.0, help="pos_weight cap — must match training config")

	# Output
	p.add_argument("--output_dir", "-o", default="./eval_outputs", help="Directory for JSON / LaTeX / CSV outputs")

	# Misc
	p.add_argument("--verbose", "-v", action="store_true")

	return p.parse_args()

def main() -> None:
		args = parse_args()
		os.makedirs(args.output_dir, exist_ok=True)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		print(f"\n{'='*80}")
		print(f"[eval_regime_stratified] Regime-Stratified Evaluation")
		print(f"  ├─ Checkpoint  : {args.checkpoint}")
		print(f"  ├─ Metadata    : {args.metadata}")
		print(f"  ├─ Supervision : {args.supervision}")
		print(f"  ├─ CLIP model  : {args.clip_model}")
		print(f"  ├─ PEFT method : {args.peft_method}")
		print(f"  ├─ Device      : {device}")
		print(f"  └─ Output dir  : {args.output_dir}")
		print(f"{'='*80}\n")

		# ── 1. DataLoaders ────────────────────────────────────────────────────────
		# We only need the val_loader for evaluation.
		# train_loader is used solely to compute loss masks (head/rare split).
		train_loader, val_loader = get_stage5_dataloaders(
				metadata_fpth=args.metadata,
				supervision_fpth=args.supervision,
				batch_size=args.batch_size,
				num_workers=args.num_workers,
				input_resolution=args.resolution,
				id_col=args.id_col,
				text_col=args.text_col,
				verbose=args.verbose,
		)
		label_dict  = train_loader.dataset.label_dict
		num_classes = len(label_dict)

		# ── 2. Loss masks (head / rare split) ─────────────────────────────────────
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

		# ── 3. Model + PEFT ───────────────────────────────────────────────────────
		model, _ = clip.load(args.clip_model, device=device)
		model.float()
		model, _ = setup_peft(
				model=model,
				peft_method=args.peft_method,
				verbose=args.verbose,
		)
		model = model.to(device)

		# ── 4. Load checkpoint ────────────────────────────────────────────────────
		epoch, ckpt_metrics = load_checkpoint(
				ckpt_path=args.checkpoint,
				model=model,
				device=device,
				verbose=True,
		)
		print(f"[eval] Evaluating checkpoint from epoch {epoch}")
		if ckpt_metrics:
				print(
						f"  ├─ Checkpoint val_loss : {ckpt_metrics.get('val_loss', float('nan')):.6f}"
				)
				print(
						f"  └─ Checkpoint mAP-all  : {ckpt_metrics.get('val_map_all', float('nan')):.4f}"
				)

		# ── 5. Build class embeddings ─────────────────────────────────────────────
		all_class_embeds = build_class_embeddings(
				model=model,
				label_dict=label_dict,
				device=device,
				verbose=args.verbose,
		).to(device)

		# ── 6. Regime-stratified evaluation ───────────────────────────────────────
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

		# Attach checkpoint provenance to results
		results["checkpoint"] = args.checkpoint
		results["checkpoint_epoch"] = epoch
		results["clip_model"]  = args.clip_model
		results["peft_method"] = args.peft_method

		# ── 7. Print table ────────────────────────────────────────────────────────
		print_results_table(results)

		# ── 8. Save outputs ───────────────────────────────────────────────────────
		json_path  = os.path.join(args.output_dir, "regime_stratified_results.json")
		latex_path = os.path.join(args.output_dir, "regime_stratified_table.tex")
		csv_path   = os.path.join(args.output_dir, "regime_stratified_results.csv")

		save_json(results,  json_path)
		save_latex_table(results, latex_path)
		save_csv(results,   csv_path)

		print(f"\n[eval_regime_stratified] Done.")
		print(f"  ├─ JSON   → {json_path}")
		print(f"  ├─ LaTeX  → {latex_path}")
		print(f"  └─ CSV    → {csv_path}")
		print(f"{'='*80}\n")

if __name__ == "__main__":
		main()