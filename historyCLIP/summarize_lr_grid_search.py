#!/usr/bin/env python3
"""
Aggregate and visualize the isolated equal-budget Learning Rate Grid Search results.

Reads per-run `run_metrics.json` files from:
    <sweep_dir>/<strategy>/lr_<lr>/run_metrics.json

Selection criterion (scale-invariant across cosine-similarity & logit-space PEFT heads):
    Composite Score = 0.5 * (I2T mAP@10 [overall] + T2I mAP@10 [overall])
on the `shared` evaluation protocol (falling back to `standard` if `shared` is absent).

Outputs written directly inside <sweep_dir>/:
    - optimal_lrs.json          : winning LR + full metrics per strategy
    - lr_grid_summary.csv       : flat table of all (strategy, lr) runs
    - lr_sensitivity_curve.png  : 3-panel publication figure (Composite / I2T / T2I vs log LR)
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple


DEFAULT_STRATEGIES = [
	"probe",
	"full",
	"lora",
	"lora_plus",
	"rslora",
	"dora",
	"vera",
	"ia3",
	"tip_adapter_f",
	"clip_adapter_v",
]

DEFAULT_LRS = [
	"5.0e-06",
	"2.0e-05",
	"1.0e-04",
	"5.0e-04",
	"1.0e-03",
	"5.0e-03",
	"1.0e-02",
	"2.0e-02",
	"5.0e-02",
]


def _get_metric(block: Optional[dict], direction: str, tier: str, metric: str, k: str) -> float:
	if not isinstance(block, dict):
		return float("nan")
	val = block.get(direction, {}).get(tier, {}).get(metric, {}).get(str(k))
	if val is None:
		val = block.get(direction, {}).get(tier, {}).get(metric, {}).get(int(k))
	return float(val) if val is not None else float("nan")


def load_sweep_results(
	sweep_dir: str,
	strategies: List[str],
	lr_strs: List[str],
	protocol: str = "shared",
	k: str = "10",
) -> Tuple[Dict[str, Dict[str, dict]], List[dict]]:
	matrix: Dict[str, Dict[str, dict]] = {}
	rows: List[dict] = []

	for strat in strategies:
		matrix[strat] = {}
		for lr_s in lr_strs:
			run_json = os.path.join(sweep_dir, strat, f"lr_{lr_s}", "run_metrics.json")
			rec = {
				"strategy": strat,
				"lr_str": lr_s,
				"lr": float(lr_s),
				"status": "missing",
				"score": float("nan"),
				"i2t_ovr": float("nan"),
				"i2t_head": float("nan"),
				"i2t_rare": float("nan"),
				"t2i_ovr": float("nan"),
				"t2i_head": float("nan"),
				"t2i_rare": float("nan"),
				"i2t_rec": float("nan"),
				"t2i_rec": float("nan"),
				"elapsed_sec": float("nan"),
				"error": "",
				"json_path": run_json,
			}

			if os.path.exists(run_json):
				try:
					with open(run_json, "r") as f:
						payload = json.load(f)
					status = payload.get("status", "ok")
					rec["status"] = status
					rec["elapsed_sec"] = payload.get("elapsed_sec", float("nan"))
					rec["error"] = payload.get("error", "")

					if status == "ok":
						block = payload.get(protocol) or payload.get("standard")
						rec["i2t_ovr"] = _get_metric(block, "i2t", "overall", "mAP", k)
						rec["i2t_head"] = _get_metric(block, "i2t", "head", "mAP", k)
						rec["i2t_rare"] = _get_metric(block, "i2t", "rare", "mAP", k)
						rec["t2i_ovr"] = _get_metric(block, "t2i", "overall", "mAP", k)
						rec["t2i_head"] = _get_metric(block, "t2i", "head", "mAP", k)
						rec["t2i_rare"] = _get_metric(block, "t2i", "rare", "mAP", k)
						rec["i2t_rec"] = _get_metric(block, "i2t", "overall", "Recall", k)
						rec["t2i_rec"] = _get_metric(block, "t2i", "overall", "Recall", k)
						if not (math.isnan(rec["i2t_ovr"]) or math.isnan(rec["t2i_ovr"])):
							rec["score"] = 0.5 * (rec["i2t_ovr"] + rec["t2i_ovr"])
				except Exception as e:
					rec["status"] = "corrupt"
					rec["error"] = str(e)

			matrix[strat][lr_s] = rec
			rows.append(rec)

	return matrix, rows


def plot_lr_sensitivity(
	matrix: Dict[str, Dict[str, dict]],
	strategies: List[str],
	lr_strs: List[str],
	out_path: str,
	protocol: str,
	k: str,
) -> bool:
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError:
		print("[INFO] matplotlib not installed — skipping lr_sensitivity_curve.png")
		return False

	lrs_float = [float(x) for x in lr_strs]
	fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True)
	panels = [
		("score", f"Composite Score: 0.5·(I2T + T2I) mAP@{k} [{protocol}]"),
		("i2t_ovr", f"I2T Overall mAP@{k} [{protocol}]"),
		("t2i_ovr", f"T2I Overall mAP@{k} [{protocol}]"),
	]
	cmap = plt.get_cmap("tab10")

	for s_idx, strat in enumerate(strategies):
		color = cmap(s_idx % 10)
		for ax, (key, title) in zip(axes, panels):
			xs, ys = [], []
			for lr_s, lr_f in zip(lr_strs, lrs_float):
				val = matrix[strat][lr_s][key]
				if not math.isnan(val):
					xs.append(lr_f)
					ys.append(val)
			if xs:
				ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=5, label=strat, color=color)
				best_i = max(range(len(ys)), key=lambda i: ys[i])
				ax.scatter([xs[best_i]], [ys[best_i]], marker="*", s=160, color=color, edgecolors="black", zorder=5)
			ax.set_xscale("log")
			ax.set_title(title, fontsize=11)
			ax.set_xlabel("Learning Rate (log scale)", fontsize=10)
			ax.grid(True, which="both", linestyle="--", alpha=0.4)

	axes[0].set_ylabel(f"mAP@{k}", fontsize=11)
	handles, labels = axes[0].get_legend_handles_labels()
	if handles:
		fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=9)
	plt.tight_layout(rect=[0, 0, 1, 0.93])
	plt.savefig(out_path, dpi=220, bbox_inches="tight")
	plt.close(fig)
	return True


def main():
	parser = argparse.ArgumentParser(description="Summarize isolated LR Grid Search results")
	parser.add_argument(
		"--sweep_dir",
		type=str,
		required=True,
		help="Path to sweep root, e.g. .../HISTORY_X4/lr_grid_search/ViT-L-14-336px/multimodal_canonical_labels/seed_42",
	)
	parser.add_argument("--protocol", type=str, default="shared", choices=["shared", "standard"])
	parser.add_argument("--k", type=str, default="10", help="Top-K for mAP@K selection (default: 10)")
	parser.add_argument("--strategies", type=str, nargs="+", default=DEFAULT_STRATEGIES)
	parser.add_argument("--lrs", type=str, nargs="+", default=DEFAULT_LRS)
	args = parser.parse_args()

	matrix, rows = load_sweep_results(
		sweep_dir=args.sweep_dir,
		strategies=args.strategies,
		lr_strs=args.lrs,
		protocol=args.protocol,
		k=args.k,
	)

	# 1. Determine winner per strategy
	winners: Dict[str, Optional[dict]] = {}
	for strat in args.strategies:
		valid_recs = [
			matrix[strat][lr_s]
			for lr_s in args.lrs
			if matrix[strat][lr_s]["status"] == "ok" and not math.isnan(matrix[strat][lr_s]["score"])
		]
		winners[strat] = max(valid_recs, key=lambda r: r["score"]) if valid_recs else None

	# 2. Print Composite Score Matrix
	header = f"{'Strategy':<16} | " + " | ".join(f"{lr_s:^11}" for lr_s in args.lrs)
	banner_w = max(102, len(header))
	print("\n" + "=" * banner_w)
	print(f"LR GRID SEARCH MATRIX — Composite = 0.5*(I2T + T2I) mAP@{args.k} [{args.protocol}]")
	print(f"Sweep root: {args.sweep_dir}")
	print("=" * banner_w)
	print(header)
	print("-" * len(header))

	n_ok = n_fail = n_miss = 0
	for strat in args.strategies:
		best_lr = winners[strat]["lr_str"] if winners[strat] else None
		cells = []
		for lr_s in args.lrs:
			rec = matrix[strat][lr_s]
			if rec["status"] == "ok" and not math.isnan(rec["score"]):
				n_ok += 1
				star = "*" if lr_s == best_lr else " "
				cells.append(f" {rec['score']:8.4f}{star} ")
			elif rec["status"] == "missing":
				n_miss += 1
				cells.append(f"{'MISS':^11}")
			else:
				n_fail += 1
				cells.append(f"{rec['status'].upper():^11}")
		print(f"{strat:<16} | " + " | ".join(cells))

	print("-" * len(header))
	print(f"Status summary: {n_ok} ok | {n_fail} failed/corrupt | {n_miss} missing  (total={len(rows)})")

	# 3. Print Detailed Winner Table + Boundary Warnings
	print("\n" + "=" * 116)
	print(f"OPTIMAL LEARNING RATES PER STRATEGY (Selection: 0.5*(I2T + T2I) mAP@{args.k} [{args.protocol}])")
	print("=" * 116)
	print(
		f"{'Strategy':<15} | {'Best LR':>9} | {'Score':>7} | "
		f"{'I2T mAP@10 (ovr/head/rare)':^27} | {'T2I mAP@10 (ovr/head/rare)':^27} | {'Boundary'}"
	)
	print("-" * 116)

	optimal_payload = {
		"sweep_dir": args.sweep_dir,
		"protocol": args.protocol,
		"selection_metric": f"0.5 * (I2T_mAP@{args.k}_overall + T2I_mAP@{args.k}_overall)",
		"lr_grid": args.lrs,
		"optimal_lrs": {},
		"details": {},
	}

	for strat in args.strategies:
		w = winners[strat]
		if w is None:
			print(f"{strat:<15} | {'N/A':>9} | {'N/A':>7} | {'—':^27} | {'—':^27} | NO VALID RUNS")
			continue

		edge_flag = "OK (interior)"
		if w["lr_str"] == args.lrs[0]:
			edge_flag = "⚠ LOW EDGE (consider lower)"
		elif w["lr_str"] == args.lrs[-1]:
			edge_flag = "⚠ HIGH EDGE (consider higher)"

		print(
			f"{strat:<15} | {w['lr_str']:>9} | {w['score']:7.4f} | "
			f"{w['i2t_ovr']:7.4f} / {w['i2t_head']:7.4f} / {w['i2t_rare']:7.4f} | "
			f"{w['t2i_ovr']:7.4f} / {w['t2i_head']:7.4f} / {w['t2i_rare']:7.4f} | {edge_flag}"
		)
		optimal_payload["optimal_lrs"][strat] = w["lr_str"]
		optimal_payload["details"][strat] = {
			"best_lr": w["lr_str"],
			"composite_score": w["score"],
			"i2t_mAP_overall": w["i2t_ovr"],
			"i2t_mAP_head": w["i2t_head"],
			"i2t_mAP_rare": w["i2t_rare"],
			"t2i_mAP_overall": w["t2i_ovr"],
			"t2i_mAP_head": w["t2i_head"],
			"t2i_mAP_rare": w["t2i_rare"],
			"boundary": edge_flag,
		}

	# 4. Print copy-pasteable Bash map for roihu_finetuner.sh / lumi_finetuner.sh
	print("\n# Ready-to-paste Bash LR mapping for finetuner scripts:")
	print("declare -A STRATEGY_LRS=(")
	for strat in args.strategies:
		if strat in optimal_payload["optimal_lrs"]:
			print(f'\t["{strat}"]="{optimal_payload["optimal_lrs"][strat]}"')
	print(")")

	# 5. Save JSON, CSV, and Sensitivity Plot inside sweep_dir
	os.makedirs(args.sweep_dir, exist_ok=True)
	opt_json_path = os.path.join(args.sweep_dir, "optimal_lrs.json")
	with open(opt_json_path, "w") as f:
		json.dump(optimal_payload, f, indent=2)

	csv_path = os.path.join(args.sweep_dir, "lr_grid_summary.csv")
	fieldnames = [
		"strategy", "lr_str", "lr", "status", "score",
		"i2t_ovr", "i2t_head", "i2t_rare",
		"t2i_ovr", "t2i_head", "t2i_rare",
		"i2t_rec", "t2i_rec", "elapsed_sec", "error",
	]
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for r in rows:
			writer.writerow({k: r[k] for k in fieldnames})

	plot_path = os.path.join(args.sweep_dir, "lr_sensitivity_curve.png")
	plotted = plot_lr_sensitivity(matrix, args.strategies, args.lrs, plot_path, args.protocol, args.k)

	print(f"\nSaved artifacts inside {args.sweep_dir}:")
	print(f"  ├─ {opt_json_path}")
	print(f"  ├─ {csv_path}")
	if plotted:
		print(f"  └─ {plot_path}")


if __name__ == "__main__":
	main()
