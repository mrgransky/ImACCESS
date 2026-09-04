# """
# Aggregate retrieval metrics from three seed JSON files.

# Usage:
# 		python aggregate_metrics.py seed_1.json seed_2.json seed_42.json

# Output:
# 		- A CSV file 'aggregated_metrics.csv' with all combinations:
# 			label_type, strategy, split, direction, subset, metric, k, mean, std
# 		- A compact summary table printed to stdout for overall subset only
# 			(mAP and Recall at k = 1, 5, 10 for i2t and t2i).
# """

import json
import sys
import os
import statistics
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
# ----------------------------------------------------------------------
# CONFIGURATION – adjust these lists to select which data to aggregate
# ----------------------------------------------------------------------
LABEL_TYPES = [
		# "llm_canonical_labels",
		# "vlm_canonical_labels",
		"multimodal_canonical_labels"
]
STRATEGIES = ["zero_shot", "probe", "full", "lora", "lora_plus", "rslora",
							"dora", "vera", "ia3", "tip_adapter_f", "clip_adapter_v"]
SPLITS = [
		"standard",
		# "shared",
]
DIRECTIONS = ["i2t", "t2i"]
SUBSETS = ["overall", "head", "rare"]          # all three are now active
METRICS = [
		"mAP",
		# "Recall",          # uncomment to include Recall
]
KS = [
		# "1", "3", "5",
		"10",
		# "15", "20",
]

def extract_value(data, keys):
		"""Traverse nested dict; return 0.0 if any key is missing."""
		try:
				for k in keys:
						data = data[k]
				return float(data)
		except (KeyError, TypeError):
				return 0.0

def collect_values(file_paths):
		"""Collect all leaf values across seeds."""
		collection = defaultdict(list)  # key: (label_type, strategy, split, direction, subset, metric, k)

		for path in file_paths:
				with open(path, 'r') as f:
						data = json.load(f)

				for lt in LABEL_TYPES:
						if lt not in data:
								continue
						for st in STRATEGIES:
								if st not in data[lt]:
										continue
								for sp in SPLITS:
										if sp not in data[lt][st]:
												continue
										for dr in DIRECTIONS:
												if dr not in data[lt][st][sp]:
														continue
												for su in SUBSETS:
														if su not in data[lt][st][sp][dr]:
																continue
														for met in METRICS:
																met_dict = data[lt][st][sp][dr][su].get(met, {})
																for k in KS:
																		val = met_dict.get(k, 0.0)
																		key = (lt, st, sp, dr, su, met, k)
																		collection[key].append(val)
		return collection

def compute_stats(collection):
		"""Compute mean and std for each key."""
		results = []
		for key, values in collection.items():
				if len(values) < 3:
						values += [0.0] * (3 - len(values))
				mean = statistics.mean(values)
				std = statistics.stdev(values) if len(values) > 1 else 0.0
				results.append((*key, mean, std))
		return results

def print_summary(results):
		"""
		Print a compact summary table.
		Now includes ALL subsets (overall, head, rare) and respects the
		active METRICS and KS lists.
		"""
		# Build a dictionary keyed by (lt, st, sp, dr, su, met, k) -> (mean, std)
		summary_dict = {}
		for row in results:
				lt, st, sp, dr, su, met, k, mean, std = row
				# No filtering by subset – we keep everything
				key = (lt, st, sp, dr, su, met, k)
				summary_dict[key] = (mean, std)

		print("\n=== Summary Table (all subsets) ===")
		print("Values are mean ± std (averaged over seeds).\n")

		keys = summary_dict.keys()
		for key in keys:
			lt, st, sp, dr, su, met, k = key
			mean, std = summary_dict[key]
			print(f"{lt:<30} {st:<20} {sp:<15} {dr:<5} {su:<8} {met:7s}@{int(k):2d} = {mean:.3f} ± {std:.6f}")

def cell(summary_dict, lt, st, sp, dr, su, met="mAP", k="10", bold=False, underline=False):
	mean, std = summary_dict.get((lt, st, sp, dr, su, met, k), (0.0, 0.0))
	s = f"{mean:.3f} ± {std:.4f}"
	if bold:
		s = f"\\textbf{{{s}}}"
	elif underline:
		s = f"\\underline{{{s}}}"
	return s

def build_summary_dict(results):
		return {(lt, st, sp, dr, su, met, k): (mean, std)
						for (lt, st, sp, dr, su, met, k, mean, std) in results}

def print_table5_row(name, summary_dict, st):
	lt, sp = "multimodal_canonical_labels", "standard"
	cols = []
	for dr in ("i2t", "t2i"):
		for su in ("overall", "head", "rare"):
			cols.append(cell(summary_dict, lt, st, sp, dr, su))
	print(f"{name:<20} " + " | ".join(cols))

def print_table6_row(name, summary_dict, lt):
		sp, st = "shared", "lora"
		cols = []
		for dr in ("i2t", "t2i"):
				for su in ("overall", "head", "rare"):
						cols.append(cell(summary_dict, lt, st, sp, dr, su))
		print(f"{name} & " + " & ".join(cols) + " \\\\")

def select_table6_strategy_and_print_rows(summary_dict):
		"""
		Select the strategy for Table 6 using Table 5 conditions:
			- multimodal canonical supervision
			- standard tier protocol
			- rare-class performance at mAP@10

		Selection criterion:
			mean of I2T-rare and T2I-rare mean mAP@10.

		After selecting the strategy, print the corresponding shared-protocol
		Table 6 retrieval values for LLM, VLM, and multimodal supervision.
		"""
		selection_label_type = "multimodal_canonical_labels"
		selection_split = "standard"
		metric = "mAP"
		k = "10"

		candidate_scores = []

		for strategy in STRATEGIES:
				i2t_key = (
						selection_label_type, strategy, selection_split,
						"i2t", "rare", metric, k
				)
				t2i_key = (
						selection_label_type, strategy, selection_split,
						"t2i", "rare", metric, k
				)

				if i2t_key not in summary_dict or t2i_key not in summary_dict:
						print(
								f"[WARNING] Skipping '{strategy}': missing standard-split "
								f"multimodal rare I2T and/or T2I mAP@10."
						)
						continue

				i2t_mean, i2t_std = summary_dict[i2t_key]
				t2i_mean, t2i_std = summary_dict[t2i_key]
				rare_score = (i2t_mean + t2i_mean) / 2.0

				candidate_scores.append({
						"strategy": strategy,
						"i2t_mean": i2t_mean,
						"i2t_std": i2t_std,
						"t2i_mean": t2i_mean,
						"t2i_std": t2i_std,
						"rare_score": rare_score,
				})

		if not candidate_scores:
				raise ValueError(
						"Could not select a Table 6 strategy: no strategy has both "
						"standard-split multimodal I2T-rare and T2I-rare mAP@10 values."
				)

		candidate_scores.sort(
				key=lambda item: item["rare_score"],
				reverse=True,
		)
		selected = candidate_scores[0]
		selected_strategy = selected["strategy"]

		print("\n=== Table 6 Strategy Selection ===")
		print(
				"Criterion: highest mean of standard-split multimodal "
				"I2T-rare and T2I-rare mAP@10.\n"
		)

		for rank, item in enumerate(candidate_scores, start=1):
				print(
						f"{rank:>2}. {item['strategy']:<16} "
						f"I2T-rare = {item['i2t_mean']:.3f} ± {item['i2t_std']:.3f}, "
						f"T2I-rare = {item['t2i_mean']:.3f} ± {item['t2i_std']:.3f}, "
						f"mean rare score = {item['rare_score']:.3f}"
				)

		print(
				f"\nSelected Table 6 strategy: {selected_strategy} "
				f"(mean rare score = {selected['rare_score']:.3f})"
		)

		print("\n=== Table 6 Retrieval Rows (shared-vocabulary protocol) ===")
		print("Values are mAP@10, mean ± std over the three seeds.\n")

		table6_label_types = [
				("LLM", "llm_canonical_labels"),
				("VLM", "vlm_canonical_labels"),
				("Multimodal (Ours)", "multimodal_canonical_labels"),
		]

		for display_name, label_type in table6_label_types:
				values = []

				for direction in ("i2t", "t2i"):
						for subset in ("overall", "head", "rare"):
								key = (
										label_type, selected_strategy, "shared",
										direction, subset, metric, k
								)

								if key not in summary_dict:
										values.append(r"\texttt{⟨missing⟩}")
										print(
												f"[WARNING] Missing Table 6 value: "
												f"label_type={label_type}, strategy={selected_strategy}, "
												f"split=shared, direction={direction}, subset={subset}"
										)
										continue

								mean, std = summary_dict[key]
								values.append(f"{mean:.3f} $\\pm$ {std:.3f}")

				print(f"{display_name} & " + " & ".join(values) + r" \\")
		
		return selected_strategy

def aggregate_seed_performance(
		json_paths: List[str],
		verbose: bool = True,
		impute_missing_as_zero: bool = False,
		required_seed_count: Optional[int] = None,
) -> Tuple[
		Dict[Tuple[str, ...], Tuple[float, float]],
		Dict[Tuple[str, ...], int],
		Dict[Tuple[str, ...], int],
]:
		"""
		Aggregate retrieval metrics across multiple seed JSON files.

		Parameters
		----------
		impute_missing_as_zero:
				If False (default), a metric missing from a seed file is simply
				excluded from that key's mean/std — this is the scientifically
				correct behavior for reporting final results.

				If True, any key that is not present in ALL of ``json_paths`` is
				padded with 0.0 for each missing seed before computing mean/std.
				This is a TEMPORARY placeholder mode intended only for previewing
				table structure while incomplete runs (e.g. an unfinished DoRA
				fine-tune) are being regenerated. Values produced this way MUST
				NOT be reported as final results.

		required_seed_count:
				Number of seeds a key is expected to have. Defaults to
				``len(json_paths)``. Used only to determine how many zeros to
				pad in when ``impute_missing_as_zero=True``.

		Returns
		-------
		summary_dict:
				(label_type, strategy, split, direction, subset, metric, k) ->
				(mean, sample_std)

		seed_counts:
				Same keys -> number of seed files that ACTUALLY contained the
				metric (never inflated by imputation).

		imputed_counts:
				Same keys -> number of zeros padded in for that key. A key with
				imputed_counts[key] > 0 contains at least one fabricated 0.0 and
				must be treated as provisional.

		Notes
		-----
		- seed_counts always reflects real, observed data only.
		- When impute_missing_as_zero=True, summary_dict[key] may be computed
			over a mix of real values and injected zeros. Use imputed_counts to
			detect this before reporting any number externally.
		"""
		if required_seed_count is None:
				required_seed_count = len(json_paths)

		raw_values = defaultdict(list)
		all_keys_seen = set()

		for path in json_paths:
				with open(path, "r", encoding="utf-8") as file:
						data = json.load(file)

				for label_type, strategies_data in data.items():
						if not isinstance(strategies_data, dict):
								continue

						for strategy, splits_data in strategies_data.items():
								if not isinstance(splits_data, dict):
										continue

								for split, directions_data in splits_data.items():
										if not isinstance(directions_data, dict):
												continue

										for direction, subsets_data in directions_data.items():
												if not isinstance(subsets_data, dict):
														continue

												for subset, metrics_data in subsets_data.items():
														if not isinstance(metrics_data, dict):
																continue

														for metric, ks_data in metrics_data.items():
																if not isinstance(ks_data, dict):
																		continue

																for k, value in ks_data.items():
																		key = (
																				str(label_type),
																				str(strategy),
																				str(split),
																				str(direction),
																				str(subset),
																				str(metric),
																				str(k),
																		)
																		all_keys_seen.add(key)

																		try:
																				raw_values[key].append(float(value))
																		except (TypeError, ValueError):
																				print(
																						"[WARNING] Skipping non-numeric value: "
																						f"file={path}, key={key}, value={value!r}"
																				)

		summary_dict = {}
		seed_counts = {}
		imputed_counts = {}

		for key in all_keys_seen:
				real_values = raw_values.get(key, [])
				n_real = len(real_values)
				n_missing = max(required_seed_count - n_real, 0)

				seed_counts[key] = n_real

				if impute_missing_as_zero and n_missing > 0:
						values_for_stats = real_values + [0.0] * n_missing
						imputed_counts[key] = n_missing
				else:
						values_for_stats = real_values
						imputed_counts[key] = 0

				if not values_for_stats:
						continue

				values_array = np.asarray(values_for_stats, dtype=np.float64)
				mean = float(values_array.mean())
				std = (
						float(values_array.std(ddof=1))
						if len(values_array) > 1
						else 0.0
				)

				summary_dict[key] = (mean, std)

		if verbose:
				count_distribution = defaultdict(int)
				for n_real in seed_counts.values():
						count_distribution[n_real] += 1

				incomplete_keys = sum(
						1 for n_real in seed_counts.values()
						if n_real != len(json_paths)
				)
				imputed_key_count = sum(1 for n in imputed_counts.values() if n > 0)

				print("\n[AGGREGATE] Seed performance summary")
				print(f"  ├─ JSON files           : {len(json_paths)}")
				print(f"  ├─ Total unique keys    : {len(summary_dict)}")

				for n_real, number_of_keys in sorted(count_distribution.items()):
						print(f"  ├─ Keys with {n_real} real seed(s) : {number_of_keys}")

				print(f"  ├─ Incomplete keys      : {incomplete_keys}")

				if impute_missing_as_zero:
						print(
								"  ├─ [PLACEHOLDER MODE] impute_missing_as_zero=True — "
								f"{imputed_key_count} key(s) contain fabricated 0.0 seed(s)."
						)
						print(
								"  └─ These values are PROVISIONAL and must not be reported "
								"as final results."
						)
				else:
						print("  └─ No imputation performed (scientifically valid mode).")

		return summary_dict, seed_counts, imputed_counts

def diagnose_table6_metric_coverage(
		json_paths: List[str],
		strategy: str = "dora",
		metric: str = "mAP",
		k: str = "10",
) -> None:
		"""
		Report per-file availability of every required Table 6 shared-protocol
		metric for a fixed strategy.

		This identifies exactly which seed JSON file lacks each required
		LLM/VLM/multimodal Table 6 result.
		"""
		split = "shared"

		table6_label_types = [
				("LLM", "llm_canonical_labels"),
				("VLM", "vlm_canonical_labels"),
				("Multimodal (Ours)", "multimodal_canonical_labels"),
		]

		required_keys = [
				(
						label_type,
						strategy,
						split,
						direction,
						subset,
						metric,
						k,
				)
				for _, label_type in table6_label_types
				for direction in ("i2t", "t2i")
				for subset in ("overall", "head", "rare")
		]

		print("\n=== Table 6 Per-Seed Coverage Diagnostic ===")
		print(
				f"Required protocol: split={split!r}, strategy={strategy!r}, "
				f"metric={metric}@{k}"
		)

		for path in json_paths:
				with open(path, "r", encoding="utf-8") as file:
						data = json.load(file)

				present_keys = set()

				for label_type, strategies_data in data.items():
						if not isinstance(strategies_data, dict):
								continue

						for current_strategy, splits_data in strategies_data.items():
								if not isinstance(splits_data, dict):
										continue

								for current_split, directions_data in splits_data.items():
										if not isinstance(directions_data, dict):
												continue

										for direction, subsets_data in directions_data.items():
												if not isinstance(subsets_data, dict):
														continue

												for subset, metrics_data in subsets_data.items():
														if not isinstance(metrics_data, dict):
																continue

														for current_metric, ks_data in metrics_data.items():
																if not isinstance(ks_data, dict):
																		continue

																for current_k in ks_data:
																		present_keys.add(
																				(
																						str(label_type),
																						str(current_strategy),
																						str(current_split),
																						str(direction),
																						str(subset),
																						str(current_metric),
																						str(current_k),
																				)
																		)

				missing_keys = [
						key for key in required_keys
						if key not in present_keys
				]

				print(f"\nFile: {path}")
				print(
						f"  ├─ Required Table 6 cells present : "
						f"{len(required_keys) - len(missing_keys)}/{len(required_keys)}"
				)
				print(f"  └─ Required Table 6 cells missing : {len(missing_keys)}")

				if missing_keys:
						for key in missing_keys:
								(
										label_type,
										current_strategy,
										current_split,
										direction,
										subset,
										current_metric,
										current_k,
								) = key

								print(
										"     ├─ "
										f"label_type={label_type}, "
										f"strategy={current_strategy}, "
										f"split={current_split}, "
										f"direction={direction}, "
										f"subset={subset}, "
										f"metric={current_metric}@{current_k}"
								)

def print_table6_shared_aggregation(
		summary_dict,
		seed_counts,
		imputed_counts=None,
		strategy="dora",
		metric="mAP",
		k="10",
		expected_n_seeds=3,
		allow_incomplete=False,
):
		"""
		Print LaTeX-ready Table 6 rows for a fixed strategy under the
		shared-vocabulary protocol.

		allow_incomplete:
				If False (default), raises on any incomplete/imputed cell —
				the strict, publication-safe behavior.

				If True, prints incomplete/imputed cells inline with a visible
				marker instead of raising, so the table structure can be
				previewed while missing evaluations (e.g. an unfinished DoRA
				fine-tune for seed 2) are being regenerated. This mode must
				never be used for the final reported table.
		"""
		imputed_counts = imputed_counts or {}
		split = "shared"

		table6_label_types = [
				("LLM", "llm_canonical_labels"),
				("VLM", "vlm_canonical_labels"),
				("Multimodal (Ours)", "multimodal_canonical_labels"),
		]

		invalid_keys = []

		print(f"\n--- Table 6 rows (shared split, fixed strategy = {strategy}) ---")
		print(f"Metric: {metric}@{k}, mean $\\pm$ sample std.\n")

		for display_name, label_type in table6_label_types:
				values = []

				for direction in ("i2t", "t2i"):
						for subset in ("overall", "head", "rare"):
								key = (label_type, strategy, split, direction, subset, metric, k)

								if key not in summary_dict:
										values.append(r"\texttt{⟨missing⟩}")
										invalid_keys.append((key, 0, 0))
										continue

								n_seeds = seed_counts.get(key, 0)
								n_imputed = imputed_counts.get(key, 0)
								mean, std = summary_dict[key]

								if n_seeds == expected_n_seeds and n_imputed == 0:
										values.append(f"${mean:.3f} ± {std:.3f}$")
								elif allow_incomplete:
										values.append(
												f"${mean:.3f} ± {std:.3f}$"
												f"\\textsuperscript{{[{n_seeds}/{expected_n_seeds} real"
												f"{', +' + str(n_imputed) + ' zero' if n_imputed else ''}]}}"
										)
										invalid_keys.append((key, n_seeds, n_imputed))
								else:
										values.append(r"\texttt{⟨incomplete⟩}")
										invalid_keys.append((key, n_seeds, n_imputed))

				print(f"{display_name} & " + " & ".join(values) + r" \\")

		print(
				"\nColumn order: "
				"I2T Overall, I2T Head, I2T Rare, T2I Overall, T2I Head, T2I Rare."
		)

		if invalid_keys:
				if allow_incomplete:
						print(
								"\n[PLACEHOLDER MODE] The following cells are PROVISIONAL "
								"(incomplete real-seed coverage and/or zero-imputed):"
						)
						for key, n_seeds, n_imputed in invalid_keys:
								print(f"  real_seeds={n_seeds}, imputed_zeros={n_imputed}: {key}")
						print(
								"\n[ACTION REQUIRED] Regenerate the missing seed-2 dora "
								"LLM/VLM shared evaluations, then re-run with "
								"allow_incomplete=False before reporting Table 6."
						)
						return

				print("\n[ERROR] Table 6 must not be reported as a three-seed result yet.")
				print("The following required cells are missing or incomplete:")
				for key, n_seeds, n_imputed in invalid_keys:
						print(f"  seeds={n_seeds}: {key}")

				raise ValueError(
						"Table 6 contains incomplete seed coverage. "
						"Regenerate the missing evaluation output before reporting it."
				)

		print(f"\n[OK] All 18 Table 6 values are present for all {expected_n_seeds} seeds.")

def main():
	if len(sys.argv) != 4:
		print("Usage: python aggregate_metrics.py seed1.json seed2.json seed3.json")
		sys.exit(1)

	file_paths = sys.argv[1:4]
	collection = collect_values(file_paths)
	results = compute_stats(collection)
	# Print summary to console
	print_summary(results)

	summary_dict = build_summary_dict(results)

	print("\n--- Table 5 rows (standard split, multimodal supervision) ---")
	for st in STRATEGIES:
			print_table5_row(st, summary_dict, st)

	print("\n--- Table 6 rows (shared split, LoRA) ---")
	for lt in LABEL_TYPES:
			print_table6_row(lt, summary_dict, lt)


	SEEDS_DIR = "/home/farid/datasets/trash/results/h4/output_roihu"
	seed_json_paths = [
		os.path.join(SEEDS_DIR, "seed_1_performance.json"),
		os.path.join(SEEDS_DIR, "seed_2_performance.json"),
		os.path.join(SEEDS_DIR, "seed_42_performance.json"),
	]

	for stg in STRATEGIES:
		diagnose_table6_metric_coverage(
			json_paths=seed_json_paths,
			strategy=stg,
			metric="mAP",
			k="10",
		)
		print("="*100)

	summary_dict, seed_counts, imputed_counts = aggregate_seed_performance(
			json_paths=seed_json_paths,
			verbose=True,
			impute_missing_as_zero=True,   # TEMPORARY — remove once seed 2 dora is regenerated
	)

	print_table6_shared_aggregation(
			summary_dict=summary_dict,
			seed_counts=seed_counts,
			imputed_counts=imputed_counts,
			strategy="dora",
			metric="mAP",
			k="10",
			allow_incomplete=True,          # TEMPORARY preview mode
	)

if __name__ == "__main__":
		main()