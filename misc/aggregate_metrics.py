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
import statistics
from collections import defaultdict

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
			print(f"{lt:<30} {st:<20} {sp:<15} {dr:<5} {su:<8} {met:7s}@{int(k):2d} = {mean:.4f} ± {std:.4f}")


def main():
		if len(sys.argv) != 4:
				print("Usage: python aggregate_metrics.py seed1.json seed2.json seed3.json")
				sys.exit(1)

		file_paths = sys.argv[1:4]

		collection = collect_values(file_paths)
		results = compute_stats(collection)

		# Print summary to console
		print_summary(results)


if __name__ == "__main__":
		main()