"""
plot_tier_cardinality.py
------------------------
Generates a per-tier label cardinality distribution figure (box plots +
strip overlay) broken down by Head / Torso / Tail frequency tiers, for
the HISTORY-X4 benchmark section of the paper.

This figure replaces the rank-frequency plot in the multi-label setting.
It directly motivates:
	(a) the asymmetric BCE reweighting decision (I2T direction)
	(b) why tail-class supervision is structurally harder regardless of model

Usage
-----
		from plot_tier_cardinality import plot_tier_cardinality_distribution

		stats = plot_tier_cardinality_distribution(
				df                = your_dataframe,
				label_col         = "multimodal_canonical_labels",
				output_path       = "plots/tier_cardinality_distribution.png",
				tau_head          = 500,    # f(l) >= 500  => HEAD
				tau_torso         = 50,     # f(l) <  50   => TAIL
				verbose           = True,
		)

Arguments
---------
df : pd.DataFrame
		Must contain the multi-label column (list-valued) and optionally
		an image identifier column.

label_col : str
		Column holding per-sample label lists (canonical labels).
		Default: "multimodal_canonical_labels".

output_path : str
		Full path for the saved PNG.

tau_head : int or None
		Frequency threshold: labels with f(l) >= tau_head are HEAD.
		If None, derived from head_pct.

tau_torso : int or None
		Frequency threshold: labels with f(l) < tau_torso are TAIL.
		If None, derived from tail_pct.

head_pct : float
		Rank-based fallback: top head_pct fraction of vocab = HEAD.

tail_pct : float
		Rank-based fallback: bottom tail_pct fraction of vocab = TAIL.

figsize : tuple
		Default (12, 7).

dpi : int
		Default 300.

verbose : bool
		Prints all placeholder values and a ready-to-paste LaTeX snippet.

Returns
-------
dict — all computed statistics keyed by placeholder description.
"""
import visualize as viz
from utils import *

if __name__ == "__main__":
	print("Running smoke-test with synthetic multi-label data …")
	sim_df = pd.read_csv("/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv")
	
	viz.plot_tier_cardinality_distribution(
		df          = sim_df,
		label_col   = "multimodal_canonical_labels",
		output_path = "smoke_test_tier_cardinality.png",
		head_pct    = 0.10,
		tail_pct    = 0.50,
		verbose     = True,
		figsize     = (13, 6),
		dpi         = 200,
	)