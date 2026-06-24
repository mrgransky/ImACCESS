# stage5_dataset_loader.py
#
# Stage 5: Regime-Aware Dataset Loader
# ─────────────────────────────────────────────────────────────────────────────
# Extends HistoricalArchivesMultiLabelDataset to consume the
# auditable_supervision_matrix.parquet produced by Stage 4.
#
# Each sample now returns a 7-tuple:
#   (image_tensor, tokenized_text, label_vector,
#    hn_vector, w_pos, w_neg, regime)
#
# Design contract
# ───────────────
# • The parquet is the SOLE authority on positive_targets, hard_negatives,
#   w_pos, w_neg, and regime.  The CSV label column is kept only as a
#   human-readable fallback for samples missing from the parquet.
# • Samples absent from the parquet are routed to MISSING_MODALITY so
#   resolve_regime_weights() in stage5_regime_aware_loss.py zeroes their
#   gradient automatically — no silent data leakage.
# • The label_dict is built from the parquet's positive_targets union
#   hard_negatives (the canonical vocabulary V from the Bridge), NOT from
#   the raw CSV column.  This guarantees that label indices are consistent
#   with Stage 4 outputs.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

HISTORY_CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "historyCLIP")
sys.path.insert(0, HISTORY_CLIP_DIR)

print(f"sys_path: {sys.path}")

from utils import *
import clip
from historyXN_dataset_loader import (
	get_preprocess,
	ImageCache,
	get_estimated_image_size_mb,
	get_cache_size,
	dtypes,
)

PARQUET_SCHEMA = {
	"doc_url",
	"positive_targets",
	"hard_negatives",
	"w_pos",
	"w_neg",
	"regime",
}

FALLBACK_REGIME   = "MISSING_MODALITY"   # used when id not in parquet
FALLBACK_W_POS    = 0.0
FALLBACK_W_NEG    = 0.0

def load_supervision_matrix(parquet_path: str, verbose: bool = True,) -> Tuple[pd.DataFrame, Dict[str, int]]:
	"""
	Load auditable_supervision_matrix.parquet and build the canonical
	label_dict from the union of all positive_targets and hard_negatives.
	Returns
	-------
	df_sup    : DataFrame indexed by id (str)
	label_dict: {canonical_label: int_index}  — deterministic, sorted
	"""
	assert os.path.isfile(parquet_path), f"[load_supervision_matrix] Parquet not found: {parquet_path}"
	df_sup = pd.read_parquet(parquet_path)
	print(df_sup.head(10))

	# Validate schema
	missing_cols = PARQUET_SCHEMA - set(df_sup.columns)
	assert not missing_cols, f"[load_supervision_matrix] Missing columns in parquet: {missing_cols}"
	
	# Ensure id is string and set as index for O(1) lookup
	df_sup["doc_url"] = df_sup["doc_url"].astype(str)
	df_sup = df_sup.set_index("doc_url")
	
	# Parse list columns if stored as strings (parquet may serialise lists as str)
	for col in ("positive_targets", "hard_negatives"):
		df_sup[col] = df_sup[col].apply(
			lambda x: list(x) if isinstance(x, (list, np.ndarray))
			else ast.literal_eval(x) if isinstance(x, str)
			else []
		)

	# Build canonical vocabulary V from all positive + hard-negative targets
	all_labels: set = set()
	for targets in df_sup["positive_targets"]:
		all_labels.update(targets)
	for targets in df_sup["hard_negatives"]:
		all_labels.update(targets)

	label_dict = {lbl: idx for idx, lbl in enumerate(sorted(all_labels))}

	if verbose:
		n_samples   = len(df_sup)
		n_labels    = len(label_dict)
		regime_dist = df_sup["regime"].value_counts().to_dict()
		n_hn        = (df_sup["hard_negatives"].apply(len) > 0).sum()
		print(f"\n[load_supervision_matrix] {parquet_path}")
		print(f"  ├─ Samples          : {n_samples:,}")
		print(f"  ├─ Canonical vocab  : {n_labels:,} labels")
		print(f"  ├─ Samples with HN  : {n_hn:,} ({n_hn/n_samples*100:.1f}%)")
		print(f"  ├─ Regime dist      : {regime_dist}")
		print(f"  ├─ w_pos range      : [{df_sup['w_pos'].min():.3f}, {df_sup['w_pos'].max():.3f}]")
		print(f"  └─ w_neg range      : [{df_sup['w_neg'].min():.3f}, {df_sup['w_neg'].max():.3f}]")
	
	return df_sup, label_dict

class Stage5RegimeAwareDataset(Dataset):
		"""
		Extends HistoricalArchivesMultiLabelDataset with Stage 4 supervision
		provenance.

		__getitem__ returns a dict (not a tuple) to keep the 7 fields named
		and avoid positional confusion in the training loop:

				{
						"image"      : FloatTensor [3, H, W]
						"text"       : LongTensor  [77]          — CLIP tokens
						"label_vec"  : FloatTensor [C]            — multi-hot positives
						"hn_vec"     : FloatTensor [C]            — multi-hot hard negatives
						"w_pos"      : float scalar tensor
						"w_neg"      : float scalar tensor
						"regime"     : str
				}

		Fallback policy (sample_id absent from parquet)
		────────────────────────────────────────────────
		label_vec  → built from CSV col (best-effort)
		hn_vec     → all zeros
		w_pos      → FALLBACK_W_POS (0.0)  → valid_mask=False in loss
		w_neg      → FALLBACK_W_NEG (0.0)
		regime     → FALLBACK_REGIME ("MISSING_MODALITY")
		"""

		def __init__(
				self,
				dataset_name:   str,
				train:          bool,
				data_frame:     pd.DataFrame,       # CSV-derived split DataFrame
				transform,
				label_dict:     Dict[str, int],     # from load_supervision_matrix()
				df_supervision: pd.DataFrame,       # parquet indexed by sample_id
				id_col:         str  = "doc_url",   # column in data_frame holding sample_id
				text_col:       str  = "multimodal_labels",
				text_augmentation: bool = True,
				cache_size:     int  = 0,
				cache_workers:  int  = 4,
				verbose:        bool = True,
		):
				self.dataset_name   = dataset_name
				self.train          = train
				self.data_frame     = data_frame.reset_index(drop=True)
				self.transform      = transform
				self.label_dict     = label_dict
				self._num_classes   = len(label_dict)
				self.df_supervision = df_supervision   # indexed by sample_id
				self.id_col         = id_col
				self.text_col       = text_col
				self.text_augmentation = text_augmentation
				self.split          = "Train" if train else "Validation"
				self.cache_size     = cache_size

				# Core arrays from CSV
				self.images     = self.data_frame["img_path"].values
				self.sample_ids = self.data_frame[id_col].astype(str).values
				self.labels     = self.data_frame[text_col].values   # raw label strings (fallback)

				# Image cache (mirrors HistoricalArchivesMultiLabelDataset)
				if self.cache_size > 0:
						self.image_cache = ImageCache(
								self.images,
								self.cache_size,
								num_workers=cache_workers,
						)
						self.cache_hits   = 0
						self.cache_misses = 0
				else:
						self.image_cache = None

				# Pre-tokenise text (same strategy as parent class)
				self.text_cache = [None] * len(self.data_frame)
				self._preload_texts()

				# Coverage diagnostic
				if verbose:
						self._print_coverage_diagnostic()

		# ── Text helpers ──────────────────────────────────────────────────────────

		def _preload_texts(self):
				for idx in range(len(self.labels)):
						self.text_cache[idx] = self._tokenize_labels(self.labels[idx])

		def _tokenize_labels(self, labels_str: str) -> torch.Tensor:
				try:
						labels = ast.literal_eval(labels_str)
						text   = self._create_text_description(labels)
						return clip.tokenize(text).squeeze(0)
				except (ValueError, SyntaxError):
						return clip.tokenize("").squeeze(0)

		def _create_text_description(self, labels: list) -> str:
				if not labels:
						return ""
				if not self.train or not self.text_augmentation:
						return " ".join(labels)
				if len(labels) == 1:
						return labels[0]
				if len(labels) == 2:
						return f"{labels[0]} and {labels[1]}"
				np.random.shuffle(labels)
				return ", ".join(labels[:-1]) + f", and {labels[-1]}"

		# ── Label vector helpers ──────────────────────────────────────────────────

		def _targets_to_vector(self, targets: List[str]) -> torch.Tensor:
				"""Convert a list of canonical label strings to a multi-hot vector."""
				vec = torch.zeros(self._num_classes, dtype=torch.float32)
				for lbl in targets:
						if lbl in self.label_dict:
								vec[self.label_dict[lbl]] = 1.0
				return vec

		def _csv_label_vector(self, labels_str: str) -> torch.Tensor:
				"""Fallback: build label_vec from raw CSV label string."""
				vec = torch.zeros(self._num_classes, dtype=torch.float32)
				try:
						for lbl in ast.literal_eval(labels_str):
								if lbl in self.label_dict:
										vec[self.label_dict[lbl]] = 1.0
				except (ValueError, SyntaxError):
						pass
				return vec

		# ── Image helpers (mirrors parent class) ─────────────────────────────────

		def _load_image(self, idx: int) -> Image.Image:
				if self.image_cache is not None:
						cached = self.image_cache.get(idx)
						if cached is not None:
								self.cache_hits += 1
								return cached
						self.cache_misses += 1
				img_path = self.images[idx]
				try:
						with Image.open(img_path) as img:
								return img.convert("RGB")
				except Exception as e:
						print(f"[Stage5Dataset] Error loading {img_path}: {e}")
						return Image.new("RGB", (224, 224), color="white")

		# ── Supervision lookup ────────────────────────────────────────────────────

		def _get_supervision(self, sample_id: str, fallback_labels_str: str) -> Dict:
				"""
				Look up Stage 4 supervision for sample_id.
				Returns a dict with keys: label_vec, hn_vec, w_pos, w_neg, regime.
				Falls back gracefully if sample_id is absent from the parquet.
				"""
				if sample_id in self.df_supervision.index:
						row        = self.df_supervision.loc[sample_id]
						label_vec  = self._targets_to_vector(row["positive_targets"])
						hn_vec     = self._targets_to_vector(row["hard_negatives"])
						w_pos      = float(row["w_pos"])
						w_neg      = float(row["w_neg"])
						regime     = str(row["regime"])
				else:
						# Fallback: use CSV labels, zero hard negatives, skip gradient
						label_vec  = self._csv_label_vector(fallback_labels_str)
						hn_vec     = torch.zeros(self._num_classes, dtype=torch.float32)
						w_pos      = FALLBACK_W_POS
						w_neg      = FALLBACK_W_NEG
						regime     = FALLBACK_REGIME

				return {
						"label_vec": label_vec,
						"hn_vec":    hn_vec,
						"w_pos":     torch.tensor(w_pos, dtype=torch.float32),
						"w_neg":     torch.tensor(w_neg, dtype=torch.float32),
						"regime":    regime,
				}

		# ── Diagnostics ───────────────────────────────────────────────────────────

		def _print_coverage_diagnostic(self):
				ids_in_parquet = set(self.df_supervision.index)
				ids_in_split   = set(self.sample_ids)
				matched        = ids_in_split & ids_in_parquet
				missing        = ids_in_split - ids_in_parquet
				print(f"\n[Stage5Dataset][{self.split}] Supervision coverage")
				print(f"  ├─ Split samples        : {len(ids_in_split):,}")
				print(f"  ├─ Matched in parquet   : {len(matched):,} ({len(matched)/max(len(ids_in_split),1)*100:.1f}%)")
				print(f"  └─ Fallback (missing)   : {len(missing):,} ({len(missing)/max(len(ids_in_split),1)*100:.1f}%)")
				if len(missing) > 0:
						print(
								f"  ⚠  {len(missing):,} samples will use MISSING_MODALITY fallback "
								f"(zero gradient contribution)."
						)

		@property
		def unique_labels(self) -> List[str]:
				return sorted(self.label_dict.keys())

		def get_cache_stats(self) -> Optional[Dict]:
				if self.image_cache is not None:
						total = self.cache_hits + self.cache_misses
						return {
								"cache_size": len(self.image_cache),
								"hits":       self.cache_hits,
								"misses":     self.cache_misses,
								"hit_rate":   self.cache_hits / max(total, 1) * 100,
						}
				return None

		def __len__(self) -> int:
				return len(self.data_frame)

		def __repr__(self) -> str:
				cache_str = (
						f"Image Cache: Size={len(self.image_cache)}/{len(self.images)}"
						if self.image_cache else "Image Cache: DISABLED"
				)
				return (
						f"{self.dataset_name} [Stage5]\n"
						f"\tSplit        : {self.split} ({len(self.data_frame):,} samples)\n"
						f"\tNum classes  : {self._num_classes:,}\n"
						f"\tParquet rows : {len(self.df_supervision):,}\n"
						f"\t{cache_str}\n"
				)

		def __getitem__(self, idx: int) -> Dict[str, Any]:
				sample_id = self.sample_ids[idx]
				try:
						image        = self._load_image(idx)
						image_tensor = self.transform(image)
						text_tensor  = self.text_cache[idx]
						supervision  = self._get_supervision(sample_id, self.labels[idx])

						return {
								"image":     image_tensor,
								"text":      text_tensor,
								"label_vec": supervision["label_vec"],
								"hn_vec":    supervision["hn_vec"],
								"w_pos":     supervision["w_pos"],
								"w_neg":     supervision["w_neg"],
								"regime":    supervision["regime"],
						}
				except Exception as e:
						print(f"[Stage5Dataset] Error at idx={idx} id={sample_id}: {e}")
						return {
								"image":     torch.zeros(3, 224, 224),
								"text":      torch.zeros(77, dtype=torch.long),
								"label_vec": torch.zeros(self._num_classes),
								"hn_vec":    torch.zeros(self._num_classes),
								"w_pos":     torch.tensor(0.0),
								"w_neg":     torch.tensor(0.0),
								"regime":    FALLBACK_REGIME,
						}

def stage5_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
		"""
		Collates a list of per-sample dicts into batched tensors.
		regime is kept as a List[str] (not a tensor) for resolve_regime_weights().
		"""
		valid = [s for s in batch if s is not None]
		if not valid:
				return {}

		return {
				"image":     torch.stack([s["image"]     for s in valid]),
				"text":      torch.stack([s["text"]      for s in valid]),
				"label_vec": torch.stack([s["label_vec"] for s in valid]),
				"hn_vec":    torch.stack([s["hn_vec"]    for s in valid]),
				"w_pos":     torch.stack([s["w_pos"]     for s in valid]),
				"w_neg":     torch.stack([s["w_neg"]     for s in valid]),
				"regime":    [s["regime"] for s in valid],          # List[str]
		}

def get_stage5_dataloaders(
		metadata_fpth:      str,
		supervision_fpth:   str,        # path to auditable_supervision_matrix.parquet
		batch_size:         int,
		num_workers:        int,
		input_resolution:   int,
		id_col:             str  = "doc_url",
		text_col:           str  = "multimodal_labels",
		cache_size:         Optional[int] = None,
		verbose:            bool = True,
) -> Tuple[DataLoader, DataLoader]:
		"""
		Factory that mirrors get_multi_label_dataloaders() but wires in the
		Stage 4 supervision matrix.

		Parameters
		----------
		metadata_fpth     : path to the full CSV (used to locate _train/_val splits)
		supervision_fpth  : path to auditable_supervision_matrix.parquet
		id_col            : column in the CSV that matches sample_id in the parquet
		text_col          : label column used for text tokenisation & fallback vectors
		"""
		ddir         = os.path.dirname(metadata_fpth)
		dataset_name = os.path.basename(ddir)

		print(f"\n[Stage5] Creating regime-aware dataloaders for {dataset_name}")
		print(f"  ├─ Metadata  : {metadata_fpth}")
		print(f"  ├─ Parquet   : {supervision_fpth}")
		print(f"  ├─ id_col    : {id_col}")
		print(f"  └─ text_col  : {text_col}")

		# ── Load supervision matrix (single shared object for train + val) ────────
		df_supervision, label_dict = load_supervision_matrix(
			parquet_path=supervision_fpth,
			verbose=verbose,
		)

		# ── Load CSV splits ───────────────────────────────────────────────────────
		train_csv = metadata_fpth.replace(".csv", "_train.csv")
		val_csv   = metadata_fpth.replace(".csv", "_val.csv")

		df_train = pd.read_csv(train_csv, on_bad_lines="skip", dtype=dtypes, low_memory=False)
		df_val   = pd.read_csv(val_csv,   on_bad_lines="skip", dtype=dtypes, low_memory=False)

		print(f"\n  TRAIN split : {df_train.shape}  |  VAL split : {df_val.shape}")

		# ── Preprocessing ─────────────────────────────────────────────────────────
		preprocess = get_preprocess(dataset_dir=ddir, input_resolution=input_resolution)

		# ── Cache sizing (mirrors get_multi_label_dataloaders) ────────────────────
		if cache_size is None:
				total_samples = len(df_train) + len(df_val)
				avg_img_mb    = get_estimated_image_size_mb(
						image_paths=(
								df_train["img_path"].tolist() + df_val["img_path"].tolist()
						),
						sample_size=min(5000, int(total_samples * 0.1)),
				)
				is_hpc = any(e in os.environ for e in ("SLURM_JOB_ID", "PBS_JOBID"))
				if is_hpc and "SLURM_MEM_PER_NODE" in os.environ:
						available_gb = float(os.environ["SLURM_MEM_PER_NODE"]) / 1024.0
				else:
						available_gb = psutil.virtual_memory().available / (1024 ** 3)

				cache_size = get_cache_size(
						dataset_size=total_samples,
						available_memory_gb=available_gb,
						average_image_size_mb=avg_img_mb,
						is_hpc=is_hpc,
				)

		train_cache = int(cache_size * 0.6)
		val_cache   = cache_size - train_cache

		# ── Datasets ──────────────────────────────────────────────────────────────
		train_dataset = Stage5RegimeAwareDataset(
			dataset_name=dataset_name,
			train=True,
			data_frame=df_train.sort_values("img_path").reset_index(drop=True),
			transform=preprocess,
			label_dict=label_dict,
			df_supervision=df_supervision,
			id_col=id_col,
			text_col=text_col,
			text_augmentation=True,
			cache_size=train_cache,
			cache_workers=min(12, num_workers),
			verbose=verbose,
		)
		print(train_dataset)

		val_dataset = Stage5RegimeAwareDataset(
			dataset_name=dataset_name,
			train=False,
			data_frame=df_val.sort_values("img_path").reset_index(drop=True),
			transform=preprocess,
			label_dict=label_dict,
			df_supervision=df_supervision,
			id_col=id_col,
			text_col=text_col,
			text_augmentation=False,
			cache_size=val_cache,
			cache_workers=min(12, num_workers),
			verbose=verbose,
		)
		print(val_dataset)

		# ── DataLoaders ───────────────────────────────────────────────────────────
		train_loader = DataLoader(
				dataset=train_dataset,
				batch_size=batch_size,
				shuffle=True,
				pin_memory=torch.cuda.is_available(),
				num_workers=num_workers,
				prefetch_factor=2 if num_workers > 0 else None,
				persistent_workers=(num_workers > 0),
				collate_fn=stage5_collate_fn,
				drop_last=False,
		)
		train_loader.name = f"{dataset_name.lower()}_stage5_train".upper()

		val_loader = DataLoader(
				dataset=val_dataset,
				batch_size=batch_size,
				shuffle=False,
				pin_memory=torch.cuda.is_available(),
				num_workers=num_workers,
				prefetch_factor=2 if num_workers > 0 else None,
				persistent_workers=(num_workers > 0),
				collate_fn=stage5_collate_fn,
				drop_last=False,
		)
		val_loader.name = f"{dataset_name.lower()}_stage5_validation".upper()

		if verbose:
			print(f"\n[Stage5] DataLoaders ready")
			print(f"  ├─ Train batches : {len(train_loader):,}")
			print(f"  ├─ Val batches   : {len(val_loader):,}")
			print(f"  └─ Canonical V   : {len(label_dict):,} labels")

		return train_loader, val_loader