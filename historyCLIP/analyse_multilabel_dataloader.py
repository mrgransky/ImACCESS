"""
analyse_multilabel_dataloader.py
==================================
Runs all dataloader checks (A, B, C + cross-split) on a REAL dataset.
No synthetic data is generated.

Usage
-----
# Minimal — point at the full metadata CSV:
python analyse_multilabel_dataloader.py \
		--csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv

# Larger dataset:
python analyse_multilabel_dataloader.py \
		--csv /home/farid/datasets/WW_DATASETs/HISTORY_X4/metadata_multi_label_multimodal.csv

# Override column (default: multimodal_canonical_labels):
python analyse_multilabel_dataloader.py \
		--csv /path/to/metadata_multi_label_multimodal.csv \
		--col multimodal_labels

# Override batch size / workers:
python analyse_multilabel_dataloader.py \
		--csv /path/to/metadata.csv --batch_size 64 --num_workers 4

Assumptions (same as your existing codebase)
---------------------------------------------
	The directory containing the CSV must also contain:
		metadata_multi_label_multimodal_train.csv
		metadata_multi_label_multimodal_val.csv
		img_rgb_mean.gz
		img_rgb_std.gz
	If mean/std files are absent, ImageNet defaults are used.
	If train/val splits are absent, a 65/35 random split is created on the fly
	and saved alongside the full CSV (so subsequent runs reuse them).
"""

import argparse
import ast
import math
import os
import random
import shutil
import sys
import threading
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Minimal CLIP tokenizer stub
# Swap for:  import clip  and  clip.tokenize(...)  in your real training code.
# ─────────────────────────────────────────────────────────────────────────────

class _FakeTokenizer:
		CONTEXT_LEN = 77
		def tokenize(self, texts):
				if isinstance(texts, str):
						texts = [texts]
				results = []
				for t in texts:
						ids = [hash(w) % 49408 for w in t.split()][:self.CONTEXT_LEN - 2]
						ids = [49406] + ids + [49407]
						ids += [0] * (self.CONTEXT_LEN - len(ids))
						results.append(torch.tensor(ids, dtype=torch.long))
				return torch.stack(results)

_clip_stub = _FakeTokenizer()
def clip_tokenize(text): return _clip_stub.tokenize(text)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

DTYPES = {
		'doc_url': str, 'img_path': str, 'title': str, 'description': str,
		'llm_based_labels': str, 'vlm_based_labels': str,
		'multimodal_labels': str, 'multimodal_canonical_labels': str,
}

def _load_pickle(fpath: str):
		import gzip, pickle
		with gzip.open(fpath, "rb") as f:
				return pickle.load(f)


def load_splits(csv_path: str, col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		"""
		Returns (df_full, df_train, df_val).
		If *_train.csv / *_val.csv don't exist, creates a 65/35 stratified split
		by the first canonical label and saves them for reproducibility.
		"""
		ddir       = os.path.dirname(csv_path)
		train_path = csv_path.replace(".csv", "_train.csv")
		val_path   = csv_path.replace(".csv", "_val.csv")

		df_full = pd.read_csv(csv_path, dtype=DTYPES, on_bad_lines="skip", low_memory=False)
		print(f"[CSV] Full dataset loaded: {df_full.shape[0]:,} rows  |  columns: {list(df_full.columns)}")

		# Drop rows where the target column is missing / unparseable
		def is_valid(raw):
				try:    return isinstance(ast.literal_eval(raw), list)
				except: return False

		before = len(df_full)
		df_full = df_full[df_full[col].apply(is_valid)].reset_index(drop=True)
		if len(df_full) < before:
				print(f"[CSV] Dropped {before - len(df_full):,} rows with unparseable '{col}' values")

		if os.path.exists(train_path) and os.path.exists(val_path):
				df_train = pd.read_csv(train_path, dtype=DTYPES, on_bad_lines="skip", low_memory=False)
				df_val   = pd.read_csv(val_path,   dtype=DTYPES, on_bad_lines="skip", low_memory=False)
				print(f"[CSV] Loaded existing splits  →  train={len(df_train):,}  val={len(df_val):,}")
		else:
				print("[CSV] No train/val splits found — creating 65/35 random split …")
				df_shuf  = df_full.sample(frac=1, random_state=42).reset_index(drop=True)
				n_train  = int(len(df_shuf) * 0.65)
				df_train = df_shuf.iloc[:n_train].reset_index(drop=True)
				df_val   = df_shuf.iloc[n_train:].reset_index(drop=True)
				df_train.to_csv(train_path, index=False)
				df_val.to_csv(val_path,     index=False)
				print(f"[CSV] Saved splits  →  {train_path}")
				print(f"                    →  {val_path}")
				print(f"[CSV] Split sizes  →  train={len(df_train):,}  val={len(df_val):,}")

		return df_full, df_train, df_val


def get_preprocess(dataset_dir: str, input_resolution: int = 224) -> T.Compose:
		try:
				mean = _load_pickle(os.path.join(dataset_dir, "img_rgb_mean.gz"))
				std  = _load_pickle(os.path.join(dataset_dir, "img_rgb_std.gz"))
				print(f"[preprocess] Loaded dataset mean={mean}  std={std}")
		except Exception:
				mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
				print(f"[preprocess] mean/std files not found — using ImageNet defaults")

		def _to_rgb(img): return img.convert("RGB")
		return T.Compose([
				T.Resize(input_resolution, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
				T.CenterCrop(input_resolution),
				_to_rgb, T.ToTensor(), T.Normalize(mean=mean, std=std),
		])


def build_label_dict(df_full: pd.DataFrame, col: str) -> dict:
		all_labels: set = set()
		for raw in df_full[col].dropna():
				try: all_labels.update(ast.literal_eval(raw))
				except (ValueError, SyntaxError): pass
		label_dict = {lbl: idx for idx, lbl in enumerate(sorted(all_labels))}
		print(f"[Labels] {len(label_dict)} unique canonical labels in full dataset")
		return label_dict


# ─────────────────────────────────────────────────────────────────────────────
# Image cache
# ─────────────────────────────────────────────────────────────────────────────

class ImageCache:
		def __init__(self, image_paths, cache_size: int, num_workers: int = 4):
				self.cache, self.lock = {}, threading.Lock()
				self.image_paths = image_paths
				self.cache_size  = cache_size
				if cache_size > 0:
						self._preload(num_workers)

		def _preload(self, nw):
				idxs = np.random.choice(len(self.image_paths),
																size=min(self.cache_size, len(self.image_paths)),
																replace=False)
				print(f"  → Preloading {len(idxs):,} images into cache …", flush=True)
				def _load(idx):
						try:
								with Image.open(self.image_paths[idx]) as img:
										return idx, np.array(img.convert("RGB"), dtype=np.uint8)
						except Exception:
								return idx, None

				with ThreadPoolExecutor(max_workers=nw) as ex:
						for idx, arr in ex.map(_load, idxs):
								if arr is not None:
										with self.lock: self.cache[idx] = arr
				print(f"  → Cached {len(self.cache):,} images")

		def get(self, idx):
				with self.lock:
						arr = self.cache.get(idx)
						return Image.fromarray(arr) if arr is not None else None

		def __len__(self): return len(self.cache)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalArchivesMultiLabelDataset(Dataset):
		def __init__(self, dataset_name, train, data_frame, transform,
								 label_dict, text_augmentation=True, cache_size=0,
								 cache_workers=4, col="multimodal_canonical_labels"):
				self.dataset_name      = dataset_name
				self.train             = train
				self.data_frame        = data_frame
				self.images            = data_frame["img_path"].values
				self.labels            = data_frame[col].values
				self.label_dict        = label_dict
				self._num_classes      = len(label_dict)
				self.transform         = transform
				self.text_augmentation = text_augmentation
				self.split             = "Train" if train else "Validation"
				self.cache_hits = self.cache_misses = 0

				self.image_cache = (ImageCache(self.images, cache_size, cache_workers)
														if cache_size > 0 else None)
				self.text_cache = [self._tokenize_labels(lbl) for lbl in self.labels]

		def _tokenize_labels(self, labels_str):
				try:
						labels = ast.literal_eval(labels_str)
						return clip_tokenize(self._make_text(labels)).squeeze(0)
				except (ValueError, SyntaxError):
						return clip_tokenize("").squeeze(0)

		def _make_text(self, labels):
				if not labels: return ""
				if not self.train or not self.text_augmentation: return " ".join(labels)
				if len(labels) == 1: return labels[0]
				if len(labels) == 2: return f"{labels[0]} and {labels[1]}"
				np.random.shuffle(labels)
				return ", ".join(labels[:-1]) + f", and {labels[-1]}"

		def _get_label_vector(self, labels_str):
				vec = torch.zeros(self._num_classes, dtype=torch.float32)
				try:
						for lbl in ast.literal_eval(labels_str):
								if lbl in self.label_dict:
										vec[self.label_dict[lbl]] = 1.0
				except (ValueError, SyntaxError): pass
				return vec

		def _load_image(self, idx):
				if self.image_cache is not None:
						cached = self.image_cache.get(idx)
						if cached is not None:
								self.cache_hits += 1; return cached
						self.cache_misses += 1
				try:
						with Image.open(self.images[idx]) as img:
								return img.convert("RGB")
				except Exception:
						return Image.new("RGB", (224, 224), "white")

		def __len__(self): return len(self.data_frame)

		def __getitem__(self, idx):
				try:
						return (self.transform(self._load_image(idx)),
										self.text_cache[idx],
										self._get_label_vector(self.labels[idx]))
				except Exception:
						return (torch.zeros(3, 224, 224),
										torch.zeros(77, dtype=torch.long),
										torch.zeros(self._num_classes))

		def get_cache_stats(self):
				total = self.cache_hits + self.cache_misses
				return dict(cache_size=len(self.image_cache) if self.image_cache else 0,
										hits=self.cache_hits, misses=self.cache_misses,
										hit_rate_pct=(self.cache_hits / total * 100) if total > 0 else 0.0)

		@property
		def unique_labels(self): return sorted(self.label_dict.keys())

		def __repr__(self):
				cs = self.get_cache_stats()
				cache_str = (f"Image Cache: {cs['cache_size']:,}/{len(self.images):,}, "
										 f"Hit Rate={cs['hit_rate_pct']:.1f}%"
										 if self.image_cache else "Image Cache: DISABLED")
				return (f"{self.dataset_name}\n"
								f"\tSplit: {self.split} ({len(self.data_frame):,} samples)\n"
								f"\tNum classes: {self._num_classes:,}\n"
								f"\t{cache_str}")


def _decide_cache(df_train, df_val) -> tuple[int, int]:
		"""Compute train/val cache sizes from available RAM."""
		total = len(df_train) + len(df_val)
		avail_gb = psutil.virtual_memory().available / 1024**3
		is_hpc   = any(k in os.environ for k in ("SLURM_JOB_ID", "PBS_JOBID"))
		coverage = 0.75 if is_hpc else 0.65
		# Sample up to 200 images to estimate avg size
		sample_paths = (df_train["img_path"].tolist() + df_val["img_path"].tolist())
		sample_paths = random.sample(sample_paths, min(200, len(sample_paths)))
		sizes = []
		for p in sample_paths:
				try:
						with Image.open(p) as img:
								w, h = img.size
								sizes.append(w * h * 3 / 1024**2)
				except Exception:
						pass
		avg_mb = np.mean(sizes) if sizes else 1.0
		print(f"[Cache] Avg image size: {avg_mb:.2f} MB  |  Available RAM: {avail_gb:.1f} GB")
		max_cache  = int(avail_gb * 0.5 * 1024 / avg_mb)
		want_cache = int(total * coverage)
		cache_size = min(want_cache, max_cache, total)
		train_cache = int(cache_size * 0.6)
		val_cache   = cache_size - train_cache
		print(f"[Cache] Total={cache_size:,}  Train={train_cache:,}  Val={val_cache:,}  "
					f"(coverage={cache_size/total*100:.1f}%)")
		return train_cache, val_cache


# ─────────────────────────────────────────────────────────────────────────────
# CHECK A — Class imbalance
# ─────────────────────────────────────────────────────────────────────────────

def _pareto_split(freq: np.ndarray, pareto: float = 0.80):
		"""
		Return (head_mask, tail_mask) using a Pareto split.

		head: smallest set of classes whose cumulative frequency covers
					`pareto` fraction of all label occurrences.
		tail: classes with zero occurrences in this split
					(genuinely unobserved — pos_weight would be undefined).

		The middle band (observed but not head) is left unmarked.
		This avoids the arbitrary 20% boundary and scales correctly
		from 60 classes (SMU) to 7,485 classes (HISTORY_X4).
		"""
		total       = freq.sum()
		sorted_idx  = np.argsort(freq)[::-1]
		cumsum      = np.cumsum(freq[sorted_idx])
		# number of classes needed to reach `pareto` of total occurrences
		n_head      = int(np.searchsorted(cumsum, pareto * total)) + 1
		n_head      = max(1, min(n_head, len(freq)))

		head_mask   = np.zeros(len(freq), dtype=bool)
		head_mask[sorted_idx[:n_head]] = True
		tail_mask   = (freq == 0)
		return head_mask, tail_mask, n_head


def analyse_class_imbalance(dataset, split_name, beta=0.9999,
														 pareto=0.80, pw_tail_threshold=20.0,
														 output_dir=".") -> dict:
		"""
		Parameters
		----------
		pareto : float
				Fraction of total label occurrences covered by the "head" set.
				Default 0.80 (80/20 Pareto rule). Head = fewest classes that
				collectively account for this fraction of all positive labels.
		pw_tail_threshold : float
				Classes whose pos_weight exceeds this value are flagged as
				"effectively rare" for loss-weighting purposes. Default 20.
		"""
		print(f"\n{'─'*60}")
		print(f"  CHECK A  ·  Class Imbalance — {split_name}  ({len(dataset):,} samples)")
		print(f"{'─'*60}")

		n     = len(dataset)
		C     = dataset._num_classes
		names = dataset.unique_labels
		freq  = np.zeros(C, dtype=np.int64)

		for raw in dataset.labels:
				try:
						for lbl in ast.literal_eval(raw):
								if lbl in dataset.label_dict:
										freq[dataset.label_dict[lbl]] += 1
				except (ValueError, SyntaxError):
						pass

		freq_ratio  = freq / max(n, 1)
		n_neg       = n - freq
		pos_weight  = torch.tensor(
				np.where(freq > 0, n_neg / np.maximum(freq, 1), 1.0), dtype=torch.float32)
		eff_num     = (1.0 - beta ** np.maximum(freq, 1)) / (1.0 - beta)
		ens_weights = (1.0 / eff_num)
		ens_weights = ens_weights / ens_weights.sum() * C
		imb_ratio   = freq.max() / max(freq.min(), 1)

		sorted_idx  = np.argsort(freq)[::-1]

		# ── Principled head / tail split ──────────────────────────────────────────
		head_mask, zero_mask, n_head = _pareto_split(freq, pareto)

		# "effectively rare" = pos_weight exceeds threshold (loss-weighting concern)
		rare_mask = (pos_weight.numpy() > pw_tail_threshold) & ~zero_mask

		head_set  = set(names[i] for i in range(C) if head_mask[i])
		zero_set  = set(names[i] for i in range(C) if zero_mask[i])
		rare_set  = set(names[i] for i in range(C) if rare_mask[i])

		# ── Console table ─────────────────────────────────────────────────────────
		# For large C, truncate: show top-40, separator, bottom-20
		col_w     = min(max(len(l) for l in names) + 2, 45)
		MAX_SHOW  = 60   # total rows to print; split top/bottom for large C
		show_all  = (C <= MAX_SHOW)

		print(f"  {'Label':<{col_w}} {'Count':>7}  {'Freq%':>6}  {'pos_weight':>10}  {'ENS_w':>8}  Note")
		print("  " + "─" * (col_w + 52))

		def _row(i):
				nm  = names[i]
				note = ""
				if nm in head_set:  note = "▲ HEAD"
				if nm in zero_set:  note = "✗ ZERO"
				if nm in rare_set:  note += (" " if note else "") + "⚠ RARE"
				return (f"  {nm[:col_w-1]:<{col_w}} {freq[i]:>7,}  "
								f"{freq_ratio[i]*100:>5.1f}%  "
								f"{pos_weight[i].item():>10.3f}  "
								f"{ens_weights[i]:>8.4f}  {note}")

		if show_all:
				for i in sorted_idx:
						print(_row(i))
		else:
				top_n, bot_n = 40, 20
				for i in sorted_idx[:top_n]:
						print(_row(i))
				print(f"  {'':.<{col_w}}  ... {C - top_n - bot_n:,} middle classes omitted ...")
				for i in sorted_idx[-(bot_n):]:
						print(_row(i))

		# ── Summary ───────────────────────────────────────────────────────────────
		n_zero = int(zero_mask.sum())
		n_rare = int(rare_mask.sum())
		pct_head_classes = n_head / C * 100
		pct_occ_covered  = freq[head_mask].sum() / max(freq.sum(), 1) * 100

		print(f"\n  Imbalance ratio (max/min freq)   : {imb_ratio:.1f}×")
		print(f"\n  Pareto head  (≥{pareto:.0%} of occurrences) :")
		print(f"    {n_head:,} of {C:,} classes ({pct_head_classes:.1f}%) cover "
					f"{pct_occ_covered:.1f}% of all positive labels")
		print(f"    Head classes: {sorted(head_set)[:15]}"
					f"{'...' if len(head_set) > 15 else ''}")
		print(f"\n  Zero-count classes (unobserved in this split) : {n_zero:,}")
		if n_zero and n_zero <= 20:
				print(f"    {sorted(zero_set)}")
		elif n_zero:
				print(f"    (first 20): {sorted(zero_set)[:20]} …")

		print(f"\n  Effectively rare  (pos_weight > {pw_tail_threshold:.0f}) : {n_rare:,} classes")
		if n_rare and n_rare <= 20:
				pw_rare = sorted([(pos_weight[dataset.label_dict[nm]].item(), nm)
													for nm in rare_set], reverse=True)
				for pw_v, nm in pw_rare:
						print(f"    {nm}: {pw_v:.1f}")
		elif n_rare:
				pw_rare = sorted([(pos_weight[dataset.label_dict[nm]].item(), nm)
													for nm in rare_set], reverse=True)
				print(f"    Top 20 by pos_weight:")
				for pw_v, nm in pw_rare[:20]:
						print(f"      {nm}: {pw_v:.1f}")

		# ── PNG ───────────────────────────────────────────────────────────────────
		_plot_imbalance(freq, pos_weight, names, sorted_idx,
										head_mask, zero_mask, rare_mask,
										split_name, n, pareto, pw_tail_threshold, output_dir)

		return dict(freq=freq, freq_ratio=freq_ratio, imb_ratio=imb_ratio,
								pos_weight=pos_weight, ens_weights=ens_weights,
								head_set=head_set, zero_set=zero_set, rare_set=rare_set,
								n_head=n_head, n_zero=n_zero, n_rare=n_rare)


def _plot_imbalance(freq, pos_weight, names, sorted_idx,
										head_mask, zero_mask, rare_mask,
										split, n, pareto, pw_threshold, out_dir):
		C = len(names)

		# For large C, cap the plot at top-150 classes so it's legible
		PLOT_CAP = 150
		if C > PLOT_CAP:
				plot_idx = sorted_idx[:PLOT_CAP]
				title_note = f"  (top {PLOT_CAP} of {C:,} shown)"
		else:
				plot_idx = sorted_idx
				title_note = ""

		fw  = max(16, len(plot_idx) * 0.42)
		fig, axes = plt.subplots(2, 1, figsize=(fw, 10), facecolor="#0f1117")
		fig.suptitle(f"Class Imbalance — {split}  ({n:,} samples){title_note}",
								 fontsize=12, color="#e8e8e8", y=0.99, fontweight="bold")

		x     = np.arange(len(plot_idx))
		xlabs = [names[i][:20] for i in plot_idx]   # truncate long names

		def _color(i):
				if zero_mask[i]:  return "#374151"   # grey  — zero count
				if head_mask[i]:  return "#ef4444"   # red   — pareto head
				if rare_mask[i]:  return "#f59e0b"   # amber — effectively rare
				return "#6366f1"                      # indigo — middle band

		colors = [_color(i) for i in plot_idx]

		# ── top panel: frequency ──────────────────────────────────────────────────
		ax = axes[0]
		ax.set_facecolor("#0f1117")
		bars = ax.bar(x, [freq[i] for i in plot_idx],
									color=colors, edgecolor="#1e2030", linewidth=0.3)
		ax.set_xticks(x)
		ax.set_xticklabels(xlabs, rotation=55, ha="right",
											 fontsize=max(4, min(7, 200 // len(plot_idx))),
											 color="#c0c0c0")
		ax.set_ylabel("Positive Count", color="#c0c0c0", fontsize=9)
		ax.set_title(
				f"Label Frequency  |  ▲ Pareto head ({pareto:.0%})  "
				f"⚠ Rare (pw>{pw_threshold:.0f})  ✗ Zero",
				color="#d0d0d0", fontsize=9, pad=6)
		ax.tick_params(colors="#808080"); ax.spines[:].set_color("#2a2a3a")
		ax.set_xlim(-0.7, len(x) - 0.3)
		# only annotate bars when they fit
		if len(plot_idx) <= 80:
				for bar, i in zip(bars, plot_idx):
						if freq[i] > 0:
								ax.text(bar.get_x() + bar.get_width() / 2,
												bar.get_height() + max(freq[plot_idx]) * 0.01,
												f"{freq[i]:,}", ha="center", va="bottom",
												fontsize=5, color="#a0a0b0")

		# ── bottom panel: pos_weight ──────────────────────────────────────────────
		ax2 = axes[1]
		ax2.set_facecolor("#0f1117")
		pw  = [pos_weight[i].item() for i in plot_idx]
		cmap = matplotlib.colormaps["RdYlGn_r"]
		norm = mcolors.Normalize(vmin=min(pw), vmax=max(pw))
		ax2.bar(x, pw, color=[cmap(norm(v)) for v in pw],
						edgecolor="#1e2030", linewidth=0.3)
		ax2.set_xticks(x)
		ax2.set_xticklabels(xlabs, rotation=55, ha="right",
												fontsize=max(4, min(7, 200 // len(plot_idx))),
												color="#c0c0c0")
		ax2.set_ylabel("pos_weight (neg/pos)", color="#c0c0c0", fontsize=9)
		ax2.set_title(
				f"BCEWithLogitsLoss pos_weight  "
				f"(dashed = balanced, dotted = threshold {pw_threshold:.0f})",
				color="#d0d0d0", fontsize=9, pad=6)
		ax2.tick_params(colors="#808080"); ax2.spines[:].set_color("#2a2a3a")
		ax2.set_xlim(-0.7, len(x) - 0.3)
		ax2.axhline(1.0, color="#fbbf24", lw=0.8, ls="--", alpha=0.7,
								label="Balanced (pw=1)")
		ax2.axhline(pw_threshold, color="#f87171", lw=0.8, ls=":",
								alpha=0.7, label=f"Rare threshold (pw={pw_threshold:.0f})")
		ax2.legend(fontsize=7, labelcolor="#e0e0e0", framealpha=0.15)

		plt.tight_layout(rect=[0, 0, 1, 0.97])
		path = os.path.join(out_dir, f"imbalance_{split.lower()}.png")
		plt.savefig(path, dpi=140, bbox_inches="tight",
								facecolor=fig.get_facecolor())
		plt.close()
		print(f"\n  [PNG] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK B — Co-occurrence
# ─────────────────────────────────────────────────────────────────────────────

# phi_coefficient is now computed fully vectorised inside analyse_cooccurrence
# using matrix operations — no per-pair Python loop needed.


def analyse_cooccurrence(dataset, split_name,
													phi_pos=0.20, phi_neg=-0.10,
													top_k=None,
													output_dir=".") -> dict:
		"""
		top_k : int or None
				If set, restrict the co-occurrence matrix to the top-K most frequent
				classes. Strongly recommended for datasets with hundreds of classes
				(e.g. HISTORY_X4) where the full C×C matrix would be huge.
		"""
		print(f"\n{'─'*60}")
		print(f"  CHECK B  ·  Co-occurrence — {split_name}  ({len(dataset):,} samples)")
		print(f"{'─'*60}")

		n_samples  = len(dataset)
		C_full     = dataset._num_classes
		names_full = dataset.unique_labels

		# ── Optionally restrict to top-K by frequency ─────────────────────────────
		if top_k is not None and top_k < C_full:
				freq_all = np.zeros(C_full, dtype=np.int64)
				for raw in dataset.labels:
						try:
								for lbl in ast.literal_eval(raw):
										if lbl in dataset.label_dict:
												freq_all[dataset.label_dict[lbl]] += 1
						except (ValueError, SyntaxError):
								pass
				top_idx   = np.argsort(freq_all)[::-1][:top_k]
				top_idx   = sorted(top_idx.tolist())           # keep alphabetical order
				names     = [names_full[i] for i in top_idx]
				idx_set   = set(top_idx)
				# remap original class indices → local [0, top_k) indices
				remap     = {orig: new for new, orig in enumerate(top_idx)}
				C         = top_k
				print(f"  [!] Restricting co-occurrence to top-{top_k} of {C_full} classes "
							f"(use --max_cooc_classes to change)")
		else:
				names   = names_full
				idx_set = set(range(C_full))
				remap   = {i: i for i in range(C_full)}
				C       = C_full

		# ── Build binary label matrix [N × C] ─────────────────────────────────────
		import time as _t
		t0 = _t.perf_counter()
		mat = np.zeros((n_samples, C), dtype=np.float32)   # float32 for BLAS matmul
		for row, raw in enumerate(dataset.labels):
				try:
						for lbl in ast.literal_eval(raw):
								if lbl in dataset.label_dict:
										orig_idx = dataset.label_dict[lbl]
										if orig_idx in idx_set:
												mat[row, remap[orig_idx]] = 1.0
				except (ValueError, SyntaxError):
						pass
		print(f"  Label matrix built in {_t.perf_counter()-t0:.1f}s  "
					f"({mat.nbytes/1024**2:.0f} MB)", flush=True)

		# ── Guard: warn if C×C matrix would exceed 2 GB ───────────────────────────
		phi_bytes = C * C * 8   # float64
		if phi_bytes > 2 * 1024**3:
				print(f"  [!] Full {C}×{C} Phi matrix would require "
							f"{phi_bytes/1024**3:.1f} GB — aborting co-occurrence.\n"
							f"      Re-run with --max_cooc_classes N (recommended N ≤ 500).")
				return dict(raw_cooc=None, jaccard=None, phi=None,
										pos_pairs=[], neg_pairs=[], density=float(mat.mean()*100))

		# ── Vectorised co-occurrence metrics (fully NumPy, no Python loops) ────────
		# raw_cooc[i,j] = number of samples where both class i and j are positive
		t1 = _t.perf_counter()
		raw_cooc = (mat.T @ mat).astype(np.float32)           # [C×C], BLAS SGEMM

		freq = raw_cooc.diagonal()                            # [C], marginal counts

		# Vectorised Phi (Matthews) coefficient for all pairs simultaneously:
		#   phi[i,j] = (tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))
		# where all quantities are derived from the raw_cooc matrix and marginals.
		#   tp[i,j] = raw_cooc[i,j]
		#   fp[i,j] = freq[j] - raw_cooc[i,j]   (j positive, i negative)
		#   fn[i,j] = freq[i] - raw_cooc[i,j]   (i positive, j negative)
		#   tn[i,j] = N - freq[i] - freq[j] + raw_cooc[i,j]
		N    = float(n_samples)
		tp   = raw_cooc
		fp   = freq[np.newaxis, :] - tp          # broadcast: shape [C, C]
		fn   = freq[:, np.newaxis] - tp
		tn   = N - freq[:, np.newaxis] - freq[np.newaxis, :] + tp

		denom = np.sqrt(
				np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0)
		).astype(np.float64)

		phi = np.where(denom > 0,
									 (tp * tn - fp * fn).astype(np.float64) / denom,
									 0.0)
		np.fill_diagonal(phi, 1.0)

		# ── Jaccard ────────────────────────────────────────────────────────────────
		union   = np.maximum(freq[:, np.newaxis] + freq[np.newaxis, :] - raw_cooc, 1.0)
		jaccard = (raw_cooc / union).astype(np.float32)

		print(f"  Phi + Jaccard computed in {_t.perf_counter()-t1:.1f}s", flush=True)

		# ── Extract interesting pairs (upper triangle only) ────────────────────────
		t2 = _t.perf_counter()
		i_idx, j_idx = np.triu_indices(C, k=1)
		phi_vals  = phi[i_idx, j_idx]
		cooc_vals = raw_cooc[i_idx, j_idx].astype(int)

		pos_mask = phi_vals >= phi_pos
		neg_mask = phi_vals <= phi_neg

		pos_pairs = sorted(
				[(phi_vals[k], names[i_idx[k]], names[j_idx[k]], int(cooc_vals[k]))
				 for k in np.where(pos_mask)[0]],
				key=lambda x: -x[0]
		)
		neg_pairs = sorted(
				[(phi_vals[k], names[i_idx[k]], names[j_idx[k]], int(cooc_vals[k]))
				 for k in np.where(neg_mask)[0]],
				key=lambda x: x[0]
		)
		print(f"  Pair extraction in {_t.perf_counter()-t2:.1f}s  "
					f"| pos_pairs={len(pos_pairs):,}  neg_pairs={len(neg_pairs):,}",
					flush=True)

		density = float(mat.mean()) * 100
		print(f"\n  Binary label matrix : {mat.shape}  ({int(mat.sum()):,} total positives)")
		print(f"  Label density       : {density:.2f}% of (sample, class) cells are positive")

		if pos_pairs:
				print(f"\n  Co-occurring pairs (phi ≥ {phi_pos:.2f}) — top 15:")
				for v, a, b, cnt in pos_pairs[:15]:
						print(f"    phi={v:+.3f}  co-occur={cnt:4d}   {a}  ↔  {b}")
		else:
				print(f"  (no pairs with phi ≥ {phi_pos:.2f})")

		if neg_pairs:
				print(f"\n  Mutually exclusive pairs (phi ≤ {phi_neg:.2f}) — top 15:")
				for v, a, b, cnt in neg_pairs[:15]:
						print(f"    phi={v:+.3f}  co-occur={cnt:4d}   {a}  ✗  {b}")
		else:
				print(f"  (no pairs with phi ≤ {phi_neg:.2f})")

		_save_phi_png(phi, names, split_name, output_dir)
		_save_html(raw_cooc, jaccard, phi, names, split_name, output_dir)

		return dict(raw_cooc=raw_cooc, jaccard=jaccard, phi=phi,
								pos_pairs=pos_pairs, neg_pairs=neg_pairs, density=density)


def _save_phi_png(phi, names, split, out_dir):
		n  = len(names)
		fs = max(10, n * 0.38)
		fig, ax = plt.subplots(figsize=(fs, fs * 0.88), facecolor="#0d1117")
		ax.set_facecolor("#0d1117")
		cmap = sns.diverging_palette(250, 15, s=90, l=40, as_cmap=True)
		sns.heatmap(phi, mask=np.eye(n, dtype=bool), cmap=cmap, center=0,
								vmin=-0.6, vmax=0.6,
								xticklabels=names, yticklabels=names,
								linewidths=0.25, linecolor="#1e2030", square=True,
								cbar_kws={"shrink": 0.55, "label": "Phi coefficient"},
								annot=(n <= 35), fmt=".2f" if n <= 35 else "",
								annot_kws={"size": max(4, 8 - n // 10), "color": "#dde"},
								ax=ax)
		ax.set_title(f"Phi Co-occurrence  ·  {split}", color="#d8d8e8",
								 fontsize=11, fontweight="bold", pad=12)
		ax.tick_params(axis="x", labelsize=max(5, 9 - n // 15),
									 colors="#a0a0b8", rotation=50)
		ax.tick_params(axis="y", labelsize=max(5, 9 - n // 15),
									 colors="#a0a0b8", rotation=0)
		ax.collections[0].colorbar.set_label("Phi", color="#a0a0b8")
		ax.collections[0].colorbar.ax.tick_params(labelcolor="#a0a0b8")
		plt.tight_layout()
		path = os.path.join(out_dir, f"cooccurrence_phi_{split.lower()}.png")
		plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
		plt.close()
		print(f"\n  [PNG] Phi heatmap → {path}")


def _save_html(raw_cooc, jaccard, phi, names, split, out_dir):
		n       = len(names)
		cell_px = max(22, min(50, 1200 // n))

		def rows(mat, fmt, diverging):
				vmax  = 0.6 if diverging else float(np.abs(mat).max() or 1)
				parts = []
				for i, rn in enumerate(names):
						cells = [f'<td class="rh">{rn}</td>']
						for j in range(n):
								v = mat[i, j]; diag = (i == j)
								if diag:
										bg, fg, txt = "#1e2030", "#445", "—"
								elif diverging:
										t = np.clip(v / vmax, -1, 1)
										r, g, b = ((int(60+180*t), 30, 30) if t >= 0
															 else (30, 30, int(60+180*(-t))))
										bg, fg, txt = f"rgb({r},{g},{b})", "#e8e8e8", format(v, fmt)
								else:
										t = np.clip(v / vmax, 0, 1)
										iv = int(20 + 200 * t)
										bg = f"rgb({int(iv*.3)},{int(iv*.5)},{iv})"
										fg, txt = "#e8e8e8", format(v, fmt)
								tip = f'{names[i]} ↔ {names[j]}: {v:.4f}'
								cells.append(f'<td style="background:{bg};color:{fg};" '
														 f'data-tip="{tip}">{txt}</td>')
						parts.append("<tr>" + "".join(cells) + "</tr>")
				return "\n".join(parts)

		hdr = ('<tr><th class="corner"></th>' +
					 "".join(f'<th class="ch">{l}</th>' for l in names) + "</tr>")

		html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Co-occurrence · {split}</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--text:#c9d1d9;--accent:#58a6ff;--cell:{cell_px}px;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:'Courier New',monospace;font-size:13px;}}
h1{{padding:18px 24px 4px;font-size:15px;font-weight:700;color:var(--accent);}}
p.sub{{padding:0 24px 10px;color:#8b949e;font-size:11px;}}
.tabs{{display:flex;gap:4px;padding:0 24px 10px;}}
.tab{{padding:5px 16px;border-radius:6px 6px 0 0;background:#1f2937;cursor:pointer;
			border:1px solid #21262d;border-bottom:none;font-size:12px;color:#8b949e;transition:.15s;}}
.tab:hover,.tab.active{{background:#374151;color:var(--accent);}}
.view{{display:none;padding:0 24px 32px;overflow:auto;}}
.view.active{{display:block;}}
table{{border-collapse:collapse;table-layout:fixed;}}
th.corner{{width:var(--cell);}}
th.ch{{width:var(--cell);writing-mode:vertical-rl;transform:rotate(180deg);
			 text-align:left;font-size:9px;color:#8b949e;padding:3px 2px;
			 max-height:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
td.rh{{font-size:9px;color:#8b949e;text-align:right;padding-right:5px;
			 white-space:nowrap;max-width:140px;overflow:hidden;text-overflow:ellipsis;}}
td[data-tip]{{width:var(--cell);height:var(--cell);text-align:center;font-size:8px;
							cursor:crosshair;transition:outline .08s;}}
td[data-tip]:hover{{outline:2px solid #f0f0f0;z-index:10;position:relative;}}
#tip{{position:fixed;pointer-events:none;display:none;background:#1e2030ee;
			border:1px solid #555;padding:6px 10px;border-radius:6px;
			font-size:11px;color:#e0e8ff;z-index:999;max-width:280px;}}
</style></head>
<body>
<h1>Label Co-occurrence · {split}</h1>
<p class="sub">{n} classes · Hover a cell for exact value · Click tabs to switch view.</p>
<div class="tabs">
	<div class="tab active" onclick="show('r',this)">Raw Co-occurrence</div>
	<div class="tab" onclick="show('j',this)">Jaccard</div>
	<div class="tab" onclick="show('p',this)">Phi Coefficient</div>
</div>
<div id="r" class="view active"><table>{hdr}{rows(raw_cooc,'.0f',False)}</table></div>
<div id="j" class="view"><table>{hdr}{rows(jaccard,'.3f',False)}</table></div>
<div id="p" class="view"><table>{hdr}{rows(phi,'.2f',True)}</table></div>
<div id="tip"></div>
<script>
function show(id,btn){{
	document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
	document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
	document.getElementById(id).classList.add('active');
	btn.classList.add('active');
}}
const tip=document.getElementById('tip');
document.querySelectorAll('td[data-tip]').forEach(td=>{{
	td.addEventListener('mousemove',e=>{{
		tip.style.display='block';
		tip.style.left=(e.clientX+14)+'px';
		tip.style.top=(e.clientY+14)+'px';
		tip.textContent=td.dataset.tip;
	}});
	td.addEventListener('mouseleave',()=>tip.style.display='none');
}});
</script>
</body></html>"""

		path = os.path.join(out_dir, f"cooccurrence_{split.lower()}.html")
		with open(path, "w", encoding="utf-8") as f:
				f.write(html)
		print(f"  [HTML] Interactive heatmap → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK C — Dataloader batch integrity
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(loader, num_classes, split_name) -> dict:
		n_samples = n_corrupt = 0
		densities = []; shape_errors = []
		for bi, (imgs, texts, labels) in enumerate(loader):
				n_corrupt += (imgs.abs().sum(dim=(1,2,3)) == 0).sum().item()
				n_samples += imgs.shape[0]
				densities.extend(labels.sum(dim=1).tolist())
				try:
						assert imgs.shape[1] == 3
						assert texts.shape[1] == 77 and texts.dtype == torch.long
						assert labels.shape[1] == num_classes and labels.dtype == torch.float32
						assert ((labels == 0) | (labels == 1)).all()
				except AssertionError as e:
						shape_errors.append(f"Batch {bi}: {e}")
		return dict(n_samples=n_samples, n_corrupt=n_corrupt,
								avg_density=np.mean(densities) if densities else 0.0,
								shape_errors=shape_errors)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
		p = argparse.ArgumentParser(description="Multi-label dataloader analysis on a real dataset")
		p.add_argument("--csv",         required=True,
									 help="Path to metadata_multi_label_multimodal.csv")
		p.add_argument("--col",         default="multimodal_canonical_labels",
									 help="Label column to use (default: multimodal_canonical_labels)")
		p.add_argument("--batch_size",  type=int, default=32)
		p.add_argument("--num_workers", type=int, default=4)
		p.add_argument("--resolution",  type=int, default=224)
		p.add_argument("--phi_pos",     type=float, default=0.20,
									 help="Phi threshold for co-occurring pairs")
		p.add_argument("--phi_neg",     type=float, default=-0.10,
									 help="Phi threshold for mutually exclusive pairs")
		p.add_argument("--pareto",      type=float, default=0.80,
									 help="Pareto fraction for head/tail split in CHECK A (default 0.80). "
												"Head = fewest classes covering this fraction of all occurrences.")
		p.add_argument("--pw_tail_threshold", type=float, default=20.0,
									 help="pos_weight threshold above which a class is flagged as "
												"effectively rare (default 20.0).")
		p.add_argument("--output_dir",  default=None,
									 help="Where to save PNGs/HTML (default: <csv_dir>/dataloader_analysis)")
		p.add_argument("--max_cooc_classes", type=int, default=None,
									 help="Limit co-occurrence to top-K most frequent classes. "
												"Essential for large-vocab datasets (e.g. HISTORY_X4: --max_cooc_classes 80). "
												"Default: all classes.")
		return p.parse_args()


def main():
		args = parse_args()
		print(args)

		csv_path   = os.path.abspath(args.csv)
		dataset_dir = os.path.dirname(csv_path)
		dataset_name = os.path.basename(dataset_dir)
		output_dir   = args.output_dir or os.path.join(dataset_dir, "outputs")
		os.makedirs(output_dir, exist_ok=True)

		sep = "=" * 70
		print(sep)
		print(f"  Multi-label Dataloader Analysis")
		print(f"  Dataset  : {dataset_name}")
		print(f"  CSV      : {csv_path}")
		print(f"  Column   : {args.col}")
		print(f"  Output   : {output_dir}")
		print(sep)

		# ── 1. Load data ──────────────────────────────────────────────────────────
		df_full, df_train, df_val = load_splits(csv_path, args.col)
		label_dict  = build_label_dict(df_full, args.col)
		n_classes   = len(label_dict)
		preprocess  = get_preprocess(dataset_dir, args.resolution)

		# ── 2. Cache ──────────────────────────────────────────────────────────────
		train_cache, val_cache = _decide_cache(df_train, df_val)

		# ── 3. Build datasets ─────────────────────────────────────────────────────
		print(f"\n── Building Train dataset ({len(df_train):,} samples) ──")
		train_ds = HistoricalArchivesMultiLabelDataset(
				dataset_name=f"{dataset_name}_TRAIN", train=True,
				data_frame=df_train.sort_values("img_path").reset_index(drop=True),
				transform=preprocess, label_dict=label_dict,
				cache_size=train_cache, cache_workers=min(4, args.num_workers or 4),
				col=args.col)
		print(train_ds)

		print(f"\n── Building Val dataset ({len(df_val):,} samples) ──")
		val_ds = HistoricalArchivesMultiLabelDataset(
				dataset_name=f"{dataset_name}_VAL", train=False,
				data_frame=df_val.sort_values("img_path").reset_index(drop=True),
				transform=preprocess, label_dict=label_dict,
				cache_size=val_cache, cache_workers=min(4, args.num_workers or 4),
				col=args.col)
		print(val_ds)

		# ── CHECK A ───────────────────────────────────────────────────────────────
		imb_tr = analyse_class_imbalance(train_ds, "TRAIN",
																			pareto=args.pareto,
																			pw_tail_threshold=args.pw_tail_threshold,
																			output_dir=output_dir)
		imb_va = analyse_class_imbalance(val_ds,   "VAL",
																			pareto=args.pareto,
																			pw_tail_threshold=args.pw_tail_threshold,
																			output_dir=output_dir)

		# ── CHECK B ───────────────────────────────────────────────────────────────
		# Auto-cap co-occurrence for large-vocab datasets if user didn't override.
		# The full Phi matrix for 7,485 classes = 7485²×8 bytes ≈ 449 GB — unusable.
		# Default cap: 500 classes (Phi matrix ≈ 2 MB, fast).
		AUTO_CAP = 500
		max_cooc = args.max_cooc_classes
		if max_cooc is None and n_classes > AUTO_CAP:
				max_cooc = AUTO_CAP
				print(f"\n[Co-occurrence] {n_classes:,} classes detected — auto-capping to top "
							f"{AUTO_CAP} for CHECK B.\n"
							f"  Override with --max_cooc_classes N (or --max_cooc_classes 0 to disable cap).")
		elif args.max_cooc_classes == 0:
				max_cooc = None   # 0 means "no cap, user accepts the risk"
		cooc_tr = analyse_cooccurrence(train_ds, "TRAIN",
																	 phi_pos=args.phi_pos, phi_neg=args.phi_neg,
																	 top_k=max_cooc,
																	 output_dir=output_dir)
		cooc_va = analyse_cooccurrence(val_ds,   "VAL",
																	 phi_pos=args.phi_pos, phi_neg=args.phi_neg,
																	 top_k=max_cooc,
																	 output_dir=output_dir)

		# ── CHECK C ───────────────────────────────────────────────────────────────
		# NOTE: num_workers=0 is intentional here.
		# With num_workers>0, DataLoader spawns subprocesses that each hold their
		# own copy of the dataset object. Cache hits/misses would accumulate on those
		# copies and be invisible to the main process. Using num_workers=0 keeps
		# everything in the main process so hit/miss counters are accurate.
		print(f"\n{'─'*60}")
		print("  CHECK C  ·  Dataloader Batch Integrity  (num_workers=0, single-process)")
		print(f"{'─'*60}")
		integrity_kw = dict(batch_size=args.batch_size, num_workers=0,
												pin_memory=False, drop_last=False)
		train_loader = DataLoader(train_ds, shuffle=True,  **integrity_kw)
		val_loader   = DataLoader(val_ds,   shuffle=False, **integrity_kw)

		import time as _time
		for name, loader, expected in [("TRAIN", train_loader, len(df_train)),
																		("VAL",   val_loader,   len(df_val))]:
				t0 = _time.perf_counter()
				r  = run_epoch(loader, n_classes, name)
				elapsed = _time.perf_counter() - t0
				throughput = r["n_samples"] / elapsed if elapsed > 0 else 0
				ok = r["n_corrupt"] == 0 and r["n_samples"] == expected and not r["shape_errors"]
				print(f"  [{'✓' if ok else '✗'}] {name}: "
							f"{r['n_samples']:,} samples | corrupt={r['n_corrupt']} | "
							f"avg_density={r['avg_density']:.2f} | "
							f"throughput={throughput:.0f} img/s | "
							f"shape_errors={len(r['shape_errors'])}")
				if r["shape_errors"]:
						for e in r["shape_errors"]: print(f"       {e}")

		# ── Cache stats (accurate because CHECK C ran single-process) ─────────────
		print(f"\n{'─'*60}")
		print("  Cache Stats after epoch  (single-process, hits/misses are accurate)")
		print(f"{'─'*60}")
		for ds_name, ds in [("TRAIN", train_ds), ("VAL", val_ds)]:
				cs = ds.get_cache_stats()
				print(f"  [{ds_name}]  hits={cs['hits']:,}  misses={cs['misses']:,}  "
							f"hit_rate={cs['hit_rate_pct']:.1f}%  cache_size={cs['cache_size']:,}")

		# ── Cross-split checks ────────────────────────────────────────────────────
		print(f"\n{'─'*60}")
		print("  Cross-split Consistency Checks")
		print(f"{'─'*60}")

		assert train_ds._num_classes == val_ds._num_classes
		print(f"  [✓] Shared vocabulary: {n_classes} classes")

		val_present   = {l for raw in val_ds.labels
										 for l in (ast.literal_eval(raw) if _parseable(raw) else [])}
		train_present = {l for raw in train_ds.labels
										 for l in (ast.literal_eval(raw) if _parseable(raw) else [])}
		unseen = val_present - train_present
		if unseen:
				print(f"  [!] {len(unseen)} val labels absent from train: {sorted(unseen)}")
				print(f"      → pos_weight for these will be 1.0 (not penalised)")
		else:
				print(f"  [✓] All val labels present in train split")

		r_tr, r_va = imb_tr["imb_ratio"], imb_va["imb_ratio"]
		print(f"  [✓] Imbalance ratio  train={r_tr:.1f}×  val={r_va:.1f}×")
		if abs(r_tr - r_va) > r_tr * 0.5:
				print(f"      [!] Ratios differ >50% — consider stratified splitting")

		n_rare_tr = imb_tr["n_rare"]
		n_zero_tr = imb_tr["n_zero"]
		n_zero_va = imb_va["n_zero"]
		print(f"  [{'!' if n_rare_tr > 0 else '✓'}] {n_rare_tr:,} train labels with "
					f"pos_weight > {args.pw_tail_threshold:.0f}  (effectively rare)")
		print(f"  [{'!' if n_zero_tr > 0 else '✓'}] {n_zero_tr:,} zero-count labels in train split")
		if n_zero_va > 0:
				zero_va_only = imb_va["zero_set"] - imb_tr["zero_set"]
				print(f"  [!] {n_zero_va:,} zero-count labels in val split "
							f"({len(zero_va_only):,} absent from train too — "
							f"model will never predict these)")

		# ── Summary ───────────────────────────────────────────────────────────────
		outputs = ["imbalance_train.png", "imbalance_val.png",
							 "cooccurrence_phi_train.png", "cooccurrence_phi_val.png",
							 "cooccurrence_train.html", "cooccurrence_val.html"]
		print(f"\n{sep}")
		print("  ✅  ALL CHECKS COMPLETE")
		print(f"  Output files in: {output_dir}")
		for f in outputs:
				path = os.path.join(output_dir, f)
				print(f"    [{'✓' if os.path.exists(path) else '✗'}] {f}")
		print(sep)


def _parseable(raw):
		try: ast.literal_eval(raw); return True
		except: return False


if __name__ == "__main__":
		main()