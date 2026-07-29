# eval_ood_transfer.py
#
# Out-of-Distribution (OOD) Transfer Experiment — Claim C3
# "Provenance over Overfitting"
# ─────────────────────────────────────────────────────────────────────────────
# Design: LEAVE-ONE-ARCHIVE-OUT on the merged HISTORY-X4 dataset.
#
# The model is trained on the full HISTORY-X4 merged dataset (all 4 archives).
# At evaluation time, the val split is partitioned by archive identity
# (extracted from doc_url) and each archive's subset is evaluated
# independently as an OOD probe.
#
# Archive identity is derived from doc_url — the path already encodes the
# archive folder name (e.g. "SMU_1900-01-01_1970-12-31").
# No second CSV is needed.  No cross-dataset loading.
#
# Three systems are compared on each archive subset:
#   B0  — Vanilla CLIP (zero-shot, no fine-tuning)
#   B2  — Standard CLIP fine-tune (no regime conditioning, optional)
#   RACL — Regime-Aware Contrastive Learning (ours)
#
# Domain shift per archive is quantified via:
#   FID   — Fréchet distance between archive image-embedding distributions
#   Δcos  — cosine distance between archive embedding centroids
#
# Outputs (under <ddir>/outputs/ood_transfer/)
# ─────────────────────────────────────────────
#   ood_transfer_results.json          — full metrics dict
#   ood_transfer_table.tex             — LaTeX booktabs table
#   ood_transfer_results.csv           — flat CSV for plotting
#   ood_domain_shift.png               — FID / Δcos per archive
#   ood_map_comparison.png             — mAP-all per archive × system
#   ood_gap_rel_comparison.png         — Gap_rel per archive × system
#
# How to run:
#   python eval_ood_transfer.py \
#     -csv /home/farid/datasets/WW_DATASETs/HISTORY_X4/metadata_multi_label_multimodal.csv \
#     --clip_model ViT-L/14 \
#     --peft_method lora \
#     -v
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
VALID_REGIMES = ["AGREEMENT", "SOFT_CONFLICT", "HARD_CONFLICT"]
SKIP_REGIMES  = {"MISSING_MODALITY", "INVALID_JSON", FALLBACK_REGIME}
GAP_REL_EPS   = 1e-8

# Known HISTORY-X4 archive identifiers (substring match against doc_url)
KNOWN_ARCHIVES = ["SMU", "NATIONAL_ARCHIVES", "EUROPEANA", "WWII"]

SYSTEM_LABELS = {
		"racl":     "RACL (Ours)",
		"baseline": "Vanilla CLIP (B0)",
		"b2":       "Std. Fine-tune (B2)",
}

REGIME_DISPLAY = {
		"AGREEMENT":     "Agreement",
		"SOFT_CONFLICT": "Soft Conflict",
		"HARD_CONFLICT": "Hard Conflict",
		"ALL":           "All (Global)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Archive identity extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_archive_name(doc_url: str, known_archives: List[str] = KNOWN_ARCHIVES) -> str:
		"""
		Derive archive identity from doc_url path.
		Matches the first known archive keyword (case-insensitive) found in the
		path components.  Falls back to the top-level directory name.

		Examples
		--------
		/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/images/foo.jpg
				→ "SMU"
		/home/farid/datasets/WW_DATASETs/NATIONAL_ARCHIVES_1939-01-01_1945-12-31/images/bar.jpg
				→ "NATIONAL_ARCHIVES"
		"""
		parts = Path(doc_url).parts
		for part in parts:
				for archive in known_archives:
						if archive.upper() in part.upper():
								return archive.upper()
		# Fallback: use the immediate parent directory name
		return Path(doc_url).parent.parent.name.upper()


def partition_val_loader_by_archive(
		val_loader: DataLoader,
		known_archives: List[str] = KNOWN_ARCHIVES,
		verbose: bool = True,
) -> Dict[str, List[int]]:
		"""
		Scan val_loader.dataset.sample_ids and group indices by archive.
		Returns {archive_name: [dataset_indices]}.
		"""
		archive_indices: Dict[str, List[int]] = defaultdict(list)
		dataset = val_loader.dataset

		for idx, doc_url in enumerate(dataset.sample_ids):
				archive = extract_archive_name(doc_url, known_archives)
				archive_indices[archive].append(idx)

		if verbose:
				print(f"\n[partition_val_loader_by_archive] Archive distribution in val split:")
				total = sum(len(v) for v in archive_indices.values())
				for arch, idxs in sorted(archive_indices.items()):
						print(f"  ├─ {arch:<22s}: {len(idxs):>6,} ({len(idxs)/max(total,1)*100:.1f}%)")
				print(f"  └─ Total: {total:,}")

		return dict(archive_indices)


def build_archive_subset_loader(
		val_loader:  DataLoader,
		indices:     List[int],
		batch_size:  int,
		num_workers: int,
) -> DataLoader:
		"""
		Build a DataLoader over a subset of val_loader.dataset defined by indices.
		Reuses the same RegimeAwareDataset (no data copy) via Subset.
		"""
		subset = torch.utils.data.Subset(val_loader.dataset, indices)
		return DataLoader(
				dataset=subset,
				batch_size=batch_size,
				shuffle=False,
				pin_memory=torch.cuda.is_available(),
				num_workers=num_workers,
				prefetch_factor=2 if num_workers > 0 else None,
				persistent_workers=(num_workers > 0),
				collate_fn=customized_collate_fn,
				drop_last=False,
		)


# ─────────────────────────────────────────────────────────────────────────────
# Domain-shift quantifiers
# ─────────────────────────────────────────────────────────────────────────────
def _compute_fid(
		mu1: np.ndarray, sigma1: np.ndarray,
		mu2: np.ndarray, sigma2: np.ndarray,
) -> float:
		"""
		Fréchet distance between two multivariate Gaussians (embedding-FID).
		FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2·sqrt(Σ1·Σ2))
		"""
		from scipy.linalg import sqrtm as mat_sqrtm
		diff    = mu1 - mu2
		covmean, _ = mat_sqrtm(sigma1 @ sigma2, disp=False)
		if np.iscomplexobj(covmean):
				covmean = covmean.real
		fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
		return fid


def _compute_cosine_distance(c1: np.ndarray, c2: np.ndarray) -> float:
		"""Cosine distance = 1 − cosine_similarity between two centroid vectors."""
		n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
		if n1 < 1e-12 or n2 < 1e-12:
				return float("nan")
		return float(1.0 - (c1 @ c2) / (n1 * n2))


@torch.no_grad()
def extract_image_embeddings(
		model:   torch.nn.Module,
		loader:  DataLoader,
		device:  torch.device,
		verbose: bool = False,
) -> np.ndarray:
		"""
		Forward-pass all images through the vision encoder.
		Returns L2-normalised embeddings: (N, D) float32 numpy array.
		"""
		model.eval()
		all_embeds = []
		for batch in loader:
				if not batch:
						continue
				images = batch["image"].to(device, non_blocking=True)
				embeds = torch.nn.functional.normalize(
						model.encode_image(images).float(), dim=-1
				).cpu().numpy()
				all_embeds.append(embeds)
		return np.concatenate(all_embeds, axis=0) if all_embeds else np.empty((0, 0))


def compute_domain_shift(
		global_embeds:  np.ndarray,
		archive_embeds: np.ndarray,
		archive_name:   str,
		verbose:        bool = True,
) -> Dict[str, float]:
		"""
		Quantify domain shift between the global (all-archive) embedding
		distribution and a single archive's embedding distribution.
		"""
		if global_embeds.shape[0] < 2 or archive_embeds.shape[0] < 2:
				return {"fid": float("nan"), "delta_cos": float("nan")}

		mu_g, mu_a   = global_embeds.mean(0),  archive_embeds.mean(0)
		eps          = 1e-6
		sigma_g      = np.cov(global_embeds,  rowvar=False) + np.eye(global_embeds.shape[1])  * eps
		sigma_a      = np.cov(archive_embeds, rowvar=False) + np.eye(archive_embeds.shape[1]) * eps

		fid   = _compute_fid(mu_g, sigma_g, mu_a, sigma_a)
		d_cos = _compute_cosine_distance(mu_g, mu_a)

		if verbose:
				print(f"  [domain_shift][{archive_name}]  FID={fid:.4f}  Δcos={d_cos:.4f}")

		return {"fid": fid, "delta_cos": d_cos}


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval evaluation on an archive subset
# ─────────────────────────────────────────────────────────────────────────────
def _compute_gap_rel(map_head: float, map_rare: float) -> float:
		if np.isnan(map_head) or np.isnan(map_rare):
				return float("nan")
		return (map_rare - map_head) / (map_head + GAP_REL_EPS)


@torch.no_grad()
def evaluate_archive_subset(
		model:            torch.nn.Module,
		subset_loader:    DataLoader,
		class_embeds_norm: torch.Tensor,   # [C, D] normalised, on device
		active_mask:      torch.Tensor,    # [C] bool
		head_mask:        torch.Tensor,    # [C] bool
		rare_mask:        torch.Tensor,    # [C] bool
		device:           torch.device,
		system_tag:       str  = "racl",
		archive_name:     str  = "UNKNOWN",
		verbose:          bool = True,
) -> Dict[str, Any]:
		"""
		Evaluate `model` on a single archive's val subset.
		Returns a metrics dict with global + per-regime breakdowns.
		"""
		model.eval()

		bucket_scores:  Dict[str, List[torch.Tensor]] = defaultdict(list)
		bucket_targets: Dict[str, List[torch.Tensor]] = defaultdict(list)
		regime_counts:  Dict[str, int]                = defaultdict(int)
		n_total = 0

		for batch in subset_loader:
				if not batch:
						continue

				images    = batch["image"].to(device, non_blocking=True)
				label_vec = batch["label_vec"]
				regimes   = batch["regime"]

				img_embeds = torch.nn.functional.normalize(
						model.encode_image(images).float(), dim=-1
				)  # [B, D]
				scores = torch.matmul(img_embeds, class_embeds_norm.T).cpu()  # [B, C]

				for i, regime in enumerate(regimes):
						n_total += 1
						regime_counts[regime] += 1
						s = scores[i].unsqueeze(0)
						t = label_vec[i].unsqueeze(0)

						bucket_scores["ALL"].append(s)
						bucket_targets["ALL"].append(t)

						if regime not in SKIP_REGIMES:
								canonical = regime.upper().replace(" ", "_")
								if canonical in VALID_REGIMES:
										bucket_scores[canonical].append(s)
										bucket_targets[canonical].append(t)

		# ── Global metrics ───────────────────────────────────────────────────────
		def _metrics_for_bucket(bucket: str) -> Dict[str, Any]:
				if not bucket_scores.get(bucket):
						return {"n_samples": 0, "map_all": float("nan"),
										"map_head": float("nan"), "map_rare": float("nan"),
										"gap_rel": float("nan"), "p@1": float("nan"),
										"p@5": float("nan"), "ndcg@5": float("nan")}
				s_cat = torch.cat(bucket_scores[bucket],  dim=0)
				t_cat = torch.cat(bucket_targets[bucket], dim=0)
				m = _compute_retrieval_metrics(
						scores=s_cat, targets=t_cat,
						active_mask=active_mask.cpu(),
						head_mask=head_mask.cpu(),
						rare_mask=rare_mask.cpu(),
				)
				m = {k.replace("val_", ""): v for k, v in m.items()}
				m["gap_rel"]   = _compute_gap_rel(m.get("map_head", float("nan")), m.get("map_rare", float("nan")))
				m["n_samples"] = len(bucket_scores[bucket])
				return m

		global_m = _metrics_for_bucket("ALL")
		per_regime = {r: _metrics_for_bucket(r) for r in VALID_REGIMES}

		if verbose:
				print(
						f"  [{system_tag}][{archive_name}]  N={n_total:,}  "
						f"mAP-all={global_m.get('map_all', float('nan')):.4f}  "
						f"mAP-rare={global_m.get('map_rare', float('nan')):.4f}  "
						f"Gap_rel={global_m.get('gap_rel', float('nan')):.4f}"
				)

		return {
				"system":        system_tag,
				"archive":       archive_name,
				"n_samples":     n_total,
				"global":        global_m,
				"per_regime":    per_regime,
				"regime_counts": dict(regime_counts),
		}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_domain_shift(
		shift_by_archive: Dict[str, Dict[str, float]],
		output_path: str,
) -> None:
		"""
		Two-panel bar chart: FID and Δcos per archive.
		Each bar = one archive's shift relative to the global distribution.
		"""
		archives = sorted(shift_by_archive.keys())
		fid_vals  = [shift_by_archive[a]["fid"]       for a in archives]
		dcos_vals = [shift_by_archive[a]["delta_cos"]  for a in archives]

		fig, axes = plt.subplots(1, 2, figsize=(max(9, len(archives) * 2.5), 4.5))
		x = np.arange(len(archives))

		for ax, vals, label, color in zip(
				axes,
				[fid_vals, dcos_vals],
				["FID (embedding)", "Δcos (centroids)"],
				["#1f77b4", "#d62728"],
		):
				bars = ax.bar(x, vals, color=color, edgecolor="black", alpha=0.85)
				ax.set_xticks(x)
				ax.set_xticklabels(archives, rotation=20, ha="right", fontsize=9)
				ax.set_title(label)
				ax.set_ylabel(label)
				for bar, v in zip(bars, vals):
						if not np.isnan(v):
								ax.text(
										bar.get_x() + bar.get_width() / 2,
										bar.get_height() * 1.02,
										f"{v:.3f}", ha="center", va="bottom", fontsize=8,
								)

		fig.suptitle("Domain Shift: Each Archive vs. Global HISTORY-X4 Distribution", fontsize=12)
		fig.tight_layout()
		fig.savefig(output_path, dpi=200, bbox_inches="tight")
		plt.close(fig)
		print(f"[plot_domain_shift] {output_path}")


def plot_map_comparison(
		results: Dict[str, Dict[str, Dict]],
		output_path: str,
) -> None:
		"""
		Grouped bar chart: mAP-all per archive × system.
		results[archive][system_tag] = metrics dict
		"""
		archives = sorted(results.keys())
		systems  = [s for s in ["baseline", "b2", "racl"] if any(s in results[a] for a in archives)]
		colors   = {"baseline": "#1f77b4", "b2": "#ff7f0e", "racl": "#2ca02c"}

		x     = np.arange(len(archives))
		width = 0.22
		fig, ax = plt.subplots(figsize=(max(10, len(archives) * 2.5), 5))

		for i, sys_key in enumerate(systems):
				vals = [results[a].get(sys_key, {}).get("global", {}).get("map_all", float("nan"))
								for a in archives]
				bars = ax.bar(
						x + i * width, vals, width,
						label=SYSTEM_LABELS.get(sys_key, sys_key),
						color=colors.get(sys_key, "grey"),
						edgecolor="black", alpha=0.85,
				)
				for bar, v in zip(bars, vals):
						if not np.isnan(v):
								ax.text(
										bar.get_x() + bar.get_width() / 2,
										bar.get_height() + 0.003,
										f"{v:.3f}", ha="center", va="bottom", fontsize=7,
								)

		ax.set_xticks(x + width * (len(systems) - 1) / 2)
		ax.set_xticklabels(archives, rotation=15, ha="right", fontsize=9)
		ax.set_ylabel("mAP-all")
		ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
		ax.set_title("OOD Transfer: mAP-all per Archive × System (HISTORY-X4 Leave-One-Archive-Out)")
		ax.legend(loc="upper right")
		fig.tight_layout()
		fig.savefig(output_path, dpi=200, bbox_inches="tight")
		plt.close(fig)
		print(f"[plot_map_comparison] {output_path}")


def plot_gap_rel_comparison(
		results: Dict[str, Dict[str, Dict]],
		output_path: str,
) -> None:
		"""
		Grouped horizontal bar chart: Gap_rel per archive × system.
		Gap_rel closer to 0 = better tail recovery.
		"""
		archives = sorted(results.keys())
		systems  = [s for s in ["baseline", "b2", "racl"] if any(s in results[a] for a in archives)]
		colors   = {"baseline": "#1f77b4", "b2": "#ff7f0e", "racl": "#2ca02c"}

		n_arch   = len(archives)
		n_sys    = len(systems)
		fig, axes = plt.subplots(1, n_arch, figsize=(max(10, n_arch * 3), 4), sharey=False)
		if n_arch == 1:
				axes = [axes]

		for ax, archive in zip(axes, archives):
				vals   = [results[archive].get(s, {}).get("global", {}).get("gap_rel", float("nan")) for s in systems]
				labels = [SYSTEM_LABELS.get(s, s) for s in systems]
				bar_colors = [colors.get(s, "grey") for s in systems]
				bars = ax.barh(labels, vals, color=bar_colors, edgecolor="black", alpha=0.85)
				ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
				for bar, v in zip(bars, vals):
						if not np.isnan(v):
								ax.text(
										v + (0.005 if v >= 0 else -0.005),
										bar.get_y() + bar.get_height() / 2,
										f"{v:.3f}", va="center",
										ha="left" if v >= 0 else "right", fontsize=8,
								)
				ax.set_title(archive, fontsize=10)
				ax.set_xlabel("Gap$_{\\mathrm{rel}}$")

		fig.suptitle("Tail Recovery (Gap$_{\\mathrm{rel}}$) per Archive × System", fontsize=12)
		fig.tight_layout()
		fig.savefig(output_path, dpi=200, bbox_inches="tight")
		plt.close(fig)
		print(f"[plot_gap_rel_comparison] {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────
def _clean_for_json(obj: Any) -> Any:
		if isinstance(obj, float) and np.isnan(obj):
				return None
		if isinstance(obj, (np.floating,)):
				return float(obj)
		if isinstance(obj, (np.integer,)):
				return int(obj)
		if isinstance(obj, np.ndarray):
				return obj.tolist()
		if isinstance(obj, dict):
				return {k: _clean_for_json(v) for k, v in obj.items()}
		if isinstance(obj, list):
				return [_clean_for_json(v) for v in obj]
		return obj


def save_json(results: Dict[str, Any], output_path: str) -> None:
		with open(output_path, "w", encoding="utf-8") as f:
				json.dump(_clean_for_json(results), f, indent=2, ensure_ascii=False)
		print(f"[save_json] {output_path}")


def save_latex_table(
		results:          Dict[str, Dict[str, Dict]],
		shift_by_archive: Dict[str, Dict[str, float]],
		output_path:      str,
) -> None:
		"""
		LaTeX booktabs table.
		Rows = archive × system.  Columns = mAP-all/head/rare, Gap_rel, P@1, nDCG@5.
		Domain-shift (FID, Δcos) appended as sub-header per archive block.
		"""
		def _fmt(v: Any) -> str:
				if isinstance(v, float) and np.isnan(v):
						return "—"
				return f"{v:.4f}"

		archives = sorted(results.keys())
		systems  = [s for s in ["baseline", "b2", "racl"] if any(s in results[a] for a in archives)]

		lines = [
				r"\begin{table}[t]",
				r"\centering",
				r"\caption{OOD Transfer: Leave-One-Archive-Out on HISTORY-X4. "
				r"Domain shift (FID, $\Delta\cos$) is computed between each archive "
				r"and the global embedding distribution.}",
				r"\label{tab:ood_transfer}",
				r"\begin{tabular}{llrrrrrr}",
				r"\toprule",
				r"Archive & System & mAP-all & mAP-head & mAP-rare "
				r"& $\text{Gap}_{\text{rel}}$ & P@1 & nDCG@5 \\",
				r"\midrule",
		]

		for archive in archives:
				shift = shift_by_archive.get(archive, {})
				fid   = shift.get("fid",       float("nan"))
				dcos  = shift.get("delta_cos", float("nan"))
				fid_str  = f"{fid:.2f}"  if not np.isnan(fid)  else "—"
				dcos_str = f"{dcos:.4f}" if not np.isnan(dcos) else "—"
				lines.append(
						rf"\multicolumn{{8}}{{l}}{{\textit{{{archive}}} "
						rf"(FID$={fid_str}$, $\Delta\cos={dcos_str}$)}} \\"
				)
				for sys_key in systems:
						m = results[archive].get(sys_key, {}).get("global", {})
						label  = SYSTEM_LABELS.get(sys_key, sys_key)
						prefix = r"\textbf{" if sys_key == "racl" else ""
						suffix = r"}"        if sys_key == "racl" else ""
						lines.append(
								f"& {prefix}{label}{suffix} & "
								f"{prefix}{_fmt(m.get('map_all'))}{suffix} & "
								f"{prefix}{_fmt(m.get('map_head'))}{suffix} & "
								f"{prefix}{_fmt(m.get('map_rare'))}{suffix} & "
								f"{prefix}{_fmt(m.get('gap_rel'))}{suffix} & "
								f"{prefix}{_fmt(m.get('p@1'))}{suffix} & "
								f"{prefix}{_fmt(m.get('ndcg@5'))}{suffix} \\\\"
						)
				lines.append(r"\midrule")

		lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

		with open(output_path, "w", encoding="utf-8") as f:
				f.write("\n".join(lines) + "\n")
		print(f"[save_latex_table] {output_path}")


def save_csv(
		results:          Dict[str, Dict[str, Dict]],
		shift_by_archive: Dict[str, Dict[str, float]],
		output_path:      str,
) -> None:
		fieldnames = [
				"archive", "system", "n_samples",
				"fid", "delta_cos",
				"map_all", "map_head", "map_rare", "gap_rel", "p@1", "p@5", "ndcg@5",
		]
		rows = []
		for archive in sorted(results.keys()):
				shift = shift_by_archive.get(archive, {})
				for sys_key in ["baseline", "b2", "racl"]:
						if sys_key not in results[archive]:
								continue
						res = results[archive][sys_key]
						m   = res.get("global", {})
						rows.append({
								"archive":   archive,
								"system":    SYSTEM_LABELS.get(sys_key, sys_key),
								"n_samples": res.get("n_samples", 0),
								"fid":       shift.get("fid",       float("nan")),
								"delta_cos": shift.get("delta_cos", float("nan")),
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
		print(f"[save_csv] {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_model(
		clip_model_name: str,
		peft_method:     str,
		ckpt_path:       Optional[str],
		device:          torch.device,
		ddir:            str,
		verbose:         bool = True,
) -> Tuple[torch.nn.Module, int]:
		model, _ = clip.load(
				name=clip_model_name,
				device=device,
				jit=False,
				random_weights=False,
				dropout=0.0,
				download_root=get_model_directory(path=ddir),
		)
		model.name = clip_model_name
		model, _ = setup_peft(model=model, peft_method=peft_method, verbose=verbose)
		model = model.to(device)
		epoch = 0
		if ckpt_path:
				epoch, _ = load_checkpoint(ckpt_path=ckpt_path, model=model, device=device, verbose=verbose)
		return model, epoch


def _discover_checkpoint(outputs_dir: str) -> Optional[str]:
		ckpt_dir = os.path.join(outputs_dir, "checkpoints")
		if not os.path.isdir(ckpt_dir):
				return None
		pt_files = sorted(Path(ckpt_dir).glob("*.pt"))
		return str(pt_files[0]) if pt_files else None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="OOD Transfer — Leave-One-Archive-Out on HISTORY-X4 (Claim C3)",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument("--metadata", "-csv", required=True,
								 help="Path to merged HISTORY-X4 metadata CSV")
	p.add_argument("--clip_model",  "-cm",   default="ViT-B/32")
	p.add_argument(
		"--peft_method", 
		"-peft", 
		default="lora", 
		choices=[
			"lora", "lora_plus", "dora", "rslora", "ia3", "vera",
			"tip_adapter", "tip_adapter_f",
			"clip_adapter_v", "clip_adapter_t", "clip_adapter_vt",
			"probe", "full",
		]
	)
	p.add_argument("--racl_ckpt", "-ckpt", default=None, help="RACL checkpoint (.pt). Auto-discovered from outputs/checkpoints/ if omitted.")
	p.add_argument("--baseline_ckpt", "-b2", default=None, help="Std. fine-tune checkpoint (.pt). Skipped if omitted.")
	p.add_argument("--batch_size",  "-bs", type=int, default=256)
	p.add_argument("--num_workers", "-nw", type=int, default=8)
	p.add_argument("--id_col",   default="doc_url")
	p.add_argument("--text_col", default="multimodal_labels")
	p.add_argument("--pw_mode", default="sqrt", choices=["log", "sqrt", "linear"])
	p.add_argument("--pw_max_cap", type=float, default=50.0)
	p.add_argument("--archives", nargs="+", default=KNOWN_ARCHIVES, help="Archive keywords to probe (matched against doc_url)")
	p.add_argument("--verbose", "-v", action="store_true")

	return p.parse_args()

def main():
		args = parse_args()
		if args.verbose:
				print(args)

		ddir        = os.path.dirname(args.metadata)
		outputs_dir = os.path.join(ddir, "outputs")
		ood_dir     = os.path.join(outputs_dir, "ood_transfer")
		os.makedirs(ood_dir, exist_ok=True)

		racl_ckpt = args.racl_ckpt or _discover_checkpoint(outputs_dir)
		if racl_ckpt is None:
				raise FileNotFoundError(
						f"[main] No RACL checkpoint found under {outputs_dir}/checkpoints/. "
						"Pass --racl_ckpt explicitly."
				)

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		print(f"\n{'='*80}")
		print(f"[eval_ood_transfer] Leave-One-Archive-Out OOD Transfer (HISTORY-X4)")
		print(f"  ├─ Metadata    : {args.metadata}")
		print(f"  ├─ CLIP model  : {args.clip_model}")
		print(f"  ├─ PEFT method : {args.peft_method}")
		print(f"  ├─ RACL ckpt   : {racl_ckpt}")
		print(f"  ├─ B2 ckpt     : {args.baseline_ckpt or '(skipped)'}")
		print(f"  ├─ Archives    : {args.archives}")
		print(f"  ├─ Output dir  : {ood_dir}")
		print(f"  └─ Device      : {device}")
		print(f"{'='*80}")

		# ── 1. Load dataloaders (merged HISTORY-X4) ───────────────────────────────
		_tmp, _ = clip.load(
				name=args.clip_model, device="cpu", jit=False,
				download_root=get_model_directory(path=ddir),
		)
		input_resolution = getattr(_tmp.visual, "input_resolution", None)
		del _tmp

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

		# ── 2. Loss masks (head / rare split — same as training) ──────────────────
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

		# ── 3. Partition val split by archive ─────────────────────────────────────
		archive_indices = partition_val_loader_by_archive(
				val_loader=val_loader,
				known_archives=args.archives,
				verbose=args.verbose,
		)
		archive_loaders = {
				arch: build_archive_subset_loader(
						val_loader=val_loader,
						indices=idxs,
						batch_size=args.batch_size,
						num_workers=args.num_workers,
				)
				for arch, idxs in archive_indices.items()
				if len(idxs) > 0
		}

		# ── 4. Load RACL model and extract global embeddings for domain shift ─────
		print(f"\n[main] Loading RACL model ...")
		model_racl, racl_epoch = _load_model(
				clip_model_name=args.clip_model,
				peft_method=args.peft_method,
				ckpt_path=racl_ckpt,
				device=device,
				ddir=ddir,
				verbose=args.verbose,
		)

		print(f"\n[main] Extracting global val embeddings for domain-shift baseline ...")
		global_embeds = extract_image_embeddings(model_racl, val_loader, device, verbose=args.verbose)

		# ── 5. Build class embeddings (shared across all archive subsets) ─────────
		print(f"\n[main] Building class embeddings ({num_classes:,} labels) ...")
		all_class_embeds = build_class_embeddings(
				model=model_racl, label_dict=label_dict, device=device, verbose=args.verbose,
		)
		class_embeds_norm = torch.nn.functional.normalize(all_class_embeds, dim=-1).to(device)

		# ── 6. Domain shift per archive ───────────────────────────────────────────
		print(f"\n[main] Computing domain shift per archive ...")
		shift_by_archive: Dict[str, Dict[str, float]] = {}
		for arch, loader in archive_loaders.items():
				arch_embeds = extract_image_embeddings(model_racl, loader, device, verbose=False)
				shift_by_archive[arch] = compute_domain_shift(
						global_embeds=global_embeds,
						archive_embeds=arch_embeds,
						archive_name=arch,
						verbose=args.verbose,
				)

		# ── 7. Evaluate all systems per archive ───────────────────────────────────
		# results[archive][system_tag] = metrics dict
		results: Dict[str, Dict[str, Dict]] = defaultdict(dict)

		# ── RACL ──────────────────────────────────────────────────────────────────
		print(f"\n[main] ── Evaluating RACL (Ours) ──")
		for arch, loader in archive_loaders.items():
				results[arch]["racl"] = evaluate_archive_subset(
						model=model_racl,
						subset_loader=loader,
						class_embeds_norm=class_embeds_norm,
						active_mask=active_mask,
						head_mask=head_mask,
						rare_mask=rare_mask,
						device=device,
						system_tag="racl",
						archive_name=arch,
						verbose=args.verbose,
				)
		del model_racl

		# ── Vanilla CLIP B0 ───────────────────────────────────────────────────────
		print(f"\n[main] ── Evaluating Vanilla CLIP (B0) ──")
		model_b0, _ = _load_model(
				clip_model_name=args.clip_model,
				peft_method=args.peft_method,
				ckpt_path=None,
				device=device,
				ddir=ddir,
				verbose=args.verbose,
		)
		b0_class_embeds = torch.nn.functional.normalize(
				build_class_embeddings(model=model_b0, label_dict=label_dict, device=device, verbose=False),
				dim=-1,
		).to(device)
		for arch, loader in archive_loaders.items():
				results[arch]["baseline"] = evaluate_archive_subset(
						model=model_b0,
						subset_loader=loader,
						class_embeds_norm=b0_class_embeds,
						active_mask=active_mask,
						head_mask=head_mask,
						rare_mask=rare_mask,
						device=device,
						system_tag="baseline",
						archive_name=arch,
						verbose=args.verbose,
				)
		del model_b0

		# ── Baseline B2 (optional) ────────────────────────────────────────────────
		if args.baseline_ckpt:
				print(f"\n[main] ── Evaluating Std. Fine-tune (B2) ──")
				model_b2, _ = _load_model(
						clip_model_name=args.clip_model,
						peft_method=args.peft_method,
						ckpt_path=args.baseline_ckpt,
						device=device,
						ddir=ddir,
						verbose=args.verbose,
				)
				b2_class_embeds = torch.nn.functional.normalize(
						build_class_embeddings(model=model_b2, label_dict=label_dict, device=device, verbose=False),
						dim=-1,
				).to(device)
				for arch, loader in archive_loaders.items():
						results[arch]["b2"] = evaluate_archive_subset(
								model=model_b2,
								subset_loader=loader,
								class_embeds_norm=b2_class_embeds,
								active_mask=active_mask,
								head_mask=head_mask,
								rare_mask=rare_mask,
								device=device,
								system_tag="b2",
								archive_name=arch,
								verbose=args.verbose,
						)
				del model_b2

		# ── 8. Print summary table ────────────────────────────────────────────────
		print(f"\n{'─'*110}")
		print(f"{'Archive':<22s} {'System':<26s} {'N':>7s} {'FID':>8s} {'Δcos':>7s} "
					f"{'mAP-all':>9s} {'mAP-rare':>9s} {'Gap_rel':>9s} {'P@1':>7s} {'nDCG@5':>8s}")
		print(f"{'─'*110}")

		def _f(v): return f"{v:.4f}" if not np.isnan(v) else "—"

		for arch in sorted(results.keys()):
				shift = shift_by_archive.get(arch, {})
				for sys_key in ["baseline", "b2", "racl"]:
						if sys_key not in results[arch]:
								continue
						res = results[arch][sys_key]
						m   = res.get("global", {})
						print(
								f"  {arch:<20s} {SYSTEM_LABELS.get(sys_key, sys_key):<26s} "
								f"{res.get('n_samples', 0):>7,} "
								f"{_f(shift.get('fid', float('nan'))):>8s} "
								f"{_f(shift.get('delta_cos', float('nan'))):>7s} "
								f"{_f(m.get('map_all')):>9s} "
								f"{_f(m.get('map_rare')):9s} "
								f"{_f(m.get('gap_rel')):>9s} "
								f"{_f(m.get('p@1')):>7s} "
								f"{_f(m.get('ndcg@5')):>8s}"
						)
				print(f"{'─'*110}")

		# ── 9. Verdict ────────────────────────────────────────────────────────────
		print(f"\n{'='*80}")
		print(f"[VERDICT]  OOD Transfer — Leave-One-Archive-Out (HISTORY-X4)")
		all_racl_wins = []
		for arch in sorted(results.keys()):
				racl_map = results[arch].get("racl",     {}).get("global", {}).get("map_all", float("nan"))
				b0_map   = results[arch].get("baseline", {}).get("global", {}).get("map_all", float("nan"))
				if np.isnan(racl_map) or np.isnan(b0_map):
						continue
				delta  = racl_map - b0_map
				symbol = "✓" if delta > 0 else "⚠"
				all_racl_wins.append(delta > 0)
				print(f"  {symbol}  [{arch}]  RACL={racl_map:.4f}  B0={b0_map:.4f}  Δ={delta:+.4f}")
		if all_racl_wins:
				n_wins = sum(all_racl_wins)
				print(f"\n  RACL outperforms B0 on {n_wins}/{len(all_racl_wins)} archives.")
				if n_wins == len(all_racl_wins):
						print(f"  ✓  Consistent OOD advantage — strongly supports Claim C3.")
				elif n_wins > 0:
						print(f"  ~  Partial OOD advantage — discuss per-archive breakdown in paper.")
				else:
						print(f"  ⚠  No OOD advantage — investigate and discuss honestly.")
		print(f"{'='*80}\n")

		# ── 10. Save outputs ──────────────────────────────────────────────────────
		full_results = {
				"clip_model":       args.clip_model,
				"peft_method":      args.peft_method,
				"racl_ckpt":        racl_ckpt,
				"racl_epoch":       racl_epoch,
				"archives_probed":  sorted(results.keys()),
				"domain_shift":     shift_by_archive,
				"results":          results,
		}
		save_json(full_results, os.path.join(ood_dir, "ood_transfer_results.json"))
		save_latex_table(results, shift_by_archive, os.path.join(ood_dir, "ood_transfer_table.tex"))
		save_csv(results, shift_by_archive, os.path.join(ood_dir, "ood_transfer_results.csv"))
		plot_domain_shift(shift_by_archive, os.path.join(ood_dir, "ood_domain_shift.png"))
		plot_map_comparison(results, os.path.join(ood_dir, "ood_map_comparison.png"))
		plot_gap_rel_comparison(results, os.path.join(ood_dir, "ood_gap_rel_comparison.png"))


if __name__ == "__main__":
		main()