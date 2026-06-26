# stage5_regime_aware_loss.py
#
# Stage 5: Regime-Aware Loss Functions
# ─────────────────────────────────────────────────────────────────────────────
# Replaces compute_multilabel_contrastive_loss() from loss.py with a
# regime-conditioned objective that operates on two orthogonal axes:
#
#   Axis 1 — Class-Level Balance  : pos_weight (inherited from loss.py)
#   Axis 2 — Sample-Level Regime  : ω_pos (throttle), ω_neg (repulsion)
#
# Total loss per sample i:
#
#   L_i = λ_i2t * ω_pos_i * L_i2t(pos_weight)
#       + λ_t2i * ω_pos_i * L_t2i
#       + λ_repel * ω_neg_i * L_repulsion(hard_negatives)
#
# Regime → weight mapping (from Stage 4 blueprint):
#
#   AGREEMENT      : ω_pos = 1.0,          ω_neg = 0.0
#   SOFT_CONFLICT  : ω_pos = 1 - |Δ|,      ω_neg = 0.0
#   HARD_CONFLICT  : ω_pos = ω_hard (0.3), ω_neg = 1 - mean(G(T⁻))
#   MISSING/INVALID: ω_pos = 0.0,          ω_neg = 0.0  (sample skipped)
#
# ─────────────────────────────────────────────────────────────────────────────

from utils import *

REGIME_OMEGA_POS_HARD   = 0.3   # ω_pos for HARD_CONFLICT positive targets
REGIME_OMEGA_POS_FLOOR  = 0.05  # minimum ω_pos for SOFT_CONFLICT (prevents zero)
REGIME_LAMBDA_I2T       = 0.5   # λ_i2t  (mirrors loss.py default)
REGIME_LAMBDA_T2I       = 0.5   # λ_t2i
REGIME_LAMBDA_REPEL     = 1.0   # λ_repel (repulsion arm weight)

VALID_REGIMES = {"AGREEMENT", "SOFT_CONFLICT", "HARD_CONFLICT"}
SKIP_REGIMES  = {"MISSING_MODALITY", "INVALID_JSON"}

def resolve_regime_weights(
	regimes:      List[str],
	w_pos_raw:    torch.Tensor,   # [B] — Stage 4 derived ω_pos (float)
	w_neg_raw:    torch.Tensor,   # [B] — Stage 4 derived ω_neg (float)
	device:       torch.device,
	verbose:      bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Converts per-sample regime strings + Stage 4 raw weights into
	three clean tensors used by the loss functions.
	
	Returns
	-------
	omega_pos  : [B] float32  — positive target scaling weight
	omega_neg  : [B] float32  — repulsion scaling weight
	valid_mask : [B] bool     — False for MISSING/INVALID samples (skip entirely)
	
	Note
	───────────
	Stage 4 already encodes the regime logic into w_pos / w_neg.
	This function's job is to:
		(a) enforce the valid_mask so MISSING/INVALID samples contribute
				zero gradient regardless of what Stage 4 stored, and
		(b) clamp ω_pos to [REGIME_OMEGA_POS_FLOOR, 1.0] so SOFT_CONFLICT
				samples never fully vanish from the gradient.
	"""
	B = len(regimes)
	omega_pos  = w_pos_raw.clone().float().to(device) # [B]
	omega_neg  = w_neg_raw.clone().float().to(device) # [B]
	valid_mask = torch.ones(B, dtype=torch.bool, device=device)
	for i, regime in enumerate(regimes):
		if regime in SKIP_REGIMES:
			omega_pos[i]  = 0.0
			omega_neg[i]  = 0.0
			valid_mask[i] = False
		elif regime not in VALID_REGIMES:
			# Unknown regime — treat conservatively as SOFT_CONFLICT
			omega_pos[i]  = REGIME_OMEGA_POS_FLOOR
			omega_neg[i]  = 0.0
			if verbose:
				print(
					f"[WARN][resolve_regime_weights] Unknown regime '{regime}' at idx {i}. "
					f"Defaulting to ω_pos={REGIME_OMEGA_POS_FLOOR}, ω_neg=0.0"
				)

	# Clamp ω_pos for valid samples to [floor, 1.0]
	omega_pos = torch.where(
		valid_mask,
		omega_pos.clamp(min=REGIME_OMEGA_POS_FLOOR, max=1.0),
		torch.zeros_like(omega_pos),
	)

	# Clamp ω_neg to [0, 1]
	omega_neg = omega_neg.clamp(min=0.0, max=1.0)
	if verbose:
		n_valid = valid_mask.sum().item()
		n_hard  = sum(r == "HARD_CONFLICT"  for r in regimes)
		n_soft  = sum(r == "SOFT_CONFLICT"  for r in regimes)
		n_agree = sum(r == "AGREEMENT"      for r in regimes)
		n_skip  = B - n_valid
		print(
			f"[resolve_regime_weights] B={B} | "
			f"AGREE={n_agree} SOFT={n_soft} HARD={n_hard} SKIP={n_skip} | "
			f"ω_pos [{omega_pos[valid_mask].min():.3f}, {omega_pos[valid_mask].max():.3f}] "
			f"ω_neg [{omega_neg.min():.3f}, {omega_neg.max():.3f}]"
		)

	return omega_pos, omega_neg, valid_mask

def compute_regime_weighted_i2t_loss(
		i2t_sim:      torch.Tensor,   # [B, C]  — image-to-class cosine similarities / T
		label_vectors: torch.Tensor,  # [B, C]  — multi-hot ground truth (float)
		criterion_i2t: torch.nn.BCEWithLogitsLoss,  # with pos_weight [C]
		active_mask:  torch.Tensor,   # [C] bool
		omega_pos:    torch.Tensor,   # [B] float32
		valid_mask:   torch.Tensor,   # [B] bool
) -> torch.Tensor:
		"""
		Axis 1 × Axis 2 combined I2T loss.

		Per-sample loss = ω_pos_i × mean_over_active_classes(BCE_i2t_i)

		Inactive classes are masked out (identical to loss.py).
		MISSING/INVALID samples (valid_mask=False) contribute zero.
		"""
		# Raw per-element BCE loss: [B, C]
		raw = criterion_i2t(i2t_sim, label_vectors)

		# Mask inactive classes: [B, C_active]
		raw_active = raw[:, active_mask]                    # [B, C_active]

		# Per-sample mean over active classes: [B]
		per_sample = raw_active.mean(dim=1)                 # [B]

		# Apply ω_pos and valid_mask
		weighted = omega_pos * per_sample * valid_mask.float()  # [B]

		# Mean over valid samples only (avoid dividing by skipped samples)
		n_valid = valid_mask.sum().clamp(min=1)
		return weighted.sum() / n_valid

def compute_regime_weighted_t2i_loss(
		t2i_sim:      torch.Tensor,   # [C, B]  — class-to-image cosine similarities / T
		label_vectors: torch.Tensor,  # [B, C]  — multi-hot ground truth (float)
		criterion_t2i: torch.nn.BCEWithLogitsLoss,  # no pos_weight
		active_mask:  torch.Tensor,   # [C] bool
		omega_pos:    torch.Tensor,   # [B] float32
		valid_mask:   torch.Tensor,   # [B] bool
) -> torch.Tensor:
		"""
		T2I direction: rows are classes, cols are batch images.
		ω_pos is applied per-image (column) to mirror I2T weighting.
		"""
		# Raw per-element BCE loss: [C, B]
		raw = criterion_t2i(t2i_sim, label_vectors.T)

		# Mask inactive class rows: [C_active, B]
		raw_active = raw[active_mask, :]                    # [C_active, B]

		# Per-image mean over active classes: [B]
		per_sample = raw_active.mean(dim=0)                 # [B]

		# Apply ω_pos and valid_mask
		weighted = omega_pos * per_sample * valid_mask.float()  # [B]

		n_valid = valid_mask.sum().clamp(min=1)
		return weighted.sum() / n_valid

def compute_hard_negative_repulsion_loss(
		image_embeds:    torch.Tensor,   # [B, D]  — L2-normalised image embeddings
		all_class_embeds: torch.Tensor,  # [C, D]  — L2-normalised class text embeddings
		hn_vectors:      torch.Tensor,   # [B, C]  — multi-hot hard-negative mask (float)
		omega_neg:       torch.Tensor,   # [B]     — repulsion weight
		valid_mask:      torch.Tensor,   # [B]     — bool
		temperature:     float,
		verbose:         bool = False,
) -> torch.Tensor:
	"""
	Repulsion loss for HARD_CONFLICT samples.
	For each image i and each hard-negative class j (hn_vectors[i,j] == 1),
	we want the cosine similarity φ(I_i, T_j) to be LOW.
	
	Formulation (mirrors the "negative" arm of BCE):
		L_repel_i = -mean_j[ hn_vectors[i,j] * log(1 - σ(φ(I_i, T_j) / T)) ]
	
	Weighted by ω_neg_i so that:
		- AGREEMENT / SOFT_CONFLICT samples: ω_neg = 0 → zero gradient
		- HARD_CONFLICT samples: ω_neg = 1 - mean(G(T⁻)) ∈ (0, 1]
	
	Returns scalar loss (0.0 if no hard negatives in batch).
	"""

	# Check if any hard negatives exist in this batch
	hn_count = hn_vectors.sum()
	if hn_count == 0:
		if verbose:
			print("[compute_hard_negative_repulsion_loss] No hard negatives in batch — skipping.")
		return torch.tensor(0.0, device=image_embeds.device, requires_grad=False)

	# Cosine similarity: [B, C]  (already normalised)
	sim = torch.matmul(image_embeds, all_class_embeds.T) / temperature  # [B, C]

	# σ(sim): probability of alignment — we want this LOW for hard negatives
	prob = torch.sigmoid(sim)                                            # [B, C]

	# Clamp for numerical stability: log(1 - prob) → avoid log(0)
	prob_clamped = prob.clamp(max=1.0 - 1e-7)

	# Per-element repulsion: -log(1 - σ(sim))
	repel_elem = -torch.log(1.0 - prob_clamped)                         # [B, C]

	# Mask to hard-negative positions only
	repel_masked = repel_elem * hn_vectors                              # [B, C]

	# Per-sample mean over hard-negative classes (avoid div-by-zero)
	hn_count_per_sample = hn_vectors.sum(dim=1).clamp(min=1)            # [B]
	per_sample = repel_masked.sum(dim=1) / hn_count_per_sample          # [B]

	# Apply ω_neg and valid_mask
	weighted = omega_neg * per_sample * valid_mask.float()              # [B]
	n_valid = valid_mask.sum().clamp(min=1)
	loss = weighted.sum() / n_valid

	if verbose:
		n_hard_samples = (hn_vectors.sum(dim=1) > 0).sum().item()
		print(
			f"[repulsion] hard-neg samples: {n_hard_samples}/{len(image_embeds)} | "
			f"hn_labels: {int(hn_count.item())} | "
			f"loss: {loss.item():.6f}"
		)

	return loss

def compute_loss(
	model:             torch.nn.Module,
	images:            torch.Tensor,        # [B, 3, H, W]
	all_class_embeds:  torch.Tensor,        # [C, D]  frozen text embeddings
	label_vectors:     torch.Tensor,        # [B, C]  multi-hot positives
	hn_vectors:        torch.Tensor,        # [B, C]  multi-hot hard negatives
	regimes:           List[str],           # [B]     regime strings from Stage 4
	w_pos_raw:         torch.Tensor,        # [B]     ω_pos from Stage 4
	w_neg_raw:         torch.Tensor,        # [B]     ω_neg from Stage 4
	criterion_i2t:     torch.nn.BCEWithLogitsLoss,
	criterion_t2i:     torch.nn.BCEWithLogitsLoss,
	active_mask:       torch.Tensor,        # [C] bool
	temperature:       float,
	split:             str,
	loss_weights:      Optional[Dict[str, float]] = None,
	verbose:           bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Loss with three arms:
		L_total = λ_i2t   * L_i2t(ω_pos)
						+ λ_t2i   * L_t2i(ω_pos)
						+ λ_repel * L_repulsion(ω_neg)
	Returns
	-------
	total_loss, loss_i2t, loss_t2i, loss_repel
	"""
	if loss_weights is None:
		loss_weights = {
			"i2t":   REGIME_LAMBDA_I2T,
			"t2i":   REGIME_LAMBDA_T2I,
			"repel": REGIME_LAMBDA_REPEL,
		}
	if verbose:
		print(f"[REGIME AWARE LOSS] {split.upper()}")

	# Embeddings
	image_embeds = torch.nn.functional.normalize(model.encode_image(images), dim=-1).float() # [B, D] FP32
	class_embeds = torch.nn.functional.normalize(all_class_embeds, dim=-1) # [C, D] already FP32

	# Similarities
	i2t_sim = torch.matmul(image_embeds, class_embeds.T) / temperature  # [B, C]
	t2i_sim = torch.matmul(class_embeds, image_embeds.T) / temperature  # [C, B]

	# Regime weights
	omega_pos, omega_neg, valid_mask = resolve_regime_weights(
		regimes=regimes,
		w_pos_raw=w_pos_raw,
		w_neg_raw=w_neg_raw,
		device=images.device,
		verbose=verbose,
	)

	# Arm 1: I2T
	loss_i2t = compute_regime_weighted_i2t_loss(
		i2t_sim=i2t_sim,
		label_vectors=label_vectors.float(),
		criterion_i2t=criterion_i2t,
		active_mask=active_mask,
		omega_pos=omega_pos,
		valid_mask=valid_mask,
	)

	# Arm 2: T2I
	loss_t2i = compute_regime_weighted_t2i_loss(
		t2i_sim=t2i_sim,
		label_vectors=label_vectors.float(),
		criterion_t2i=criterion_t2i,
		active_mask=active_mask,
		omega_pos=omega_pos,
		valid_mask=valid_mask,
	)

	# Arm 3: Repulsion
	loss_repel = compute_hard_negative_repulsion_loss(
		image_embeds=image_embeds,
		all_class_embeds=class_embeds,
		hn_vectors=hn_vectors.float(),
		omega_neg=omega_neg,
		valid_mask=valid_mask,
		temperature=temperature,
		verbose=verbose,
	)

	# Combine
	weighted_loss_i2t = loss_weights["i2t"] * loss_i2t
	weighted_loss_t2i = loss_weights["t2i"] * loss_t2i
	weighted_loss_repel = loss_weights["repel"] * loss_repel
	total_loss = weighted_loss_i2t + weighted_loss_t2i + weighted_loss_repel

	if verbose:
		print(
			f"[LOSS] "
			f"total={total_loss.item():.6f} | "
			f"i2t={loss_i2t.item():.6f} | "
			f"t2i={loss_t2i.item():.6f} | "
			f"repel={loss_repel.item():.6f} | "
			f"valid={valid_mask.sum().item()}/{len(regimes)}"
		)

	return total_loss, loss_i2t, loss_t2i, loss_repel