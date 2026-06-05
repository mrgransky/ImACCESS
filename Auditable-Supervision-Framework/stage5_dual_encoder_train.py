import os
import sys
import json
import argparse
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Workspace setup
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

print(f"sys.path: {sys.path}")

from utils import *

# =========================================================================
# PYTORCH DATASET IMPLEMENTATION
# =========================================================================

class HistoryX4AuditableDataset(Dataset):
	def __init__(
		self,
		parquet_path: str,
		target_vocab_path: str,
		images_dir: str,
		processor: CLIPProcessor,
		verbose: bool = True
	):
		self.df = pd.read_parquet(parquet_path)
		self.images_dir = images_dir
		self.processor = processor
		self.verbose = verbose

		# FIX-1: Support both the metadata-wrapped format and legacy bare lists
		with open(target_vocab_path, 'r') as f:
			vocab_payload = json.load(f)
		if isinstance(vocab_payload, dict):
			self.target_vocabulary = vocab_payload["vocabulary"]
		else:
			self.target_vocabulary = vocab_payload

		self.num_classes = len(self.target_vocabulary)
		self.label_to_idx = {label: idx for idx, label in enumerate(self.target_vocabulary)}

		# Cache labels as raw strings for compute_loss_masks helper compatibility
		self.labels = self.df['positive_targets'].apply(str).tolist()
		self.label_dict = self.label_to_idx

		if self.verbose:
			print(f"[DATASET] Loaded {len(self.df):,} rows from Parquet")
			print(f"[DATASET] Target Vocabulary has {self.num_classes} classes")

	def __len__(self) -> int:
		return len(self.df)

	def _to_multi_hot(self, labels: List[str]) -> torch.Tensor:
		"""Converts string labels list to a FloatTensor multi-hot vector."""
		vec = torch.zeros(self.num_classes, dtype=torch.float32)
		for lbl in labels:
			if lbl in self.label_to_idx:
				vec[self.label_to_idx[lbl]] = 1.0
		return vec

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		sample_id = row['id']
		
		# Resolve image path (handles full paths and relative filenames)
		img_name = os.path.basename(row['id']) if '/' in str(row['id']) else f"{row['id']}.jpg"
		img_path = os.path.join(self.images_dir, img_name)

		# Load and preprocess image using CLIP's official processor
		try:
			image = Image.open(img_path).convert("RGB")
			pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
		except Exception as e:
			if self.verbose:
				print(f"[WARN] Failed to load image {img_path}: {e} — returning blank tensor")
			pixel_values = torch.zeros((3, 336, 336), dtype=torch.float32) # Fallback blank image

		# Convert target lists to multi-hot representations
		pos_targets = self._to_multi_hot(row['positive_targets'])
		hn_targets = self._to_multi_hot(row['hard_negatives'])

		w_pos = torch.tensor(row['w_pos'], dtype=torch.float32)
		w_neg = torch.tensor(row['w_neg'], dtype=torch.float32)

		return pixel_values, pos_targets, hn_targets, w_pos, w_neg


# =========================================================================
# LOSS MASKS COMPUTATION (Your exact original long-tail balancing logic)
# =========================================================================

def compute_loss_masks(
	loader: DataLoader,
	num_classes: int,
	device: torch.device,
	pw_mode: str = "log",
	pw_max_cap: Optional[float] = None,
	pareto_threshold: float = 0.8,
	rare_percentile: float = 0.2,
	verbose: bool = True,
) -> Dict[str, torch.Tensor]:
	N = len(loader.dataset)
	train_freq = torch.zeros(num_classes, dtype=torch.float32)
	
	for raw in loader.dataset.labels:
		try:
			# Safely evaluate string representation of lists
			labels_list = ast.literal_eval(raw) if isinstance(raw, str) else raw
			for lbl in labels_list:
				if lbl in loader.dataset.label_dict:
					idx = loader.dataset.label_dict[lbl]
					train_freq[idx] += 1
		except (ValueError, SyntaxError):
			pass

	active_mask = (train_freq > 0).to(device)
	ratio = (N - train_freq) / train_freq.clamp(min=1)

	if pw_mode == "log":
		scaled = torch.log1p(ratio)
	elif pw_mode == "sqrt":
		scaled = torch.sqrt(ratio)
	elif pw_mode == "linear":
		scaled = ratio
	else:
		raise ValueError(f"Unknown pw_mode '{pw_mode}'")

	if pw_max_cap:
		scaled = scaled.clamp(min=1.0, max=pw_max_cap)

	pos_weight = torch.where(
		train_freq > 0,
		scaled,
		torch.ones(num_classes),
	).to(device)

	sorted_freq, sorted_idx = torch.sort(train_freq, descending=True)
	cumsum = sorted_freq.cumsum(0)
	pareto_cutoff = int((cumsum <= cumsum[-1] * pareto_threshold).sum().item()) + 1
	head_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
	head_mask[sorted_idx[:pareto_cutoff]] = True

	active_freq = train_freq[active_mask.cpu()]
	if active_freq.numel() > 1:
		freq_threshold = torch.quantile(active_freq, rare_percentile)
		rare_mask = ((train_freq <= freq_threshold) & (train_freq > 0)).to(device)
	else:
		rare_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
	
	if verbose:
		print(f"\n[LONG-TAIL MASKS] Computed from {N:,} samples over {num_classes} classes:")
		print(f"  ├─ pos_weight range : [{pos_weight.min().item():.2f}, {pos_weight.max().item():.2f}]")
		print(f"  ├─ Active classes   : {active_mask.sum().item()} / {num_classes}")
		print(f"  ├─ Pareto Head classes: {head_mask.sum().item()}")
		print(f"  └─ Rare Tail classes  : {rare_mask.sum().item()}")

	return {
		"active_mask": active_mask,
		"head_mask":   head_mask,
		"rare_mask":   rare_mask,
		"train_freq":  train_freq,
		"pos_weight":  pos_weight,
		"N":           N,
	}


# =========================================================================
# CORE TRAINING ENGINE
# =========================================================================

def train_regime_aware_clip(
	parquet_file: str,
	target_vocab_file: str,
	images_dir: str,
	model_id: str = "openai/clip-vit-large-patch14-336",
	epochs: int = 5,
	batch_size: int = 32,
	lr: float = 2e-6,
	alpha_repulsion: float = 1.5,
	device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
	verbose: bool = True
):
	device = torch.device(device)
	
	# Load processor and CLIP model
	print(f"\n[INIT] Loading CLIP Model & Processor: {model_id}...")
	processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_directory[os.getenv('USER')])
	model = CLIPModel.from_pretrained(model_id, cache_dir=cache_directory[os.getenv('USER')]).to(device)

	# Set up Dataset & DataLoader
	dataset = HistoryX4AuditableDataset(
		parquet_path=parquet_file,
		target_vocab_path=target_vocab_file,
		images_dir=images_dir,
		processor=processor,
		verbose=verbose
	)
	
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

	# Compute class-level balancing pos_weight masks
	masks = compute_loss_masks(
		loader=train_loader,
		num_classes=dataset.num_classes,
		device=device,
		pw_mode="sqrt",
		pw_max_cap=50.0,
		verbose=verbose
	)
	pos_weight = masks["pos_weight"]

	# Load target vocabulary strings to pre-compute textual classification embeddings
	class_prompts = [f"a photograph of {label}" for label in dataset.target_vocabulary]
	
	print(f"[INIT] Pre-tokenizing {len(class_prompts)} class-level prompts...")
	text_inputs = processor(text=class_prompts, padding=True, return_tensors="pt").to(device)

	# Set up dual-encoder BCE criteria (Image-to-Text and Text-to-Image)
	# reduction='none' is mandatory so we can scale gradients on a row-by-row sample level!
	criterion_i2t = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
	criterion_t2i = nn.BCEWithLogitsLoss(reduction='none')

	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

	print(f"\n[TRAINING] Starting Regime-Aware Fine-Tuning for {epochs} epochs...")
	
	for epoch in range(epochs):
		model.train()
		epoch_loss = 0.0
		epoch_pos_loss = 0.0
		epoch_hn_loss = 0.0
		
		# Pre-compute text features once per epoch (or batch if we fine-tune text encoder)
		with torch.set_grad_enabled(model.text_model.training):
			text_outputs = model.get_text_features(**text_inputs)
			# L2 normalize textual embeddings
			text_embeds = text_outputs / text_outputs.norm(dim=-1, keepdim=True)

		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=120)
		for step, (images, pos_targets, hn_targets, w_pos, w_neg) in enumerate(pbar):
			images = images.to(device)
			pos_targets = pos_targets.to(device) # Shape: [B, C]
			hn_targets = hn_targets.to(device)   # Shape: [B, C]
			
			# Shape: [B, 1] — necessary for broadcast multiplying over loss matrices
			w_pos = w_pos.to(device).view(-1, 1)
			w_neg = w_neg.to(device).view(-1, 1)

			optimizer.zero_grad()

			# Extract image embeddings
			image_outputs = model.get_image_features(pixel_values=images)
			image_embeds = image_outputs / image_outputs.norm(dim=-1, keepdim=True) # L2 normalize

			# Logit generation: scaled cosine similarity via dot product
			logit_scale = model.logit_scale.exp()
			
			# Image-to-Text Logits [B, C]
			logits_i2t = (image_embeds @ text_embeds.T) * logit_scale
			# Text-to-Image Logits [C, B]
			logits_t2i = (text_embeds @ image_embeds.T) * logit_scale

			# ── POSITIVE LOSS: Class Balanced AND Regime Conditioned (ω_pos) ────
			loss_pos_i2t = criterion_i2t(logits_i2t, pos_targets) * w_pos
			loss_pos_t2i = criterion_t2i(logits_t2i, pos_targets.T) * w_pos.T # Symmetrical transpose
			
			total_pos_loss = loss_pos_i2t.mean() + loss_pos_t2i.mean()

			# ── HARD NEGATIVE LOSS: Explicit Repulsion on Text Orphans (ω_neg) ──
			# We force logits of mined hard negatives to be heavily pushed toward zero (BCE with 0 targets)
			hn_loss_base = nn.functional.binary_cross_entropy_with_logits(
				logits_i2t * hn_targets, 
				torch.zeros_like(logits_i2t), 
				reduction='none'
			)
			total_hn_loss = (hn_loss_base * w_neg).mean()

			# Joint objective formulation
			batch_loss = total_pos_loss + (alpha_repulsion * total_hn_loss)

			batch_loss.backward()
			optimizer.step()

			epoch_loss += batch_loss.item()
			epoch_pos_loss += total_pos_loss.item()
			epoch_hn_loss += total_hn_loss.item()

			pbar.set_postfix({
				"Loss": f"{batch_loss.item():.4f}", 
				"Pos": f"{total_pos_loss.item():.4f}", 
				"HN": f"{total_hn_loss.item():.4f}"
			})

		avg_loss = epoch_loss / len(train_loader)
		avg_pos = epoch_pos_loss / len(train_loader)
		avg_hn = epoch_hn_loss / len(train_loader)
		print(f"[*] Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} [Positive: {avg_pos:.4f}, Repulsion: {avg_hn:.4f}]")

	# Save fine-tuned weights
	save_path = os.path.join(outputs_dir, "fine_tuned_clip_regime_aware")
	model.save_pretrained(save_path)
	processor.save_pretrained(save_path)
	print(f"\n[SUCCESS] Saved fine-tuned, conflict-resilient model to: {save_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Stage 5: Downstream Regime-Aware CLIP Fine-Tuning")
	parser.add_argument("--parquet_file", "-p", type=str, required=True, help="Path to Stage 4 auditable_matrix.parquet")
	parser.add_argument("--target_vocab", "-t", type=str, required=True, help="Path to Stage 3 target_vocabulary.json")
	parser.add_argument("--images_dir", "-i", type=str, required=True, help="Path to raw images folder")
	parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of training epochs")
	parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
	parser.add_argument("--lr", "-lr", type=float, default=2e-6, help="Learning rate")
	parser.add_argument("--alpha", "-a", type=float, default=1.5, help="Alpha weight for hard negative repulsion loss")
	parser.add_argument("--verbose", "-v", action='store_true', help="Verbose logging")

	args = parser.parse_args()
	set_seeds(seed=42)

	train_regime_aware_clip(
		parquet_file=args.parquet_file,
		target_vocab_file=args.target_vocab,
		images_dir=args.images_dir,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		alpha_repulsion=args.alpha,
		verbose=args.verbose
	)