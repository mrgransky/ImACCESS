import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")
CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

from utils import *
from historyXN_dataset_loader import (
	get_multi_label_dataloaders, 
	get_preprocess
)

# from clip directory:
from peft import get_injected_peft_clip, get_adapter_peft_clip
from probe import get_probe_clip
from loss import compute_loss_masks
from evals import evaluate_best_model, get_validation_metrics, compute_tiered_retrieval_metrics
import visualize as viz

# "https://pbs.twimg.com/media/GowwFwkbQAAaMs-?format=jpg"
# "https://pbs.twimg.com/media/Gowu5zDaYAAZ2YK?format=jpg"
# "https://pbs.twimg.com/media/Go0qRhvWEAAIxpn?format=png"
# "https://pbs.twimg.com/media/Go2T7FJbIAApElq?format=jpg"
# https://pbs.twimg.com/media/GowwFwkbQAAaMs-?format=jpg

# # run in local and Puhti for all fine-tuned models with image and label:
# local:
# $ python multi_label_inference.py -pth_dir /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/multi_label/ -a 'ViT-B/32' -v

# Puhti:
# $ python multi_label_inference.py -pth_dir /scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4/multi_label/ -a 'ViT-L/14@336px' -v

def get_multi_label_head_torso_tail_samples(
	metadata_path: str,
	metadata_train_path: str,
	metadata_val_path: str,
	col:str,
	num_samples_per_segment: int = 5,
	seed: int = 42,  # Add seed parameter
) -> Tuple[List[Dict], List[str], Dict]:
	"""
	Sample from head, torso, tail distributions for multi-label datasets.
	Fully deterministic with seed parameter.
	"""
	try:
			# Create a local random generator for this function only
			local_rng = random.Random(seed)
			
			df_val = pd.read_csv(
					filepath_or_buffer=metadata_val_path, 
					on_bad_lines='skip', 
					dtype=dtypes, 
					low_memory=False
			)
			
			# Sort dataframe by index to ensure consistent iteration order
			df_val = df_val.sort_index()
			
			print(f"VAL: {type(df_val)} {df_val.shape} {list(df_val.columns)}")
			
			# For multi-label, we need to count label frequencies differently
			all_labels = []
			label_to_images = {}  # Track which images have which labels
			
			# Iterate in sorted order for consistency
			for idx, row in df_val.iterrows():
					try:
							labels = ast.literal_eval(row[col])
							# Sort labels for consistency
							labels = sorted(labels)
							all_labels.extend(labels)
							
							# Build reverse index
							for label in labels:
									if label not in label_to_images:
											label_to_images[label] = []
									label_to_images[label].append({
											'image_path': row['img_path'],
											'row_index': idx,
											'all_labels': labels
									})
					except Exception as e:
							continue
			
			# Count label frequencies - use stable sort
			label_counts = pd.Series(all_labels).value_counts()
			
			# Sort label_counts index for consistency
			label_counts = label_counts.sort_index()
			
			# Calculate distribution statistics
			total_labels = len(label_counts)
			total_frequency = label_counts.sum()
			
			# Define head, torso, tail based on frequency distribution
			head_threshold = int(total_labels * 0.2)  # Top 20%
			tail_threshold = int(total_labels * 0.8)   # Bottom 20%
			
			# Get head labels (most frequent)
			head_labels = set(label_counts.head(head_threshold).index)
			
			# Get tail labels (least frequent) - need to sort tail properly
			tail_labels = set(label_counts.tail(total_labels - tail_threshold).index)
			
			# Torso is the remainder
			torso_labels = set(label_counts.index) - head_labels - tail_labels
			
			# Convert to sorted lists for deterministic iteration
			head_labels = sorted(head_labels)
			torso_labels = sorted(torso_labels)
			tail_labels = sorted(tail_labels)
			
			# Calculate frequency statistics for each segment
			head_freq = label_counts[head_labels].sum() if head_labels else 0
			torso_freq = label_counts[torso_labels].sum() if torso_labels else 0
			tail_freq = label_counts[tail_labels].sum() if tail_labels else 0
			
			# Compile detailed distribution info
			distribution_info = {
				'label_counts': label_counts.to_dict(),
				'segment_info': {
					'head': {
						'count': len(head_labels),
						'percentage': len(head_labels) / total_labels * 100 if total_labels > 0 else 0,
						'frequency': head_freq,
						'frequency_percentage': head_freq / total_frequency * 100 if total_frequency > 0 else 0,
						'labels': head_labels,
						'frequency_range': {
							'min': label_counts[head_labels].min() if head_labels else 0,
							'max': label_counts[head_labels].max() if head_labels else 0,
							'mean': label_counts[head_labels].mean() if head_labels else 0
						}
					},
					'torso': {
						'count': len(torso_labels),
						'percentage': len(torso_labels) / total_labels * 100 if total_labels > 0 else 0,
						'frequency': torso_freq,
						'frequency_percentage': torso_freq / total_frequency * 100 if total_frequency > 0 else 0,
						'labels': torso_labels,
						'frequency_range': {
							'min': label_counts[torso_labels].min() if torso_labels else 0,
							'max': label_counts[torso_labels].max() if torso_labels else 0,
							'mean': label_counts[torso_labels].mean() if torso_labels else 0
						}
					},
					'tail': {
						'count': len(tail_labels),
						'percentage': len(tail_labels) / total_labels * 100 if total_labels > 0 else 0,
						'frequency': tail_freq,
						'frequency_percentage': tail_freq / total_frequency * 100 if total_frequency > 0 else 0,
						'labels': tail_labels,
						'frequency_range': {
							'min': label_counts[tail_labels].min() if tail_labels else 0,
							'max': label_counts[tail_labels].max() if tail_labels else 0,
							'mean': label_counts[tail_labels].mean() if tail_labels else 0
						}
					}
				},
				'total_labels': total_labels,
				'total_frequency': total_frequency,
				'head_threshold': head_threshold,
				'tail_threshold': tail_threshold,
				'seed_used': seed  # Track which seed was used
			}
			
			print("\nLABEL DISTRIBUTION ANALYSIS")
			print(f"Total unique labels: {total_labels}")
			print(f"Total label occurrences: {total_frequency}")
			print(f"Using seed: {seed}")
			print(f"\nHead threshold (top 20%): {head_threshold} labels")
			print(f"Tail threshold (bottom 20%): {total_labels - tail_threshold} labels")
			
			print("\nHEAD (most frequent)")
			head_info = distribution_info['segment_info']['head']
			print(f"Count: {head_info['count']} labels ({head_info['percentage']:.1f}%)")
			print(f"Frequency: {head_info['frequency']} occurrences ({head_info['frequency_percentage']:.1f}%)")
			print(f"Frequency range: {head_info['frequency_range']['min']} - {head_info['frequency_range']['max']} (mean: {head_info['frequency_range']['mean']:.1f})")
			print(f"Sample head labels: {head_labels[:10]}")
			
			print("\nTORSO (medium frequency)")
			torso_info = distribution_info['segment_info']['torso']
			print(f"Count: {torso_info['count']} labels ({torso_info['percentage']:.1f}%)")
			print(f"Frequency: {torso_info['frequency']} occurrences ({torso_info['frequency_percentage']:.1f}%)")
			print(f"Frequency range: {torso_info['frequency_range']['min']} - {torso_info['frequency_range']['max']} (mean: {torso_info['frequency_range']['mean']:.1f})")
			print(f"Sample torso labels: {torso_labels[:10]}")
			
			print("\nTAIL (least frequent)")
			tail_info = distribution_info['segment_info']['tail']
			print(f"Count: {tail_info['count']} labels ({tail_info['percentage']:.1f}%)")
			print(f"Frequency: {tail_info['frequency']} occurrences ({tail_info['frequency_percentage']:.1f}%)")
			print(f"Frequency range: {tail_info['frequency_range']['min']} - {tail_info['frequency_range']['max']} (mean: {tail_info['frequency_range']['mean']:.1f})")
			print(f"Sample tail labels: {tail_labels[:10]}")
			
			# Sample images with head, torso, tail labels
			i2t_samples = []
			t2i_samples = []
			
			# Image sampling (image-to-text)
			print("\nSAMPLING IMAGES (I2T)")
			
			# Process segments in fixed order
			for segment_name, segment_labels in [("head", head_labels), ("torso", torso_labels), ("tail", tail_labels)]:
				segment_samples = []
				
				# Find all images with labels from this segment
				# Iterate through labels in sorted order
				for label in sorted(segment_labels):
					if label in label_to_images:
						# Sort images by path for consistency
						images_for_label = sorted(label_to_images[label], key=lambda x: x['image_path'])
						segment_samples.extend(images_for_label)
				
				# Remove duplicates while preserving order (use OrderedDict for deterministic behavior)
				unique_samples = {}
				for sample in segment_samples:
					if sample['image_path'] not in unique_samples:
						unique_samples[sample['image_path']] = sample
				
				segment_samples = list(unique_samples.values())
				
				print(f"\n{segment_name.upper()} segment:")
				print(f"  Total unique images available: {len(segment_samples)}")
				
				# Randomly sample from this segment using local RNG
				if len(segment_samples) >= num_samples_per_segment:
					# Sort segment_samples by image path for deterministic selection before sampling
					segment_samples = sorted(segment_samples, key=lambda x: x['image_path'])
					sampled = local_rng.sample(segment_samples, num_samples_per_segment)
					
					for sample in sampled:
						sample['segment'] = segment_name
						i2t_samples.append(sample)
					
					print(f"  Sampled {num_samples_per_segment} images:")
					for i, sample in enumerate(sampled, 1):
						# Get which specific labels from this segment are in this image
						image_labels = set(sample['all_labels'])
						segment_labels_in_image = image_labels & set(segment_labels)
						
						print(f"    {i}. {os.path.basename(sample['image_path'])}")
						print(f"       All labels: {sample['all_labels']}")
						print(f"       {segment_name} labels: {sorted(segment_labels_in_image)}")
				else:
					print(f"  WARNING: Only {len(segment_samples)} images available, need {num_samples_per_segment}")
					if segment_samples:
						# Sort for consistency
						segment_samples = sorted(segment_samples, key=lambda x: x['image_path'])
						sampled = segment_samples  # Take all available
						for sample in sampled:
							sample['segment'] = segment_name
							i2t_samples.append(sample)
						print(f"  Using all {len(segment_samples)} available images instead")
			
			# Sort final i2t_samples for consistency
			i2t_samples = sorted(i2t_samples, key=lambda x: x['image_path'])
			
			# Sample text queries from head, torso, tail
			print("\nSAMPLING TEXT QUERIES (T2I)")
			
			for segment_name, segment_labels in [("head", head_labels), ("torso", torso_labels), ("tail", tail_labels)]:
				segment_label_list = list(segment_labels)
				
				print(f"\n{segment_name.upper()} segment:")
				print(f"  Total labels available: {len(segment_label_list)}")
				
				if len(segment_label_list) >= num_samples_per_segment:
					# Sort for deterministic selection
					segment_label_list = sorted(segment_label_list)
					sampled_labels = local_rng.sample(segment_label_list, num_samples_per_segment)
					t2i_samples.extend(sampled_labels)
					
					print(f"  Sampled {num_samples_per_segment} labels:")
					for i, label in enumerate(sampled_labels, 1):
						freq = label_counts[label]
						print(f"    {i}. '{label}' (frequency: {freq})")
						
						# Show sample images with this label
						if label in label_to_images:
							sample_images = sorted(label_to_images[label], key=lambda x: x['image_path'])[:2]
							print(f"       Example images: {[os.path.basename(img['image_path']) for img in sample_images]}")
				else:
					print(f"[WARNING] Only {len(segment_label_list)} labels available, need {num_samples_per_segment}")
					if segment_label_list:
						t2i_samples.extend(sorted(segment_label_list))
						print(f"  Using all {len(segment_label_list)} available labels instead")
			
			# Sort final t2i_samples for consistency
			t2i_samples = sorted(t2i_samples)
			
			print(f">> TOTAL SAMPLES: {len(i2t_samples)} images, {len(t2i_samples)} text queries")
			
			return i2t_samples, t2i_samples
			
	except Exception as e:
		print(f"Error in multi-label sampling: {e}")
		import traceback
		traceback.print_exc()
		return [], []

def _parse_checkpoint_strategy(ft_path: str) -> Tuple[str, Dict]:
	fname = os.path.basename(ft_path)
	
	# Order matters — more specific patterns before general ones
	if fname.startswith("zero_shot"):
		strategy = "zero_shot"
	elif fname.startswith("lora_plus"):
		strategy = "lora_plus"
	elif fname.startswith("rslora"):
		strategy = "rslora"
	elif fname.startswith("lora"):
		strategy = "lora"
	elif fname.startswith("dora"):
		strategy = "dora"
	elif fname.startswith("vera"):
		strategy = "vera"
	elif fname.startswith("ia3"):
		strategy = "ia3"
	elif fname.startswith("clip_adapter_vt"):
		strategy = "clip_adapter_vt"
	elif fname.startswith("clip_adapter_t"):
		strategy = "clip_adapter_t"
	elif fname.startswith("clip_adapter_v"):
		strategy = "clip_adapter_v"
	elif fname.startswith("tip_adapter_f"):
		strategy = "tip_adapter_f"
	elif fname.startswith("tip_adapter"):
		strategy = "tip_adapter"
	elif fname.startswith("probe"):
		strategy = "probe"
	elif fname.startswith("full"):
		strategy = "full"
	else:
		strategy = "unknown"

	# Extract hyperparams from filename if present
	params = {}
	batch_size = re.search(r'bs_(\d+)', fname) # Batch size
	lor_match = re.search(r'lor_(\d+)', fname) # LoRA rank
	loa_match = re.search(r'loa_([\d.]+)', fname) # LoRA alpha
	lod_match = re.search(r'lod_([\d.]+)', fname) # LoRA dropout
	lmbd_match = re.search(r'lmbd_([\d.]+)', fname) # LoRA+ lambda
	cbd_match = re.search(r'cbd_(\d+)', fname) # Bottleneck dim
	act_match = re.search(r'act_(\w+?)_', fname)
	init_alpha_match = re.search(r'init_alpha_([\d.]+)', fname)
	init_beta_match = re.search(r'init_beta_([\d.]+)', fname)

	if batch_size:
		params["batch_size"] = int(batch_size.group(1))
	if lor_match:
		params["lora_rank"]    = int(lor_match.group(1))
	if loa_match:
		params["lora_alpha"]   = float(loa_match.group(1))
	if lod_match:
		params["lora_dropout"] = float(lod_match.group(1))
	if lmbd_match:
		params["lora_plus_lambda"] = float(lmbd_match.group(1))
	if cbd_match:
		params["bottleneck_dim"] = int(cbd_match.group(1))
	if act_match:
		params["activation"] = act_match.group(1)
	if init_alpha_match:
		params["init_alpha"] = float(init_alpha_match.group(1))
	if init_beta_match:
		params["init_beta"] = float(init_beta_match.group(1))

	return strategy, params

def _load_checkpoint_into_model(
	model: torch.nn.Module,
	ft_path: str,
	device: torch.device,
	verbose: bool = False,
) -> torch.nn.Module:
	checkpoint = torch.load(ft_path, map_location=device)
	state_dict = checkpoint.get('model_state_dict', checkpoint)

	# Key translation for probe checkpoints (saved as bare linear layer)
	print(f"Loading {ft_path} into {type(model)}")
	# print(f"Checkpoint keys (sample): {list(state_dict.keys())}")
	try:
		missing, unexpected = model.load_state_dict(state_dict, strict=False)
		if verbose and (missing or unexpected):
			print(f"{len(missing)} Missing: {missing}")
			print(f"{len(unexpected)} Unexpected: {unexpected}")
	except Exception as e:
		print(f"[WARNING] load_state_dict failed:\n{e}")

	return model

def get_tail_only_samples(
	metadata_val_path: str,
	col:str,
	num_samples: int = 5,
	seed: int = 42,
) -> Tuple[List[Dict], List[str]]:
	"""
	Sample exclusively from the tail distribution (bottom 20% by frequency).
	Returns i2t_samples (images whose ALL labels are tail) and t2i_samples (tail label strings).
	"""
	print(f"Sampling from Tail Distribution: {metadata_val_path}")
	print(f"Column: {col}")
	local_rng = random.Random(seed)
	df_val = pd.read_csv(
		filepath_or_buffer=metadata_val_path,
		on_bad_lines='skip',
		dtype=dtypes,
		low_memory=False,
	).sort_index()
	all_labels = []
	label_to_images = {}
	for idx, row in df_val.iterrows():
			try:
					labels = sorted(ast.literal_eval(row[col]))
					all_labels.extend(labels)
					for label in labels:
							label_to_images.setdefault(label, []).append({
									'image_path': row['img_path'],
									'row_index': idx,
									'all_labels': labels,
									'segment': 'tail',
							})
			except Exception:
					continue
	label_counts = pd.Series(all_labels).value_counts().sort_index()
	total_labels = len(label_counts)
	tail_threshold = int(total_labels * 0.8)
	tail_labels = sorted(label_counts.tail(total_labels - tail_threshold).index)
	print(
		f"\nTAIL distribution: {len(tail_labels)} labels "
		f"(freq range: {label_counts[tail_labels].min()}–{label_counts[tail_labels].max()}, "
		f"mean: {label_counts[tail_labels].mean():.1f})"
	)
	# --- I2T samples: images that contain at least one tail label ---
	candidate_pool = {}
	for label in tail_labels:
			for entry in sorted(label_to_images.get(label, []), key=lambda x: x['image_path']):
					candidate_pool.setdefault(entry['image_path'], entry)
	candidates = sorted(candidate_pool.values(), key=lambda x: x['image_path'])
	if len(candidates) >= num_samples:
			i2t_samples = local_rng.sample(candidates, num_samples)
	else:
			print(f"[WARNING] Only {len(candidates)} tail images available; using all.")
			i2t_samples = candidates
	i2t_samples = sorted(i2t_samples, key=lambda x: x['image_path'])
	# --- T2I samples: tail label strings ---
	if len(tail_labels) >= num_samples:
		t2i_samples = sorted(local_rng.sample(tail_labels, num_samples))
	else:
		print(f"[WARNING] Only {len(tail_labels)} tail labels available; using all.")
		t2i_samples = tail_labels
	print(f"{len(i2t_samples)} selected Query Images from Tail Distribution:")
	print(json.dumps(i2t_samples, indent=4, ensure_ascii=False))
	print(f"{len(t2i_samples)} Query Labels from Tail Distribution: {t2i_samples}")
	
	print("-"*130)

	return i2t_samples, t2i_samples

def get_top_k_strategies(
	results_json_path: str,
	top_k: int = 5,
	distribution: str = "rare", # overall, head, rare
	metric_key: str = "mAP", # mAP, Recall
	k_value: int = 10,
	mode: str = "i2t", # "i2t" or "t2i"
) -> Tuple[List[str], List[str]]:
	if not os.path.exists(results_json_path):
		print(f"WARNING: {results_json_path} not found.")
		return [], []
	with open(results_json_path) as f:
		all_results = json.load(f)

	direction = mode.lower()   # "i2t" or "t2i"
	scores = {}
	print(f"\n>> Ranking {len(all_results)} strategies — {direction.upper()} {distribution} {metric_key}@{k_value}")
	for strategy, metrics in all_results.items():
		try:
			scores[strategy] = metrics[direction][distribution][metric_key][str(k_value)]
		except KeyError:
			scores[strategy] = -1.0
	
	sorted_strategies = sorted(scores, key=scores.get, reverse=True)[:top_k]
	print(f"\nTop-{len(sorted_strategies)} ({direction.upper()} {distribution}):")
	for i, s in enumerate(sorted_strategies):
		print(f"  {i+1}. {s:<20} ({metric_key}@{k_value}: {scores[s]:.4f})")
	print("-" * 60)

	pth_dir = os.path.dirname(results_json_path)

	checkpoints = []
	
	for strategy in sorted_strategies:
		all_pths = [f for f in os.listdir(pth_dir) if f.endswith('.pth')]
		valid_match = None
		
		for f in all_pths:
			# Use existing parser to see what the file ACTUALLY is
			detected_strategy, _ = _parse_checkpoint_strategy(f)
			if detected_strategy == strategy:
				valid_match = f
				break
						
		if valid_match:
			checkpoints.append(os.path.join(pth_dir, valid_match))
		else:
			print(f"WARNING: no checkpoint for '{strategy}'")
	return sorted_strategies, checkpoints

def run_qualitative_retrieval(
	model: torch.nn.Module,
	i2t_samples: List[Dict],
	t2i_samples: List[str],
	class_names: List[str],
	all_val_image_paths: List[str],
	preprocess,
	device: torch.device,
	i2t_topk: int = 3,
	t2i_topk: int = 1,
	canonical_class_names: List[str] = None,
	probe_train_freq: Optional[dict] = None,
	text_batch_size: int = 256,   # chunk size for text encoding
	image_batch_size: int = 32,   # chunk size for image encoding
	verbose: bool = False,
) -> Dict:

	model.eval()

	is_probe = hasattr(model, 'probe') and hasattr(model, 'clip_model')

	if is_probe:
		print(
			f"[qualitative] Probe detected — using trained W ({model.probe.weight.shape}) "
			f"as class embeddings instead of encode_text."
		)
		all_text_embeds = torch.nn.functional.normalize(model.probe.weight.detach().float(), dim=-1).cpu()
		label_vocab = canonical_class_names if canonical_class_names is not None else class_names

		if probe_train_freq is not None and canonical_class_names is not None:
			active_indices = torch.tensor([
				i for i, name in enumerate(canonical_class_names)
				if probe_train_freq.get(name, 0) > 0
			], dtype=torch.long)
			all_text_embeds_active = all_text_embeds[active_indices]
			label_vocab_active     = [canonical_class_names[i] for i in active_indices.tolist()]
			print(
				f"[qualitative] Probe active classes: {len(active_indices)}/{len(canonical_class_names)} "
				f"(excluded {len(canonical_class_names)-len(active_indices)} zero-freq classes)"
			)
		else:
			all_text_embeds_active = all_text_embeds
			label_vocab_active     = label_vocab
	else:
		all_text_embeds = []
		for start in range(0, len(class_names), text_batch_size):
			chunk = class_names[start: start + text_batch_size]
			with torch.no_grad():
				tokens = clip.tokenize(chunk, truncate=True).to(device)
				chunk_embeds = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
			all_text_embeds.append(chunk_embeds.cpu())
			del tokens, chunk_embeds
			torch.cuda.empty_cache()

		all_text_embeds = torch.cat(all_text_embeds, dim=0)
		label_vocab        = class_names        # raw vocabulary for non-probe
		all_text_embeds_active = all_text_embeds  # ← no masking for non-probe
		label_vocab_active     = label_vocab      # ← same vocab, no masking

	# I2T — use label_vocab for index→name mapping
	i2t_results = []
	for sample in i2t_samples:
			image = preprocess(
					Image.open(sample['image_path']).convert('RGB')
			).unsqueeze(0).to(device)
			with torch.no_grad():
					img_embed = torch.nn.functional.normalize(
							model.encode_image(image).float(), dim=-1
					).cpu()
			sims     = (img_embed @ all_text_embeds_active.T).squeeze(0)   # [C_active]
			topk_idx = sims.topk(i2t_topk).indices.tolist()
			i2t_results.append({
					'image_path':       sample['image_path'],
					'ground_truth':     sample['all_labels'],
					'segment':          sample['segment'],
					'retrieved_labels': [label_vocab_active[j] for j in topk_idx],
					'retrieved_scores': [sims[j].item() for j in topk_idx],
			})
			del image, img_embed
			torch.cuda.empty_cache()


	# T2I: tail label → top-t2i_topk images from FULL val pool
	# Encode val images in chunks, accumulate embeddings on CPU,          #
	# then do the similarity matmul entirely on CPU to avoid OOM.         #
	# For 41K images × 768-dim: ~120 MB on CPU — totally fine.           #
	all_img_embeds = []   # accumulate on CPU
	for start in range(0, len(all_val_image_paths), image_batch_size):
		batch_paths = all_val_image_paths[start : start + image_batch_size]
		imgs = torch.stack(
			[
				preprocess(Image.open(p).convert('RGB'))
				for p in batch_paths
			]
		).to(device)
		with torch.no_grad():
			embs = torch.nn.functional.normalize(model.encode_image(imgs).float(), dim=-1)
		all_img_embeds.append(embs.cpu())
		del imgs, embs
		torch.cuda.empty_cache()

	all_img_embeds = torch.cat(all_img_embeds, dim=0)  # [N_val, D] on CPU
	
	# Build once before the T2I loop, after all_text_embeds_active is defined
	if is_probe:
		label_to_embed_idx = {name: i for i, name in enumerate(label_vocab_active)}

	# T2I loop
	t2i_results = []
	for query_label in t2i_samples:
		if is_probe:
			if query_label in label_to_embed_idx:
				idx = label_to_embed_idx[query_label]
				txt_embed = all_text_embeds_active[idx].unsqueeze(0)  # [1, D] already normalised
			else:
				# query_label is a zero-freq class — W row exists but was never trained.
				# Options: (a) skip, (b) fall back to frozen text encoder with a warning.
				# Option (b) is more informative for the qualitative figure:
				print(f"  [probe T2I] '{query_label}' not in active vocab — falling back to ZS text encoder")
				with torch.no_grad():
					token = clip.tokenize([query_label], truncate=True).to(device)
					txt_embed = torch.nn.functional.normalize(model.clip_model.encode_text(token).float(), dim=-1).cpu()
		else:
			with torch.no_grad():
				token = clip.tokenize([query_label], truncate=True).to(device)
				txt_embed = torch.nn.functional.normalize(model.encode_text(token).float(), dim=-1).cpu()
		sims     = (txt_embed @ all_img_embeds.T).squeeze(0)
		topk_idx = sims.topk(t2i_topk).indices.tolist()
		t2i_results.append(
			{
				'query_label':      query_label,
				'retrieved_paths':  [all_val_image_paths[j] for j in topk_idx],
				'retrieved_scores': [sims[j].item() for j in topk_idx],
			}
		)


	if verbose:
		print(f"i2t_results: {len(i2t_results)}")
		print(f"t2i_results: {len(t2i_results)}")

	return {'i2t': i2t_results, 't2i': t2i_results}

def _load_models(
	checkpoint_path: str,
	model_architecture: str,
	device: torch.device,
	dataset_directory: str,
	validation_loader: DataLoader,
	class_names: List[str],
	verbose: bool = False,
):
	fine_tuned_models = {}
	ft_start = time.time()
	probe_class_names = None   # populated only for probe strategy
	probe_train_freq = None

	strategy, params = _parse_checkpoint_strategy(checkpoint_path)
	print(f"\nStrategy: {strategy} Params: {params}")
	print(f"Loading {checkpoint_path}")

	# Fresh base model for each checkpoint
	base_model, _ = clip.load(
		name=model_architecture,
		device=device,
		download_root=get_model_directory(path=dataset_directory),
	)
	base_model = base_model.float()
	base_model.name = model_architecture

	try:
		if strategy == "zero_shot":
			ft_model = base_model
		elif strategy in ("lora", "lora_plus", "dora", "rslora"):
			rank    = params.get("lora_rank")
			alpha   = params.get("lora_alpha")
			dropout = params.get("lora_dropout")

			ft_model = get_injected_peft_clip(
				clip_model=base_model,
				method=strategy,
				rank=rank,
				alpha=alpha,
				dropout=dropout,
				lora_plus_lambda=params.get("lora_plus_lambda"),
				target_text_modules=[],
				target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
				verbose=verbose,
			)
		elif strategy in ("ia3", "vera"):
			rank    = params.get("lora_rank")
			alpha   = params.get("lora_alpha")
			dropout = params.get("lora_dropout")
			ft_model = get_injected_peft_clip(
					clip_model=base_model,
					method=strategy,
					rank=rank,
					alpha=alpha,
					dropout=dropout,
					target_text_modules=[],
					target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
					verbose=verbose,
			)
		elif strategy in ("clip_adapter_v", "clip_adapter_t", "clip_adapter_vt"):
				bottleneck_dim = params.get("bottleneck_dim")
				activation = params.get("activation")
				ft_model = get_adapter_peft_clip(
					clip_model=base_model,
					method=strategy,
					bottleneck_dim=bottleneck_dim,
					activation=activation,
					verbose=verbose,
				)
		elif strategy in ("tip_adapter", "tip_adapter_f"):
			# Tip-Adapter cache is not reconstructible from checkpoint alone —
			# the cache depends on support set features extracted at training time.
			# For inference, we load the trainable projection weights only.
			ft_model = get_adapter_peft_clip(
				clip_model=base_model,
				method=strategy,
				bottleneck_dim=None,
				activation=None,
				initial_beta=params.get("init_beta", 1.0),
				initial_alpha=params.get("init_alpha", 1.0),
				verbose=verbose,
			)
			# Load checkpoint first to get cache size
			checkpoint = torch.load(checkpoint_path, map_location=device)
			state_dict = checkpoint.get('model_state_dict', checkpoint)
			
			# Extract cache dimensions from checkpoint
			cache_key = f"visual.{strategy.replace('-', '_')}_proj.cache_keys"
			if cache_key in state_dict:
				if verbose:
					print(f"  Found cache in checkpoint: {cache_key}")
	
				cache_size = state_dict[cache_key].shape[0]
				cache_dim = state_dict[cache_key].shape[1]

				if verbose:
					print(f"  Cache size: {cache_size}, Cache dim: {cache_dim}")
				
				# Resize the cache buffers to match checkpoint
				adapter_module = getattr(ft_model.visual, f"{strategy.replace('-', '_')}_proj")
				adapter_module.cache_keys = torch.zeros(cache_size, cache_dim, device=device)
				adapter_module.cache_values = torch.zeros(cache_size, cache_dim, device=device)
				
				if verbose:
					print(f"  Resized cache to [{cache_size}, {cache_dim}]")
		elif strategy == "probe":
			# ── Peek at checkpoint to get the correct probe output size ──────────
			_ckpt = torch.load(checkpoint_path, map_location=device)
			_sd   = _ckpt.get('model_state_dict', _ckpt)
			# probe weight shape: [num_classes, feature_dim]
			probe_weight_key = next(
					(k for k in _sd if k.endswith('probe.weight') or k.endswith('probe.bias')), 
					None
			)
			if probe_weight_key and probe_weight_key.endswith('probe.weight'):
					ckpt_num_classes = _sd[probe_weight_key].shape[0]
					if verbose:
							print(f"  Checkpoint probe output size: {ckpt_num_classes}")
			else:
					# Fallback: derive from validation loader
					ckpt_num_classes = None
					if verbose:
							print("  Could not detect probe size from checkpoint — using validation loader")
			del _ckpt, _sd  # free memory before building model

			ft_model = get_probe_clip(
				clip_model=base_model,
				validation_loader=validation_loader,
				device=device,
				num_classes_override=ckpt_num_classes, # checkpoint size
				verbose=verbose,
			)
		elif strategy == "full":
			ft_model = base_model  # weights loaded directly below
		else:
			print(f"  [SKIP] Unknown strategy '{strategy}' for {checkpoint_path}")
			return {}, None, None

		ft_model = ft_model.to(device).float()
		ft_model.name = model_architecture
		ft_model = _load_checkpoint_into_model(
			model=ft_model, 
			ft_path=checkpoint_path, 
			device=device, 
			verbose=verbose
		)

		# Probe W-matrix geometry diagnostics
		if strategy == "probe" and hasattr(ft_model, "probe"):
			W_raw = ft_model.probe.weight.detach().float()
			W     = torch.nn.functional.normalize(W_raw, dim=-1)
			C, D  = W.shape
			# Reconstruct canonical class names from training metadata
			full_meta_path  = os.path.join(
					os.path.dirname(os.path.dirname(checkpoint_path)),
					"metadata_multi_label_multimodal.csv"   # full dataset
			)
			train_meta_path = os.path.join(
					os.path.dirname(os.path.dirname(checkpoint_path)),
					"metadata_multi_label_multimodal_train.csv"
			)

			canonical_vocab = set()
			for _, row in pd.read_csv(full_meta_path, on_bad_lines='skip', low_memory=False).iterrows():
					try:
							canonical_vocab.update(ast.literal_eval(row['multimodal_canonical_labels']))
					except Exception:
							continue
			probe_class_names = sorted(canonical_vocab)[:C]
			print(f"  [probe] Reconstructed {len(probe_class_names)} canonical class names from full dataset metadata")

			# Training frequencies from train split only
			all_train_labels = []
			for _, row in pd.read_csv(train_meta_path, on_bad_lines='skip', low_memory=False).iterrows():
					try:
							all_train_labels.extend(ast.literal_eval(row['multimodal_canonical_labels']))
					except Exception:
							continue
			probe_train_freq = pd.Series(all_train_labels).value_counts()
			print(f"  [probe] Train frequency computed from {len(probe_train_freq)} unique canonical classes")

			inter_sim = W @ W.T
			mask      = ~torch.eye(C, dtype=torch.bool, device=W.device)
			off_diag  = inter_sim[mask]
			print(f"\n[PROBE DIAGNOSTICS] W shape: [{C} classes × {D} dims]")
			print(f"  Inter-row cosine sim : mean={off_diag.mean():.4f}  "
						f"std={off_diag.std():.4f}  max={off_diag.max():.4f}")
			print(f"  (Healthy: mean≈0.0, std≈0.3 | Collapsed: mean>0.8, std<0.05)")
			raw_norms = W_raw.norm(dim=-1)
			print(f"  Raw W row norms      : mean={raw_norms.mean():.4f}  "
						f"std={raw_norms.std():.4f}  "
						f"min={raw_norms.min():.4f}  max={raw_norms.max():.4f}")
			print(f"  (Healthy: high std | Collapsed: all norms similar, low std)")
			try:
					_, S, _ = torch.svd(W)
					cumvar  = (S / S.sum()).cumsum(0)
					dims_90 = int((cumvar < 0.90).sum().item()) + 1
					dims_99 = int((cumvar < 0.99).sum().item()) + 1
					print(f"  Effective rank       : "
								f"{dims_90} dims → 90% variance, "
								f"{dims_99} dims → 99% variance")
					print(f"  (Healthy: 50+ dims for 90% | Collapsed: 1–5 dims for 90%)")
			except Exception as e:
					print(f"  [SVD failed: {e}]")
			# 4. Drift from zero-shot initialization
			try:
					zs_embeds = []
					with torch.no_grad():
							for start in range(0, len(probe_class_names), 64):
									chunk  = probe_class_names[start: start + 64]
									tokens = clip.tokenize(chunk, truncate=True).to(device)
									zs_chunk = torch.nn.functional.normalize(
											ft_model.clip_model.encode_text(tokens).float(), dim=-1
									).cpu()
									zs_embeds.append(zs_chunk)
					zs_embeds     = torch.cat(zs_embeds, dim=0)          # [C, D]
					W_norm        = torch.nn.functional.normalize(W_raw.cpu(), dim=-1)
					per_class_sim = (W_norm * zs_embeds).sum(dim=-1)     # [C]
					print(f"  ZS init alignment    : mean={per_class_sim.mean():.4f}  "
								f"std={per_class_sim.std():.4f}  "
								f"min={per_class_sim.min():.4f}  max={per_class_sim.max():.4f}")
					print(f"  (mean≈1.0 = no learning | mean<0.5 = significant drift)")
					del zs_embeds, W_norm, per_class_sim
			except Exception as e:
					print(f"  [ZS alignment failed: {e}]")
			# del W_raw, W, inter_sim, mask, off_diag, raw_norms
			print()

			# 5. Per-tier drift analysis — requires label frequency info
			# Add inside the probe diagnostics block, after ZS alignment
			try:
					# Recompute per_class_sim before this block (or keep it in scope)
					zs_embeds_check = []
					with torch.no_grad():
							for start in range(0, len(probe_class_names), 64):
									chunk  = probe_class_names[start: start + 64]
									tokens = clip.tokenize(chunk, truncate=True).to(device)
									zs_chunk = torch.nn.functional.normalize(
											ft_model.clip_model.encode_text(tokens).float(), dim=-1
									).cpu()
									zs_embeds_check.append(zs_chunk)
					zs_embeds_check = torch.cat(zs_embeds_check, dim=0)
					W_norm_check    = torch.nn.functional.normalize(W_raw.cpu(), dim=-1)
					per_class_sim   = (W_norm_check * zs_embeds_check).sum(dim=-1)  # [C]
					# Sort by alignment score to find which classes drifted most/least
					sorted_idx   = per_class_sim.argsort()                  # ascending = most drifted first
					most_drifted = [(probe_class_names[i], per_class_sim[i].item()) for i in sorted_idx[:5]]
					least_drifted= [(probe_class_names[i], per_class_sim[i].item()) for i in sorted_idx[-5:]]
					print(f"  Most drifted classes (learned most from data):")
					for name, sim in most_drifted:
							print(f"    {name:<30} ZS-align={sim:.4f}")
					print(f"  Least drifted classes (stayed close to ZS init):")
					for name, sim in least_drifted:
							print(f"    {name:<30} ZS-align={sim:.4f}")
					del zs_embeds_check, W_norm_check, per_class_sim, sorted_idx
			except Exception as e:
					print(f"  [Per-tier drift failed: {e}]")

			# 6. Cross-reference drift with training frequency
			try:
					# Load training label frequencies from the loss masks
					# (already computed during training via compute_loss_masks)
					# Alternative: recompute from train_loader directly
					train_metadata = pd.read_csv(
							os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "metadata_multi_label_multimodal_train.csv"),
							on_bad_lines='skip',
							low_memory=False,
					)
					all_train_labels = []
					for _, row in train_metadata.iterrows():
							try:
									labels = sorted(ast.literal_eval(row['multimodal_canonical_labels']))
									all_train_labels.extend(labels)
							except Exception:
									continue
					print(f"\n  Drift vs. frequency cross-reference:")
					print(f"  {'Class':<30} {'ZS-align':>10} {'Train freq':>12} {'Tier':>8}")
					print(f"  {'-'*62}")
					# Recompute per_class_sim for all probe classes
					zs_tmp = []
					with torch.no_grad():
							for start in range(0, len(probe_class_names), 64):
									chunk  = probe_class_names[start: start + 64]
									tokens = clip.tokenize(chunk, truncate=True).to(device)
									zs_chunk = torch.nn.functional.normalize(
											ft_model.clip_model.encode_text(tokens).float(), dim=-1
									).cpu()
									zs_tmp.append(zs_chunk)
					zs_tmp  = torch.cat(zs_tmp, dim=0)
					W_tmp   = torch.nn.functional.normalize(W_raw.cpu(), dim=-1)
					sim_all = (W_tmp * zs_tmp).sum(dim=-1)  # [C]
					# Show top-5 most drifted with their frequencies
					sorted_idx = sim_all.argsort()
					for idx in sorted_idx[:10]:
							cname = probe_class_names[idx]
							freq  = probe_train_freq.get(cname, 0)
							total = len(all_train_labels)
							n_classes = len(probe_train_freq)
							pct_rank  = (probe_train_freq.rank(ascending=False).get(cname, n_classes)) / n_classes
							tier = "head" if pct_rank <= 0.2 else ("tail" if pct_rank >= 0.8 else "torso")
							print(f"  {cname:<30} {sim_all[idx].item():>10.4f} {freq:>12} {tier:>8}")
					del zs_tmp, W_tmp, sim_all, sorted_idx
			except Exception as e:
					print(f"  [Drift-frequency cross-ref failed: {e}]")


			# Extended: full drift-frequency table saved to file
			try:
					zs_full = []
					with torch.no_grad():
						for start in range(0, len(probe_class_names), 64):
							chunk  = probe_class_names[start: start + 64]
							tokens = clip.tokenize(chunk, truncate=True).to(device)
							zs_chunk = torch.nn.functional.normalize(ft_model.clip_model.encode_text(tokens).float(), dim=-1).cpu()
							zs_full.append(zs_chunk)
					zs_full  = torch.cat(zs_full, dim=0)
					W_tmp    = torch.nn.functional.normalize(W_raw.cpu(), dim=-1)
					sim_all  = (W_tmp * zs_full).sum(dim=-1)   # [C]
					
					# Build full table
					rows = []
					for idx in range(len(probe_class_names)):
							cname    = probe_class_names[idx]
							freq     = probe_train_freq.get(cname, 0)
							n_classes = len(probe_train_freq)
							rank     = probe_train_freq.rank(ascending=False).get(cname, n_classes)
							pct_rank = rank / n_classes
							tier     = "head" if pct_rank <= 0.2 else ("tail" if pct_rank >= 0.8 else "torso")
							rows.append({
									'class':      cname,
									'zs_align':   sim_all[idx].item(),
									'probe_train_freq': freq,
									'tier':       tier,
									'pct_rank':   pct_rank,
							})
					df_drift = pd.DataFrame(rows).sort_values('zs_align')
					# Summary statistics per tier
					print(f"\n  Drift summary by tier:")
					print(f"  {'Tier':<8} {'N':>5} {'mean ZS-align':>15} {'mean freq':>12} {'zero-freq classes':>18}")
					print(f"  {'-'*60}")
					for tier in ['head', 'torso', 'tail']:
							sub = df_drift[df_drift['tier'] == tier]
							zero_freq = (sub['probe_train_freq'] == 0).sum()
							print(f"  {tier:<8} {len(sub):>5} {sub['zs_align'].mean():>15.4f} "
										f"{sub['probe_train_freq'].mean():>12.1f} {zero_freq:>18}")
					# Save full table
					drift_csv = os.path.join(
							os.path.dirname(checkpoint_path),
							"inference",
							"probe_drift_analysis.csv"
					)
					df_drift.to_csv(drift_csv, index=False)
					print(f"\n  Full drift table saved → {drift_csv}")
					del zs_full, W_tmp, sim_all
			except Exception as e:
					print(f"  [Full drift table failed: {e}]")




		# Use strategy as key — append index if duplicate (e.g. two lora checkpoints)
		key = strategy
		if key in fine_tuned_models:
			key = f"{strategy}_{i}"
		fine_tuned_models[key] = ft_model
		print(f"[OK] Loaded {key} in {time.time()-ft_start:.1f}s")
		print("-"*100)
	except Exception as e:
		print(f"[ERROR] Failed to load {checkpoint_path}: {e}")
		return {}, None, None

	return fine_tuned_models, probe_class_names, probe_train_freq.to_dict() if probe_train_freq is not None else None

def run_inference(
	pth_files_directory: str,
	model_architecture: str,
	metadata_csv: str,
	device: torch.device,
	batch_size: int,
	num_workers: int,
	dataset_dir: str,
	inference_dir: str,
	i2t_topk: int = 3,
	t2i_topk: int = 1,
	verbose: bool = True,
):
	dataset_name = os.path.basename(dataset_dir)
	column = "multimodal_canonical_labels" if "multi_label" in metadata_csv else "label"

	# list of all available checkpoints in directory of file.pth:
	available_checkpoints = glob.glob(os.path.join(pth_files_directory, "*.pth"))
	assert len(available_checkpoints) > 0, f"No checkpoints found in {pth_files_directory}"

	if verbose:
		print(f"{len(available_checkpoints)} Available checkpoints")
		for i, ft_path in enumerate(available_checkpoints):
			print(f"Checkpoint[{i}]: {ft_path}")

	# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	# print(clip.available_models()) # ViT-[size]/[patch_size][@resolution] or RN[depth]x[width_multiplier]
	print(f">> CLIP model configuration: {model_architecture}...")
	model_config = get_config(architecture=model_architecture)
	print(json.dumps(model_config, indent=4, ensure_ascii=False))

	train_loader, validation_loader = get_multi_label_dataloaders(
		metadata_fpth=metadata_csv,
		batch_size=batch_size,
		num_workers=num_workers,
		input_resolution=model_config["image_resolution"],
		col=column,
	)

	print_loader_info(loader=train_loader)
	print_loader_info(loader=validation_loader)

	customized_preprocess = get_preprocess(
		dataset_dir=dataset_dir, 
		input_resolution=model_config["image_resolution"],
	)

	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		try:
			class_names = validation_loader.dataset.dataset.classes
		except:
			class_names = train_loader.dataset.unique_labels
	print(f"Number of classes: {len(class_names)}")

	####################################### Quantitative Analysis #######################################
	results_json_path = os.path.join(pth_files_directory, f"{dataset_name}_retrieval_metrics_accumulated.json")
	if os.path.exists(results_json_path):
		with open(results_json_path) as f:
			all_results = json.load(f)
		print(f"[Quantitative Analysis] {len(all_results)} methods: {list(all_results.keys())}")
		viz.plot_quantitative_retrieval(
			all_results=all_results,
			output_dir=inference_dir,
			dataset_name=dataset_name,
		)
	else:
		print(f"WARNING: {results_json_path} not found. Skipping plotting...")
	####################################### Quantitative Analysis #######################################

	####################################### Qualitative Analysis #######################################
	i2t_samples, t2i_samples = get_tail_only_samples(
		metadata_val_path=metadata_csv.replace('.csv', '_val.csv'),
		col=column,
		num_samples=5,
		seed=42,
	)

	# Build full val image path list for T2I pool
	all_val_image_paths = sorted(
		validation_loader.dataset.data_frame['img_path'].tolist()
	)

	# Separate top-K strategies selection I2T
	i2t_strategies, i2t_checkpoints = get_top_k_strategies(
		results_json_path,
		top_k=3,
		distribution="rare",
		metric_key="mAP", 
		k_value=i2t_topk,
		mode="i2t",
	)

	# Separate top-K strategies selection T2I
	t2i_strategies, t2i_checkpoints = get_top_k_strategies(
		results_json_path, 
		top_k=3,
		distribution="rare",
		metric_key="mAP",
		k_value=t2i_topk,
		mode="t2i",
	)

	# Union of checkpoints to avoid loading the same model twice
	all_checkpoints = dict.fromkeys(i2t_checkpoints + t2i_checkpoints)
	if verbose:
		print("i2t_checkpoints:", i2t_checkpoints)
		print("t2i_checkpoints:", t2i_checkpoints)
		print(f"all_checkpoints: {len(all_checkpoints)} = (i2t: {len(i2t_checkpoints)}) + (t2i: {len(t2i_checkpoints)})")
		print(json.dumps(all_checkpoints, indent=4, ensure_ascii=False))

	qualitative_results = {}
	for ckpt_path in all_checkpoints:
		strategy_name, _ = _parse_checkpoint_strategy(ckpt_path)

		model_dict, canonical_class_names, probe_train_freq = _load_models(
			checkpoint_path=ckpt_path,
			model_architecture=model_architecture,
			device=device,
			dataset_directory=dataset_dir,
			validation_loader=validation_loader,
			class_names=class_names,
			verbose=verbose,
		)
		
		if not model_dict:
			continue
		
		ft_model = list(model_dict.values())[0]
		
		qualitative_results[strategy_name] = run_qualitative_retrieval(
			model=ft_model,
			i2t_samples=i2t_samples,
			t2i_samples=t2i_samples,
			class_names=class_names,
			all_val_image_paths=all_val_image_paths,
			preprocess=customized_preprocess,
			device=device,
			i2t_topk=i2t_topk,
			t2i_topk=t2i_topk,
			canonical_class_names=canonical_class_names,  # None for non-probe, correct for probe
			probe_train_freq=probe_train_freq,
			text_batch_size=128, # tune down to 128 if ViT-L/14@336px still OOMs
			image_batch_size=16, # tune down to 16 for 32GB GPUs under memory pressure
			verbose=verbose,
		)

		del ft_model, model_dict
		torch.cuda.empty_cache()

	print("="*100)
	print(f"{len(qualitative_results)} Qualitative Results:")
	print(json.dumps(qualitative_results, indent=2, ensure_ascii=False))
	print("="*100)

	# Two separate plots
	viz.plot_qualitative_retrieval_i2t(
		results_by_strategy={s: qualitative_results[s] for s in i2t_strategies if s in qualitative_results},
		output_dir=inference_dir,
		topk=i2t_topk,
		verbose=verbose,
	)
	viz.plot_qualitative_retrieval_t2i(
		results_by_strategy={s: qualitative_results[s] for s in t2i_strategies if s in qualitative_results},
		output_dir=inference_dir,
		topk=t2i_topk,
		verbose=verbose,
	)
	####################################### Qualitative Analysis #######################################

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Evaluate CLIP for Historical Archives Dataset [Inference]")
	parser.add_argument('--pth_files_directory', '-pth_dir', type=str, required=True, help='Directory containing the .pth files')
	parser.add_argument('--model_architecture', '-a', type=str, required=True, help='CLIP architecture')
	parser.add_argument('--batch_size', '-bs', type=int, default=8, help='Batch size for training')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=4, help='Number of CPUs')

	parser.add_argument('--query_image', '-qi', type=str, default=None, help='image path for zero shot classification')
	parser.add_argument('--query_label', '-ql', type=str, default=None, help='image path for zero shot classification')
	parser.add_argument('--t2i_topk', type=int, default=1, help='TopK results for Text-to-Image retrieval')
	parser.add_argument('--i2t_topk', type=int, default=3, help='TopK results for Image-to-Text retrieval')
	parser.add_argument('--topK_values', '-k', type=int, nargs='+', default=[1, 3, 5, 10, 15, 20], help='Top K values for retrieval metrics')

	parser.add_argument('--temperature', '-t', type=float, default=0.07, help='Temperature for evaluation')
	parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	args.pth_files_directory = os.path.normpath(args.pth_files_directory)
	print_args_table(args=args, parser=parser)
	print(args)
	set_seeds(seed=42)
	DATASET_DIRECTORY = os.path.dirname(args.pth_files_directory)
	print(f"DATASET_DIRECTORY: {DATASET_DIRECTORY}")
	
	dataset_type = "multi_label"
	metadata_csv = os.path.join(DATASET_DIRECTORY, f"metadata_{dataset_type}_multimodal.csv")
	assert os.path.exists(metadata_csv), f"{metadata_csv} not found!"

	INFERENCE_DIRECTORY = os.path.join(args.pth_files_directory, f"inference")
	os.makedirs(INFERENCE_DIRECTORY, exist_ok=True)

	run_inference(
		pth_files_directory=args.pth_files_directory,
		model_architecture=args.model_architecture,
		metadata_csv=metadata_csv,
		device=args.device,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		dataset_dir=DATASET_DIRECTORY,
		inference_dir=INFERENCE_DIRECTORY,
		i2t_topk=args.i2t_topk,
		t2i_topk=args.t2i_topk,
		verbose=args.verbose,
	)

if __name__ == "__main__":
	main()