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

# # run in local for all fine-tuned models with image and label:
# $ python multi_label_inference.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv -a 'ViT-B/32' -v

def pretrain_multi_label(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		device: torch.device,
		results_dir: str,
		active_mask: torch.Tensor,
		head_mask: torch.Tensor,
		rare_mask: torch.Tensor,
		topk_values: List[int] = [1, 3, 5, 10, 15, 20],
		temperature: float = 0.07,
		verbose: bool = True,
) -> Dict:
		"""
		Evaluate pretrained (zero-shot) CLIP on the validation set.
		Routes through the same get_validation_metrics + compute_tiered_retrieval_metrics
		pipeline as evaluate_best_model, so numbers are directly comparable in the table.
		"""
		dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
		model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown'

		if verbose:
				print(f"Zero-Shot CLIP Evaluation: {model.__class__.__name__} {model_arch} on {dataset_name}")

		# Identical pipeline to evaluate_best_model — no separate implementation
		validation_results = get_validation_metrics(
				model=model,
				validation_loader=validation_loader,
				device=device,
				topK_values=topk_values,
				finetune_strategy="pretrained",
				cache_dir=results_dir,
				verbose=verbose,
				is_training=False,
				model_hash=get_model_hash(model),
				temperature=temperature,
		)

		i2t_similarity = validation_results["i2t_similarity"]
		t2i_similarity = validation_results["t2i_similarity"]
		device_labels   = validation_results["device_labels"]

		if verbose:
				print("\nComputing tiered retrieval metrics (Overall / Head / Rare)...")

		tiered_i2t = compute_tiered_retrieval_metrics(
				similarity_matrix=i2t_similarity,
				query_labels=device_labels,
				topK_values=topk_values,
				head_mask=head_mask,
				rare_mask=rare_mask,
				active_mask=active_mask,
				mode="Image-to-Text",
				verbose=verbose,
		)
		tiered_t2i = compute_tiered_retrieval_metrics(
				similarity_matrix=t2i_similarity,
				query_labels=device_labels,
				topK_values=topk_values,
				head_mask=head_mask,
				rare_mask=rare_mask,
				active_mask=active_mask,
				mode="Text-to-Image",
				min_val_support=10,
				verbose=verbose,
		)

		del i2t_similarity, t2i_similarity
		torch.cuda.empty_cache()

		if verbose:
				print("\n>> Zero-Shot Tiered I2T Retrieval")
				for tier, m in tiered_i2t.items():
						print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
				print("\n>> Zero-Shot Tiered T2I Retrieval")
				for tier, m in tiered_t2i.items():
						print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")

		return {
				"full_metrics":    validation_results["full_metrics"],
				"img2txt_metrics": validation_results["img2txt_metrics"],
				"txt2img_metrics": validation_results["txt2img_metrics"],
				"tiered_i2t":      tiered_i2t,
				"tiered_t2i":      tiered_t2i,
		}

def get_multi_label_head_torso_tail_samples(
	metadata_path: str,
	metadata_train_path: str,
	metadata_val_path: str,
	num_samples_per_segment: int = 5,
) -> Tuple[List[Dict], List[str]]:
	"""
	Sample from head, torso, tail distributions for multi-label datasets.
	Only used when dataset_type == 'multi_label'
	"""
	try:
		df_val = pd.read_csv(
		filepath_or_buffer=metadata_val_path, 
			on_bad_lines='skip', 
			dtype=dtypes, 
			low_memory=False
		)
		print(f"VAL: {type(df_val)} {df_val.shape} {list(df_val.columns)}")
		# For multi-label, we need to count label frequencies differently
		all_labels = []
		for labels_str in df_val['multimodal_labels']:
			try:
				labels = ast.literal_eval(labels_str)
				all_labels.extend(labels)
			except:
				continue
		
		# Count label frequencies
		label_counts = pd.Series(all_labels).value_counts()
		
		# Define head, torso, tail based on frequency distribution
		total_labels = len(label_counts)
		head_threshold = int(total_labels * 0.2)  # Top 20%
		tail_threshold = int(total_labels * 0.8)   # Bottom 20%
		
		head_labels = set(label_counts.head(head_threshold).index)
		tail_labels = set(label_counts.tail(total_labels - tail_threshold).index)
		torso_labels = set(label_counts.index) - head_labels - tail_labels
		
		# Sample images and labels
		i2t_samples = []
		t2i_samples = []
		
		# Sample images with head, torso, tail labels
		for segment_name, segment_labels in [("head", head_labels), ("torso", torso_labels), ("tail", tail_labels)]:
			segment_samples = []
			for _, row in df_val.iterrows():
				try:
						row_labels = set(ast.literal_eval(row['multimodal_labels']))
						if row_labels & segment_labels:  # If any intersection
								segment_samples.append(
									{
										'image_path': row['img_path'],
										'labels': list(row_labels),
										'segment': segment_name
									}
								)
				except:
					continue
			# Randomly sample from this segment
			if len(segment_samples) >= num_samples_per_segment:
				sampled = random.sample(segment_samples, num_samples_per_segment)
				i2t_samples.extend(sampled)

		# Sample text queries from head, torso, tail
		for segment_name, segment_labels in [("head", head_labels), ("torso", torso_labels), ("tail", tail_labels)]:
			segment_label_list = list(segment_labels)
			if len(segment_label_list) >= num_samples_per_segment:
				sampled_labels = random.sample(segment_label_list, num_samples_per_segment)
				t2i_samples.extend(sampled_labels)
		
		return i2t_samples, t2i_samples
	except Exception as e:
		print(f"Error in multi-label sampling: {e}")
		return [], []

def _parse_checkpoint_strategy(ft_path: str) -> Tuple[str, Dict]:
		"""
		Parse checkpoint filename to determine fine-tuning strategy and hyperparameters.
		Returns (strategy_name, params_dict).

		Checkpoint naming conventions:
			lora_*          → strategy="lora"
			lora_plus_*     → strategy="lora_plus"
			dora_*          → strategy="dora"
			vera_*          → strategy="vera"
			ia3_*           → strategy="ia3"
			clip_adapter_v_* / clip_adapter_t_* / clip_adapter_vt_* → strategy=that variant
			tip_adapter_f_* → strategy="tip_adapter_f"
			tip_adapter_*   → strategy="tip_adapter"
			probe_*         → strategy="probe"
			full_*          → strategy="full"
		"""
		fname = os.path.basename(ft_path)

		# Order matters — more specific patterns before general ones
		if fname.startswith("lora_plus"):
				strategy = "lora_plus"
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

		# Extract LoRA hyperparams from filename if present
		params = {}
		lor_match = re.search(r'lor_(\d+)', fname)
		loa_match = re.search(r'loa_([\d.]+)', fname)
		lod_match = re.search(r'lod_([\d.]+)', fname)
		cbd_match = re.search(r'cbd_(\d+)', fname)
		act_match = re.search(r'act_(\w+?)_', fname)

		if lor_match:
				params["lora_rank"]    = int(lor_match.group(1))
		if loa_match:
				params["lora_alpha"]   = float(loa_match.group(1))
		if lod_match:
				params["lora_dropout"] = float(lod_match.group(1))
		if cbd_match:
				params["bottleneck_dim"] = int(cbd_match.group(1))
		if act_match:
				params["activation"] = act_match.group(1)

		return strategy, params

def _load_checkpoint_into_model(
		model: torch.nn.Module,
		ft_path: str,
		device: torch.device,
		verbose: bool = False,
) -> torch.nn.Module:
		"""Load checkpoint weights into an already-constructed model."""
		checkpoint = torch.load(ft_path, map_location=device)
		state_dict = checkpoint.get('model_state_dict', checkpoint)

		# Key translation for probe checkpoints (saved as bare linear layer)
		try:
				missing, unexpected = model.load_state_dict(state_dict, strict=False)
				if verbose and (missing or unexpected):
						print(f"  Missing keys : {len(missing)}")
						print(f"  Unexpected   : {len(unexpected)}")
		except Exception as e:
				print(f"  [WARNING] load_state_dict failed: {e}")

		return model

def load_finetuned_models(
		available_checkpoints: List[str],
		model_architecture: str,
		device: torch.device,
		dataset_directory: str,
		validation_loader: DataLoader,
		verbose: bool = False,
) -> Dict[str, torch.nn.Module]:
		"""
		Load all fine-tuned model checkpoints found in results_dir.
		Returns dict mapping strategy_name → model.
		"""
		fine_tuned_models = {}
		ft_start = time.time()
		print(f">> Loading {len(available_checkpoints)} fine-tuned checkpoints...")

		for i, ft_path in enumerate(available_checkpoints):
				strategy, params = _parse_checkpoint_strategy(ft_path)
				print(f"  [{i+1}/{len(available_checkpoints)}] strategy={strategy}  {os.path.basename(ft_path)}")

				# Fresh base model for each checkpoint
				base_model, _ = clip.load(
						name=model_architecture,
						device=device,
						download_root=get_model_directory(path=dataset_directory),
				)
				base_model = base_model.float()
				base_model.name = model_architecture

				try:
						if strategy in ("lora", "lora_plus", "dora"):
								rank    = params.get("lora_rank", 16)
								alpha   = params.get("lora_alpha", 1.0)
								dropout = params.get("lora_dropout", 0.0)
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

						elif strategy in ("ia3", "vera"):
								rank    = params.get("lora_rank", 16)
								alpha   = params.get("lora_alpha", 1.0)
								dropout = params.get("lora_dropout", 0.0)
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
								bottleneck_dim = params.get("bottleneck_dim", 256)
								activation     = params.get("activation", "relu")
								ft_model = get_adapter_peft_clip(
										clip_model=base_model,
										method=strategy,
										cache_dim=None,
										bottleneck_dim=bottleneck_dim,
										activation=activation,
										verbose=verbose,
								)

						elif strategy in ("tip_adapter", "tip_adapter_f"):
								# Tip-Adapter cache is not reconstructible from checkpoint alone —
								# the cache depends on support set features extracted at training time.
								# For inference, we load the trainable projection weights only.
								try:
										text_dim = base_model.encode_text(
												clip.tokenize(["a"]).to(device)
										).shape[-1]
								except Exception:
										text_dim = 768
								ft_model = get_adapter_peft_clip(
										clip_model=base_model,
										method=strategy,
										cache_dim=text_dim,
										bottleneck_dim=None,
										activation=None,
										verbose=verbose,
								)

						elif strategy == "probe":
								ft_model = get_probe_clip(
										clip_model=base_model,
										validation_loader=validation_loader,
										device=device,
										verbose=verbose,
								)

						elif strategy == "full":
								ft_model = base_model  # weights loaded directly below

						else:
								print(f"  [SKIP] Unknown strategy '{strategy}' for {ft_path}")
								continue

						ft_model = ft_model.to(device).float()
						ft_model.name = model_architecture
						ft_model = _load_checkpoint_into_model(ft_model, ft_path, device, verbose)

						# Use strategy as key — append index if duplicate (e.g. two lora checkpoints)
						key = strategy
						if key in fine_tuned_models:
								key = f"{strategy}_{i}"
						fine_tuned_models[key] = ft_model
						print(f"  ✓ Loaded {key}")

				except Exception as e:
						print(f"  [ERROR] Failed to load {ft_path}: {e}")
						continue

		print(f">> {len(fine_tuned_models)} {type(fine_tuned_models)} models loaded in {time.time()-ft_start:.1f}s")
		return fine_tuned_models

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Evaluate CLIP for Historical Archives Dataset [Inference]")
	parser.add_argument('--metadata_csv', '-csv', type=str, required=True, help='Metadata CSV file')
	parser.add_argument('--model_architecture', '-a', type=str, required=True, help='CLIP architecture')
	parser.add_argument('--batch_size', '-bs', type=int, default=16, help='Batch size for training')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=4, help='Number of CPUs')

	parser.add_argument('--query_image', '-qi', type=str, default=None, help='image path for zero shot classification')
	parser.add_argument('--query_label', '-ql', type=str, default=None, help='image path for zero shot classification')
	parser.add_argument('--topK', '-k', type=int, default=3, help='TopK results')
	parser.add_argument('--topK_values', type=int, nargs='+', default=[1, 3, 5, 10, 15, 20], help='Top K values for retrieval metrics')
	parser.add_argument('--temperature', '-t', type=float, default=0.07, help='Temperature for evaluation')
	parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)
	print(args)
	set_seeds(seed=42)
	DATASET_DIRECTORY = os.path.dirname(args.metadata_csv)
	dataset_name = os.path.basename(DATASET_DIRECTORY)
	dataset_type = "single_label" if "single_label" in args.metadata_csv else "multi_label"
	RESULT_DIRECTORY = os.path.join(DATASET_DIRECTORY, f"{dataset_type}")
	INFERENCE_DIRECTORY = os.path.join(RESULT_DIRECTORY, f"inference")
	CACHES_DIRECTORY = os.path.join(INFERENCE_DIRECTORY, "caches")

	os.makedirs(INFERENCE_DIRECTORY, exist_ok=True)
	os.makedirs(CACHES_DIRECTORY, exist_ok=True)

	# list of all available checkpoints in RESULT_DIRECTORY file.pth:
	available_checkpoints = glob.glob(os.path.join(RESULT_DIRECTORY, "*.pth"))
	print(f"{len(available_checkpoints)} Available checkpoints")
	for i, ft_path in enumerate(available_checkpoints):
		print(f"Checkpoint[{i}]: {ft_path}")

	if "probe" in available_checkpoints:
		params = get_probe_params(args.probe_checkpoint)
		if params:
			print(f">> {args.probe_checkpoint}\n\tProbe parameters: {params}")
			args.probe_dropout = params['probe_dropout']
		else:
			raise ValueError("Probe parameters not found in the provided checkpoint path!")

	if "lora" in available_checkpoints:
		params = get_lora_params(args.lora_checkpoint)
		if params:
			print(f">> {args.lora_checkpoint}\n\tLoRA parameters: {params}")
			args.lora_rank = params['lora_rank']
			args.lora_alpha = params['lora_alpha']
			args.lora_dropout = params['lora_dropout']
		else:
			raise ValueError("LoRA parameters not found in the provided checkpoint path!")

	# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	# print(clip.available_models()) # ViT-[size]/[patch_size][@resolution] or RN[depth]x[width_multiplier]
	models_to_plot = {}
	print(f">> CLIP model configuration: {args.model_architecture}...")
	model_config = get_config(architecture=args.model_architecture)
	print(json.dumps(model_config, indent=4, ensure_ascii=False))

	pretrained_model, pretrained_preprocess = clip.load(
		name=args.model_architecture,
		device=args.device,
		download_root=get_model_directory(path=DATASET_DIRECTORY),
	)
	pretrained_model = pretrained_model.float() # Convert model parameters to FP32
	pretrained_model_name = pretrained_model.__class__.__name__ # CLIP
	pretrained_model.name = args.model_architecture # ViT-B/32
	pretrained_model_arch = re.sub(r'[/@]', '-', args.model_architecture)
	print(f">> Pretrained model: {pretrained_model_name} {pretrained_model_arch}")

	print(f"Temperature used in evaluation: {getattr(args, 'temperature', 'Not set')}")

	models_to_plot["pretrained"] = pretrained_model

	train_loader, validation_loader = get_multi_label_dataloaders(
		metadata_fpth=args.metadata_csv,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		input_resolution=model_config["image_resolution"],
	)

	print_loader_info(loader=train_loader, batch_size=args.batch_size)
	print_loader_info(loader=validation_loader, batch_size=args.batch_size)

	print("DEBUGGING: Ground Truth Examination")
	# Check if ground truth extraction is correct
	for sample in validation_loader:
		images, _, labels = sample
		print(f"Batch size: {images.shape[0]}")
		print(f"Image shape: {images.shape}")
		print(f"Label shape: {labels.shape}")  # Should be [batch_size, num_classes]
		print(f"Labels dtype: {labels.dtype}")
		print(f"Number of positive labels in 1st sample: {labels[0].sum().item()}")
		print(f"Non-zero label indices: {torch.where(labels[0] == 1)[0].tolist()}")
		print(f"Number of positive labels in 2nd sample: {labels[1].sum().item()}")
		break  # Only check first batch	
	print("="*80)

	customized_preprocess = get_preprocess(
		dataset_dir=DATASET_DIRECTORY, 
		input_resolution=model_config["image_resolution"],
	)

	# GET CLASS INFORMATION
	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		try:
			class_names = validation_loader.dataset.dataset.classes
		except:
			class_names = train_loader.dataset.unique_labels
	
	num_classes = len(class_names)

	# Compute masks once from training set — reused for all models
	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		device=args.device,
		verbose=args.verbose,
	)

	# Zero-shot baseline
	pretrained_results = pretrain_multi_label(
		model=pretrained_model,
		validation_loader=validation_loader,
		device=args.device,
		results_dir=INFERENCE_DIRECTORY,
		active_mask=masks["active_mask"],
		head_mask=masks["head_mask"],
		rare_mask=masks["rare_mask"],
		topk_values=args.topK_values,
		temperature=args.temperature,
		verbose=args.verbose,
	)

	if args.query_image is None or args.query_label is None:
		print("\nSystematic selection of samples from validation set: Head, Torso, Tail...")
		i2t_samples, t2i_samples = get_multi_label_head_torso_tail_samples(
			metadata_path=args.metadata_csv,
			metadata_train_path=args.metadata_csv.replace('.csv', '_train.csv'),
			metadata_val_path=args.metadata_csv.replace('.csv', '_val.csv'),
			num_samples_per_segment=2,
		)
		if i2t_samples and t2i_samples:
			QUERY_IMAGES = [sample['image_path'] for sample in i2t_samples]
			QUERY_LABELS = t2i_samples  # Already a list of strings
	else:
		QUERY_IMAGES = [args.query_image]
		QUERY_LABELS = [args.query_label]

	print("\nQUERY IMAGES & LABELS")
	print(f">> {len(QUERY_IMAGES)} QUERY IMAGES:")
	for i, v in enumerate(QUERY_IMAGES):
		print(f"{i}. {v}")
	print(f">> {len(QUERY_LABELS)} QUERY LABELS:")
	for i, v in enumerate(QUERY_LABELS):
		print(f"{i}. {v}")
	print("-"*160)

	# clear cache
	torch.cuda.empty_cache()

	fine_tuned_models = load_finetuned_models(
		available_checkpoints=available_checkpoints,
		model_architecture=args.model_architecture,
		device=args.device,
		dataset_directory=DATASET_DIRECTORY,
		validation_loader=validation_loader,
		verbose=args.verbose,
	)
	models_to_plot.update(fine_tuned_models)

	if args.verbose:
		print(f"\nEvaluating {len(fine_tuned_models)} Fine-tuned Models: {list(fine_tuned_models.keys())}")
	finetuned_img2txt_dict = {args.model_architecture: {}}
	finetuned_txt2img_dict = {args.model_architecture: {}}
	ft_eval_start = time.time()
	# clear cache
	torch.cuda.empty_cache()
	for strategy, ft_model in fine_tuned_models.items():
		print(f"\n>> Evaluating: {strategy}")
		evaluation_results = evaluate_best_model(
			model=ft_model,
			validation_loader=validation_loader,
			active_mask=masks["active_mask"],
			head_mask=masks["head_mask"],
			rare_mask=masks["rare_mask"],
			early_stopping=None,
			checkpoint_path=None,
			finetune_strategy=strategy,
			device=args.device,
			cache_dir=CACHES_DIRECTORY,
			topk_values=args.topK_values,
			verbose=args.verbose,
			clean_cache=False, # keep cache across models
			lora_params={
				"lora_rank": args.lora_rank,
				"lora_alpha": args.lora_alpha,
				"lora_dropout": args.lora_dropout,
			} if strategy == "lora" else None,
			temperature=args.temperature,
		)
		finetuned_img2txt_dict[args.model_architecture][strategy] = evaluation_results["img2txt_metrics"]
		finetuned_txt2img_dict[args.model_architecture][strategy] = evaluation_results["txt2img_metrics"]
		# clear cache
		torch.cuda.empty_cache()

	print(f"{len(fine_tuned_models)} Fine-tuned Models evaluated in {time.time() - ft_eval_start:.5f} sec")

	####################################### Qualitative Analysis #######################################
	if args.verbose:
		print(f"Qualitative Analysis".center(160, " "))
	for query_image in QUERY_IMAGES:
		viz.plot_image_to_texts_pretrained(
			best_pretrained_model=pretrained_model,
			validation_loader=validation_loader,
			# preprocess=pretrained_preprocess, # customized_preprocess,
			preprocess=customized_preprocess,
			img_path=query_image,
			topk=args.topK,
			device=args.device,
			results_dir=INFERENCE_DIRECTORY,
		)
		viz.plot_image_to_texts_stacked_horizontal_bar(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			img_path=query_image,
			topk=args.topK,
			device=args.device,
			results_dir=INFERENCE_DIRECTORY,
		)
		viz.plot_image_to_texts_separate_horizontal_bars(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			img_path=query_image,
			topk=args.topK,
			device=args.device,
			results_dir=INFERENCE_DIRECTORY,
		)

	for query_label in QUERY_LABELS:
		viz.plot_text_to_images(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			query_text=query_label,
			topk=args.topK,
			device=args.device,
			results_dir=INFERENCE_DIRECTORY,
			cache_dir=CACHES_DIRECTORY,
			embeddings_cache=embeddings_cache,
		)
	####################################### Qualitative Analysis #######################################

	####################################### Quantitative Analysis #######################################
	if args.plot and os.path.exists(results_json_path):
		with open(results_json_path) as f:
			all_results = json.load(f)
		print(f">> Plotting {len(all_results)} methods: {list(all_results.keys())}")
		plot_retrieval_curves(
				all_results=all_results,
				output_dir=os.path.join(RESULTS_DIRECTORY, "plots"),
				dataset_name=dataset_name,
				verbose=args.verbose,
		)
	elif args.plot:
		print(f"[WARNING] --plot requested but no results file found at {results_json_path}")
	####################################### Quantitative Analysis #######################################

if __name__ == "__main__":
	main()