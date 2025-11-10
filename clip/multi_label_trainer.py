from utils import *
from early_stopper import EarlyStopping
from peft import get_injected_peft_clip, get_adapter_peft_clip
from evals import *
import visualize as viz

def full_finetune_multi_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		patience: int,
		min_delta: float,
		cumulative_delta: float,
		minimum_epochs: int,
		volatility_threshold: float,
		slope_threshold: float,
		pairwise_imp_threshold: float,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
		temperature: float = 0.07,  # Temperature for contrastive learning
		label_smoothing: float = 0.0,  # Label smoothing for multi-label
		use_lamb: bool = False,
		verbose: bool=True,
	):
	"""
	Full fine-tuning for multi-label CLIP classification.
	
	Key changes from single-label version:
	1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
	2. Handles bidirectional multi-label targets (I2T and T2I)
	3. Proper multi-label evaluation metrics
	4. Custom loss computation for contrastive multi-label learning
	
	Args:
			model: CLIP model to fine-tune
			train_loader: Training DataLoader (must provide multi-label vectors)
			validation_loader: Validation DataLoader  
			num_epochs: Number of training epochs
			print_every: Print loss every N batches
			learning_rate: Learning rate
			weight_decay: Weight decay for regularization
			device: Training device (cuda/cpu)
			results_dir: Directory to save results
			window_size: Window size for early stopping
			patience: Early stopping patience
			min_delta: Minimum change for improvement
			cumulative_delta: Cumulative delta for early stopping
			minimum_epochs: Minimum epochs before early stopping
			topk_values: K values for evaluation metrics
			loss_weights: Optional weights for I2T and T2I losses
			temperature: Temperature scaling for similarities
			label_smoothing: Label smoothing factor (0.0 = no smoothing)
	"""

	window_size = minimum_epochs + 1
	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)
	
	# Set default loss weights
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name


	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	print(f"{mode} {model_name} {model_arch} {dataset_name} {num_epochs} Epoch(s) batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))
	
	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)
	
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]

	print(f"\nNon-zero dropout detected in base {model_name} {model_arch} during {mode}:")
	print(non_zero_dropouts)
	print()

	# Unfreeze all layers for full fine-tuning
	for name, param in model.named_parameters():
		param.requires_grad = True

	get_parameters_info(model=model, mode=mode)

	# Use BCEWithLogitsLoss for multi-label classification
	if label_smoothing > 0:
		if verbose:
			print(f"Using label smoothing: {label_smoothing}")
		criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
	else:
		criterion = torch.nn.BCEWithLogitsLoss()
	if verbose:
		print(f"Using {criterion.__class__.__name__} for multi-label classification with {num_classes} classes")

	all_class_embeds = []
	model.eval()  # Ensure model is in eval mode
	text_batch_size = validation_loader.batch_size
	if verbose:
		print(f"Pre-encoding {num_classes} class texts in batch_size: {text_batch_size}")
	with torch.no_grad():
		with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			for i in tqdm(range(0, num_classes, text_batch_size), desc="Pre-encoding class texts"):
				end_idx = min(i + text_batch_size, num_classes)
				batch_class_names = class_names[i:end_idx]

				batch_class_texts = clip.tokenize(batch_class_names).to(device)
				batch_embeds = model.encode_text(batch_class_texts)
				batch_embeds = F.normalize(batch_embeds, dim=-1)
				all_class_embeds.append(batch_embeds.cpu())  # Move to CPU immediately to save GPU memory
				
				# Clean up
				del batch_class_texts, batch_embeds
				torch.cuda.empty_cache()
	
	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device)
	if verbose:
		print(f"all_class_embeds: {type(all_class_embeds)} {all_class_embeds.shape} {all_class_embeds.dtype} {all_class_embeds.device}")

	if use_lamb:
		optimizer = LAMB(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			weight_decay=weight_decay,
		)
	else:
		optimizer = torch.optim.AdamW(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,
		anneal_strategy='cos',
	)

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"do_{dropout_val}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdelta_{min_delta:.1e}_"
		f"cdelta_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)
	print(f"Best model will be saved in: {mdl_fpth}")
	training_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	train_start_time = time.time()
	best_val_loss = float('inf')
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	# model.train()
	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0
		for bidx, batch_data in enumerate(train_loader):
			if len(batch_data) == 3:
				images, _, label_vectors = batch_data  # Ignore tokenized_labels, use pre-encoded
			else:
				raise ValueError(f"Expected 3 items from DataLoader, got {len(batch_data)}")
			batch_size = images.size(0)
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			# Validate label_vectors shape
			if label_vectors.shape != (batch_size, num_classes):
				raise ValueError(f"Label vectors shape {label_vectors.shape} doesn't match expected ({batch_size}, {num_classes})")
			optimizer.zero_grad(set_to_none=True)
			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				# ================================
				# CRITICAL CHANGE: Multi-label loss computation
				# ================================
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion=criterion,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)
			# Check for NaN loss
			if torch.isnan(total_loss):
				print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
				continue
			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			# Track losses
			batch_loss_total = total_loss.item()
			batch_loss_i2t = loss_i2t.item()
			batch_loss_t2i = loss_t2i.item()
			
			epoch_loss_total += batch_loss_total
			epoch_loss_i2t += batch_loss_i2t
			epoch_loss_t2i += batch_loss_t2i
			num_batches += 1
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\t\tBatch [{bidx + 1}/{len(train_loader)}] "
					f"Total Loss: {batch_loss_total:.6f} "
					f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})"
				)
		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0
		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)
		print(f">> Training completed in {time.time() - train_and_val_st_time:.2f} sec. Validating Epoch {epoch+1}")
		# clear cache before validation
		torch.cuda.empty_cache()
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			max_batches=10
		)
		
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,  # Now uses BCEWithLogitsLoss
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=True,
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
		)
		
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
		in_batch_loss_acc_metrics_per_epoch["val_loss"] = current_val_loss
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}
		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]
		print(
			f'@ Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}:\n'
			f'\t\tTraining - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f})\n'
			f'\t\tValidation: {current_val_loss:.6f}\n'
			f'\tMulti-label Validation Metrics:\n'
			f'\t\tIn-batch Top-K Accuracy:\n'
			f'\t\t\t[Image→Text]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\t\tFull Validation Set:\n'
			f'\t\t\t[Image→Text]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'\tMulti-label Metrics:')
			print(f'\t\tHamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'\t\tPartial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'\t\tF1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')
		
		print(f"\tRetrieval Metrics:")
		print(
			f"\t\tImage-to-Text: mAP@10={retrieval_metrics_per_epoch['img2txt'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['img2txt'].get('Recall', {}).get('10', 'N/A')}"
		)
		print(
			f"\t\tText-to-Image: mAP@10={retrieval_metrics_per_epoch['txt2img'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['txt2img'].get('Recall', {}).get('10', 'N/A')}"
		)
		if hasattr(train_loader.dataset, 'get_cache_stats'):
			print(f"#"*100)
			cache_stats = train_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Train Cache Stats: {cache_stats}")
		if hasattr(validation_loader.dataset, 'get_cache_stats'):
			cache_stats = validation_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Validation Cache Stats: {cache_stats}")
			print(f"#"*100)
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\nEarly stopping at epoch {epoch + 1} "
				f"with best loss: {early_stopping.get_best_score()} "
				f"obtained in epoch {early_stopping.get_best_epoch()+1}")
			break
		print(f"Epoch {epoch+1} Duration [Train + Validation]: {time.time() - train_and_val_st_time:.2f}s".center(150, "="))

	print(f"[{mode}] Total Training  Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

	# ================================
	# FINAL EVALUATION
	# ================================
	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=True,
		max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
	)

	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	model_source = evaluation_results["model_loaded_from"]

	print(f"Final evaluation used model weights from: {model_source}")

	print("\nGenerating result plots...")

	actual_trained_epochs = len(training_losses)

	file_base_name = (
		f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}"
	)
	
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print(f"Best model will be renamed to: {mdl_fpth}")

	# ================================
	# PLOTTING: Enhanced for multi-label
	# ================================
	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}
	
	viz.plot_multilabel_loss_breakdown(
			training_losses_breakdown=training_losses_breakdown,
			filepath=plot_paths["losses_breakdown"]
	)
	viz.plot_loss_accuracy_metrics(
			dataset_name=dataset_name,
			train_losses=training_losses,
			val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
			in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
			full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
			losses_file_path=plot_paths["losses"],
			in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
			in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
			full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
			full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
	)
	viz.plot_retrieval_metrics_per_epoch(
			dataset_name=dataset_name,
			image_to_text_metrics_list=img2txt_metrics_all_epochs,
			text_to_image_metrics_list=txt2img_metrics_all_epochs,
			fname=plot_paths["retrieval_per_epoch"],
	)
	viz.plot_retrieval_metrics_best_model(
			dataset_name=dataset_name,
			image_to_text_metrics=final_img2txt_metrics,
			text_to_image_metrics=final_txt2img_metrics,
			fname=plot_paths["retrieval_best"],
	)

	return final_metrics_in_batch, final_metrics_full, final_img2txt_metrics, final_txt2img_metrics

def progressive_finetune_multi_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		window_size: int = 10,
		patience: int = 10,
		min_delta: float = 1e-4,  # Make slightly less sensitive than default
		cumulative_delta: float = 5e-3,  # Keep cumulative check reasonable
		minimum_epochs: int = 20,  # Minimum epochs before ANY early stop
		min_epochs_per_phase: int = 5,  # Minimum epochs within a phase before transition check
		volatility_threshold: float = 15.0,  # Allow slightly more volatility
		slope_threshold: float = 1e-4,  # Allow very slightly positive slope before stopping/transitioning
		pairwise_imp_threshold: float = 1e-4,  # Stricter requirement for pairwise improvement
		accuracy_plateau_threshold: float = 5e-4,  # For phase transition based on accuracy
		min_phases_before_stopping: int = 3,  # Ensure significant unfreezing before global stop
		topk_values: list[int] = [1, 5, 10],
		layer_groups_to_unfreeze: list[str] = ['visual_transformer', 'text_transformer', 'projections'],  # Focus on key layers
		loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
		temperature: float = 0.07,  # Temperature for contrastive learning
		label_smoothing: float = 0.0,  # Label smoothing for multi-label
		use_lamb: bool = False,
	):
	# Set default loss weights
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	initial_learning_rate = learning_rate
	initial_weight_decay = weight_decay

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',  # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,  # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		min_phases_before_stopping=min_phases_before_stopping,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	print(f"{mode} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	try:
		num_classes = len(validation_loader.dataset.unique_labels)
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		num_classes = len(validation_loader.dataset.dataset.classes)
		class_names = validation_loader.dataset.dataset.classes
	print(f"Multi-label progressive fine-tuning: {num_classes} classes")
	# print(f"Class names sample:\n{class_names}")

	# Find dropout value
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break

	# dropout layers inspection:
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	print(f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during {mode} fine-tuning:")
	print(non_zero_dropouts)
	print()

	# Get the detailed layer unfreeze schedule
	total_num_phases = min_phases_before_stopping + 5
	unfreeze_schedule = get_unfreeze_schedule(
		model=model,
		num_phases=total_num_phases,
		layer_groups_to_unfreeze=layer_groups_to_unfreeze,
	)

	# Use BCEWithLogitsLoss for multi-label classification
	if label_smoothing > 0:
		print(f"Using label smoothing: {label_smoothing}")
		criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
	else:
		criterion = torch.nn.BCEWithLogitsLoss()
	print(f"Using {criterion.__class__.__name__} for multi-label classification")

	# Pre-encode all class texts (for efficiency)
	model.eval()
	print(f"Pre-encoding {num_classes} class texts...")
	all_class_texts = clip.tokenize(class_names).to(device)
	with torch.no_grad():
		with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			all_class_embeds = model.encode_text(all_class_texts)
			all_class_embeds = F.normalize(all_class_embeds, dim=-1)

	model.train()
	if use_lamb:
		optimizer = LAMB(
			params=filter(lambda p: p.requires_grad, model.parameters()),  # Initially might be empty if phase 0 has no unfrozen layers
			lr=initial_learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=initial_weight_decay,
		)
	else:
		optimizer = torch.optim.AdamW(
			params=filter(lambda p: p.requires_grad, model.parameters()),  # Initially might be empty if phase 0 has no unfrozen layers
			lr=initial_learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=initial_weight_decay,
		)

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=initial_learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,  # Standard pct_start
		anneal_strategy='cos'  # Cosine annealing
	)

	print(f"Using {scheduler.__class__.__name__} for learning rate scheduling")
	print(f"Using {criterion.__class__.__name__} as the loss function")

	scaler = torch.amp.GradScaler(device=device) # automatic mixed precision
	print(f"Using {scaler.__class__.__name__} for automatic mixed precision training")

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"do_{dropout_val}_"
		f"ilr_{initial_learning_rate:.1e}_"
		f"iwd_{initial_weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}_"
		f"mepph_{min_epochs_per_phase}_"
		f"mpbs_{min_phases_before_stopping}_"
		f".pth"
	)
	print(f"Best model will be saved in: {mdl_fpth}")

	current_phase = 0
	epochs_in_current_phase = 0
	training_losses = list()  # History of average training loss per epoch
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}  # Multi-label loss breakdown
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()  # History of [in-batch] validation metrics dicts per epoch
	full_val_loss_acc_metrics_all_epochs = list()  # History of [full] validation metrics dicts per epoch
	best_val_loss = None  # Track the absolute best validation loss
	layer_cache = {}  # Cache for layer status (optional, used by get_status)
	last_lr = initial_learning_rate  # Track current LR
	last_wd = initial_weight_decay  # Track current WD
	phase_just_changed = False  # Flag to signal optimizer refresh needed

	# --- Main Training Loop ---
	train_start_time = time.time()

	for epoch in range(num_epochs):
		epoch_start_time = time.time()
		print(f"Epoch {epoch+1}/{num_epochs} Phase {current_phase}/{total_num_phases} current LR: {last_lr} current WD: {last_wd}")
		torch.cuda.empty_cache()
		
		# --- Phase Transition Check ---
		# Check only if enough epochs *overall* and *within the phase* have passed,
		# and if we are not already in the last phase.
		if (epoch >= minimum_epochs and  # Overall min epochs check
			epochs_in_current_phase >= min_epochs_per_phase and
			current_phase < total_num_phases - 1 and
			len(early_stopping.value_history) >= window_size):
			print(f"Checking phase transition ({epochs_in_current_phase} elapsed epochs in phase {current_phase})")

			val_losses = early_stopping.value_history
			
			# For multi-label, we can use average accuracy across I2T and T2I
			val_accs_in_batch = list()
			for m in in_batch_loss_acc_metrics_all_epochs:
				i2t_acc = m.get('img2txt_acc', 0.0)
				t2i_acc = m.get('txt2img_acc', 0.0)
				avg_acc = (i2t_acc + t2i_acc) / 2.0
				val_accs_in_batch.append(avg_acc)

			should_trans = should_transition_to_next_phase(
				current_phase=current_phase,
				losses=val_losses,
				window=window_size,
				best_loss=early_stopping.get_best_score(),  # Use best score from early stopping state
				best_loss_threshold=min_delta,  # Use min_delta for closeness check
				volatility_threshold=volatility_threshold,
				slope_threshold=slope_threshold,  # Use positive threshold for worsening loss
				pairwise_imp_threshold=pairwise_imp_threshold,
				accuracies=val_accs_in_batch,  # Pass average accuracy for multi-label
				accuracy_plateau_threshold=accuracy_plateau_threshold,
			)
			
			if should_trans:
				current_phase, last_lr, last_wd = handle_phase_transition(
					current_phase=current_phase,
					last_lr=last_lr,
					last_wd=last_wd,
				)
				epochs_in_current_phase = 0  # Reset phase epoch counter
				early_stopping.reset()  # <<< CRITICAL: Reset early stopping state for the new phase
				print(f"Transitioned to Phase {current_phase}. Early stopping reset.")

				phase_just_changed = True  # Signal that optimizer needs refresh after unfreeze
				print(f"Phase transition triggered. Optimizer/Scheduler refresh pending after unfreeze.")
				print(f"Current Phase: {current_phase}")

		unfreeze_layers(
			model=model,
			strategy=unfreeze_schedule,
			phase=current_phase,
			cache=layer_cache,
		)
		
		if phase_just_changed or epoch == 0:
			print("Refreshing optimizer parameter groups...")
			optimizer.param_groups.clear()
			optimizer.add_param_group(
				{
					'params': [p for p in model.parameters() if p.requires_grad],
					'lr': last_lr,  # Use the new LR
					'weight_decay': last_wd,  # Use the new WD
				}
			)
			print(f"Optimizer parameter groups refreshed. LR set to {last_lr}, WD set to {last_wd}.")
			print("Re-initializing OneCycleLR scheduler for new phase/start...")
			steps_per_epoch = len(train_loader)
			# Schedule over remaining epochs (more adaptive)
			# max: Ensure scheduler_epochs is at least 1
			scheduler_epochs = max(1, num_epochs - epoch)
			scheduler = torch.optim.lr_scheduler.OneCycleLR(
				optimizer=optimizer,
				max_lr=last_lr,  # Use the new LR as the peak for the new cycle
				steps_per_epoch=steps_per_epoch,
				epochs=scheduler_epochs,
				pct_start=0.1,  # Consider if this needs adjustment in later phases
				anneal_strategy='cos',
				# last_epoch = -1 # Ensures it starts fresh
			)
			print(f"Scheduler re-initialized with max_lr={last_lr} for {scheduler_epochs} epochs.")
			phase_just_changed = False  # Reset the flag

		# --- Training Epoch ---
		model.train()
		epoch_train_loss_total = 0.0
		epoch_train_loss_i2t = 0.0
		epoch_train_loss_t2i = 0.0
		num_train_batches = len(train_loader)
		num_processed_batches = 0
		
		trainable_params_exist = any(p.requires_grad for p in model.parameters())
		if not trainable_params_exist:
			print("Warning: No trainable parameters found for the current phase. Skipping training steps.")
		else:
			for bidx, batch_data in enumerate(train_loader):
				if len(batch_data) == 3:
					images, _, label_vectors = batch_data  # Ignore tokenized_labels, use pre-encoded
				else:
					raise ValueError(f"Expected 3 items from DataLoader, got {len(batch_data)}")
				
				batch_size = images.size(0)
				images = images.to(device, non_blocking=True)
				label_vectors = label_vectors.to(device, non_blocking=True).float()
				
				# Validate label_vectors shape
				if label_vectors.shape != (batch_size, num_classes):
					raise ValueError(f"Label vectors shape {label_vectors.shape} doesn't match expected ({batch_size}, {num_classes})")

				optimizer.zero_grad(set_to_none=True)
				
				with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
					# Multi-label contrastive loss computation
					total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
						model=model,
						images=images,
						all_class_embeds=all_class_embeds,
						label_vectors=label_vectors,
						criterion=criterion,
						temperature=temperature,
						loss_weights=loss_weights,
						verbose=verbose,
					)
				
				if torch.isnan(total_loss):
					print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
					continue  # Skip optimizer step if loss is NaN
				
				scaler.scale(total_loss).backward()
				scaler.unscale_(optimizer)  # Unscale before clipping
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				scaler.step(optimizer)
				scaler.update()
				scheduler.step()  # Step the scheduler
				
				# Track losses
				batch_loss_total = total_loss.item()
				batch_loss_i2t = loss_i2t.item()
				batch_loss_t2i = loss_t2i.item()
				
				epoch_train_loss_total += batch_loss_total
				epoch_train_loss_i2t += batch_loss_i2t
				epoch_train_loss_t2i += batch_loss_t2i
				num_processed_batches += 1

				if bidx % print_every == 0:
					print(
						f"\tBatch [{bidx+1}/{num_train_batches}] "
						f"Total Loss: {batch_loss_total:.6f} "
						f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})"
					)
				elif bidx == num_train_batches - 1 and batch_loss_total > 0:
					print(
						f"\tBatch [{bidx+1}/{num_train_batches}] "
						f"Total Loss: {batch_loss_total:.6f} "
						f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})"
					)

		# Calculate average losses
		avg_total_loss = epoch_train_loss_total / num_processed_batches if num_processed_batches > 0 and trainable_params_exist else 0.0
		avg_i2t_loss = epoch_train_loss_i2t / num_processed_batches if num_processed_batches > 0 and trainable_params_exist else 0.0
		avg_t2i_loss = epoch_train_loss_t2i / num_processed_batches if num_processed_batches > 0 and trainable_params_exist else 0.0
		
		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)

		# --- Validation ---
		print(f">> Training Completed in {time.time() - epoch_start_time:.2f} sec. Validating Epoch: {epoch+1}")

		# clear cache before validation
		torch.cuda.empty_cache()

		# Compute validation loss using the same multi-label loss function
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			max_batches=10
		)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,  # Now uses BCEWithLogitsLoss
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=True,
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
		)
		
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
		in_batch_loss_acc_metrics_per_epoch["val_loss"] = current_val_loss  # Use computed validation loss
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		
		current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]

		print(
			f'@ Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}:\n'
			f'\t\tTraining - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f})\n'
			f'\t\tValidation: {current_val_loss:.6f}\n'
			f'\tMulti-label Validation Metrics:\n'
			f'\t\tIn-batch Top-K Accuracy:\n'
			f'\t\t\t[Image→Text]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\t\tFull Validation Set:\n'
			f'\t\t\t[Image→Text]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'\tMulti-label Metrics:')
			print(f'\t\tHamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'\t\tPartial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'\t\tF1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')
		
		print(f"\tRetrieval Metrics:")
		print(
			f"\t\tImage-to-Text: mAP@10={retrieval_metrics_per_epoch['img2txt'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['img2txt'].get('Recall', {}).get('10', 'N/A')}"
		)
		print(
			f"\t\tText-to-Image: mAP@10={retrieval_metrics_per_epoch['txt2img'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['txt2img'].get('Recall', {}).get('10', 'N/A')}"
		)

		if hasattr(train_loader.dataset, 'get_cache_stats'):
			cache_stats = train_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"#"*100)
				print(f"Train Cache Stats: {cache_stats}")

		if hasattr(validation_loader.dataset, 'get_cache_stats'):
			cache_stats = validation_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"#"*100)
				print(f"Validation Cache Stats: {cache_stats}")
		
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
			current_phase=current_phase
		):
			print(f"--- Training stopped early at epoch {epoch+1} ---")
			break  # Exit the main training loop

		# --- End of Epoch ---
		epochs_in_current_phase += 1

		if epoch+1 > minimum_epochs: 
			print(f"EarlyStopping Status:\n{json.dumps(early_stopping.get_status(), indent=2, ensure_ascii=False)}")
		print(f"Epoch {epoch+1} Elapsed_t: {time.time()-epoch_start_time:.2f} sec".center(170, "-"))

	# --- End of Training ---
	total_training_time = time.time() - train_start_time
	print(f"\n--- Training Finished ---")
	print(f"Total Epochs Run: {epoch + 1}")
	print(f"Final Phase Reached: {current_phase}")
	print(f"Best Validation Loss Achieved: {early_stopping.get_best_score()} @ Epoch {early_stopping.get_best_epoch() + 1}")
	print(f"Total Training Time: {total_training_time:.2f}s")

	# Final evaluation with best model
	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=True,
		max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
	)

	# Access individual metrics as needed
	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]

	model_source = evaluation_results["model_loaded_from"]
	print(f"Final evaluation used model weights from: {model_source}")

	print("\nGenerating result plots...")
	actual_trained_epochs = len(training_losses)

	file_base_name = (
		f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"last_phase_{current_phase}_"
		f"ep_{actual_trained_epochs}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}_"
		f"temp_{temperature}_"
		f"ilr_{initial_learning_rate:.1e}_"
		f"iwd_{initial_weight_decay:.1e}"
	)
	if last_lr is not None:
		file_base_name += f"_final_lr_{last_lr:.1e}"

	if last_wd is not None:
		file_base_name += f"_final_wd_{last_wd:.1e}"

	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth,
		actual_epochs=actual_trained_epochs,
		additional_info={
			'last_phase': current_phase,
			'flr': last_lr,
			'fwd': last_wd
		}
	)

	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
	)

	viz.plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
	)

	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)

	return final_metrics_in_batch, final_metrics_full, final_img2txt_metrics, final_txt2img_metrics

def lora_finetune_multi_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		lora_rank: int,
		lora_alpha: float,
		lora_dropout: float,
		patience: int,
		min_delta: float,
		cumulative_delta: float,
		minimum_epochs: int,
		volatility_threshold: float,
		slope_threshold: float,
		pairwise_imp_threshold: float,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
		temperature: float = 0.07,  # Temperature for contrastive learning
		label_smoothing: float = 0.0,  # Label smoothing for multi-label
		use_lamb: bool = False,
		quantization_bits: int = 8,
		quantized: bool = False,
		verbose: bool = True,
	):
	"""
	LoRA fine-tuning for multi-label CLIP classification.
	
	Key differences from single-label LoRA version:
	1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
	2. Handles bidirectional multi-label targets (I2T and T2I)
	3. Pre-encodes class embeddings for efficiency
	4. Uses multi-label specific loss computation
	5. Proper multi-label evaluation metrics
	
	Args:
			model: CLIP model to fine-tune with LoRA
			train_loader: Training DataLoader (must provide multi-label vectors)
			validation_loader: Validation DataLoader  
			num_epochs: Number of training epochs
			print_every: Print loss every N batches
			learning_rate: Learning rate for LoRA parameters
			weight_decay: Weight decay for regularization
			device: Training device (cuda/cpu)
			results_dir: Directory to save results
			window_size: Window size for early stopping
			lora_rank: LoRA rank parameter
			lora_alpha: LoRA alpha parameter
			lora_dropout: LoRA dropout parameter
			patience: Early stopping patience
			min_delta: Minimum change for improvement
			cumulative_delta: Cumulative delta for early stopping
			minimum_epochs: Minimum epochs before early stopping
			topk_values: K values for evaluation metrics
			loss_weights: Optional weights for I2T and T2I losses
			temperature: Temperature scaling for similarities
			label_smoothing: Label smoothing factor (0.0 = no smoothing)
	"""
	# Adaptive window size based on minimum epochs
	window_size = minimum_epochs + 1

	# Set default loss weights
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	# Inspect the model for dropout layers and validate for LoRA
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	# Check for non-zero dropout in the base model
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during LoRA fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for LoRA practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during LoRA fine-tuning "
			"which gives you more control over LoRA-specific regularization without affecting the base model.\n"
		)

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',  # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,  # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	print(f"{mode} | Rank: {lora_rank} | Alpha: {lora_alpha} | Dropout: {lora_dropout} | {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	# Get dataset information
	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)

	if verbose:
		print(f"{mode.upper()} Rank: {lora_rank} Alpha: {lora_alpha} Dropout: {lora_dropout} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}")

		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

		if torch.cuda.is_available():
			gpu_name = torch.cuda.get_device_name(device)
			gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
			cuda_capability = torch.cuda.get_device_capability()
			print(f"   ├─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

		print(f"Multi-label LoRA fine-tuning: {num_classes} classes")

	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	)
	
	model.to(device)
	get_parameters_info(model=model, mode=mode)

	# Use BCEWithLogitsLoss for multi-label classification
	if label_smoothing > 0:
		if verbose:
			print(f"Using label smoothing: {label_smoothing}")
		criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
	else:
		criterion = torch.nn.BCEWithLogitsLoss()
	if verbose:
		print(f"Using {criterion.__class__.__name__} for multi-label classification")

	all_class_embeds = []
	model.eval()  # Ensure model is in eval mode
	text_batch_size = validation_loader.batch_size
	print(f"Pre-encoding {num_classes} class texts in batch_size: {text_batch_size}")
	with torch.no_grad():
		with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			for i in tqdm(range(0, num_classes, text_batch_size), desc="Pre-encoding class texts"):
				end_idx = min(i + text_batch_size, num_classes)
				batch_class_names = class_names[i:end_idx]

				batch_class_texts = clip.tokenize(batch_class_names).to(device)
				batch_embeds = model.encode_text(batch_class_texts)
				batch_embeds = F.normalize(batch_embeds, dim=-1)
				all_class_embeds.append(batch_embeds.cpu())  # Move to CPU immediately to save GPU memory
				
				# Clean up
				del batch_class_texts, batch_embeds
				if torch.cuda.is_available():
						torch.cuda.empty_cache()
	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device)
	print(f"all_class_embeds: {type(all_class_embeds)} {all_class_embeds.shape} {all_class_embeds.dtype} {all_class_embeds.device}")

	if use_lamb:
		optimizer = LAMB(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)
	else:
		optimizer = torch.optim.AdamW(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,
		anneal_strategy='cos',
	)

	scaler = torch.amp.GradScaler(device=device)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)

	training_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	train_start_time = time.time()
	best_val_loss = float('inf')
	final_img2txt_metrics = None
	final_txt2img_metrics = None

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0
		
		for bidx, batch_data in enumerate(train_loader):
			if len(batch_data) == 3:
				images, _, label_vectors = batch_data  # Ignore tokenized_labels, use pre-encoded
			else:
				raise ValueError(f"Expected 3 items from DataLoader, got {len(batch_data)}")
			
			batch_size = images.size(0)
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			# Validate label_vectors shape
			if label_vectors.shape != (batch_size, num_classes):
				raise ValueError(f"Label vectors shape {label_vectors.shape} doesn't match expected ({batch_size}, {num_classes})")

			optimizer.zero_grad(set_to_none=True)
			
			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				# Multi-label contrastive loss computation
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion=criterion,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)
			
			# Check for NaN loss
			if torch.isnan(total_loss):
				print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
				continue
			
			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			# Track losses
			batch_loss_total = total_loss.item()
			batch_loss_i2t = loss_i2t.item()
			batch_loss_t2i = loss_t2i.item()
			
			epoch_loss_total += batch_loss_total
			epoch_loss_i2t += batch_loss_i2t
			epoch_loss_t2i += batch_loss_t2i
			num_batches += 1
			
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\t\tBatch [{bidx + 1}/{len(train_loader)}] "
					f"Total Loss: {batch_loss_total:.6f} "
					f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})"
				)
		
		# Calculate average losses
		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0
		
		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)

		print(f">> Validating Epoch {epoch+1} ...")
		
		# Compute validation loss using the same multi-label loss function
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			max_batches=10
		)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,  # Now uses BCEWithLogitsLoss
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=True,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
			},
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
		)
		
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
		in_batch_loss_acc_metrics_per_epoch["val_loss"] = current_val_loss  # Use computed validation loss
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		
		current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]

		print(
			f'@ Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}:\n'
			f'\t\tTraining - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f})\n'
			f'\t\tValidation: {current_val_loss:.6f}\n'
			f'\tMulti-label Validation Metrics:\n'
			f'\t\tIn-batch Top-K Accuracy:\n'
			f'\t\t\t[Image→Text]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\t\tFull Validation Set:\n'
			f'\t\t\t[Image→Text]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		
		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'\tMulti-label Metrics:')
			print(f'\t\tHamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'\t\tPartial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'\t\tF1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')
		
		print(f"\tRetrieval Metrics:")
		print(
			f"\t\tImage-to-Text: mAP@10={retrieval_metrics_per_epoch['img2txt'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['img2txt'].get('Recall', {}).get('10', 'N/A')}"
		)
		print(
			f"\t\tText-to-Image: mAP@10={retrieval_metrics_per_epoch['txt2img'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['txt2img'].get('Recall', {}).get('10', 'N/A')}"
		)

		if hasattr(train_loader.dataset, 'get_cache_stats'):
			print(f"#"*100)
			cache_stats = train_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Train Cache Stats: {cache_stats}")

		if hasattr(validation_loader.dataset, 'get_cache_stats'):
			cache_stats = validation_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Validation Cache Stats: {cache_stats}")
			print(f"#"*100)

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\nEarly stopping triggered at epoch {epoch + 1} "
				f"with best loss: {early_stopping.get_best_score()} "
				f"obtained in epoch {early_stopping.get_best_epoch()+1}")
			break

		print(f"Epoch {epoch+1} Duration [Train + Validation]: {time.time() - train_and_val_st_time:.2f} sec".center(150, "="))
	
	print(f"[{mode}] Total Elapsed_t: {time.time() - train_start_time:.1f} sec")

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=verbose,
		max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
	)

	# Access individual metrics
	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	model_source = evaluation_results["model_loaded_from"]

	if verbose:
		print(f"Final evaluation used model weights from: {model_source}")
		print("--- Final Metrics [In-batch Validation] ---")
		print(json.dumps(final_metrics_in_batch, indent=2, ensure_ascii=False))
		print("--- Final Metrics [Full Validation Set] ---")
		print(json.dumps(final_metrics_full, indent=2, ensure_ascii=False))
		print("--- Image-to-Text Retrieval ---")
		print(json.dumps(final_img2txt_metrics, indent=2, ensure_ascii=False))
		print("--- Text-to-Image Retrieval ---")
		print(json.dumps(final_txt2img_metrics, indent=2, ensure_ascii=False))

	print("\nGenerating result plots...")
	actual_trained_epochs = len(training_losses)

	file_base_name = (
		f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)
	
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
	)

	viz.plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
	)

	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)

	return final_metrics_in_batch, final_metrics_full, final_img2txt_metrics, final_txt2img_metrics

def lora_plus_finetune_multi_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		lora_rank: int,
		lora_alpha: float,
		lora_dropout: float,
		patience: int,
		min_delta: float,
		cumulative_delta: float,
		minimum_epochs: int,
		volatility_threshold: float,
		slope_threshold: float,
		pairwise_imp_threshold: float,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		quantization_bits: int = 8,
		lora_plus_lambda: float = 32.0,
		quantized: bool = False,
		use_lamb: bool = False,
		loss_weights: Dict[str, float] = None,
		temperature: float = 0.07,
		label_smoothing: float = 0.0,
		verbose: bool = True,
	):
	"""
	LoRA+ fine-tuning for multi-label CLIP classification.
	
	Key differences from single-label LoRA+ version:
	1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
	2. Handles bidirectional multi-label targets (I2T and T2I)
	3. Pre-encodes class embeddings for efficiency
	4. Uses multi-label specific loss computation
	5. Proper multi-label evaluation metrics
	6. Differential learning rates for LoRA A (base) and LoRA B (λ * base)
	
	Args:
		model: CLIP model to fine-tune with LoRA+
		train_loader: Training DataLoader (must provide multi-label vectors)
		validation_loader: Validation DataLoader  
		num_epochs: Number of training epochs
		print_every: Print loss every N batches
		learning_rate: Base learning rate for LoRA A parameters
		weight_decay: Weight decay for regularization
		device: Training device (cuda/cpu)
		results_dir: Directory to save results
		lora_rank: LoRA rank parameter
		lora_alpha: LoRA alpha parameter
		lora_dropout: LoRA dropout parameter
		patience: Early stopping patience
		min_delta: Minimum change for improvement
		cumulative_delta: Cumulative delta for early stopping
		minimum_epochs: Minimum epochs before early stopping
		volatility_threshold: Threshold for validation loss volatility
		slope_threshold: Threshold for validation loss slope
		pairwise_imp_threshold: Threshold for pairwise improvement
		topk_values: K values for evaluation metrics
		quantization_bits: Bits for quantization (4 or 8)
		lora_plus_lambda: Learning rate multiplier for LoRA B parameters
		quantized: Whether to use quantized base weights (QLoRA+)
		use_lamb: Use LAMB optimizer instead of AdamW
		loss_weights: Optional weights for I2T and T2I losses
		temperature: Temperature scaling for similarities
		label_smoothing: Label smoothing factor (0.0 = no smoothing)
		verbose: Enable detailed logging
	"""
	
	window_size = minimum_epochs + 1
	
	# Set default loss weights
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	# Check for non-zero dropout in the base model
	dropout_values = []
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))
	
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during LoRA+ fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for LoRA practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during LoRA+ fine-tuning "
			"which gives you more control over LoRA-specific regularization without affecting the base model.\n"
		)
	
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
	)
	
	# Dataset and model setup
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name
	
	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)
	
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	
	if verbose:
		print(f"{mode.upper()} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}")
		print(f"   ├─ Rank: {lora_rank}")
		print(f"   ├─ Alpha: {lora_alpha}")
		print(f"   ├─ Dropout: {lora_dropout}")
		print(f"   ├─ LoRA+ Learning Rate Multiplier (λ): {lora_plus_lambda}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")
		
		if torch.cuda.is_available():
			gpu_name = torch.cuda.get_device_name(device)
			gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
			cuda_capability = torch.cuda.get_device_capability()
			print(f"   ├─ {gpu_name} {device}")
			print(f"   ├─ Total Memory: {gpu_total_mem:.2f}GB")
			print(f"   └─ CUDA Capability: {cuda_capability}")
	
	# Get dataset information
	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)
	
	if verbose:
		print(f"Multi-label LoRA+ fine-tuning: {num_classes} classes")
	
	# Apply LoRA+ to the model
	model = get_injected_peft_clip(
		clip_model=model,
		method='lora',  # Use 'lora' method, we'll handle LoRA+ in optimizer
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		lora_plus_lambda=lora_plus_lambda,
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	)
	
	model.to(device)
	get_parameters_info(model=model, mode=mode)
	
	# Separate LoRA A and LoRA B parameters for differential learning rates
	lora_a_params = []
	lora_b_params = []
	for name, param in model.named_parameters():
		if param.requires_grad:
			if "lora_A" in name:
				lora_a_params.append(param)
			elif "lora_B" in name:
				lora_b_params.append(param)
			else:
				# Should not happen in pure LoRA+, but safe fallback
				lora_a_params.append(param)
	
	if verbose:
		print(f"LoRA+ Parameter Groups:")
		print(f"  ├─ lora_A params: {len(lora_a_params)} tensors")
		print(f"  ├─ lora_B params: {len(lora_b_params)} tensors")
		print(f"  └─ LR multiplier (λ): {lora_plus_lambda}")
	
	# Setup optimizer with differential learning rates
	loraA_lr = learning_rate
	loraB_lr = learning_rate * lora_plus_lambda
	
	if use_lamb:
		optimizer = LAMB(
			params=[
				{'params': lora_a_params, 'lr': loraA_lr},
				{'params': lora_b_params, 'lr': loraB_lr},
			],
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)
	else:
		optimizer = torch.optim.AdamW(
			params=[
				{'params': lora_a_params, 'lr': loraA_lr},
				{'params': lora_b_params, 'lr': loraB_lr},
			],
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)
	
	if verbose:
		print(f"{optimizer.__class__.__name__} optimizer")
		print(f"  ├─ LR: lora_A = {loraA_lr} lora_B = {loraB_lr}")
		print(f"  └─ Weight Decay: {weight_decay}")
	
	# Setup scheduler
	estimated_epochs = min(num_epochs, 15)
	total_training_steps = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=total_training_steps,
		eta_min=eta_min,
		last_epoch=-1,
	)
	
	if verbose:
		print(f"{scheduler.__class__.__name__} scheduler")
		print(f"  ├─ T_max = {total_training_steps} steps")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100:.1f}% of initial LR)")
	
	# Use BCEWithLogitsLoss for multi-label classification
	if label_smoothing > 0:
		if verbose:
			print(f"Using label smoothing: {label_smoothing}")
		criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
	else:
		criterion = torch.nn.BCEWithLogitsLoss()
	
	if verbose:
		print(f"Using {criterion.__class__.__name__} for multi-label classification")
	
	# Pre-encode all class texts for efficiency
	all_class_embeds = []
	model.eval()
	text_batch_size = validation_loader.batch_size
	if verbose:
		print(f"Pre-encoding {num_classes} class texts in batch_size: {text_batch_size}")
	
	with torch.no_grad():
		with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			for i in tqdm(range(0, num_classes, text_batch_size), desc="Pre-encoding class texts"):
				end_idx = min(i + text_batch_size, num_classes)
				batch_class_names = class_names[i:end_idx]
				
				batch_class_texts = clip.tokenize(batch_class_names).to(device)
				batch_embeds = model.encode_text(batch_class_texts)
				batch_embeds = F.normalize(batch_embeds, dim=-1)
				all_class_embeds.append(batch_embeds.cpu())
				
				del batch_class_texts, batch_embeds
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
	
	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device)
	if verbose:
		print(f"all_class_embeds: {type(all_class_embeds)} {all_class_embeds.shape} {all_class_embeds.dtype} {all_class_embeds.device}")
	
	scaler = torch.amp.GradScaler(device=device)
	
	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"loraA_lr_{loraA_lr:.1e}_"
		f"lmbd_{lora_plus_lambda}_"
		f"loraB_lr_{loraB_lr:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)
	
	training_losses = []
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	in_batch_loss_acc_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history = []
	train_start_time = time.time()
	
	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0
		
		for bidx, batch_data in enumerate(train_loader):
			if len(batch_data) == 3:
				images, _, label_vectors = batch_data
			else:
				raise ValueError(f"Expected 3 items from DataLoader, got {len(batch_data)}")
			
			batch_size = images.size(0)
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			# Validate label_vectors shape
			if label_vectors.shape != (batch_size, num_classes):
				raise ValueError(f"Label vectors shape {label_vectors.shape} doesn't match expected ({batch_size}, {num_classes})")
			
			optimizer.zero_grad(set_to_none=True)
			
			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				# Multi-label contrastive loss computation
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion=criterion,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)
			
			# Check for NaN loss
			if torch.isnan(total_loss):
				print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
				continue
			
			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			# Track losses
			batch_loss_total = total_loss.item()
			batch_loss_i2t = loss_i2t.item()
			batch_loss_t2i = loss_t2i.item()
			
			epoch_loss_total += batch_loss_total
			epoch_loss_i2t += batch_loss_i2t
			epoch_loss_t2i += batch_loss_t2i
			num_batches += 1
			
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\t\tBatch [{bidx + 1}/{len(train_loader)}] "
					f"Total Loss: {batch_loss_total:.6f} "
					f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})"
				)
		
		# Calculate average losses
		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0
		
		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)
		
		# Track learning rates (now we have multiple parameter groups)
		learning_rates_history.append([group['lr'] for group in optimizer.param_groups])
		weight_decays_history.append([group['weight_decay'] for group in optimizer.param_groups])
		
		if verbose and epoch == 0:
			print(f"[Epoch {epoch+1}] {len(learning_rates_history[-1])} LR groups: {learning_rates_history[-1]}")
		
		print(f">> Validating Epoch {epoch+1} ...")
		
		# Compute validation loss
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			all_class_embeds=all_class_embeds,
			temperature=temperature,
			max_batches=10
		)
		
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=True,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
				"lora_plus_lambda": lora_plus_lambda,
			},
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
		)
		
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
		in_batch_loss_acc_metrics_per_epoch["val_loss"] = current_val_loss
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}
		
		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		
		current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]
		
		print(
			f'Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}:\n'
			f'\t\tTraining - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f})\n'
			f'\t\tValidation: {current_val_loss:.6f}\n'
			f'\tMulti-label Validation Metrics:\n'
			f'\t\tIn-batch Top-K Accuracy:\n'
			f'\t\t\t[Image→Text]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\t\tFull Validation Set:\n'
			f'\t\t\t[Image→Text]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		
		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'\tMulti-label Metrics:')
			print(f'\t\tHamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'\t\tPartial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'\t\tF1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')
		
		print(f"\tRetrieval Metrics:")
		print(
			f"\t\tImage-to-Text: mAP@10={retrieval_metrics_per_epoch['img2txt'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['img2txt'].get('Recall', {}).get('10', 'N/A')}"
		)
		print(
			f"\t\tText-to-Image: mAP@10={retrieval_metrics_per_epoch['txt2img'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['txt2img'].get('Recall', {}).get('10', 'N/A')}"
		)
		
		if hasattr(train_loader.dataset, 'get_cache_stats'):
			print(f"#"*100)
			cache_stats = train_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Train Cache Stats: {cache_stats}")
		
		if hasattr(validation_loader.dataset, 'get_cache_stats'):
			cache_stats = validation_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Validation Cache Stats: {cache_stats}")
			print(f"#"*100)
		
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\nEarly stopping triggered at epoch {epoch + 1} "
				f"with best loss: {early_stopping.get_best_score()} "
				f"obtained in epoch {early_stopping.get_best_epoch()+1}")
			break
		
		print(f"Epoch {epoch+1} Duration [Train + Validation]: {time.time() - train_and_val_st_time:.2f} sec".center(150, "="))
	
	print(f"[{mode}] Total Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))
	
	# Final evaluation
	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=True,
		max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
	)
	
	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	model_source = evaluation_results["model_loaded_from"]
	
	print(f"Final evaluation used model weights from: {model_source}")
	print("--- Final Metrics [In-batch Validation] ---")
	print(json.dumps(final_metrics_in_batch, indent=2, ensure_ascii=False))
	print("--- Final Metrics [Full Validation Set] ---")
	print(json.dumps(final_metrics_full, indent=2, ensure_ascii=False))
	print("--- Image-to-Text Retrieval ---")
	print(json.dumps(final_img2txt_metrics, indent=2, ensure_ascii=False))
	print("--- Text-to-Image Retrieval ---")
	print(json.dumps(final_txt2img_metrics, indent=2, ensure_ascii=False))
	
	print("\nGenerating result plots...")
	actual_trained_epochs = len(training_losses)
	
	file_base_name = (
		f"{mode}_"
		f"{CLUSTER}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"loraA_lr_{loraA_lr:.1e}_"
		f"lmbd_{lora_plus_lambda}_"
		f"loraB_lr_{loraB_lr:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}"
	)
	
	mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)
	print(f"Best model will be renamed to: {mdl_fpth}")
	
	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}
	
	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
	)
	
	viz.plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
	)
	
	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)
	
	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)
	
	return final_metrics_in_batch, final_metrics_full, final_img2txt_metrics, final_txt2img_metrics

def ia3_finetune_multi_label():
	pass

def dora_finetune_multi_label():
	pass

def vera_finetune_multi_label():
	pass

def clip_adapter_finetune_multi_label():
	pass

def tip_adapter_finetune_multi_label():
	pass

def probe_finetune_multi_label(
		model: torch.nn.Module,
		train_loader,
		validation_loader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		verbose: bool = True,
		patience: int = 10,
		min_delta: float = 1e-4,
		cumulative_delta: float = 5e-3,
		minimum_epochs: int = 20,
		topk_values: List[int] = [1, 3, 5, 10, 15, 20],
		loss_weights: Dict[str, float] = None,
		temperature: float = 0.07,
		label_smoothing: float = 0.0,
		volatility_threshold: float = 15.0,
		slope_threshold: float = 1e-4,
		pairwise_imp_threshold: float = 1e-4,
		use_lamb: bool = False,
		probe_hidden_dim: int = None,  # Optional: add hidden layer
		probe_dropout: float = 0.1,
		cache_features: bool = True,  # Optional: cache features for efficiency
):
		"""
		Enhanced Linear probing fine-tuning for multi-label CLIP classification with robust ViT support.
		Automatically handles different ViT architectures and fixes positional embedding issues.
		"""

		window_size = minimum_epochs + 1
		# Set default loss weights
		if loss_weights is None:
				loss_weights = {"i2t": 0.5, "t2i": 0.5}

		early_stopping = EarlyStopping(
				patience=patience,
				min_delta=min_delta,
				cumulative_delta=cumulative_delta,
				window_size=window_size,
				mode='min',
				min_epochs=minimum_epochs,
				restore_best_weights=True,
				volatility_threshold=volatility_threshold,
				slope_threshold=slope_threshold,
				pairwise_imp_threshold=pairwise_imp_threshold,
				# min_phases_before_stopping=1,
		)

		try:
				dataset_name = validation_loader.dataset.dataset.__class__.__name__
		except AttributeError:
				dataset_name = validation_loader.dataset.dataset_name

		mode = inspect.stack()[0].function
		mode = re.sub(r'_finetune_multi_label', '', mode)
		model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
		model_name = model.__class__.__name__

		print(f"{mode} | {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
		if torch.cuda.is_available():
				gpu_name = torch.cuda.get_device_name(device)
				total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
				print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

		# Get dataset information
		try:
				num_classes = len(validation_loader.dataset.unique_labels)
				class_names = validation_loader.dataset.unique_labels
		except:
				num_classes = len(validation_loader.dataset.dataset.classes)
				class_names = validation_loader.dataset.dataset.classes
		print(f"Multi-label Linear Probe fine-tuning: {num_classes} classes")

		# =====================================
		# STEP 1: FREEZE ALL CLIP PARAMETERS AND CREATE ROBUST PROBE
		# =====================================
		for param in model.parameters():
				param.requires_grad = False

		print("\nCreating robust probe model...")
		probe = get_probe_clip(
			clip_model=model,
			validation_loader=validation_loader,
			device=torch.device(device),
			# hidden_dim=256,  # Optional: creates MLP probe
			dropout=probe_dropout,
			zero_shot_init=True, # faster convergence
			verbose=True
		)
		print(f"Multi-label dataset: {isinstance(probe, MultiLabelProbe)}")

		embed_dim = probe.input_dim  # Get the detected feature dimension
		probe_params = sum(p.numel() for p in probe.parameters())
		probe_type = probe.probe_type

		print(f"CLIP embedding dimension: {embed_dim}")
		print(f"Probe type: {probe_type} | Parameters: {probe_params:,}")

		# Use BCEWithLogitsLoss for multi-label classification
		if label_smoothing > 0:
				print(f"Using label smoothing: {label_smoothing}")
				criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
		else:
				criterion = torch.nn.BCEWithLogitsLoss()
		print(f"Using {criterion.__class__.__name__} for multi-label classification")

		# Pre-encode all class texts (for evaluation)
		print(f"Pre-encoding {num_classes} class texts...")
		all_class_texts = clip.tokenize(class_names).to(device)
		print(f"all_class_texts: {type(all_class_texts)} {all_class_texts.shape} {all_class_texts.dtype} {all_class_texts.device}")
		model.eval()
		with torch.no_grad():
			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				all_class_embeds = model.encode_text(all_class_texts)
				all_class_embeds = F.normalize(all_class_embeds, dim=-1)
		print(f"all_class_embeds: {type(all_class_embeds)} {all_class_embeds.shape} {all_class_embeds.dtype} {all_class_embeds.device}")

		# Optimizer setup
		if use_lamb:
				optimizer = LAMB(
						params=probe.parameters(),
						lr=learning_rate,
						betas=(0.9, 0.98),
						eps=1e-6,
						weight_decay=weight_decay,
				)
		else:
				optimizer = torch.optim.AdamW(
						params=probe.parameters(),
						lr=learning_rate,
						betas=(0.9, 0.98),
						eps=1e-6,
						weight_decay=weight_decay,
				)

		scheduler = torch.optim.lr_scheduler.OneCycleLR(
				optimizer=optimizer,
				max_lr=learning_rate,
				steps_per_epoch=len(train_loader),
				epochs=num_epochs,
				pct_start=0.1,
				anneal_strategy='cos',
		)

		scaler = torch.amp.GradScaler(device=device)

		mdl_fpth = os.path.join(
				results_dir,
				f"{mode}_"
				f"{model_arch}_"
				f"{optimizer.__class__.__name__}_"
				f"{scheduler.__class__.__name__}_"
				f"{criterion.__class__.__name__}_"
				f"probe_{probe_type}_"
				f"ieps_{num_epochs}_"
				f"lr_{learning_rate:.1e}_"
				f"wd_{weight_decay:.1e}_"
				f"temp_{temperature}_"
				f"bs_{train_loader.batch_size}_"
				f"mep_{minimum_epochs}_"
				f"pat_{patience}_"
				f"mdt_{min_delta:.1e}_"
				f"cdt_{cumulative_delta:.1e}_"
				f"vt_{volatility_threshold}_"
				f"st_{slope_threshold:.1e}_"
				f"pit_{pairwise_imp_threshold:.1e}"
				f".pth"
		)

		# Optional: Cache features for efficiency
		train_features_cache = None
		val_features_cache = None
		
		if cache_features:
				print("Pre-extracting features for efficient training...")
				# Extract training features
				train_features = list()
				train_labels = list()
				model.eval()
				with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
					for batch_data in tqdm(train_loader, desc="Extracting train features"):
						if len(batch_data) == 3:
							images, _, label_vectors = batch_data
						else:
							raise ValueError(f"Expected 3 items, got {len(batch_data)}")
						
						images = images.to(device, non_blocking=True)
						image_embeds = model.encode_image(images)
						image_embeds = F.normalize(image_embeds, dim=-1)
						
						train_features.append(image_embeds.cpu())
						train_labels.append(label_vectors.cpu())
				
				train_features_cache = (torch.cat(train_features, dim=0), torch.cat(train_labels, dim=0))
				
				# Extract validation features
				val_features = list()
				val_labels = list()
				with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
					for batch_data in tqdm(validation_loader, desc="Extracting val features"):
						if len(batch_data) == 3:
							images, _, label_vectors = batch_data
						else:
							raise ValueError(f"Expected 3 items, got {len(batch_data)}")
						
						images = images.to(device, non_blocking=True)
						image_embeds = model.encode_image(images)
						image_embeds = F.normalize(image_embeds, dim=-1)
						
						val_features.append(image_embeds.cpu())
						val_labels.append(label_vectors.cpu())
				
				val_features_cache = (torch.cat(val_features, dim=0), torch.cat(val_labels, dim=0))
				
				print(f"Cached features - Train: {train_features_cache[0].shape}, Val: {val_features_cache[0].shape}")
				
				# Create feature dataloaders
				from torch.utils.data import TensorDataset
				train_feature_dataset = TensorDataset(train_features_cache[0], train_features_cache[1])
				val_feature_dataset = TensorDataset(val_features_cache[0], val_features_cache[1])
				
				train_feature_loader = DataLoader(
						train_feature_dataset,
						batch_size=train_loader.batch_size,
						shuffle=True,
						num_workers=0
				)
				val_feature_loader = DataLoader(
						val_feature_dataset,
						batch_size=validation_loader.batch_size,
						shuffle=False,
						num_workers=0
				)

		training_losses = list()
		training_losses_breakdown = {"total": []}
		img2txt_metrics_all_epochs = list()
		txt2img_metrics_all_epochs = list()
		in_batch_loss_acc_metrics_all_epochs = list()
		full_val_loss_acc_metrics_all_epochs = list()
		train_start_time = time.time()

		for epoch in range(num_epochs):
				train_and_val_st_time = time.time()
				torch.cuda.empty_cache()
				probe.train()

				print(f"Epoch [{epoch + 1}/{num_epochs}]")
				
				epoch_loss_total = 0.0
				num_batches = 0
				
				# Choose data source
				data_loader = train_feature_loader if cache_features else train_loader
				
				for bidx, batch_data in enumerate(data_loader):
						if cache_features:
								# Using cached features
								image_embeds, label_vectors = batch_data
								image_embeds = image_embeds.to(device, non_blocking=True)
								label_vectors = label_vectors.to(device, non_blocking=True).float()
						else:
								# Extract features on-the-fly
								if len(batch_data) == 3:
										images, _, label_vectors = batch_data
								else:
										raise ValueError(f"Expected 3 items, got {len(batch_data)}")
								
								images = images.to(device, non_blocking=True)
								label_vectors = label_vectors.to(device, non_blocking=True).float()
								
								# Extract image embeddings (frozen)
								with torch.no_grad():
										image_embeds = model.encode_image(images)
										image_embeds = F.normalize(image_embeds, dim=-1)
						
						optimizer.zero_grad(set_to_none=True)
						
						with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
								# Linear probe forward (multi-label logits)
								logits = probe(image_embeds)
								
								# Multi-label loss
								loss = criterion(logits, label_vectors)
						
						# Check for NaN loss
						if torch.isnan(loss):
								print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
								continue
						
						scaler.scale(loss).backward()
						scaler.unscale_(optimizer)
						torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
						scaler.step(optimizer)
						scaler.update()
						scheduler.step()
						
						# Track losses
						batch_loss_total = loss.item()
						epoch_loss_total += batch_loss_total
						num_batches += 1
						
						if bidx % print_every == 0 or bidx + 1 == len(data_loader):
								print(f"\t\tBatch [{bidx + 1}/{len(data_loader)}] Loss: {batch_loss_total:.6f}")
				
				# Calculate average losses
				avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
				training_losses.append(avg_total_loss)
				training_losses_breakdown["total"].append(avg_total_loss)

				print(f">> Validating Epoch {epoch+1} ...")
				
				# Validation with probe
				probe.eval()
				val_loss = 0.0
				val_preds = list()
				val_labels_list = list()
				
				with torch.no_grad():
						data_loader = val_feature_loader if cache_features else validation_loader
						
						for batch_data in data_loader:
								if cache_features:
										image_embeds, label_vectors = batch_data
										image_embeds = image_embeds.to(device, non_blocking=True)
										label_vectors = label_vectors.to(device, non_blocking=True).float()
								else:
										if len(batch_data) == 3:
												images, _, label_vectors = batch_data
										else:
												raise ValueError(f"Expected 3 items, got {len(batch_data)}")
										
										images = images.to(device, non_blocking=True)
										label_vectors = label_vectors.to(device, non_blocking=True).float()
										
										# Extract features
										image_embeds = model.encode_image(images)
										image_embeds = F.normalize(image_embeds, dim=-1)
								
								# Get predictions from probe
								logits = probe(image_embeds)
								loss = criterion(logits, label_vectors)
								val_loss += loss.item()
								
								# Store predictions for metrics
								probs = torch.sigmoid(logits)
								preds = (probs > 0.5).float()
								val_preds.append(preds.cpu())
								val_labels_list.append(label_vectors.cpu())
				
				avg_val_loss = val_loss / len(data_loader)
				
				# Calculate multi-label metrics
				val_preds = torch.cat(val_preds, dim=0)
				val_labels = torch.cat(val_labels_list, dim=0)
				
				hamming = hamming_loss(val_labels.numpy(), val_preds.numpy())
				f1 = f1_score(val_labels.numpy(), val_preds.numpy(), average='weighted', zero_division=0)
				exact_match = (val_preds == val_labels).all(dim=1).float().mean().item()
				partial_match = (val_preds == val_labels).float().mean().item()
				
				print(f"Validation - Loss: {avg_val_loss:.6f}, Hamming: {hamming:.4f}, F1: {f1:.4f}, Exact Match: {exact_match:.4f}")
				
				# Create metrics for compatibility
				current_val_loss = avg_val_loss
				
				# Simple in-batch metrics
				in_batch_metrics = {
						"val_loss": avg_val_loss,
						"hamming_loss": hamming,
						"f1_score": f1,
						"exact_match_acc": exact_match,
						"partial_acc": partial_match,
				}
				in_batch_loss_acc_metrics_all_epochs.append(in_batch_metrics)
				full_val_loss_acc_metrics_all_epochs.append(in_batch_metrics)

				if early_stopping.should_stop(
						current_value=current_val_loss,
						model=probe,  # Save probe weights
						epoch=epoch,
						optimizer=optimizer,
						scheduler=scheduler,
						checkpoint_path=mdl_fpth,
				):
						print(f"\nEarly stopping at epoch {epoch + 1}")
						break

				print(f"Epoch {epoch+1} Duration: {time.time() - train_and_val_st_time:.2f} sec".center(150, "="))
		
		print(f"[{mode}] Total Time: {time.time() - train_start_time:.1f} sec".center(170, "-"))

		# Load best probe weights
		if os.path.exists(mdl_fpth):
				print(f"Loading best probe weights from {mdl_fpth}")
				checkpoint = torch.load(mdl_fpth, map_location=device)
				if 'model_state_dict' in checkpoint:
						probe.load_state_dict(checkpoint['model_state_dict'])
				else:
						probe.load_state_dict(checkpoint)

		# Final evaluation
		print("\nFinal Evaluation:")
		probe.eval()
		final_preds = list()
		final_labels = list()
		
		with torch.no_grad():
				data_loader = val_feature_loader if cache_features else validation_loader
				
				for batch_data in data_loader:
						if cache_features:
								image_embeds, label_vectors = batch_data
								image_embeds = image_embeds.to(device, non_blocking=True)
								label_vectors = label_vectors.to(device, non_blocking=True).float()
						else:
								if len(batch_data) == 3:
										images, _, label_vectors = batch_data
								else:
										raise ValueError(f"Expected 3 items, got {len(batch_data)}")
								
								images = images.to(device, non_blocking=True)
								label_vectors = label_vectors.to(device, non_blocking=True).float()
								
								image_embeds = model.encode_image(images)
								image_embeds = F.normalize(image_embeds, dim=-1)
						
						logits = probe(image_embeds)
						probs = torch.sigmoid(logits)
						preds = (probs > 0.5).float()
						
						final_preds.append(preds.cpu())
						final_labels.append(label_vectors.cpu())
		
		final_preds = torch.cat(final_preds, dim=0)
		final_labels = torch.cat(final_labels, dim=0)
		
		# Final metrics
		final_hamming = hamming_loss(final_labels.numpy(), final_preds.numpy())
		final_f1 = f1_score(final_labels.numpy(), final_preds.numpy(), average='weighted', zero_division=0)
		final_exact = (final_preds == final_labels).all(dim=1).float().mean().item()
		final_partial = (final_preds == final_labels).float().mean().item()
		best_val_loss = early_stopping.get_best_score() or 0.0

		print("\n" + "="*80)
		print("ENHANCED LINEAR PROBE MULTI-LABEL TRAINING SUMMARY")
		print("="*80)
		print(f"Method: {mode}")
		print(f"Model: {getattr(model, 'name', 'Unknown')}")
		print(f"Probe Type: {probe_type}")
		print(f"Probe Parameters: {probe_params:,}")
		print(f"CLIP Parameters (frozen): {sum(p.numel() for p in model.parameters()):,}")
		print(f"Total Epochs: {len(training_losses)}")
		print(f"Best Val Loss: {best_val_loss}")
		print(f"Best Epoch: {early_stopping.get_best_epoch() + 1}")
		print("-"*80)
		print("Final Metrics:")
		print(f"  Hamming Loss: {final_hamming:.4f}")
		print(f"  F1 Score: {final_f1:.4f}")
		print(f"  Exact Match: {final_exact:.4f}")
		print(f"  Partial Match: {final_partial:.4f}")
		print("="*80)

		evaluation_results = evaluate_best_model(
				model=model,
				validation_loader=validation_loader,
				criterion=criterion,
				early_stopping=early_stopping,
				checkpoint_path=mdl_fpth,
				finetune_strategy=mode,
				device=device,
				cache_dir=results_dir,
				topk_values=topk_values,
				verbose=verbose,
				max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
		)

		# Access individual metrics
		final_metrics_in_batch = evaluation_results["in_batch_metrics"]
		final_metrics_full = evaluation_results["full_metrics"]
		final_img2txt_metrics = evaluation_results["img2txt_metrics"]
		final_txt2img_metrics = evaluation_results["txt2img_metrics"]

		if verbose:
				print(f"Final evaluation used model weights from: {evaluation_results['model_loaded_from']}")
				print("--- Final Metrics [In-batch Validation] ---")
				print(json.dumps(final_metrics_in_batch, indent=2, ensure_ascii=False))
				print("--- Final Metrics [Full Validation Set] ---")
				print(json.dumps(final_metrics_full, indent=2, ensure_ascii=False))
				print("--- Image-to-Text Retrieval ---")
				print(json.dumps(final_img2txt_metrics, indent=2, ensure_ascii=False))
				print("--- Text-to-Image Retrieval ---")
				print(json.dumps(final_txt2img_metrics, indent=2, ensure_ascii=False))

		print("\nGenerating result plots...")
		actual_trained_epochs = len(training_losses)

		file_base_name = (
			f"{dataset_name}_"
			f"{mode}_"
			f"{CLUSTER}_"
			f"{optimizer.__class__.__name__}_"
			f"{scheduler.__class__.__name__}_"
			f"{criterion.__class__.__name__}_"
			f"{scaler.__class__.__name__}_"
			f"{model_name}_"
			f"{model_arch}_"
			f"ep_{actual_trained_epochs}_"
			f"lr_{learning_rate:.1e}_"
			f"wd_{weight_decay:.1e}_"
			f"temp_{temperature}_"
			f"bs_{train_loader.batch_size}"
		)
		
		# Update model path
		mdl_fpth = get_updated_model_name(
				original_path=mdl_fpth, 
				actual_epochs=actual_trained_epochs
		)
		
		print(f"Model renamed to: {mdl_fpth}")

		# Plotting
		plot_paths = {
				"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
				"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
				"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
				"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
				"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
				"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
				"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
				"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		}

		viz.plot_multilabel_loss_breakdown(
				training_losses_breakdown=training_losses_breakdown,
				filepath=plot_paths["losses_breakdown"]
		)

		viz.plot_loss_accuracy_metrics(
				dataset_name=dataset_name,
				train_losses=training_losses,
				val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
				in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
				in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
				full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
				full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
				losses_file_path=plot_paths["losses"],
				in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
				in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
				full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
				full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
		)

		viz.plot_retrieval_metrics_per_epoch(
				dataset_name=dataset_name,
				image_to_text_metrics_list=img2txt_metrics_all_epochs,
				text_to_image_metrics_list=txt2img_metrics_all_epochs,
				fname=plot_paths["retrieval_per_epoch"],
		)

		viz.plot_retrieval_metrics_best_model(
				dataset_name=dataset_name,
				image_to_text_metrics=final_img2txt_metrics,
				text_to_image_metrics=final_txt2img_metrics,
				fname=plot_paths["retrieval_best"],
		)

		return in_batch_loss_acc_metrics_all_epochs

def compute_multilabel_contrastive_loss(
		model: torch.nn.Module,
		images: torch.Tensor,
		all_class_embeds: torch.Tensor,
		label_vectors: torch.Tensor,
		criterion: torch.nn.Module,
		temperature: float,
		loss_weights: Dict[str, float] = None,
		verbose: bool = False,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Compute bidirectional multi-label contrastive loss.
	
	Args:
		model: CLIP model
		images: [batch_size, 3, 224, 224]
		all_class_embeds: [num_classes, embed_dim] - pre-computed text embeddings
		label_vectors: [batch_size, num_classes] - binary label matrix
		criterion: Loss function (BCEWithLogitsLoss)
		temperature: Temperature scaling for similarities
		loss_weights: Weights for I2T and T2I losses
		verbose: Print debug info
	Returns:
		Tuple of (total_loss, i2t_loss, t2i_loss)
	"""
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	batch_size, num_classes = label_vectors.shape
	if verbose:
		print(f"batch_size: {batch_size}, num_classes: {num_classes}")

	# Encode images
	image_embeds = model.encode_image(images)  # [batch_size, embed_dim]
	image_embeds = F.normalize(image_embeds, dim=-1)
	if verbose:
		print(f"image_embeds: {image_embeds.shape} {image_embeds.dtype} {image_embeds.device}")
	
	all_class_embeds = F.normalize(all_class_embeds, dim=-1)
	if verbose:
		print(f"all_class_embeds: {all_class_embeds.shape} {all_class_embeds.dtype} {all_class_embeds.device}")

	# ================================
	# Image-to-Text Loss
	# ================================
	# Compute similarity matrix: [batch_size, num_classes]
	i2t_similarities = torch.matmul(image_embeds, all_class_embeds.T) / temperature
	
	if verbose:
		print(f"i2t_similarities: {i2t_similarities.shape} {i2t_similarities.dtype} {i2t_similarities.device}")

	# I2T targets: label_vectors directly [batch_size, num_classes]
	i2t_targets = label_vectors.float()
	
	# Compute I2T loss
	loss_i2t = criterion(i2t_similarities, i2t_targets)
	if verbose:
		print(f"loss_i2t: {loss_i2t.item()}")
	
	# ================================
	# Text-to-Image Loss  
	# ================================
	# Compute similarity matrix: [num_classes, batch_size]
	t2i_similarities = torch.matmul(all_class_embeds, image_embeds.T) / temperature

	if verbose:
		print(f"t2i_similarities: {t2i_similarities.shape} {t2i_similarities.dtype} {t2i_similarities.device}")
	
	# T2I targets: transpose of label_vectors [num_classes, batch_size]
	t2i_targets = label_vectors.T.float()
	
	# Compute T2I loss
	loss_t2i = criterion(t2i_similarities, t2i_targets)
	if verbose:
		print(f"loss_t2i: {loss_t2i.item()}")
	
	# ================================
	# Combine losses
	# ================================
	total_loss = loss_weights["i2t"] * loss_i2t + loss_weights["t2i"] * loss_t2i
	if verbose:
		print(f"total_loss: {total_loss.item()}")
	
	return total_loss, loss_i2t, loss_t2i

class LabelSmoothingBCELoss(torch.nn.Module):
	def __init__(self, smoothing: float = 0.1):
		super().__init__()
		self.smoothing = smoothing
			
	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			logits: [batch_size, num_classes] - raw logits
			targets: [batch_size, num_classes] - binary targets (0 or 1)
		"""
		# Apply label smoothing
		# Positive labels: 1 -> (1 - smoothing)
		# Negative labels: 0 -> smoothing
		smooth_targets = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
		
		# Apply BCE loss with logits
		loss = F.binary_cross_entropy_with_logits(logits, smooth_targets)
		return loss