from utils import *
from early_stopper import EarlyStopping
from loss_analyzer import LossAnalyzer
from peft import get_injected_peft_clip, get_adapter_peft_clip
from evals import *
from apft import *
import visualize as viz
from model import LAMB

def progressive_finetune_single_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		patience: int,															# patience for epochs without improvement
		min_delta: float,														# Make slightly less sensitive than default
		cumulative_delta: float,										# Keep cumulative check reasonable
		minimum_epochs: int,												# Minimum epochs before ANY early stop
		min_epochs_per_phase: int,									# Minimum epochs within a phase before transition check
		volatility_threshold: float,								# Allow slightly more volatility
		slope_threshold: float, 										# Allow very slightly positive slope before stopping/transitioning
		pairwise_imp_threshold: float,							# Stricter requirement for pairwise improvement
		min_phases_before_stopping: int,						# Ensure significant unfreezing before global stop
		total_num_phases: int,
		accuracy_plateau_threshold: float = 5e-4,		# For phase transition based on accuracy
		topk_values: list[int]=None,
		layer_groups_to_unfreeze: list[str]=None,
		use_lamb: bool=False,
		verbose: bool=False,
	):
	if layer_groups_to_unfreeze is None:
		layer_groups_to_unfreeze = ['visual_transformer', 'text_transformer', 'projections'] # key layers
	if topk_values is None:
		topk_values = [1, 5, 10, 15, 20]

	window_size = min_epochs_per_phase + 3
	estimated_epochs_per_phase = min_epochs_per_phase * 3
	
	initial_learning_rate = learning_rate
	initial_weight_decay = weight_decay

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold, # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		min_phases_before_stopping=min_phases_before_stopping,
	)
	early_stopping_triggered = False

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	total_model_params = sum(p.numel() for p in model.parameters())

	print(f"{mode} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		print(f"{CLUSTER} | {HOST} | {gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	if USER == "farid":
		build_arch_flowchart(
			model,
			filename=os.path.join(results_dir, f"{model_arch}_flowchart"),
			format="png",      # you can also use "svg" for a vector image
			view=False,         # opens the image automatically (optional)
			rankdir="LR"       # top‑to‑bottom (feel free to try "LR")
		)

	if verbose:
		for n, m in model.named_modules():
			print(f"{n:<60} {type(m).__name__:<50} Training: {m.training:<10} Weights Grad: {m.weight.requires_grad if hasattr(m, 'weight') else ''}")
		print("-"*100)

	# Find dropout value
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break

	# Inspect the model for dropout layers
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		print(f"Non-zero dropout detected in {model_name} {model_arch}:")
		for i, v in enumerate(non_zero_dropouts):
			print(f"{i:02d} {v[0]:<60}p={v[1]}")
		print("-"*100)
	else:
		print(f"No non-zero dropout detected in {model_name} {model_arch}")

	unfreeze_schedule = get_unfreeze_schedule(
		model=model,
		num_phases=total_num_phases,
		layer_groups_to_unfreeze=layer_groups_to_unfreeze,
	)

	layer_cache = {} # Cache for layer status (optional, used by get_status)

	# unfreeze layers for Phase 0 to correctly initialize the optimizer
	unfreeze_layers(
		model=model,
		strategy=unfreeze_schedule,
		phase=0,
		cache=layer_cache,
	)
	
	optimizer = None
	scheduler = None

	criterion = torch.nn.CrossEntropyLoss()
	print(f"Using {criterion.__class__.__name__} as the loss function")

	scaler = torch.amp.GradScaler(device=device) # For mixed precision
	print(f"Using {scaler.__class__.__name__} for mixed precision training")

	mdl_fpth = os.path.join(
		results_dir,
		# f"{dataset_name}_"
		f"{mode}_"
		f"{model_arch}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"do_{dropout_val}_"
		f"ilr_{initial_learning_rate:.1e}_"
		f"iwd_{initial_weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}_"
		f"mepph_{min_epochs_per_phase}_"
		f"mphb4stp_{min_phases_before_stopping}_"
		f"tph_{total_num_phases}"
		f".pth"
	)

	# Embedding drift: get a fixed batch of validation data and original embeddings
	val_subset_loader = DataLoader(
		validation_loader.dataset, 
		batch_size=validation_loader.batch_size,
		shuffle=False
	)

	fixed_val_batch = next(iter(val_subset_loader))

	with torch.no_grad():
		model.eval()
		initial_images, _, _ = fixed_val_batch
		initial_images = initial_images.to(device)
		pretrained_embeds = model.encode_image(initial_images)
		pretrained_embeds = F.normalize(pretrained_embeds, dim=-1)
	
	current_phase = 0
	epochs_in_current_phase = 0
	training_losses = list()
	validation_losses = list()
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list() # History of [in-batch] validation metrics dicts per epoch
	full_val_loss_acc_metrics_all_epochs = list() # History of [full] validation metrics dicts per epoch
	
	planned_next_lr = initial_learning_rate # What we PLAN to use in next phase initialization
	planned_next_wd = initial_weight_decay # What we PLAN to use in next phase initialization

	learning_rates_history = list()
	weight_decays_history = list()
	phases_history = list()
	trainable_params_per_phase = list()
	phase_transitions_epochs = list()
	embedding_drift_history = list()
	batches_per_epoch = len(train_loader)

	train_start_time = time.time()
	print(f"Training: {num_epochs} epochs | {total_num_phases} phases | {min_epochs_per_phase} minimum epochs per phase".center(170, "-"))

	for epoch in range(num_epochs):
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		epoch_start_time = time.time()

		unfreeze_layers(
			model=model,
			strategy=unfreeze_schedule,
			phase=current_phase,
			cache=layer_cache,
		)

		if epochs_in_current_phase == 0:
			# 1. Create optimizer for the current phase
			trainable_params = [p for p in model.parameters() if p.requires_grad]
			
			if not trainable_params:
				raise ValueError("No trainable parameters found!")
			
			optimizer = torch.optim.AdamW(
				trainable_params,
				lr=planned_next_lr, 					# Use planned LR as starting point
				weight_decay=planned_next_wd,	# Use planned WD as starting point
				betas=(0.9, 0.98),
				eps=1e-6
			)

			# 2. Configure the main scheduler to take over *after* the warm-up
			# minimum LR to be X% of what PLANNED LR is for this phase: planned_next_lr * X 
			if current_phase >= 3:
				ANNEALING_RATIO = 1e-2
			elif current_phase >= 2:
				ANNEALING_RATIO = 5e-2
			else:
				ANNEALING_RATIO = 1e-1
			eta_min = planned_next_lr * ANNEALING_RATIO			

			# 3. Estimate the number of training steps for the current phase
			total_training_steps = estimated_epochs_per_phase * batches_per_epoch
			trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
			trainable_params_percent = trainable_params/total_model_params*100
			trainable_params_per_phase.append(trainable_params_percent)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
				optimizer=optimizer,
				T_max=total_training_steps,
				eta_min=eta_min,
				last_epoch=-1
			)
			print(f"Phase {current_phase} scheduler Annealing ratio: {ANNEALING_RATIO}")
			print(f"  ├─ T_max = {total_training_steps} steps [({(min_epochs_per_phase)} x 3) estimated epochs in phase x {batches_per_epoch} batches/epoch]")
			print(f"  ├─ LR PLANNED: {planned_next_lr} annealing to eta_min: {eta_min}")
			print(f"  ├─ Trainable params: {trainable_params:,} / {total_model_params:,} ({trainable_params/total_model_params*100:.3f}%)")
			print(f"  └─ Main scheduler ({scheduler.__class__.__name__}) configured.")

		model.train()
		epoch_train_loss = 0.0
		num_train_batches = len(train_loader)
		trainable_params_exist = any(p.requires_grad for p in model.parameters())

		for bidx, batch_data in enumerate(train_loader):
			images, tokenized_labels, _ = batch_data # Adjust unpacking as needed
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)
			optimizer.zero_grad(set_to_none=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				batch_loss = 0.5 * (loss_img + loss_txt)

			scheduler.step()

			scaler.scale(batch_loss).backward()
			scaler.unscale_(optimizer) # Unscale before clipping
			torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()

			batch_loss_item = batch_loss.item()
			epoch_train_loss += batch_loss_item

			if bidx % print_every == 0:
				print(f"Batch [{bidx+1}/{num_train_batches}] Loss: {batch_loss_item}")
			elif bidx == num_train_batches - 1 and batch_loss_item > 0:
				print(f"Batch [{bidx+1}/{num_train_batches}] Loss: {batch_loss_item}")
			else:
				pass

		avg_training_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 and trainable_params_exist else 0.0
		training_losses.append(avg_training_loss)
		if hasattr(early_stopping, "train_loss_history"):
			early_stopping.train_loss_history.append(avg_training_loss)

		current_lr = optimizer.param_groups[0]['lr']
		current_wd = optimizer.param_groups[0]['weight_decay']

		learning_rates_history.append(current_lr)    # Record what was ACTUALLY used
		weight_decays_history.append(current_wd)     # Record what was ACTUALLY used
		phases_history.append(current_phase)         # Record current phase

		drift_value = compute_embedding_drift(
			model, 
			fixed_val_batch, # Pass the fixed batch
			pretrained_embeds, 
			device, 
			current_phase, 
			epoch + 1
		)
		embedding_drift_history.append(drift_value)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=False,
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
		)

		if epoch == 0:
			print("="*60)
			print(f"DEBUG: After epoch 0, learning_rates length: {len(learning_rates_history)}")
			print(f"DEBUG: After epoch 0, weight_decays length: {len(weight_decays_history)}")
			print(f"DEBUG: LR: {learning_rates_history}")
			print(f"DEBUG: WD: {weight_decays_history}")
			print("="*60)

		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
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
		validation_losses.append(current_val_loss)

		print(f"\nEpoch {epoch+1:3d} Phase: {current_phase} current_epochs_in_phase: {epochs_in_current_phase}")
		print(f'\t[LOSS] Train: {avg_training_loss} Val(in-batch): {current_val_loss}')
		print(f"\t[Hyperparameters]")
		print(f"\t\t[ACTUAL] current_lr: {current_lr} current_wd: {current_wd}")
		print(f"\t\t[PLANNED] planned_next_lr: {planned_next_lr} planned_next_wd: {planned_next_wd}")
		# print(
		# 	f'\tValidation Top-k Accuracy:\n'
		# 	f'\tIn-batch:\n'
		# 	f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
		# 	f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
		# 	f'\tFull Validation Set:\n'
		# 	f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
		# 	f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		# )

		# print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
		# print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

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

		print(f"{len(validation_losses)} validation losses: {validation_losses}")
		print(f"early stopping validation loss history: {len(early_stopping.value_history)}: {early_stopping.value_history}")

		should_trans = should_transition_to_next_phase(
			current_phase=current_phase,
			losses=validation_losses,
			window=window_size,
			epochs_in_phase=epochs_in_current_phase,
			min_epochs_per_phase=min_epochs_per_phase,
			num_phases=total_num_phases,
			volatility_threshold=volatility_threshold,
		)
		if should_trans:
			phase_transitions_epochs.append(epoch+1)
			current_phase, planned_next_lr, planned_next_wd = handle_phase_transition(current_phase, optimizer)
			print(f"Transition to Phase {current_phase} @ epoch {epoch+1}: New planned LR: {planned_next_lr}, New WD: {planned_next_wd}")
			epochs_in_current_phase = 0 # Reset phase epoch counter
			early_stopping.reset() # <<< CRITICAL: Reset early stopping state for the new phase
			validation_losses = list() # Reset validation losses for the new phase
		else:
			epochs_in_current_phase += 1

		training_should_stop = early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
			current_phase=current_phase
		)
		if training_should_stop:
			print(f"EarlyStopping Status:\n{json.dumps(early_stopping.get_status(), indent=2, ensure_ascii=False)}")
			early_stopping_triggered = True
			print(f"--- Training stopped early at epoch {epoch+1} ---")
			break
		print(f"Epoch {epoch+1} total elapsed time: {time.time() - epoch_start_time:.1f} sec.".center(170, " "))

	# --- End of Training ---
	total_training_time = time.time() - train_start_time
	print(f"Training Finished Total Epochs Run: {epoch + 1} Elapsed_t: {total_training_time:.1f} sec".center(170, "-"))
	print(f"Final Phase Reached: {current_phase}")
	print(f"Best Validation Loss Achieved: {early_stopping.get_best_score()} @ Epoch {early_stopping.get_best_epoch() + 1}")

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
		# f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"ilr_{initial_learning_rate:.1e}_"
		f"iwd_{initial_weight_decay:.1e}_"
		f"ep_{actual_trained_epochs}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}_"
		f"mep_{minimum_epochs}_"
		f"mepph_{min_epochs_per_phase}_"
		f"mphb4stp_{min_phases_before_stopping}_"
		f"vt_{volatility_threshold}_"
		f"slp_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"tph_{total_num_phases}_"
		f"fph_{current_phase}"
	)

	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth,
		actual_epochs=actual_trained_epochs,
		additional_info={
			'fph': current_phase,
			'flr': planned_next_lr,
			'fwd': planned_next_wd
		}
	)

	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retr_perEP.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retr_perK.png"),
		"progressive_dynamics": os.path.join(results_dir, f"{file_base_name}_dyn.png"),
		"phase_analysis": os.path.join(results_dir, f"{file_base_name}_ph_anls.png"),
		"unfreeze_heatmap": os.path.join(results_dir, f"{file_base_name}_unf_hmap.png"),
		"training_summary": os.path.join(results_dir, f"{file_base_name}_train_summary.txt"),
		"loss_evolution": os.path.join(results_dir, f"{file_base_name}_loss_evol.png"),
		"lr_evolution": os.path.join(results_dir, f"{file_base_name}_lr_evol.png"),
		"wd_evolution": os.path.join(results_dir, f"{file_base_name}_wd_evol.png"),
		"phase_efficiency": os.path.join(results_dir, f"{file_base_name}_ph_eff.png"),
		"hyperparameter_correlation": os.path.join(results_dir, f"{file_base_name}_hypp_corr.png"),
		"trainable_layers": os.path.join(results_dir, f"{file_base_name}_train_lyrs.png"),
		"grad_norm": os.path.join(results_dir, f"{file_base_name}_grad_norm.png"),
		"loss_volatility": os.path.join(results_dir, f"{file_base_name}_loss_vol.png"),
		"loss_analyzer": os.path.join(results_dir, f"{file_base_name}_loss_analyzer.png"),
	}

	training_history = collect_progressive_training_history(
		training_losses=training_losses,
		in_batch_metrics_all_epochs=in_batch_loss_acc_metrics_all_epochs,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		phases=phases_history,
		trainable_params_per_phase=trainable_params_per_phase,
		total_model_params=total_model_params,
		embedding_drifts=embedding_drift_history,
		phase_transitions=phase_transitions_epochs,
		early_stop_epoch=epoch+1 if early_stopping_triggered else None,
		best_epoch=early_stopping.best_epoch if hasattr(early_stopping, 'best_epoch') else None
	)

	plot_phase_transition_analysis(
		training_history=training_history,
		file_path=plot_paths["phase_analysis"],
	)

	print(phases_history, total_num_phases)
	plot_unfreeze_heatmap(
		unfreeze_schedule=unfreeze_schedule,
		layer_groups=get_layer_groups(model),
		max_phase=total_num_phases,
		fname=plot_paths["unfreeze_heatmap"],
		# auto_trim=False,
		used_phases=max(phases_history),
	)

	analyzer = LossAnalyzer(
		epochs=training_history['epochs'], 
		train_loss=training_history['train_losses'], 
		val_loss=training_history['val_losses']
	)
	analyzer.plot_analysis(fpth=plot_paths["loss_analyzer"])
	signals = analyzer.get_training_signals()
	print(f"\nEMA signal summary: {signals}\n")

	plot_phase_transition_analysis_individual(
		training_history=training_history,
		file_path=plot_paths["phase_analysis"],
	)

	plot_loss_accuracy_metrics(
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

	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)

	return in_batch_loss_acc_metrics_all_epochs

def full_finetune_single_label(
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
		use_lamb: bool = False,
	):
	window_size = minimum_epochs + 1
	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold, # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__  # CIFAR10, ImageNet, etc.
	except AttributeError as e:
		dataset_name = validation_loader.dataset.dataset_name

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	print(f"{mode} {model_name} {model_arch} {dataset_name} {num_epochs} Epoch(s) | batch_size: {train_loader.batch_size} | {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	# Extract dropout value from the model (if any)
	dropout_val = None
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break

	if dropout_val is None:
		dropout_val = 0.0  # Default to 0.0 if no Dropout layers are found

	# Inspect the model for dropout layers
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	print(f"Non-zero dropout detected in base {model_name} {model_arch} during {mode}:")
	print(non_zero_dropouts)

	# Unfreeze all layers for full fine-tuning:
	for name, param in model.named_parameters():
		param.requires_grad = True # all parameters are trainable

	get_parameters_info(model=model, mode=mode)

	if use_lamb:
		param_names = {id(p): n for n, p in model.named_parameters()}
		optimizer = LAMB(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			weight_decay=weight_decay,
		)
		optimizer.param_names = param_names
	else:
		optimizer = torch.optim.AdamW(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			weight_decay=weight_decay,
			betas=(0.9, 0.98),
			eps=1e-6,
		)

	estimated_epochs = min(num_epochs, 15)
	total_training_steps = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=total_training_steps,
		eta_min=eta_min,
		last_epoch=-1,
	)
	print(f"{scheduler.__class__.__name__} scheduler configured")
	print(f"  ├─ T_max = {total_training_steps} steps [({min(num_epochs, 15)} estimated epochs x {len(train_loader)} batches/epoch)]")
	print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")

	criterion = torch.nn.CrossEntropyLoss()

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
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"do_{dropout_val}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
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

	print(f"Best model will be saved in: {mdl_fpth}")

	training_losses = list()
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	
	train_start_time = time.time()
	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad() # Clear gradients from previous batch
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)

			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Stabilize training
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()

			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\t\tBatch [{bidx + 1}/{len(train_loader)}] Loss: {total_loss.item()}")

				if hasattr(optimizer, 'get_trust_ratio_stats'):
					trust_stats = optimizer.get_trust_ratio_stats()
					if trust_stats:
						print(f"\t\tTrust Ratio Summary:")
						print(f"\t\t{trust_stats}\n")
						optimizer.adaptive_lr_adjustment()

						# Periodic visualization
						if epoch % 10 == 0:
							optimizer.visualize_stats(save_path=os.path.join(results_dir, f"trust_ratio_epoch_{epoch}"))

			epoch_loss += total_loss.item()

		if hasattr(optimizer, 'get_trust_ratio_stats'):
			trust_stats = optimizer.get_trust_ratio_stats()
			if trust_stats:
				print(f"\nEpoch {epoch+1}: Trust Ratio Summary:")
				print(trust_stats)
				print("-"*120)

		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)
		
		learning_rates_history.append(optimizer.param_groups[0]['lr'])
		weight_decays_history.append(optimizer.param_groups[0]['weight_decay'])

		# all metrics in one using caching mechanism:
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=True,
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
		)
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
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
			f'Epoch {epoch + 1:3d}:\n'
			f'\t[LOSS] {mode}'
			f'(Training): {avg_training_loss} '
			f'Validation(in-batch): {in_batch_loss_acc_metrics_per_epoch["val_loss"]}\n'
			f'\tValidation Top-k Accuracy:\n'
			f'\tIn-batch:\n'
			f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\tFull Validation Set:\n'
			f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

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

		# --- Early Stopping Check ---
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

		print(f"Epoch {epoch+1} Duration [Train + Validation]: {time.time() - train_and_val_st_time:.2f} sec".center(170, "-"))
	
	print(f"[{mode}] Total Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

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
		# f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}"
	)

	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	viz.plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		# mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		# cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
		# mean_reciprocal_rank_file_path=plot_paths["mrr"],
		# cosine_similarity_file_path=plot_paths["cs"],
	)

	retrieval_metrics_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png")
	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=retrieval_metrics_fpth,
	)

	retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png")
	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
	)

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	)

def vera_finetune_single_label(
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
		topk_values: List[int]=[1, 5, 10, 15, 20],
		quantization_bits: int=8,
		quantized: bool=False,
		use_lamb: bool=False,
		verbose: bool=True,
	):
	window_size = minimum_epochs + 1

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
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold, # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1, # Not really needed for LoRA finetune, but for consistency
	)

	# Dataset and directory setup (same as finetune())
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"{mode.upper()} Rank: {lora_rank} Alpha: {lora_alpha} Dropout: {lora_dropout} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

		if torch.cuda.is_available():
			gpu_name = torch.cuda.get_device_name(device)
			gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
			cuda_capability = torch.cuda.get_device_capability()
			print(f"   ├─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	# Apply VeRA to the model
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

	estimated_epochs = min(num_epochs, 15)
	total_training_steps = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=total_training_steps,
		eta_min=eta_min,
		last_epoch=-1,
	)
	if verbose:
		print(f"{scheduler.__class__.__name__} scheduler")
		print(f"  ├─ T_max = {total_training_steps} steps [({min(num_epochs, 15)} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100:.1f}% of initial LR)")

	criterion = torch.nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(device=device)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
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
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	train_start_time = time.time()
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	learning_rates_history = list()
	weight_decays_history = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)
			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\t\tBatch [{bidx + 1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)

		learning_rates_history.append(optimizer.param_groups[0]['lr'])
		weight_decays_history.append(optimizer.param_groups[0]['weight_decay'])

		print(f">> Validating Epoch {epoch+1} ...")
		# all metrics in one using caching mechanism:
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
			},
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
		)
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
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
			f'\t[LOSS] {mode}'
			f'(Training): {avg_training_loss} '
			f'Validation(in-batch): {current_val_loss}\n'
			f'\tValidation Top-k Accuracy:\n'
			f'\tIn-batch:\n'
			f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\tFull Validation Set:\n'
			f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

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
		# f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)
	
	mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)
	
	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	viz.plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		# mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		# cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
		# mean_reciprocal_rank_file_path=plot_paths["mrr"],
		# cosine_similarity_file_path=plot_paths["cs"],
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

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	)

def dora_finetune_single_label(
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
		topk_values: List[int]=[1, 5, 10, 15, 20],
		quantization_bits: int=8,
		quantized: bool=False,
		use_lamb: bool=False,
		verbose: bool=True,
	):
	window_size = minimum_epochs + 1

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
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold, # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1, # Not really needed for LoRA finetune, but for consistency
	)

	# Dataset and directory setup (same as finetune())
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"{mode.upper()} Rank: {lora_rank} Alpha: {lora_alpha} Dropout: {lora_dropout} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

		if torch.cuda.is_available():
			gpu_name = torch.cuda.get_device_name(device)
			gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
			cuda_capability = torch.cuda.get_device_capability()
			print(f"   ├─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

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

	estimated_epochs = min(num_epochs, 15)
	total_training_steps = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=total_training_steps,
		eta_min=eta_min,
		last_epoch=-1,
	)
	if verbose:
		print(f"{scheduler.__class__.__name__} scheduler")
		print(f"  ├─ T_max = {total_training_steps} steps [({min(num_epochs, 15)} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100:.1f}% of initial LR)")

	criterion = torch.nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(device=device)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
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
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	train_start_time = time.time()
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	learning_rates_history = list()
	weight_decays_history = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)
			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\t\tBatch [{bidx + 1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)

		learning_rates_history.append(optimizer.param_groups[0]['lr'])
		weight_decays_history.append(optimizer.param_groups[0]['weight_decay'])

		print(f">> Validating Epoch {epoch+1} ...")
		# all metrics in one using caching mechanism:
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
			},
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
		)
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
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
			f'\t[LOSS] {mode}'
			f'(Training): {avg_training_loss} '
			f'Validation(in-batch): {current_val_loss}\n'
			f'\tValidation Top-k Accuracy:\n'
			f'\tIn-batch:\n'
			f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\tFull Validation Set:\n'
			f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

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
		# f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)
	
	mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)
	
	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	viz.plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		# mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		# cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
		# mean_reciprocal_rank_file_path=plot_paths["mrr"],
		# cosine_similarity_file_path=plot_paths["cs"],
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

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	)

def lora_plus_finetune_single_label(
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
		topk_values: List[int]=[1, 5, 10, 15, 20],
		quantization_bits: int=8,
		lora_plus_lambda: float=32.0,
		quantized: bool=False,
		use_lamb: bool=False,
		verbose: bool=True,
	):
	window_size = minimum_epochs + 1

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
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold, # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1, # Not really needed for LoRA finetune, but for consistency
	)

	# Dataset and directory setup (same as finetune())
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"{mode.upper()}")
		print(f"   ├─ Rank: {lora_rank}")
		print(f"   ├─ Alpha: {lora_alpha}")
		print(f"   ├─ Dropout: {lora_dropout}")
		print(f"   ├─ LoRA+ Learning Rate Multiplier (λ): {lora_plus_lambda}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

		if torch.cuda.is_available():
			gpu_name = torch.cuda.get_device_name(device)
			gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
			cuda_capability = torch.cuda.get_device_capability()
			print(f"   ├─ {gpu_name} {device}")
			print(f"   ├─ Total Memory: {gpu_total_mem:.2f}GB")
			print(f"   └─ CUDA Capability: {cuda_capability}")

	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
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

	lora_a_params = []
	lora_b_params = []
	for name, param in model.named_parameters():
		# print(f"{name} {param.shape} {param.requires_grad}")
		if param.requires_grad:
			if "lora_A" in name:
				# print(f"  ├─ lora_A: {name} {param.shape}")
				lora_a_params.append(param)
			elif "lora_B" in name:
				# print(f"  └─ lora_B: {name} {param.shape}")
				lora_b_params.append(param)
			else:
				# Should not happen in pure LoRA+, but safe fallback
				lora_a_params.append(param)
	if verbose:
		print(f"LoRA+ Parameter Groups:")
		print(f"  ├─ lora_A params: {len(lora_a_params)} tensors: {lora_a_params[0].shape}")
		print(f"  ├─ lora_B params: {len(lora_b_params)} tensors: {lora_b_params[0].shape}")
		print(f"  └─ LR multiplier (λ): {lora_plus_lambda}")

	loraA_lr = learning_rate
	loraB_lr = learning_rate * lora_plus_lambda
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

	# setup scheduler
	estimated_epochs = min(num_epochs, 15)
	total_training_steps = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=total_training_steps,
		eta_min=eta_min,
		last_epoch=-1,
	)
	if verbose:
		print(f"{scheduler.__class__.__name__} scheduler")
		print(f"  ├─ T_max = {total_training_steps} steps [({min(num_epochs, 15)} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100:.1f}% of initial LR)")

	criterion = torch.nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(device=device)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"loraA_lr_{loraA_lr:.1e}_"
		f"lmbd_{lora_plus_lambda}_"
		f"loraB_lr_{loraB_lr:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
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
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	train_start_time = time.time()
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	learning_rates_history = list()
	weight_decays_history = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)
			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\t\tBatch [{bidx + 1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)

		learning_rates_history.append([group['lr'] for group in optimizer.param_groups])
		weight_decays_history.append([group['weight_decay'] for group in optimizer.param_groups])
		print(f"[Epoch {epoch+1}] {len(learning_rates_history[-1])} LRs (first 10): {learning_rates_history[-1][:10]}")

		print(f">> Validating Epoch {epoch+1} ...")
		# all metrics in one using caching mechanism:
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
			},
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
		)
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
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
			f'\t[LOSS] {mode}'
			f'(Training): {avg_training_loss} '
			f'Validation(in-batch): {current_val_loss}\n'
			f'\tValidation Top-k Accuracy:\n'
			f'\tIn-batch:\n'
			f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\tFull Validation Set:\n'
			f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

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
		# f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"loraA_lr_{loraA_lr:.1e}_"
		f"lmbd_{lora_plus_lambda}_"
		f"loraB_lr_{loraB_lr:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)
	
	mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)
	
	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	viz.plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		# mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		# cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
		# mean_reciprocal_rank_file_path=plot_paths["mrr"],
		# cosine_similarity_file_path=plot_paths["cs"],
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

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	)

def lora_finetune_single_label(
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
		quantization_bits: int=8,
		quantized: bool=False,
		use_lamb: bool=False,
		verbose: bool=True,
	):
	window_size = minimum_epochs + 1

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
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold, # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1, # Not really needed for LoRA finetune, but for consistency
	)

	# Dataset and directory setup (same as finetune())
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"{mode.upper()} Rank: {lora_rank} Alpha: {lora_alpha} Dropout: {lora_dropout} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

		if torch.cuda.is_available():
			gpu_name = torch.cuda.get_device_name(device)
			gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
			cuda_capability = torch.cuda.get_device_capability()
			print(f"   ├─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

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

	estimated_epochs = min(num_epochs, 15)
	total_training_steps = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=total_training_steps,
		eta_min=eta_min,
		last_epoch=-1,
	)
	if verbose:
		print(f"{scheduler.__class__.__name__} scheduler")
		print(f"  ├─ T_max = {total_training_steps} steps [({min(num_epochs, 15)} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100:.1f}% of initial LR)")

	criterion = torch.nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(device=device)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
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
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	train_start_time = time.time()
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	learning_rates_history = list()
	weight_decays_history = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)
			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\t\tBatch [{bidx + 1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)

		learning_rates_history.append(optimizer.param_groups[0]['lr'])
		weight_decays_history.append(optimizer.param_groups[0]['weight_decay'])

		print(f">> Validating Epoch {epoch+1} ...")
		# all metrics in one using caching mechanism:
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
			},
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
		)
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
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
			f'\t[LOSS] {mode}'
			f'(Training): {avg_training_loss} '
			f'Validation(in-batch): {current_val_loss}\n'
			f'\tValidation Top-k Accuracy:\n'
			f'\tIn-batch:\n'
			f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\tFull Validation Set:\n'
			f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

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
		# f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)
	
	mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)
	
	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	viz.plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		# mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		# cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
		# mean_reciprocal_rank_file_path=plot_paths["mrr"],
		# cosine_similarity_file_path=plot_paths["cs"],
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

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	)

def ia3_finetune_single_label(
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
		lora_rank: int=1000,
		lora_alpha: float=26.3,
		lora_dropout: float=0.056,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		quantization_bits: int=8,
		quantized: bool=False,
		use_lamb: bool=False,
		verbose: bool=True,
	):
	window_size = minimum_epochs + 1

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
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold, # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1, # Not really needed for LoRA finetune, but for consistency
	)

	# Dataset and directory setup (same as finetune())
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"{mode.upper()} Rank: {lora_rank} Alpha: {lora_alpha} Dropout: {lora_dropout} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

		if torch.cuda.is_available():
			gpu_name = torch.cuda.get_device_name(device)
			gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
			cuda_capability = torch.cuda.get_device_capability()
			print(f"   ├─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

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

	estimated_epochs = min(num_epochs, 15)
	total_training_steps = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=total_training_steps,
		eta_min=eta_min,
		last_epoch=-1,
	)
	if verbose:
		print(f"{scheduler.__class__.__name__} scheduler")
		print(f"  ├─ T_max = {total_training_steps} steps [({min(num_epochs, 15)} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100:.1f}% of initial LR)")

	criterion = torch.nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(device=device)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
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
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	train_start_time = time.time()
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	learning_rates_history = list()
	weight_decays_history = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)
			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\t\tBatch [{bidx + 1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)

		learning_rates_history.append(optimizer.param_groups[0]['lr'])
		weight_decays_history.append(optimizer.param_groups[0]['weight_decay'])

		print(f">> Validating Epoch {epoch+1} ...")
		# all metrics in one using caching mechanism:
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
			},
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
		)
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
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
			f'\t[LOSS] {mode}'
			f'(Training): {avg_training_loss} '
			f'Validation(in-batch): {current_val_loss}\n'
			f'\tValidation Top-k Accuracy:\n'
			f'\tIn-batch:\n'
			f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\tFull Validation Set:\n'
			f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

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
		# f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)
	
	mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)
	
	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	viz.plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		# mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		# cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
		# mean_reciprocal_rank_file_path=plot_paths["mrr"],
		# cosine_similarity_file_path=plot_paths["cs"],
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

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	)

def clip_adapter_finetune_single_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		clip_adapter_method: str, # "clip_adapter_v", "clip_adapter_t", "clip_adapter_vt"
		bottleneck_dim: int=256,
		activation: str = "relu",
		patience: int = 7,
		min_delta: float = 1e-4,
		cumulative_delta: float = 1e-3,
		minimum_epochs: int = 10,
		volatility_threshold: float = 0.02,
		slope_threshold: float = 1e-3,
		pairwise_imp_threshold: float = 0.01,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		use_lamb: bool = False,
		verbose: bool = True,
):
		"""
		Fine-tunes a CLIP model using CLIP-Adapter technique.

		Args:
				model: Pre-trained CLIP model.
				train_loader: DataLoader for training data.
				validation_loader: DataLoader for validation data.
				num_epochs: Number of training epochs.
				print_every: Print training stats every N batches.
				learning_rate: Learning rate for the optimizer.
				weight_decay: Weight decay for the optimizer.
				device: Device to run the model on.
				results_dir: Directory to save results and checkpoints.
				clip_adapter_method: CLIP-Adapter variant - "clip_adapter_v", "clip_adapter_t", "clip_adapter_vt".
				bottleneck_dim: Dimension of the CLIP-Adapter bottleneck layer.
				activation: Activation function for the adapter ("relu" or "gelu").
				patience: Patience for early stopping.
				min_delta: Minimum change to qualify as an improvement for early stopping.
				cumulative_delta: Cumulative change threshold for early stopping.
				minimum_epochs: Minimum epochs to train before applying early stopping.
				volatility_threshold: Threshold for validation loss volatility.
				slope_threshold: Threshold for the slope of validation loss.
				pairwise_imp_threshold: Threshold for pairwise improvement in early stopping.
				topk_values: List of k values for Top-K accuracy calculation.
				use_lamb: Use LAMB optimizer instead of AdamW.
				verbose: Enable detailed logging.
		"""
		window_size = minimum_epochs + 1

		# --- CLIP-ADAPTER SPECIFIC SETUP ---
		# CLIP-Adapter does not typically modify dropout in the base model,
		# but we check as a general good practice.
		dropout_values = list()
		for name, module in model.named_modules():
			if isinstance(module, torch.nn.Dropout):
				dropout_values.append((name, module.p))

		# Check for non-zero dropout in the base model (less critical for CLIP-Adapter than LoRA,
		# but good to be aware of for consistency).
		non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
		if verbose and non_zero_dropouts:
			dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
			print(
				f"[CLIP-Adapter] WARNING: Non-zero dropout detected in base model: {dropout_info}. "
				f"This might affect the frozen base model's behavior during adaptation."
			)

		early_stopping = EarlyStopping(
			patience=patience,
			min_delta=min_delta,
			cumulative_delta=cumulative_delta,
			window_size=window_size,
			mode='min', # Monitoring validation loss
			min_epochs=minimum_epochs,
			restore_best_weights=True,
			volatility_threshold=volatility_threshold,
			slope_threshold=slope_threshold, # Positive slope is bad for loss
			pairwise_imp_threshold=pairwise_imp_threshold,
		)

		# Dataset and directory setup (same as lora_finetune_single_label)
		try:
			dataset_name = validation_loader.dataset.dataset.__class__.__name__
		except AttributeError:
			dataset_name = validation_loader.dataset.dataset_name

		mode = inspect.stack()[0].function
		mode = re.sub(r'_finetune_single_label', '', mode) # Expected to be 'clip_adapter'

		model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
		model_name = model.__class__.__name__

		if verbose:
			print(f"{mode.upper()} Method: {clip_adapter_method} Bottleneck: {bottleneck_dim} Activation: {activation} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}")
			if torch.cuda.is_available():
				gpu_name = torch.cuda.get_device_name(device)
				gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
				cuda_capability = torch.cuda.get_device_capability()
				print(f"   ├─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

		# Apply CLIP-Adapter using the provided function
		model = get_adapter_peft_clip(
			clip_model=model,
			method=clip_adapter_method,
			cache_dim=None, # Not used by CLIP-Adapter, only by Tip-Adapter
			bottleneck_dim=bottleneck_dim,
			activation=activation,
			verbose=verbose,
		)

		model.to(device)

		# DEBUG: Check which parameters are trainable
		if verbose:
			trainable_params = []
			frozen_params = []
			for name, param in model.named_parameters():
				if param.requires_grad:
					trainable_params.append((name, param.numel()))
				else:
					frozen_params.append((name, param.numel()))
			
			print(f"DEBUG - Trainable parameters: {len(trainable_params)}")
			print(f"DEBUG - Frozen parameters: {len(frozen_params)}")
			
			if trainable_params:
				print("Trainable parameters:")
				for name, numel in trainable_params[:10]:  # Show first 10
					print(f"  {name}: {numel} params")
				if len(trainable_params) > 10:
					print(f"  ... and {len(trainable_params) - 10} more")
			else:
				print("WARNING: No trainable parameters found!")
					
			total_trainable = sum(numel for _, numel in trainable_params)
			total_frozen = sum(numel for _, numel in frozen_params)
			print(f"Total trainable parameters: {total_trainable:,}")
			print(f"Total frozen parameters: {total_frozen:,}")

		trainable_parameters = [p for p in model.parameters() if p.requires_grad]
		if not trainable_parameters:
			raise ValueError("No trainable parameters found in the model. Check your setup.")

		# Assuming get_parameters_info can handle CLIP-Adapter models too
		get_parameters_info(model=model, mode=mode)

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

		estimated_epochs = min(num_epochs, 15)
		total_training_steps = estimated_epochs * len(train_loader)
		ANNEALING_RATIO = 1e-2 # 1% of initial LR
		eta_min = learning_rate * ANNEALING_RATIO
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer=optimizer,
			T_max=total_training_steps,
			eta_min=eta_min,
			last_epoch=-1,
		)
		if verbose:
			print(f"{scheduler.__class__.__name__} scheduler")
			print(f"  ├─ T_max = {total_training_steps} steps [({min(num_epochs, 15)} estimated epochs x {len(train_loader)} batches/epoch)]")
			print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100:.1f}% of initial LR)")

		criterion = torch.nn.CrossEntropyLoss()
		scaler = torch.amp.GradScaler(device=device)

		mdl_fpth = os.path.join(
			results_dir,
			f"{mode}_"
			f"{model_arch}_"
			f"ieps_{num_epochs}_"
			f"lr_{learning_rate:.1e}_"
			f"wd_{weight_decay:.1e}_"
			f"bs_{train_loader.batch_size}_"
			f"cad_{clip_adapter_method}_"
			f"cbd_{bottleneck_dim}_"
			f"act_{activation}_"
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
		img2txt_metrics_all_epochs = list()
		txt2img_metrics_all_epochs = list()
		train_start_time = time.time()
		final_img2txt_metrics = None
		final_txt2img_metrics = None
		learning_rates_history = list()
		weight_decays_history = list()
		in_batch_loss_acc_metrics_all_epochs = list()
		full_val_loss_acc_metrics_all_epochs = list()

		for epoch in range(num_epochs):
				train_and_val_st_time = time.time()
				torch.cuda.empty_cache()
				model.train()
				print(f"Epoch [{epoch + 1}/{num_epochs}]")
				epoch_loss = 0.0
				for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
						optimizer.zero_grad(set_to_none=True)
						images = images.to(device, non_blocking=True)
						tokenized_labels = tokenized_labels.to(device, non_blocking=True)

						with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
								logits_per_image, logits_per_text = model(images, tokenized_labels)
								ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
								loss_img = criterion(logits_per_image, ground_truth)
								loss_txt = criterion(logits_per_text, ground_truth)
								total_loss = 0.5 * (loss_img + loss_txt)
						scaler.scale(total_loss).backward()
						# CLIP-Adapter typically has fewer parameters, gradient clipping might be less critical,
						# but keeping it for consistency and stability.
						torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
						scaler.step(optimizer)
						scaler.update()
						scheduler.step()
						if bidx % print_every == 0 or bidx + 1 == len(train_loader):
								print(f"\t\tBatch [{bidx + 1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
						epoch_loss += total_loss.item()
				avg_training_loss = epoch_loss / len(train_loader)
				training_losses.append(avg_training_loss)

				learning_rates_history.append(optimizer.param_groups[0]['lr'])
				weight_decays_history.append(optimizer.param_groups[0]['weight_decay'])

				print(f">> Validating Epoch {epoch+1} ...")
				# all metrics in one using caching mechanism:
				validation_results = get_validation_metrics(
						model=model,
						validation_loader=validation_loader,
						criterion=criterion,
						device=device,
						topK_values=topk_values,
						finetune_strategy=mode,
						cache_dir=results_dir,
						verbose=True,
						# Pass CLIP-Adapter specific parameters for logging/debugging
						clip_adapter_params={
								"method": clip_adapter_method,
								"bottleneck_dim": bottleneck_dim,
								"activation": activation,
						},
						max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
						is_training=True,
						model_hash=get_model_hash(model),
				)
				in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
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
						f'\t[LOSS] {mode}'
						f'(Training): {avg_training_loss} '
						f'Validation(in-batch): {current_val_loss}\n'
						f'\tValidation Top-k Accuracy:\n'
						f'\tIn-batch:\n'
						f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
						f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
						f'\tFull Validation Set:\n'
						f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
						f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
				)
				print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
				print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

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
				# Pass CLIP-Adapter specific parameters for logging/debugging in evaluation
				clip_adapter_params={
						"method": clip_adapter_method,
						"bottleneck_dim": bottleneck_dim,
						"activation": activation,
				},
				max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
		)

		# Access individual metrics as needed
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
				# f"{dataset_name}_"
				f"{mode}_"
				f"{CLUSTER}_"
				# f"{optimizer.__class__.__name__}_"
				# f"{scheduler.__class__.__name__}_"
				# f"{criterion.__class__.__name__}_"
				# f"{scaler.__class__.__name__}_"
				# f"{model_name}_"
				f"{model_arch}_"
				f"ep_{actual_trained_epochs}_"
				f"lr_{learning_rate:.1e}_"
				f"wd_{weight_decay:.1e}_"
				f"bs_{train_loader.batch_size}_"
				f"cad_{clip_adapter_method}_"
				f"cbd_{bottleneck_dim}_"
				f"act_{activation}"
		)
		
		mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)
		
		print(f"Best model will be renamed to: {mdl_fpth}")

		plot_paths = {
				"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
				"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
				"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
				"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
				"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
				"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
				"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
				"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
				"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		}

		viz.plot_loss_accuracy_metrics(
				dataset_name=dataset_name,
				train_losses=training_losses,
				val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
				in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
				in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
				full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
				full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
				# mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
				# cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
				losses_file_path=plot_paths["losses"],
				in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
				in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
				full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
				full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
				# mean_reciprocal_rank_file_path=plot_paths["mrr"],
				# cosine_similarity_file_path=plot_paths["cs"],
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

		viz.plot_hyperparameter_evolution(
				eta_min=eta_min,
				learning_rates=learning_rates_history,
				weight_decays=weight_decays_history,
				fname=os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
		)

def probe_finetune_single_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		learning_rate: float,
		weight_decay: float,
		probe_dropout: float,
		device: str,
		results_dir: str,
		print_every: int,
		patience: int,
		min_delta: float,
		cumulative_delta: float,
		minimum_epochs: int,
		volatility_threshold: float,
		slope_threshold: float,
		pairwise_imp_threshold: float,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		probe_hidden_dim: int = None,  # Optional hidden layer for MLP probe
		use_lamb: bool = False,
	):

	window_size = minimum_epochs + 1
	# Inspect the model for dropout layers
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
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1, # Not really needed for linear probe, but for consistency
	)
	
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name
	
	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	
	print(f"{mode} {model_name} {model_arch} {dataset_name} {num_epochs} Epoch(s) | batch_size: {train_loader.batch_size} | {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))
	
	# Get number of classes
	try:
		class_names = validation_loader.dataset.dataset.classes
		num_classes = len(class_names)
	except AttributeError:
		class_names = validation_loader.dataset.unique_labels
		num_classes = len(class_names)
	
	print(f"Number of Labels: {num_classes}")
	
	# =====================================
	# STEP 1: FREEZE ALL CLIP PARAMETERS AND CREATE PROBE
	# =====================================
	for param in model.parameters():
		param.requires_grad = False # Freeze all CLIP parameters(no fine-tuning)
	
	get_parameters_info(model=model, mode=mode)

	# Create the robust linear probe that handles everything automatically
	print("\nCreating probe model...")
	
	probe = get_probe_clip(
			clip_model=model,
			validation_loader=validation_loader,
			device=torch.device(device),
			# hidden_dim=256,  # Optional: creates MLP probe
			dropout=probe_dropout,
			zero_shot_init=True,
			verbose=True
	)
	print(f"DEBUG: Probe created successfully")
	print(f"Multi-label dataset? {isinstance(probe, MultiLabelProbe)}")

	clip_dim = probe.input_dim  # Get the detected feature dimension
	probe_params = sum(p.numel() for p in probe.parameters())

	print(f"DEBUG: Probe class: {probe.__class__.__name__}")
	print(f"DEBUG: Probe input_dim: {probe.input_dim}")
	print(f"DEBUG: Probe num_classes: {probe.num_classes}")


	print(f"CLIP output dimension: {clip_dim}")
	print(f"Probe type: {probe.probe_type} | Probe parameters: {probe_params:,}")
	
	# Debug: Print probe architecture details
	if probe.probe_type == "Linear":
		print(f"Probe weight shape: {probe.probe.weight.shape}")
		print(f"Probe bias shape: {probe.probe.bias.shape}")
	else:
		print(f"MLP Probe architecture: {probe.probe}")
	
	# =====================================
	# STEP 2: SETUP TRAINING
	# =====================================
	
	# Only optimize probe parameters
	if use_lamb:
		optimizer = LAMB(
			params=probe.parameters(),
			lr=learning_rate,
			weight_decay=weight_decay,
		)
	else:
		optimizer = torch.optim.AdamW(
			params=probe.parameters(),
			lr=learning_rate,
			betas=(0.9, 0.999),
			eps=1e-8,
			weight_decay=weight_decay,
		)
	print(f"Using {optimizer.__class__.__name__} for optimization")
	
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=num_epochs,
		eta_min=learning_rate * 0.01, # 1% of initial LR
		last_epoch=-1,
	)
	print(f"Using {scheduler.__class__.__name__} for learning rate scheduling")
	
	criterion = torch.nn.CrossEntropyLoss()
	print(f"Using {criterion.__class__.__name__} as the loss function")
	
	scaler = torch.amp.GradScaler(device=device)
	print(f"Using {scaler.__class__.__name__} for mixed precision training")
	
	mdl_fpth = os.path.join(
		results_dir,
		f"{probe.probe_type.lower()}_probe_"
		f"{model_arch}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"hdim_{probe_hidden_dim}_"
		f"pdo_{probe_dropout}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)
	
	print(f"Best model will be saved in: {mdl_fpth}")
	
	# =====================================
	# STEP 3: EXTRACT AND CACHE FEATURES (OPTIONAL)
	# =====================================
	# For efficiency, we can pre-extract all features once
	print("Extracting features from frozen CLIP model...")
	
	def extract_features(loader, model, device):
			"""Extract features from frozen CLIP model"""
			features = list()
			labels = list()
			
			model.eval()
			with torch.no_grad():
					for images, _, label_indices in tqdm(loader, desc="Extracting features"):
							images = images.to(device, non_blocking=True)
							# Extract CLIP features
							image_features = model.encode_image(images)
							image_features = F.normalize(image_features, dim=-1)
							
							features.append(image_features.cpu())
							labels.append(label_indices.cpu())
			
			return torch.cat(features, dim=0), torch.cat(labels, dim=0)
	
	# Extract features once (optional - can be skipped for online extraction)
	train_features, train_labels = extract_features(train_loader, model, device)
	val_features, val_labels = extract_features(validation_loader, model, device)
	
	print(f"Extracted features - Train: {train_features.shape}, Val: {val_features.shape}")
	print(f"DEBUG: Extracted train features shape: {train_features.shape}")
	print(f"DEBUG: Extracted val features shape: {val_features.shape}")
	print(f"DEBUG: Train labels shape: {train_labels.shape}")
	print(f"DEBUG: Val labels shape: {val_labels.shape}")	


	# Create feature datasets
	from torch.utils.data import TensorDataset
	train_feature_dataset = TensorDataset(train_features, train_labels)
	val_feature_dataset = TensorDataset(val_features, val_labels)
	
	# Create new dataloaders for features
	train_feature_loader = DataLoader(
			train_feature_dataset,
			batch_size=train_loader.batch_size,
			shuffle=True,
			num_workers=0,  # Features are already in memory
			pin_memory=False
	)
	
	val_feature_loader = DataLoader(
			val_feature_dataset,
			batch_size=validation_loader.batch_size,
			shuffle=False,
			num_workers=0,
			pin_memory=False
	)
	
	# =====================================
	# STEP 4: TRAINING LOOP
	# =====================================
	training_losses = list()
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	train_start_time = time.time()
	
	for epoch in range(num_epochs):
			train_and_val_st_time = time.time()
			torch.cuda.empty_cache()
			
			# Training
			probe.train()
			print(f"Epoch [{epoch + 1}/{num_epochs}]")
			epoch_loss = 0.0
			correct = 0
			total = 0
			
			for bidx, (features, labels) in enumerate(train_feature_loader):
					features = features.to(device, non_blocking=True)
					labels = labels.to(device, non_blocking=True)
					
					optimizer.zero_grad()
					
					with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
							# Forward pass through probe only
							logits = probe(features)
							loss = criterion(logits, labels)
					
					scaler.scale(loss).backward()
					torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
					scaler.step(optimizer)
					scaler.update()
					
					# Track accuracy
					_, predicted = torch.max(logits.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()
					
					if bidx % print_every == 0 or bidx + 1 == len(train_feature_loader):
							print(f"\t\tBatch [{bidx + 1}/{len(train_feature_loader)}] Loss: {loss.item():.6f}")
					
					epoch_loss += loss.item()
			
			avg_training_loss = epoch_loss / len(train_feature_loader)
			train_accuracy = 100 * correct / total
			training_losses.append(avg_training_loss)
			
			print(f"Training Loss: {avg_training_loss:.6f}, Training Accuracy: {train_accuracy:.2f}%")
			
			# Validation
			probe.eval()
			val_loss = 0.0
			correct = 0
			total = 0
			
			with torch.no_grad():
					for features, labels in val_feature_loader:
							features = features.to(device, non_blocking=True)
							labels = labels.to(device, non_blocking=True)
							
							logits = probe(features)
							loss = criterion(logits, labels)
							
							val_loss += loss.item()
							
							_, predicted = torch.max(logits.data, 1)
							total += labels.size(0)
							correct += (predicted == labels).sum().item()
			
			avg_val_loss = val_loss / len(val_feature_loader)
			val_accuracy = 100 * correct / total
			
			print(f"Validation Loss: {avg_val_loss:.6f}, Validation Accuracy: {val_accuracy:.2f}%")
			
			# For compatibility with existing code, create metrics dict
			in_batch_metrics = {
					"val_loss": avg_val_loss,
					"accuracy": val_accuracy / 100,  # Convert to fraction
			}
			in_batch_loss_acc_metrics_all_epochs.append(in_batch_metrics)
			
			# Update scheduler
			scheduler.step()
			
			# Early stopping
			if early_stopping.should_stop(
					current_value=avg_val_loss,
					model=probe,  # Save probe weights, not CLIP
					epoch=epoch,
					optimizer=optimizer,
					scheduler=scheduler,
					checkpoint_path=mdl_fpth,
			):
					print(f"\nEarly stopping at epoch {epoch + 1}")
					break
			
			print(f"Epoch {epoch+1} Duration: {time.time() - train_and_val_st_time:.2f} sec".center(170, "-"))
	
	print(f"[{mode}] Total Training Time: {time.time() - train_start_time:.1f} sec".center(170, "-"))
	
	# Continue with the rest of your original code (evaluation, plotting, etc.)
	# [Keep all the remaining parts of your original function unchanged]
	
	# =====================================
	# STEP 5: FINAL EVALUATION
	# =====================================
	# Load best probe weights
	if os.path.exists(mdl_fpth):
		print(f"Loading best probe weights from {mdl_fpth}")
		checkpoint = torch.load(mdl_fpth, map_location=device)
		if 'model_state_dict' in checkpoint:
			probe.load_state_dict(checkpoint['model_state_dict'])
		else:
			probe.load_state_dict(checkpoint)
	
	# Create combined model for evaluation
	class CLIPWithProbe(torch.nn.Module):
			def __init__(self, clip_model, probe):
					super().__init__()
					self.clip = clip_model
					self.probe = probe
					# Copy necessary attributes from CLIP
					self.visual = clip_model.visual
					self.encode_image = clip_model.encode_image
					self.encode_text = clip_model.encode_text
					self.name = getattr(clip_model, 'name', 'unknown')
			
			def forward(self, images, texts):
					# For compatibility with evaluation code
					image_features = self.clip.encode_image(images)
					image_features = F.normalize(image_features, dim=-1)
					
					# Get logits from probe
					logits = self.probe(image_features)
					
					# For CLIP-style evaluation, we need to return image-text similarity
					# We'll use the probe's logits as a proxy
					batch_size = images.shape[0]
					num_classes = logits.shape[1]
					
					# Create pseudo-similarity matrix
					# This is a hack for compatibility - ideally evaluation should be adapted
					if batch_size == num_classes:
							return logits, logits.T
					else:
							# Pad or truncate to make square matrix
							min_dim = min(batch_size, num_classes)
							logits_per_image = logits[:min_dim, :min_dim]
							logits_per_text = logits_per_image.T
							return logits_per_image, logits_per_text
	
	# Combine CLIP and probe for evaluation
	combined_model = CLIPWithProbe(model, probe)
	
	# Run evaluation with the combined model
	evaluation_results = evaluate_best_model(
		model=combined_model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=True,
		max_in_batch_samples=get_max_samples(
			batch_size=validation_loader.batch_size, 
			N=10, 
			device=device
		),
	)
	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	print(f"Final evaluation used model weights from: {evaluation_results['model_loaded_from']}")
	print("--- Final Metrics [In-batch Validation] ---")
	print(json.dumps(final_metrics_in_batch, indent=2, ensure_ascii=False))
	print("--- Final Metrics [Full Validation Set] ---")
	print(json.dumps(final_metrics_full, indent=2, ensure_ascii=False))
	print("--- Image-to-Text Retrieval ---")
	print(json.dumps(final_img2txt_metrics, indent=2, ensure_ascii=False))
	print("--- Text-to-Image Retrieval ---")
	print(json.dumps(final_txt2img_metrics, indent=2, ensure_ascii=False))
	# Generate plots
	print("\nGenerating result plots...")
	actual_trained_epochs = len(training_losses)
	
	file_base_name = (
		# f"{dataset_name}_"
		f"{mode}_{probe.probe_type}_"
		f"{CLUSTER}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}"
	)
	
	# Update model path
	mdl_fpth = get_updated_model_name(
			original_path=mdl_fpth,
			actual_epochs=actual_trained_epochs
	)
	
	print(f"Best model will be renamed to: {mdl_fpth}")
	
	# Print final summary
	best_val_loss = early_stopping.get_best_score() or 0.0
	print("\n" + "="*80)
	print("ENHANCED LINEAR PROBE TRAINING SUMMARY")
	print("="*80)
	print(f"Method: {mode}")
	print(f"Model: {getattr(model, 'name', 'Unknown')}")
	print(f"Probe Type: {probe.probe_type}")
	print(f"Probe Parameters: {probe_params:,}")
	print(f"CLIP Parameters (frozen): {sum(p.numel() for p in model.parameters()):,}")
	print(f"Total Epochs: {actual_trained_epochs}")
	print(f"Best Validation Loss: {best_val_loss}")
	print(f"Best Epoch: {early_stopping.get_best_epoch() + 1}")
	print("="*80)
	plot_paths = {
			"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
			"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
			"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
			"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
			"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
			"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
			"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}
	viz.plot_loss_accuracy_metrics(
			dataset_name=dataset_name,
			train_losses=training_losses,
			val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
			in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
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