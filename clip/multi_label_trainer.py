from utils import *
from early_stopper import EarlyStopping
from loss import *
import clip
from clip_peft import get_injected_peft_clip, get_adapter_peft_clip
from probe import get_probe_clip
from evals import *
import visualize as viz

def zero_shot_multi_label(
	model: torch.nn.Module,
	train_loader: DataLoader,
	validation_loader: DataLoader,
	device: str,
	results_dir: str,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
	topk_values: List[int] = [1, 3, 5, 10, 15, 20],
	temperature: float = 0.07,
	verbose: bool = True,
) -> Dict:

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	mode = inspect.stack()[0].function
	mode = re.sub(r'_multi_label', '', mode)
	if verbose:
		print(f"{mode.upper()} {model_name} {model_arch} on {dataset_name}")

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"bs_{train_loader.batch_size}_"
		f"temp_{temperature}"
		f".pth"
	)

	# Save zero-shot checkpoint (just the base model state)
	checkpoint = {
		"epoch": 0,  # No training epochs
		"model_state_dict": model.state_dict(),
		"best_val_loss": None,  # No training loss
		"strategy": "zero_shot",
		"temperature": temperature,
		"model_architecture": model_arch,
	}
	torch.save(checkpoint, mdl_fpth)
	if verbose:
		print(f"\nSaved {mode} checkpoint: {mdl_fpth}")

	validation_results = get_validation_metrics(
		model=model,
		validation_loader=validation_loader,
		device=device,
		topK_values=topk_values,
		cache_dir=results_dir,
		is_training=False,
		model_hash=get_model_hash(model),
		temperature=temperature,
		verbose=verbose,
	)

	i2t_similarity = validation_results["i2t_similarity"]
	t2i_similarity = validation_results["t2i_similarity"]
	device_labels = validation_results["device_labels"]

	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		try:
			class_names = validation_loader.dataset.dataset.classes
		except:
			class_names = train_loader.dataset.unique_labels	
	num_classes = len(class_names)

	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="log",
		device=device,
		verbose=verbose,
	)
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	train_freq = masks["train_freq"]

	val_freq = diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	final_tiered_i2t = compute_tiered_retrieval_metrics(
		similarity_matrix=i2t_similarity,
		query_labels=device_labels,
		topK_values=topk_values,
		head_mask=head_mask,
		rare_mask=rare_mask,
		active_mask=active_mask,
		mode="Image-to-Text",
		verbose=verbose,
	)

	final_tiered_t2i = compute_tiered_retrieval_metrics(
		similarity_matrix=t2i_similarity,
		query_labels=device_labels,
		topK_values=topk_values,
		head_mask=head_mask,
		rare_mask=rare_mask,
		active_mask=active_mask,
		mode="Text-to-Image",
		verbose=verbose,
	)

	del i2t_similarity, t2i_similarity
	torch.cuda.empty_cache()

	if verbose:
		print(f"{'='*50}")
		print(f"\n{mode.upper()} Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"\n{mode.upper()} Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")

	return final_tiered_i2t, final_tiered_t2i

def probe_multi_label(
	model: torch.nn.Module,
	train_loader,
	validation_loader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	patience: int = 10,
	min_delta: float = 1e-4,
	cumulative_delta: float = 5e-3,
	minimum_epochs: int = 20,
	topk_values: List[int] = [1, 3, 5, 10, 15, 20],
	loss_weights: Dict[str, float] = None,
	volatility_threshold: float = 15.0,
	slope_threshold: float = 1e-4,
	pairwise_imp_threshold: float = 1e-4,
	probe_hidden_dim: int = None,
	probe_dropout: float = 0.1,
	cache_features: bool = True,
	temperature: float = 0.07,
	verbose: bool = True,
):
	window_size = minimum_epochs + 1
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

	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes	
	num_classes = len(class_names)

	mode = inspect.stack()[0].function
	mode = re.sub(r'_multi_label', '', mode)
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"\n{mode.upper()} [Multi-Label]")
		print(f"   ├─ {model_name} {model_arch}")
		print(f"   ├─ {dataset_name} {num_classes} classes")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss Weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
	
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		print(f"   └─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	# FREEZE ALL CLIP PARAMETERS AND CREATE ROBUST PROBE
	for param in model.parameters():
		param.requires_grad = False

	# Build probe (wraps frozen CLIP, adds trainable W)
	model.eval()
	probe = get_probe_clip(
		clip_model=model,
		validation_loader=validation_loader,
		device=torch.device(device),
		hidden_dim=probe_hidden_dim,  # creates MLP probe
		dropout=probe_dropout,
		zero_shot_init=True, # faster convergence
		verbose=verbose,
	)

	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="log",
		device=device,
		verbose=verbose,
	)

	pos_weight = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask = masks["head_mask"]
	rare_mask = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	val_freq = diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	# ── Criteria
	# For the probe, loss is computed directly on logits [B, C] — same shape
	# as i2t_sim — so criterion_i2t with pos_weight applies directly.
	# No criterion_t2i needed: the probe has no T2I direction.
	criterion = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)
	if verbose:
		print(f"\n{criterion.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min()}, {pos_weight.max()}]")
		print(f"   ├─ number of samples: {N}")
		print(f"   ├─ number of classes: {num_classes}")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,} / {num_classes:,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	optimizer = torch.optim.AdamW(
		params=probe.probe.parameters(),
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=T_max,
		eta_min=eta_min,
		last_epoch=-1,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")
	
	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**11,      # 2048 (Conservative start)
		growth_factor=1.5,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=5000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'} (cuda_cap: {cuda_capability})")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_{probe.probe_type}_"
		f"{model_name}_"
		f"{model_arch}_"
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

	# Feature caching
	# wrap feature extraction in torch.no_grad() to avoid building
	# a computation graph for 74K images — CLIP is frozen, gradients useless.
	if cache_features:
		print("Pre-extracting image features (CLIP frozen — runs once)...")
		model.eval()

		def extract_features(loader, desc):
			feats, lbls = [], []
			with torch.no_grad():
				with torch.amp.autocast(
					device_type=device.type, 
					enabled=torch.cuda.is_available(),
					dtype=amp_dtype,
				):
					for images, _, label_vectors in tqdm(loader, desc=desc):
						images = images.to(device, non_blocking=True)
						emb = model.encode_image(images)
						emb = torch.nn.functional.normalize(emb, dim=-1)
						feats.append(emb.cpu())
						lbls.append(label_vectors.cpu())
			return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)
		
		train_feats, train_lbls = extract_features(train_loader, "Train features")
		val_feats, val_lbls = extract_features(validation_loader, "Val features")

		print(f"Cached — train: {train_feats.shape}, val: {val_feats.shape}")
		
		train_feature_loader = DataLoader(
			TensorDataset(train_feats, train_lbls),
			batch_size=train_loader.batch_size,
			shuffle=True,
			num_workers=0,
		)

		val_feature_loader = DataLoader(
			TensorDataset(val_feats, val_lbls),
			batch_size=validation_loader.batch_size,
			shuffle=False,
			num_workers=0,
		)
	
	training_losses = list()
	validation_losses = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		probe.probe.train()  # only the linear head
		model.eval() # CLIP stays frozen, no gradients
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss = 0.0
		num_batches = 0
		
		# Choose data source
		data_iter = train_feature_loader if cache_features else train_loader
		
		for bidx, batch_data in enumerate(data_iter):
			if cache_features:
				# Using cached features
				image_embeds, label_vectors = batch_data
				image_embeds = image_embeds.to(device, non_blocking=True)
				label_vectors = label_vectors.to(device, non_blocking=True).float()
			else:
				images, _, label_vectors = batch_data
				images = images.to(device, non_blocking=True)
				label_vectors = label_vectors.to(device, non_blocking=True).float()
				
				# Extract image embeddings (frozen)
				with torch.no_grad():
					image_embeds = model.encode_image(images)
					image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
			
			optimizer.zero_grad(set_to_none=True)
			
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				# Linear probe forward (multi-label logits)
				logits = probe.probe(image_embeds)   
				
				# Apply pos_weight BCE, mask zero-count classes
				loss_raw = criterion(logits, label_vectors)  # [B, C], reduction='none'
				loss = loss_raw[:, active_mask].mean()
											
			# Check for NaN loss
			if torch.isnan(loss):
				print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
				continue
			
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(probe.probe.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			# Track losses
			epoch_loss  += loss.item()
			num_batches += 1
			
			if bidx % print_every == 0 or bidx + 1 == len(data_iter):
				print(f"\t\tBatch [{bidx+1:04d}/{len(data_iter)}] Loss: {loss.item():.6f}")
		
		# Calculate average losses
		avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
		training_losses.append(avg_loss)
		learning_rates_history.append([g['lr'] for g in optimizer.param_groups])
		weight_decays_history.append([g['weight_decay'] for g in optimizer.param_groups])
		print(f">> Training Elapsed Time: {time.time() - train_and_val_st_time:.1f}s. Validating Epoch {epoch+1} ...")
		
		probe.probe.eval()
		val_loss = 0.0
		val_preds_list, val_labels_list = list(), list()
		
		with torch.no_grad():
			val_iter = val_feature_loader if cache_features else validation_loader
			
			for batch_data in val_iter:
				if cache_features:
					image_embeds, label_vectors = batch_data
					image_embeds = image_embeds.to(device, non_blocking=True)
					label_vectors = label_vectors.to(device, non_blocking=True).float()
				else:
					images, _, label_vectors = batch_data										
					images = images.to(device, non_blocking=True)
					label_vectors = label_vectors.to(device, non_blocking=True).float()
					
					# Extract features
					image_embeds = model.encode_image(images)
					image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
				
				# Get predictions from probe
				logits = probe.probe(image_embeds)
				loss_raw = criterion(logits, label_vectors)
				val_loss += loss_raw[:, active_mask].mean().item()
				val_preds_list.append((torch.sigmoid(logits) > 0.5).float().cpu())
				val_labels_list.append(label_vectors.cpu())
		
		avg_val_loss = val_loss / len(val_iter)
		validation_losses.append(avg_val_loss)
		# Calculate multi-label metrics
		val_preds = torch.cat(val_preds_list, dim=0)
		val_labels = torch.cat(val_labels_list, dim=0)
		
		hamming = hamming_loss(val_labels.numpy(), val_preds.numpy())
		f1 = f1_score(val_labels.numpy(), val_preds.numpy(), average='weighted', zero_division=0)
		exact_match = (val_preds == val_labels).all(dim=1).float().mean().item()
		partial_match = (val_preds == val_labels).float().mean().item()
		epoch_metrics = {
			"val_loss":       avg_val_loss,
			"hamming_loss":   hamming,
			"f1_score":       f1,
			"exact_match_acc": exact_match,
			"partial_acc":    partial_match,
		}
		full_val_loss_acc_metrics_all_epochs.append(epoch_metrics)
		if cache_features:
			sample_feats = train_feats[:512].to(device)        # [512, 768] image features
			# Get true class indices for these 512 samples from the loader
			# For each sample, average the W rows corresponding to its true labels
			sample_labels = train_lbls[:512].to(device)  # [512, C] multi-hot
			
			W = torch.nn.functional.normalize(probe.probe.weight, dim=-1)        # [C, 768]
			matched_class_vecs = torch.zeros(512, W.shape[1], device=device)

			for i in range(512):
				pos_idx = sample_labels[i].nonzero(as_tuple=True)[0]
				if pos_idx.numel() > 0:
					matched_class_vecs[i] = W[pos_idx].mean(dim=0)
			
			matched_class_vecs = torch.nn.functional.normalize(matched_class_vecs, dim=-1)
			sample_feats_norm = torch.nn.functional.normalize(sample_feats, dim=-1)
			cos_sim = torch.nn.functional.cosine_similarity(
				sample_feats_norm, 
				matched_class_vecs, 
				dim=1
			).mean().item()

			# AlignScore@5 against learned W — comparable to PEFT methods
			align_score = get_multilabel_alignment_score(
				image_embeds=sample_feats_norm,
				all_class_embeds=W, # probe W, not frozen text embeds
				labels=sample_labels,
				temperature=temperature,
				topk=5,
				verbose=verbose,
			)
		else:
			cos_sim = 0
			align_score = 0
		
		print(
			f"\nEpoch {epoch+1}:\n"
			f"  [LOSS] {mode.upper()} — Train: {avg_loss} Val: {avg_val_loss}\n"
			f"  Hamming Loss: {hamming:.4f}\n"
			f"  F1 Score: {f1:.4f}\n"
			f"  ExactMatch: {exact_match} PartialAcc: {partial_match:.4f}\n"
			f"  LR: {scheduler.get_last_lr()}"
		)
		if align_score is not None:
			print(f"  AlignScore@5 (W): {align_score:.4f}")
		if cos_sim is not None:
			print(f"  CosSim (W):: {cos_sim:.4f}")
		
		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=probe.probe,  # ← probe weights, not full CLIP
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break


		if early_stopping.should_stop(
			current_value=avg_val_loss,
			model=probe, # full MultiLabelProbe: saves {probe.weight, probe.bias, clip_model.*}
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}"
			)
			break
		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f} sec")
	
	print(f"[{mode}] Total Time: {time.time() - train_start_time:.1f} sec".center(170, "-"))
	# Load best probe weights
	if os.path.exists(mdl_fpth):
		print(f"Loading best probe weights from {mdl_fpth}")
		checkpoint = torch.load(mdl_fpth, map_location=device)
		state_dict = checkpoint.get('model_state_dict', checkpoint)
		probe.load_state_dict(state_dict, strict=False)
	elif early_stopping.best_weights is not None:
		print(f"Loading best weights from early stopping (epoch {early_stopping.best_epoch+1})")
		probe.load_state_dict(
			{k: v.to(device) for k, v in early_stopping.best_weights.items()}
		)
	else:
		print("Warning: No best weights found - using final weights")
	
	# Final evaluation
	# pass probe (not model) — probe.encode_image and probe.encode_text
	# delegate to frozen CLIP, so get_validation_metrics works correctly.
	# The probe's W matrix is now the fine-tuned one; similarity is computed
	# as probe.encode_image(img) @ probe.encode_text(class).T, which is
	# equivalent to image_embed @ W.T since W was initialised from text embeds.
	evaluation_results = evaluate_best_model(
		model=probe, # not model — probe wraps model with trained W
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		temperature=temperature,
		class_embeds_override=probe.probe.weight.detach().clone(),
		verbose=verbose,
	)

	final_metrics_full    = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	final_tiered_i2t      = evaluation_results["tiered_i2t"]
	final_tiered_t2i      = evaluation_results["tiered_t2i"]
	model_source          = evaluation_results["model_loaded_from"]

	# Update model path
	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	if verbose:
		print(f"{'='*50}")
		print(f"{mode.upper()} Final evaluation from: {model_source}")
		print(f"  Model: {model_arch}")
		print(f"  {probe.probe_type} | Params: {sum(p.numel() for p in probe.probe.parameters()):,}")
		print(f"  CLIP frozen params: {sum(p.numel() for p in model.parameters()):,}")
		print(f"  Epochs trained: {actual_trained_epochs}")
		print(f"  Best val loss: {early_stopping.get_best_score():.6f} @ Epoch {early_stopping.get_best_epoch()+1}")
		print(f"  Best model: {mdl_fpth}")
		print("\n>> Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print("\n>> Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")

	# Generate plots
	file_base_name = (
		f"{mode}_{probe.probe_type}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}"
	)
	
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
		"hp_evol": os.path.join(results_dir, f"{file_base_name}_hyperparameter_evolution.png"),
	}
	
	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)
	
	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)
	
	return final_tiered_i2t, final_tiered_t2i

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
	temperature: float = 0.07,
	topk_values: List[int] = [1, 5, 10, 15, 20],
	loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
	verbose: bool=True,
):
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)
		
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

	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)
	
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"{mode.upper()}-FT")
		print(f"  ├─ {model_name} {model_arch}")
		print(f"  ├─ {dataset_name} {num_classes} classes")
		print(f"  ├─ Epochs: {num_epochs}  Batch size: {train_loader.batch_size}  Device: {type(device)} {device}")
		print(f"  ├─ Learning rate: {learning_rate}  Weight decay: {weight_decay}  Patience: {patience}")
		print(f"  ├─ Loss weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		print(f"  ├─ Temperature: {temperature}")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		# check with cuda capability
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16

		if verbose:
			print(f"  └─ {gpu_name} {total_mem:.2f}GB VRAM cuda capability: {cuda_capability}")
	
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

	if non_zero_dropouts:
		print(f"\nNon-zero dropout detected in base {model_name} {model_arch} during {mode}:")
		print(non_zero_dropouts)
		print()

	# Freeze all parameters first
	for param in model.parameters():
		param.requires_grad = False
	
	# Unfreeze only vision encoder parameters
	for param in model.visual.parameters():
		param.requires_grad = True

	print("\nModel parameters before training:")
	for n, p in model.named_parameters():
		print(f"{n:<60}{p.requires_grad:<5}{p.dtype}\t{p.shape}")
	print("="*130)

	get_parameters_info(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="linear", 
		pw_max_cap=100.0,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	val_freq = diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)

	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min()}, {pos_weight.max()}]")
		print(f"   ├─ number of samples: {N}")
		print(f"   ├─ number of classes: {num_classes}")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,} / {num_classes:,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(
		reduction='none',
	)

	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	model.eval()
	all_class_embeds = []
	text_batch_size = validation_loader.batch_size
	if verbose:
		print(f"\nPre-encoding {num_classes} class texts in batch_size: {text_batch_size}")
	with torch.no_grad():
		with torch.amp.autocast(
			device_type=device.type, 
			enabled=torch.cuda.is_available(),
			dtype=amp_dtype,
		):
			for i in range(0, num_classes, text_batch_size):
				end_idx = min(i + text_batch_size, num_classes)
				batch_class_names = class_names[i:end_idx]

				batch_class_texts = clip.tokenize(batch_class_names).to(device)
				batch_embeds = model.encode_text(batch_class_texts)
				batch_embeds = torch.nn.functional.normalize(batch_embeds, dim=-1)
				all_class_embeds.append(batch_embeds.cpu())  # Move to CPU immediately to save GPU memory
				
				# Clean up
				del batch_class_texts, batch_embeds
				torch.cuda.empty_cache()
	
	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device).detach()
	
	if verbose:
		print(f"All {num_classes} classes Embeddings (frozen text encoder)")
		print(f"   ├─ {type(all_class_embeds)}")
		print(f"   ├─ {all_class_embeds.shape}")
		print(f"   ├─ {all_class_embeds.dtype}")
		print(f"   └─ {all_class_embeds.device}")

	# Optimizer
	full_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(
		params=full_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in full_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=T_max,
		eta_min=eta_min,
		last_epoch=-1,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**11,      # 2048 (Conservative start)
		growth_factor=1.5,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=5000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'} (cuda_cap: {cuda_capability})")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_name}_"
		f"{model_arch}_"
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
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	train_start_time = time.time()
	final_img2txt_metrics = None
	final_txt2img_metrics = None

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0

		for bidx, batch_data in enumerate(train_loader):
			images, _, label_vectors = batch_data  # Ignore tokenized_labels, use pre-encoded
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			optimizer.zero_grad(set_to_none=True)
			
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
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
			torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
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
					f"\t\tBatch [{bidx + 1:04d}/{len(train_loader)}] "
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

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f">> Training epoch {epoch+1} took {time.time() - train_and_val_st_time:.2f} sec. Validating Epoch {epoch+1}")

		# clear cache before validation
		torch.cuda.empty_cache()
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)
		
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score   = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])

		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {mode}-FT: Train — Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f}) Val: {current_val_loss:.6f}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')

		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")

	print(f"[{mode}] Total Training Elapsed Time: {time.time() - train_start_time:.1f} sec")

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params=None,
		topk_values=topk_values,
		temperature=temperature,
		verbose=verbose,
	)

	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	final_tiered_i2t = evaluation_results["tiered_i2t"]
	final_tiered_t2i = evaluation_results["tiered_t2i"]
	model_source = evaluation_results["model_loaded_from"]

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	if verbose:
		print(f"{'='*50}")
		print(f"{mode.upper()} Final evaluation from: {model_source}")
		print(f"  Model: {model_arch}")
		print(f"  CLIP frozen params: {sum(p.numel() for p in model.parameters()):,}")
		print(f"  Epochs trained: {actual_trained_epochs}")
		print(f"  Best val loss: {early_stopping.get_best_score():.6f} @ Epoch {early_stopping.get_best_epoch()+1}")
		print(f"  Best model: {mdl_fpth}")
		print("\n>> Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print("\n>> Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}"
	)
	
	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"hp_evol": os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	}
	
	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
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

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	return final_tiered_i2t, final_tiered_t2i

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
	loss_weights: Dict[str, float] = None,
	temperature: float = 0.07,
	quantization_bits: int = 8,
	quantized: bool = False,
	verbose: bool = True,
):
	window_size = minimum_epochs + 1
	if loss_weights is None:
			loss_weights = {"i2t": 0.5, "t2i": 0.5}

	# ── Dropout check 
	non_zero_dropouts = [
			(name, module.p)
			for name, module in model.named_modules()
			if isinstance(module, torch.nn.Dropout) and module.p > 0
	]
	if non_zero_dropouts:
			dropout_info = ", ".join([f"{n}: p={p}" for n, p in non_zero_dropouts])
			assert False, (
					f"Non-zero dropout in base model during LoRA: {dropout_info}\n"
					"Set dropout=0.0 in clip.load() before LoRA injection."
			)
	
	mode = re.sub(r'_finetune_multi_label', '', inspect.stack()[0].function)
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name
	
	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)
	
	if verbose:
		print(f"\n{mode.upper()}")
		print(f"   ├─ {model_name} {model_arch}")
		print(f"   ├─ Rank: {lora_rank}")
		print(f"   ├─ Alpha: {lora_alpha}")
		print(f"   ├─ Dropout: {lora_dropout}")
		print(f"   ├─ {dataset_name} classes: {num_classes}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss Weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		# check with cuda capability
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		
		if verbose:
			print(f"   └─ {gpu_name} | {total_mem:.1f}GB VRAM | cuda capability: {cuda_capability}")
		
	# LoRA injection — vision encoder only
	# Text encoder stays frozen and un-injected, consistent with full fine-tuning.
	# all_class_embeds pre-computed from frozen text encoder remains valid.
	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		target_text_modules=[], # no LoRA in text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	).to(device)
	
	get_parameters_info(model=model, mode=mode)
	
	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="sqrt",
		pw_max_cap=50.0,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	val_freq = diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min()}, {pos_weight.max()}]")
		print(f"   ├─ number of samples: {N}")
		print(f"   ├─ number of classes: {num_classes}")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,} / {num_classes:,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(
		reduction='none',
	)
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")
	
	# ── Pre-encode class texts (frozen text encoder — valid for entire run) ──
	model.eval()
	all_class_embeds = []
	text_batch_size = validation_loader.batch_size
	if verbose:
		print(f"\nPre-encoding {num_classes} class texts in batch_size: {text_batch_size}")
	with torch.no_grad():
		with torch.amp.autocast(
			device_type=device.type, 
			enabled=torch.cuda.is_available(),
			dtype=amp_dtype,
		):
			for i in range(0, num_classes, text_batch_size):
				batch_tokens = clip.tokenize(class_names[i:i+text_batch_size]).to(device)
				embeds = model.encode_text(batch_tokens)
				embeds = torch.nn.functional.normalize(embeds, dim=-1)
				all_class_embeds.append(embeds.cpu())

				del batch_tokens, embeds
				torch.cuda.empty_cache()
	
	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device).detach()
	
	if verbose:
		print(f"All {num_classes} classes Embeddings (frozen text encoder)")
		print(f"   ├─ {type(all_class_embeds)}")
		print(f"   ├─ {all_class_embeds.shape}")
		print(f"   ├─ {all_class_embeds.dtype}")
		print(f"   └─ {all_class_embeds.device}")

	# ── Early stopping
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

	# Optimizer — LoRA parameters only
	lora_params = [p for p in model.parameters() if p.requires_grad]	
	optimizer = torch.optim.AdamW(
		params=lora_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)
	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in lora_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=T_max,
		eta_min=eta_min,
		last_epoch=-1,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**11,      # 2048 (Conservative start)
		growth_factor=1.5,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=5000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'} (cuda_cap: {cuda_capability})")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ieps_{num_epochs}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_"
		f"lor_{lora_rank}_loa_{lora_alpha}_lod_{lora_dropout}_"
		f"temp_{temperature}_bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_pat_{patience}_"
		f"mdt_{min_delta:.1e}_cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}.pth"
	)

	training_losses = []
	validation_losses = []
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history = []
	train_start_time = time.time()

	for epoch in range(num_epochs):
			train_and_val_st_time = time.time()
			torch.cuda.empty_cache()
			model.train()
			print(f"Epoch [{epoch+1}/{num_epochs}]")
			epoch_loss_total = epoch_loss_i2t = epoch_loss_t2i = 0.0
			num_batches = 0

			for bidx, batch_data in enumerate(train_loader):
				images, _, label_vectors = batch_data
				images = images.to(device, non_blocking=True)
				label_vectors = label_vectors.to(device, non_blocking=True).float()
				optimizer.zero_grad(set_to_none=True)

				with torch.amp.autocast(
					device_type=device.type, 
					enabled=torch.cuda.is_available(),
					dtype=amp_dtype,
				):
					total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
						model=model,
						images=images,
						all_class_embeds=all_class_embeds,
						label_vectors=label_vectors,
						criterion_i2t=criterion_i2t,
						criterion_t2i=criterion_t2i,
						active_mask=active_mask,
						temperature=temperature,
						loss_weights=loss_weights,
						verbose=verbose,
					)

				if torch.isnan(total_loss):
					print(f"Warning: NaN loss at epoch {epoch+1}, batch {bidx+1}. Skipping.")
					continue

				scaler.scale(total_loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(
					[p for p in model.parameters() if p.requires_grad], 
					max_norm=1.0
				)

				scaler.step(optimizer)
				scaler.update()
				scheduler.step()
				epoch_loss_total += total_loss.item()
				epoch_loss_i2t   += loss_i2t.item()
				epoch_loss_t2i   += loss_t2i.item()

				num_batches += 1
				if bidx % print_every == 0 or bidx + 1 == len(train_loader):
					print(
						f"\t\tBatch [{bidx+1:04d}/{len(train_loader)}] "
						f"Total: {total_loss.item():.6f} "
						f"(I2T: {loss_i2t.item():.6f}, T2I: {loss_t2i.item():.6f})"
					)

					b_norms = [
						p.data.norm().item()
						for n, p in model.named_parameters()
						if p.requires_grad and "lora_B" in n
					]
					if b_norms:
						b_norms_t = torch.tensor(b_norms)
						print(
							f"\t\t[B weight norms] "
							f"(min, max): ({b_norms_t.min():.4f}, {b_norms_t.max():.4f}) "
							f"mean: {b_norms_t.mean():.4f} std: {b_norms_t.std():.4f}"
						)
						print()

			avg_total = epoch_loss_total / num_batches if num_batches > 0 else 0.0
			avg_i2t   = epoch_loss_i2t   / num_batches if num_batches > 0 else 0.0
			avg_t2i   = epoch_loss_t2i   / num_batches if num_batches > 0 else 0.0

			training_losses.append(avg_total)
			training_losses_breakdown["total"].append(avg_total)
			training_losses_breakdown["i2t"].append(avg_i2t)
			training_losses_breakdown["t2i"].append(avg_t2i)

			learning_rates_history.append([optimizer.param_groups[0]['lr']])
			weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

			print(f">> Training epoch {epoch+1} took {time.time() - train_and_val_st_time:.2f} sec. Validating Epoch {epoch+1} ...")

			current_val_loss = compute_multilabel_validation_loss(
				model=model,
				validation_loader=validation_loader,
				criterion_i2t=criterion_i2t,
				criterion_t2i=criterion_t2i,
				active_mask=active_mask,
				device=device,
				all_class_embeds=all_class_embeds,
				temperature=temperature,
				verbose=verbose,
			)
			validation_losses.append(current_val_loss)

			validation_results = get_validation_metrics(
				model=model,
				validation_loader=validation_loader,
				device=device,
				topK_values=topk_values,
				finetune_strategy=mode,
				cache_dir=results_dir,
				lora_params={
					"lora_rank": lora_rank,
					"lora_alpha": lora_alpha,
					"lora_dropout": lora_dropout,
				},
				is_training=True,
				model_hash=get_model_hash(model),
				temperature=temperature,
				verbose=verbose,
			)
			
			full_val_metrics = validation_results["full_metrics"]
			img2txt_metrics  = validation_results["img2txt_metrics"]
			txt2img_metrics  = validation_results["txt2img_metrics"]
			full_val_loss_acc_metrics_all_epochs.append(full_val_metrics)
			cos_sim = full_val_metrics.get("cosine_similarity")
			align_score = full_val_metrics.get("alignment_score")

			img2txt_metrics_all_epochs.append(img2txt_metrics)
			txt2img_metrics_all_epochs.append(txt2img_metrics)
			i2t_map10 = img2txt_metrics.get("mAP", {}).get("10", float("nan"))
			t2i_map10 = txt2img_metrics.get("mAP", {}).get("10", float("nan"))
			i2t_r10 = img2txt_metrics.get("Recall", {}).get("10", float("nan"))
			t2i_r10 = txt2img_metrics.get("Recall", {}).get("10", float("nan"))

			print(
				f"\nEpoch {epoch+1}:\n"
				f"  [LOSS] — {mode.upper()}-FT Train: {avg_total:.4f} (I2T: {avg_i2t}, T2I: {avg_t2i}) Val: {current_val_loss}\n"
				f"  I2T    — mAP@10: {i2t_map10:.4f}  R@10: {i2t_r10:.4f}\n"
				f"  T2I    — mAP@10: {t2i_map10:.4f}  R@10: {t2i_r10:.4f}\n"
				f"  LR     — {scheduler.get_last_lr()[0]}"
			)
			if align_score is not None:
				print(f"  Embed — AlignScore@5: {align_score:.4f}")
			elif cos_sim is not None:
				print(f"  Embed — CosSim: {cos_sim:.4f}")
			else:
				print(f"  Embed — AlignScore: N/A")


			# Training health check
			# Run after epoch 1 and at mid-warmup — all signals now available
			if epoch in {0, minimum_epochs // 2}:
				should_abort = check_training_health(
					model=model,
					epoch=epoch,
					mode=mode,
					training_losses=training_losses,
					validation_losses=validation_losses,
					align_score=align_score,
					temperature=temperature,
					learning_rate=learning_rate,
					verbose=verbose,
				)
				if should_abort:
					print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
					break

			if early_stopping.should_stop(
					current_value=current_val_loss,
					model=model,
					epoch=epoch,
					optimizer=optimizer,
					scheduler=scheduler,
					checkpoint_path=mdl_fpth,
			):
					print(
						f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
						f"@ epoch {early_stopping.get_best_epoch()+1}"
					)
					break
			
			print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time()-train_and_val_st_time:.1f}s")

	print(f"[{mode}] Total elapsed: {time.time()-train_start_time:.1f}s")

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params={
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout,
		},
		temperature=temperature,
		topk_values=topk_values,
		verbose=verbose,
	)

	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	final_tiered_i2t        = evaluation_results["tiered_i2t"]
	final_tiered_t2i        = evaluation_results["tiered_t2i"]
	model_source = evaluation_results["model_loaded_from"]

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	if verbose:
		print(f"{'='*50}")
		print(f"{mode.upper()} Final evaluation from: {model_source}")
		print(f"  Model: {model_arch}")
		print(f"  CLIP frozen params: {sum(p.numel() for p in model.parameters()):,}")
		print(f"  Epochs trained: {actual_trained_epochs}")
		print(f"  Best val loss: {early_stopping.get_best_score():.6f} @ Epoch {early_stopping.get_best_epoch()+1}")
		print(f"  Best model: {mdl_fpth}")
		print("\n>> Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print("\n>> Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_arch}_ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_temp_{temperature}_"
		f"bs_{train_loader.batch_size}_lor_{lora_rank}_loa_{lora_alpha}_lod_{lora_dropout}"
	)

	plot_paths = {
		"losses_breakdown":   os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"losses":             os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"full_val_topk_i2t":  os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i":  os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch":os.path.join(results_dir, f"{file_base_name}_retrieval_per_epoch.png"),
		"retrieval_best":     os.path.join(results_dir, f"{file_base_name}_retrieval_best.png"),
		"hp_evol":            os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"],
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
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return final_tiered_i2t, final_tiered_t2i

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
	lora_plus_lambda: float,
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
	loss_weights: Dict[str, float]=None,
	temperature: float = 0.07,
	verbose: bool=True,
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
		loss_weights: Optional weights for I2T and T2I losses
		temperature: Temperature scaling for similarities
		verbose: Enable detailed logging
	"""	
	window_size = minimum_epochs + 1	
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
	
	# Get dataset information
	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)	

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)
	
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	
	if verbose:
		print(f"\n{mode.upper()} {model_name} {model_arch}")
		print(f"   ├─ {dataset_name} {num_classes} classes")
		print(f"   ├─ Batch size: {train_loader.batch_size}  Device: {type(device)} {device}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")
		
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability(device)
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   ├─ {gpu_name} {device}")
			print(f"   ├─ AMP dtype: {amp_dtype}")
			print(f"   ├─ Total Memory: {gpu_total_mem:.2f}GB")
			print(f"   └─ CUDA Capability: {cuda_capability}")
	
	# Apply LoRA+ to the model
	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		lora_plus_lambda=lora_plus_lambda,
		target_text_modules=[], # no LoRA+ in text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	).to(device)
	
	get_parameters_info(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="sqrt",
		pw_max_cap=50.0,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)

	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min()}, {pos_weight.max()}]")
		print(f"   ├─ number of samples: {N}")
		print(f"   ├─ number of classes: {num_classes}")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,} / {num_classes:,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(
		reduction='none',
	)

	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	# Separate LoRA A and B parameters for differential LR
	lora_A_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
	lora_B_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
	# Sanity check — nothing should fall through
	all_lora_params = set(id(p) for p in lora_A_params + lora_B_params)
	ungrouped = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in all_lora_params]
	if ungrouped:
		raise ValueError(f"LoRA+: trainable params not in lora_A or lora_B groups: {ungrouped}")

	if verbose:
		print(f"\n{mode.upper()} Parameter Groups:")
		print(f"  ├─ lora_A params: {len(lora_A_params)} tensors")
		print(f"  ├─ lora_B params: {len(lora_B_params)} tensors")
		print(f"  ├─ All lora params: {len(all_lora_params)} tensors")
		print(f"  ├─ Unassigned params: {len(ungrouped)} tensors")
		print(f"  └─ LR multiplier (λ): {lora_plus_lambda}")
	
	# Optimizer with differential learning rates
	lora_A_lr = learning_rate
	lora_A_wd = weight_decay
	lora_B_lr = learning_rate * lora_plus_lambda
	lora_B_wd = 0.0

	optimizer = torch.optim.AdamW(
		params=[
			{
				'params': lora_A_params,
				'lr': lora_A_lr,
				'weight_decay': lora_A_wd,
			},
			{
				'params': lora_B_params,
				'lr': lora_B_lr,
				'weight_decay': lora_B_wd,
			},
		],
	)

	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ LR: lora_A = {lora_A_lr} lora_B = {lora_B_lr} (λ={lora_plus_lambda})")
		print(f"  ├─ WD: lora_A = {lora_A_wd} lora_B = {lora_B_wd}")
		print(f"  └─ Params: lora_A: {sum(p.numel() for p in lora_A_params):,} lora_B: {sum(p.numel() for p in lora_B_params):,}")
	
	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=T_max,
		eta_min=eta_min,
		last_epoch=-1,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")

	is_ampere_or_newer = cuda_capability[0] >= 8

	if is_ampere_or_newer:
		B_MAX_NORM = 50.0
		amp_enabled = True
		scaler = torch.amp.GradScaler(
			device=device,
			init_scale=2**11,
			growth_factor=1.5,
			backoff_factor=0.5,
			growth_interval=5000,
		)
	else:
		B_MAX_NORM = 10.0
		amp_enabled = False  # AMP disabled — FP16 on V100 is unstable for ViT-L LoRA+
		scaler = None

	if verbose:
		print(f"\nAMP Configuration:")
		print(f"  ├─ Enabled: {amp_enabled}")
		print(f"  ├─ dtype: {amp_dtype if amp_enabled else 'FP32 (AMP disabled)'}")
		print(f"  ├─ B_MAX_NORM: {B_MAX_NORM}")
		if scaler is not None:
			scaler_state = scaler.state_dict()
			print(f"  ├─ GradScaler init_scale: {scaler_state.get('scale', 'N/A')}")
			print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
			print(f"  └─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		else:
			print(f"  └─ GradScaler: disabled")

	# ── Pre-encode class texts (frozen text encoder — valid for entire run) ──
	model.eval()
	all_class_embeds = []
	text_batch_size = validation_loader.batch_size
	print(f"\nPre-encoding {num_classes} class texts in batch_size: {text_batch_size}")
	with torch.no_grad():
		with torch.amp.autocast(
			device_type=device.type, 
			enabled=amp_enabled,#torch.cuda.is_available(),
			dtype=amp_dtype,
		):
			for i in range(0, num_classes, text_batch_size):
				batch_tokens = clip.tokenize(class_names[i:i+text_batch_size]).to(device)
				embeds = model.encode_text(batch_tokens)
				embeds = torch.nn.functional.normalize(embeds, dim=-1)
				all_class_embeds.append(embeds.cpu())

				del batch_tokens, embeds
				torch.cuda.empty_cache()

	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device).detach()

	if verbose:
		print(f"All {num_classes} classes Embeddings (frozen text encoder)")
		print(f"   ├─ {type(all_class_embeds)}")
		print(f"   ├─ {all_class_embeds.shape}")
		print(f"   ├─ {all_class_embeds.dtype}")
		print(f"   └─ {all_class_embeds.device}")




	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_name}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"lr_A_{lora_A_lr:.1e}_"
		f"wd_A_{lora_A_wd:.1e}_"
		f"lr_B_{lora_B_lr:.1e}_"
		f"wd_B_{lora_B_wd}_"
		f"B_norm_max_{B_MAX_NORM}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"lmbd_{lora_plus_lambda}_"
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
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history = []
	train_start_time = time.time()
	
	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		model.train()
		print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0
		
		for bidx, batch_data in enumerate(train_loader):
			images, _, label_vectors = batch_data
			
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			if torch.isnan(images).any():
				if verbose:
					print(f"[WARNING] Corrupted image detected in batch {bidx+1}. Skipping.")
				continue

			optimizer.zero_grad(set_to_none=True)
			
			with torch.amp.autocast(
				device_type=device.type,
				enabled=amp_enabled,#torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				# Multi-label contrastive loss computation
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)
			
			# Check for NaN loss
			if torch.isnan(total_loss) or torch.isinf(total_loss):
				print(
					f"[WARNING] e{epoch+1} b{bidx+1}: "
					f"total_loss: {total_loss} "
					f"loss_i2t: {loss_i2t} "
					f"loss_t2i: {loss_t2i}"
				)

				# no zero_grad needed here — already done above
				continue # skip this batch
			
			# Backward Pass
			if scaler is not None:
				scaler.scale(total_loss).backward()
			else:
				total_loss.backward()
			
			# Gradient Clipping (Vital for LoRA+)
			if scaler is not None:
				scaler.unscale_(optimizer)

			grad_norm = torch.nn.utils.clip_grad_norm_(
				[p for p in model.parameters() if p.requires_grad], 
				max_norm=1.0
			)

			# Grad norm check — post-clipping
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				grad_norms_A, grad_norms_B = [], []
				for name, param in model.named_parameters():
					if param.requires_grad and param.grad is not None:
						gn = param.grad.norm().item()
						if "lora_A" in name:
							grad_norms_A.append(gn)
						elif "lora_B" in name:
							grad_norms_B.append(gn)
				
				if grad_norms_A and grad_norms_B:
					print(
						f"\t\t[Grad norms e{epoch+1} b{bidx+1}] "
						f"(min, max) "
						f"A: ({min(grad_norms_A):.4f}, {max(grad_norms_A):.4f}) "
						f"B: ({min(grad_norms_B):.4f}, {max(grad_norms_B):.4f})"
					)

			# Guard: skip optimizer step if grads are still corrupt post-unscale
			# fires only on FP16 (V100 and older) due to overflow.
			# On BF16/Ampere this is defensive dead code — kept for correctness.
			if torch.isnan(grad_norm) or torch.isinf(grad_norm):
				if verbose:
					print(f"[WARNING] Corrupt grad_norm at epoch {epoch+1} batch {bidx+1}: {grad_norm} | isnan(): {torch.isnan(grad_norm)} | isinf(): {torch.isinf(grad_norm)}")

				optimizer.zero_grad(set_to_none=True)

				# purge Adam state for any param whose grad was NaN
				for group in optimizer.param_groups:
					for p in group['params']:
						if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
							if p in optimizer.state:
								optimizer.state[p] = {}

				if scaler is not None:
					if cuda_capability[0] < 8:
						if hasattr(scaler, '_scale') and scaler._scale is not None:
							# scaler._scale.mul_(0.5)
							current_scale = scaler._scale.item()
							new_scale = max(current_scale * 0.5, 1.0)  # never go below 1.0
							scaler._scale.fill_(new_scale)

					scaler.update()  # must still call to reset internal state
					if verbose:
						print(f"\t\t[Scaler] scale after corrupt grad recovery={scaler.get_scale():.1f}")

				continue

			# Step the optimizer (Generic for Scaler or None)
			if scaler is not None:
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()

			# Clip B matrix norms (magnitude control)
			clipped_count = 0
			clipped_details = []
			with torch.no_grad():
				for name, param in model.named_parameters():
					if param.requires_grad and "lora_B" in name:
						norm = param.data.norm()
						if norm > B_MAX_NORM:
							clipped_count += 1
							clipped_details.append((name, norm.item()))
							param.data.mul_(B_MAX_NORM / norm)

			# Only log clipping details at print_every intervals, not every batch
			if clipped_count > 0 and (bidx % print_every == 0 or bidx + 1 == len(train_loader)):
				print(f"\t\t[B-norm clip] {clipped_count} layers clipped at e{epoch+1} b{bidx+1}:")
				for cname, cnorm in clipped_details[:5]:  # cap at 5 to avoid log spam
					print(f"\t\t  {cname}: {cnorm:.4f} → {B_MAX_NORM}")

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
					f"\t\tBatch [{bidx + 1:04d}/{len(train_loader)}] "
					f"Total Loss: {batch_loss_total:.6f} "
					f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})\n"
				)
		
		# Detect full NaN cascade — entire epoch was skipped
		if num_batches == 0:
			print(f"[CRITICAL] Epoch {epoch+1}: zero valid batches — full NaN cascade detected.")
			# Clear corrupted Adam momentum buffers
			for group in optimizer.param_groups:
					for p in group['params']:
							if p in optimizer.state:
									optimizer.state[p] = {}
			# Restore best weights if available
			if early_stopping.best_weights is not None:
					early_stopping._restore_best_weights(model)
					print(f"  Restored weights from epoch {early_stopping.get_best_epoch()+1}.")
			else:
					print(f"  No checkpoint available. Aborting.")
			break

		# average losses
		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0
		
		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)
		
		# Track learning rates (now we have multiple parameter groups)
		# for i, g in enumerate(optimizer.param_groups):
		# (['params', 'lr', 'weight_decay', 'betas', 'eps', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused', 'decoupled_weight_decay', 'initial_lr'])
		# 	print(i, g.keys())

		learning_rates_history.append([group['lr'] for group in optimizer.param_groups])
		weight_decays_history.append([group['weight_decay'] for group in optimizer.param_groups])
		
		if verbose:
			print(f"{len(learning_rates_history[-1])} LR groups: {learning_rates_history[-1]}")
			print(f"{len(weight_decays_history[-1])} WD groups: {weight_decays_history[-1]}")
		
		# Weight health check before validation
		healthy, A_norms, B_norms = check_lora_weight_health(
			model=model, 
			optimizer=optimizer,  # pass optimizer
			verbose=verbose,
		)

		if not healthy:
			# Clear corrupted Adam states first
			for group in optimizer.param_groups:
					for p in group['params']:
							if p in optimizer.state:
									optimizer.state[p] = {}
			# Then restore weights
			if early_stopping.best_weights is not None:
					early_stopping._restore_best_weights(model)
			break

		print(f">> Training epoch {epoch+1} took {time.time() - train_and_val_st_time:.2f} sec. Validating Epoch {epoch+1} ...")		
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)
		
		# empty cache:
		torch.cuda.empty_cache()

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
				"lora_plus_lambda": lora_plus_lambda,
			},
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score   = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])

		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {mode}-FT: Train - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f}) Val: {current_val_loss:.6f}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')

		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break


		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break
		
		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")
	
	print(f"[{mode}] Total Elapsed Time: {time.time() - train_start_time:.1f}s")
	
	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params={
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout,
			"lora_plus_lambda": lora_plus_lambda,
		},
		temperature=temperature,
		topk_values=topk_values,
		verbose=verbose,
	)
	
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	final_tiered_i2t = evaluation_results["tiered_i2t"]
	final_tiered_t2i = evaluation_results["tiered_t2i"]
	model_source = evaluation_results["model_loaded_from"]

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	if verbose:
		print(f"{'='*50}")
		print(f"{mode.upper()} Final evaluation from: {model_source}")
		print(f"  {model_arch} frozen params: {sum(p.numel() for p in model.parameters()):,}")
		print(f"  Rank: {lora_rank} Alpha: {lora_alpha} Dropout: {lora_dropout} Lambda: {lora_plus_lambda} B_MAX_NORM: {B_MAX_NORM}")
		print(f"  Total trained epochs: {actual_trained_epochs}")
		print(f"  Best val loss: {early_stopping.get_best_score():.6f} @ Epoch {early_stopping.get_best_epoch()+1}")
		print(f"  Best model: {mdl_fpth}")
		print("\n>> Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print("\n>> Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")
	
	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_A_{lora_A_lr:.1e}_"
		f"wd_A_{lora_A_wd:.1e}_"
		f"lmbd_{lora_plus_lambda}_"
		f"lr_B_{lora_B_lr:.1e}_"
		f"wd_B_{lora_B_wd}_"
		f"B_norm_max_{B_MAX_NORM}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}"
	)
	
	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"hp_evol": os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	}
	
	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
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
	
	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	return final_tiered_i2t, final_tiered_t2i

def rslora_finetune_multi_label(
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
	loss_weights: Dict[str, float] = None,
	temperature: float = 0.07,
	quantization_bits: int = 8,
	quantized: bool = False,
	verbose: bool = True,
):
	"""
	rsLoRA multi-label fine-tuning for CLIP.

	Identical to lora_finetune_multi_label() in every respect except that
	LoRA's standard scaling  α/r  is replaced by the rank-stabilised variant
		α / √r   (rsLoRA, Kalajdzievski 2023)

	This keeps gradient norms bounded as rank grows, which is particularly
	useful when sweeping lora_rank or when training at higher ranks (r ≥ 16)
	where standard LoRA can exhibit loss spikes or slower convergence.

	All parameters are identical to lora_finetune_multi_label(); the only
	internal difference is passing method="rslora" to get_injected_peft_clip(),
	which sets self.scale = alpha / (rank ** 0.5) inside LoRALinear.
	"""
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}

	# ── Dropout check
	non_zero_dropouts = [
		(name, module.p)
		for name, module in model.named_modules()
		if isinstance(module, torch.nn.Dropout) and module.p > 0
	]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{n}: p={p}" for n, p in non_zero_dropouts])
		assert False, (
			f"Non-zero dropout in base model during rsLoRA: {dropout_info}\n"
			"Set dropout=0.0 in clip.load() before rsLoRA injection."
		)

	mode = re.sub(r'_finetune_multi_label', '', inspect.stack()[0].function)  # → "rslora"
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)

	if verbose:
		print(f"\n{mode.upper()}")
		print(f"   ├─ {model_name} {model_arch}")
		print(f"   ├─ Rank           : {lora_rank}")
		print(f"   ├─ Alpha (input)  : {lora_alpha}")
		print(f"   ├─ Dropout        : {lora_dropout}")
		print(f"   ├─ Dataset        : {dataset_name}  classes: {num_classes}")
		print(f"   ├─ Batch size     : {train_loader.batch_size}")
		print(f"   ├─ Device         : {type(device)} {device}")
		print(f"   ├─ Temperature    : {temperature}")
		print(f"   ├─ Loss Weights   : I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	# ── rsLoRA injection — vision encoder only
	# Text encoder stays frozen and un-injected.
	# all_class_embeds pre-computed from frozen text encoder remains valid.
	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		target_text_modules=[],               # no rsLoRA in text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	).to(device)
	
	get_parameters_info(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="sqrt",
		pw_max_cap=50.0,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N           = masks["N"]
	train_freq  = masks["train_freq"]

	val_freq = diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	# Criteria
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,
		reduction='none',
	)
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min()}, {pos_weight.max()}]")
		print(f"   ├─ samples: {N} classes: {num_classes}")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item()}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item()}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min()}, {train_freq.max()}]")

	criterion_t2i = torch.nn.BCEWithLogitsLoss(reduction='none')
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	# Pre-encode class texts (frozen text encoder)
	model.eval()
	all_class_embeds = []
	text_batch_size = validation_loader.batch_size
	if verbose:
		print(f"\nPre-encoding {num_classes} class texts in batch_size: {text_batch_size}")
	with torch.no_grad():
		with torch.amp.autocast(
			device_type=device.type, 
			enabled=torch.cuda.is_available(),
			dtype=amp_dtype,
		):
			for i in range(0, num_classes, text_batch_size):
				batch_tokens = clip.tokenize(class_names[i:i+text_batch_size]).to(device)
				embeds = model.encode_text(batch_tokens)
				embeds = torch.nn.functional.normalize(embeds, dim=-1)
				all_class_embeds.append(embeds.cpu())

				del batch_tokens, embeds
				torch.cuda.empty_cache()

	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device).detach()

	if verbose:
		print(f"All {num_classes} classes Embeddings (frozen text encoder)")
		print(f"   ├─ {type(all_class_embeds)}")
		print(f"   ├─ {all_class_embeds.shape}")
		print(f"   ├─ {all_class_embeds.dtype}")
		print(f"   └─ {all_class_embeds.device}")

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

	#  Optimizer — rsLoRA parameters only 
	rslora_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(
		params=rslora_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)
	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in rslora_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=T_max,
		eta_min=eta_min,
		last_epoch=-1,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**11,      # 2048 (Conservative start)
		growth_factor=1.5,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=5000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'} (cuda_cap: {cuda_capability})")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ieps_{num_epochs}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}_bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_pat_{patience}_"
		f"mdt_{min_delta:.1e}_cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}.pth"
	)

	training_losses = []
	validation_losses = []
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history = []
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		epoch_loss_total = epoch_loss_i2t = epoch_loss_t2i = 0.0
		num_batches = 0
		nan_loss_count = 0      # track NaN loss skips
		nan_grad_count = 0      # track NaN gradient skips

		for bidx, batch_data in enumerate(train_loader):
			images, _, label_vectors = batch_data
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			optimizer.zero_grad(set_to_none=True)

			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)

			if torch.isnan(total_loss) or torch.isinf(total_loss):
				nan_loss_count += 1
				print(
					f"Warning: NaN/Inf loss at epoch {epoch+1}, "
					f"batch {bidx+1} (total skipped: {nan_loss_count}). Skipping."
				)
				
				# Abort epoch if NaN cascade is unrecoverable
				if nan_loss_count > len(train_loader) // 2:
					raise RuntimeError(
						f"[ABORT] NaN loss in >{len(train_loader)//2} batches "
						f"All {len(train_loader)} batches produced NaN/Inf loss. "
						f"Reduce learning_rate or lora_alpha and rerun."
					)
				
				continue

			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)

			# Check for NaN/Inf gradients after unscaling
			has_bad_grad = any(
				not torch.isfinite(p.grad).all()
				for p in model.parameters()
				if p.requires_grad and p.grad is not None
			)
			
			if has_bad_grad:
				nan_grad_count += 1
				if nan_grad_count % 50 == 1:
					print(
						f"Warning: NaN/Inf gradient at epoch {epoch+1}, "
						f"batch {bidx+1} (total skipped: {nan_grad_count}). Skipping step."
					)
				optimizer.zero_grad(set_to_none=True)
				scaler.update()
				continue
			
			torch.nn.utils.clip_grad_norm_(
				[p for p in model.parameters() if p.requires_grad],
				max_norm=1.0,
			)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()

			epoch_loss_total += total_loss.item()
			epoch_loss_i2t   += loss_i2t.item()
			epoch_loss_t2i   += loss_t2i.item()
			num_batches += 1

			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\t\tBatch [{bidx+1:04d}/{len(train_loader)}] "
					f"Total: {total_loss.item():.6f} "
					f"(I2T: {loss_i2t.item():.6f}, T2I: {loss_t2i.item():.6f})"
				)

		if num_batches == 0:
			raise RuntimeError(
				f"[ABORT] Epoch {epoch+1}: zero valid batches. "
				f"All {len(train_loader)} batches produced NaN/Inf loss. "
				f"Reduce learning_rate or lora_alpha and rerun."
			)

		# Report skips at end of epoch
		if nan_loss_count > 0 or nan_grad_count > 0:
			print(
				f"[Epoch {epoch+1}] Skipped batches — "
				f"NaN loss: {nan_loss_count}, NaN grad: {nan_grad_count} "
				f"/ {len(train_loader)} total"
			)

		avg_total = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t   = epoch_loss_i2t   / num_batches if num_batches > 0 else 0.0
		avg_t2i   = epoch_loss_t2i   / num_batches if num_batches > 0 else 0.0
		training_losses.append(avg_total)
		training_losses_breakdown["total"].append(avg_total)
		training_losses_breakdown["i2t"].append(avg_i2t)
		training_losses_breakdown["t2i"].append(avg_t2i)
		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f"Training Epoch {epoch+1} took {time.time() - train_and_val_st_time:.2f}s. Validating Epoch {epoch+1}...")

		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
			},
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)

		full_val_metrics = validation_results["full_metrics"]
		img2txt_metrics  = validation_results["img2txt_metrics"]
		txt2img_metrics  = validation_results["txt2img_metrics"]
		full_val_loss_acc_metrics_all_epochs.append(full_val_metrics)
		img2txt_metrics_all_epochs.append(img2txt_metrics)
		txt2img_metrics_all_epochs.append(txt2img_metrics)

		i2t_map10 = img2txt_metrics.get("mAP", {}).get("10", float("nan"))
		t2i_map10 = txt2img_metrics.get("mAP", {}).get("10", float("nan"))
		i2t_r10   = img2txt_metrics.get("Recall", {}).get("10", float("nan"))
		t2i_r10   = txt2img_metrics.get("Recall", {}).get("10", float("nan"))
		cos_sim   = full_val_metrics.get("cosine_similarity")
		align_score = full_val_metrics.get("alignment_score")
		print(
			f"\nEpoch {epoch+1}:\n"
			f"  [LOSS] — {mode.upper()}-FT Train: {avg_total} (I2T: {avg_i2t}, T2I: {avg_t2i}) Val: {current_val_loss}\n"
			f"  I2T    — mAP@10: {i2t_map10:.4f}  R@10: {i2t_r10:.4f}\n"
			f"  T2I    — mAP@10: {t2i_map10:.4f}  R@10: {t2i_r10:.4f}\n"
			f"  LR     — {scheduler.get_last_lr()[0]}"
		)
		if align_score is not None:
			print(f"  Embed — AlignScore@5: {align_score:.4f}")
		elif cos_sim is not None:
			print(f"  Embed — CosSim: {cos_sim:.4f}")
		else:
			print(f"  Embed — AlignScore: N/A")


		# ── Training health check ────────────────────────────────────────────
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}"
			)
			break

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time()-train_and_val_st_time:.1f}s")

	print(f"[{mode}] Total elapsed: {time.time()-train_start_time:.1f}s")

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params={
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout,
		},
		temperature=temperature,
		topk_values=topk_values,
		verbose=verbose,
	)

	final_metrics_full    = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	final_tiered_i2t      = evaluation_results["tiered_i2t"]
	final_tiered_t2i      = evaluation_results["tiered_t2i"]
	model_source          = evaluation_results["model_loaded_from"]

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth,
		actual_epochs=actual_trained_epochs
	)

	if verbose:
		print(f"{'='*50}")
		print(f"{mode.upper()} Final evaluation from: {model_source}")
		print(f"  Model: {model_arch}")
		print(f"  CLIP frozen params: {sum(p.numel() for p in model.parameters()):,}")
		print(f"  Epochs trained: {actual_trained_epochs}")
		print(f"  Best val loss: {early_stopping.get_best_score():.6f} @ Epoch {early_stopping.get_best_epoch()+1}")
		print(f"  Best model: {mdl_fpth}")
		print("\n>> Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print("\n>> Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")

	file_base_name = (
		f"{mode}_"
		f"{model_arch}_ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)

	plot_paths = {
		"losses_breakdown":    os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"losses":              os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"full_val_topk_i2t":   os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i":   os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_per_epoch.png"),
		"retrieval_best":      os.path.join(results_dir, f"{file_base_name}_retrieval_best.png"),
		"hp_evol":             os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"],
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
		fname=plot_paths["hp_evol"],
	)
	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return final_tiered_i2t, final_tiered_t2i

def dora_finetune_multi_label(
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
	topk_values: List[int] = [1, 3, 5, 10, 15, 20],
	quantization_bits: int = 8,
	quantized: bool = False,
	temperature: float = 0.07,
	loss_weights: Dict[str, float] = None,
	verbose: bool = True,
):
	"""
	DoRA (Weight-Decomposed Low-Rank Adaptation) fine-tuning for multi-label datasets.
	
	DoRA decomposes pre-trained weights into magnitude and direction components,
	applying LoRA to the directional component while keeping magnitude trainable.
	This improves learning capacity and stability without inference overhead.
	
	Reference: DoRA: Weight-Decomposed Low-Rank Adaptation, ICML'24
	"""
	
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	window_size = minimum_epochs + 1

	# Validate dropout configuration
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during DoRA fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for DoRA practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during DoRA fine-tuning "
			"which gives you more control over DoRA-specific regularization without affecting the base model.\n"
		)

	# Early stopping setup
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

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"\n{mode.upper()} [Multi-Label]")
		print(f"   ├─ Rank: {lora_rank}")
		print(f"   ├─ Alpha: {lora_alpha}")
		print(f"   ├─ Dropout: {lora_dropout}")
		print(f"   ├─ Model      : {model_name} {model_arch}")
		print(f"   ├─ Dataset    : {dataset_name}  classes: {num_classes}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Learning rate: {learning_rate}  Weight decay: {weight_decay}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss Weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	# Inject DoRA into model
	model = get_injected_peft_clip(
		clip_model=model,
		method=mode, # "dora"
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		target_text_modules=[], # no DoRA in text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	)
	model.to(device)
	get_parameters_info(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="sqrt",
		pw_max_cap=50.0,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	val_freq = diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)

	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min()}, {pos_weight.max()}]")
		print(f"   ├─ number of samples: {N}")
		print(f"   ├─ number of classes: {num_classes}")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,} / {num_classes:,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(
		reduction='none',
	)

	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	model.eval()
	all_class_embeds = []
	text_batch_size = validation_loader.batch_size
	if verbose:
		print(f"\nPre-encoding {num_classes} class texts in batch_size: {text_batch_size}")
	with torch.no_grad():
		with torch.amp.autocast(
			device_type=device.type, 
			enabled=torch.cuda.is_available(),
			dtype=amp_dtype,
		):
			for i in range(0, num_classes, text_batch_size):
				batch_tokens = clip.tokenize(class_names[i:i+text_batch_size]).to(device)
				embeds = model.encode_text(batch_tokens)
				embeds = torch.nn.functional.normalize(embeds, dim=-1)
				all_class_embeds.append(embeds.cpu())

				del batch_tokens, embeds
				torch.cuda.empty_cache()
	
	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device).detach()

	if verbose:
		print(f"All {num_classes} classes Embeddings (frozen text encoder)")
		print(f"   ├─ {type(all_class_embeds)}")
		print(f"   ├─ {all_class_embeds.shape}")
		print(f"   ├─ {all_class_embeds.dtype}")
		print(f"   └─ {all_class_embeds.device}")

	# Optimizer setup
	dora_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(
		params=dora_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in dora_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=T_max,
		eta_min=eta_min,
		last_epoch=-1,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**11,      # 2048 (Conservative start)
		growth_factor=1.5,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=5000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'} (cuda_cap: {cuda_capability})")
		print()

	# Model checkpoint path
	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_name}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
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

	training_losses = list()
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	
	train_start_time = time.time()
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

		for bidx, (images, _, label_vectors) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True)

			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
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
			torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\t\tBatch [{bidx + 1:04d}/{len(train_loader)}] "
					f"Loss: {total_loss.item():.7f} "
					f"(I2T: {loss_i2t.item():.7f}, T2I: {loss_t2i.item():.7f})"
				)
			
			epoch_loss_total += total_loss.item()
			epoch_loss_i2t += loss_i2t.item()
			epoch_loss_t2i += loss_t2i.item()
			num_batches += 1
		
		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss   = epoch_loss_i2t   / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss   = epoch_loss_t2i   / num_batches if num_batches > 0 else 0.0

		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f">> Training epoch {epoch+1} took {time.time() - train_and_val_st_time:.2f} sec. Validating Epoch {epoch+1} ...")
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)
		
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
			},
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])

		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {mode.upper()}: Train - Total: {avg_total_loss} (I2T: {avg_i2t_loss}, T2I: {avg_t2i_loss}) Val: {current_val_loss}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')


		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break

		# # Cache stats
		# if hasattr(train_loader.dataset, 'get_cache_stats'):
		# 	cache_stats = train_loader.dataset.get_cache_stats()
		# 	if cache_stats is not None:
		# 		print(f"Train Cache: {cache_stats}")

		# if hasattr(validation_loader.dataset, 'get_cache_stats'):
		# 	cache_stats = validation_loader.dataset.get_cache_stats()
		# 	if cache_stats is not None:
		# 		print(f"Validation Cache: {cache_stats}")

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")
	
	print(f"[{mode}] Total Training Elapsed Time: {time.time() - train_start_time:.1f}s")

	# Final evaluation
	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params={
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout,
		},
		temperature=temperature,
		topk_values=topk_values,
		verbose=verbose,
	)

	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	final_tiered_i2t = evaluation_results["tiered_i2t"]
	final_tiered_t2i = evaluation_results["tiered_t2i"]
	model_source = evaluation_results["model_loaded_from"]

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	if verbose:
		print(f"{'='*50}")
		print(f"{mode.upper()} Final evaluation from: {model_source}")
		print(f"  Model: {model_arch}")
		print(f"  CLIP frozen params: {sum(p.numel() for p in model.parameters()):,}")
		print(f"  Epochs trained: {actual_trained_epochs}")
		print(f"  Best val loss: {early_stopping.get_best_score():.6f} @ Epoch {early_stopping.get_best_epoch()+1}")
		print(f"  Best model: {mdl_fpth}")
		print("\n>> Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print("\n>> Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}"
	)
	
	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"loss_breakdown": os.path.join(results_dir, f"{file_base_name}_loss_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"hp_evol": os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	}

	# Plot loss breakdown (I2T vs T2I)
	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["loss_breakdown"],
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
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return final_tiered_i2t, final_tiered_t2i

def ia3_finetune_multi_label(
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
	temperature: float = 0.07,
	quantization_bits: int = 8,
	quantized: bool = False,
	verbose: bool = True,
):
	"""
	(IA)³ fine-tuning for multi-label CLIP classification.
	
	Key differences from single-label (IA)³ version:
	1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
	2. Handles bidirectional multi-label targets (I2T and T2I)
	3. Pre-encodes class embeddings for efficiency
	4. Uses multi-label specific loss computation
	5. Proper multi-label evaluation metrics
	
	Args:
			model: CLIP model to fine-tune with (IA)³
			train_loader: Training DataLoader (must provide multi-label vectors)
			validation_loader: Validation DataLoader  
			num_epochs: Number of training epochs
			print_every: Print loss every N batches
			learning_rate: Learning rate for (IA)³ parameters
			weight_decay: Weight decay for regularization
			device: Training device (cuda/cpu)
			results_dir: Directory to save results
			patience: Early stopping patience
			min_delta: Minimum change for improvement
			cumulative_delta: Cumulative delta for early stopping
			minimum_epochs: Minimum epochs before early stopping
			volatility_threshold: Volatility threshold for early stopping
			slope_threshold: Slope threshold for early stopping
			pairwise_imp_threshold: Pairwise improvement threshold for early stopping
			topk_values: K values for evaluation metrics
			loss_weights: Optional weights for I2T and T2I losses
			temperature: Temperature scaling for similarities
			quantization_bits: Bits for quantization (if quantized=True)
			quantized: Whether to use quantized base weights
			verbose: Enable detailed logging
	"""
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	# Inspect the model for dropout layers and validate for (IA)³
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	# Check for non-zero dropout in the base model
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during (IA)³ fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for (IA)³ practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during (IA)³ fine-tuning "
			"which gives you more control over (IA)³-specific regularization without affecting the base model.\n"
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
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	# Get dataset information
	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
		
	if verbose:
		print(f"\n{mode.upper()} [Multi-Label]")
		print(f"   ├─ Model      : {model_name} {model_arch}")
		print(f"   ├─ Dataset    : {dataset_name}  classes: {num_classes}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Learning rate: {learning_rate}  Weight decay: {weight_decay}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss Weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=1,  # Not used for (IA)³, but required by function signature
		alpha=1.0,  # Not used for (IA)³, but required by function signature
		dropout=0.0,  # Not used for (IA)³, but required by function signature
		target_text_modules=[], # no (IA)³ in text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	)
	
	model.to(device)
	get_parameters_info(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="log",
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	val_freq = diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min()}, {pos_weight.max()}]")
		print(f"   ├─ number of samples: {N}")
		print(f"   ├─ number of classes: {num_classes}")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,} / {num_classes:,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(
		reduction='none',
	)
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")
	
	model.eval()
	all_class_embeds = []
	text_batch_size = validation_loader.batch_size
	print(f"\nPre-encoding {num_classes} class texts in batch_size: {text_batch_size}")
	with torch.no_grad():
		with torch.amp.autocast(
			device_type=device.type, 
			enabled=torch.cuda.is_available(),
			dtype=amp_dtype,
		):
			for i in range(0, num_classes, text_batch_size):
				batch_tokens = clip.tokenize(class_names[i:i+text_batch_size]).to(device)
				embeds = model.encode_text(batch_tokens)
				embeds = torch.nn.functional.normalize(embeds, dim=-1)
				all_class_embeds.append(embeds.cpu())

				del batch_tokens, embeds
				torch.cuda.empty_cache()
	
	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device).detach()

	if verbose:
		print(f"All {num_classes} classes Embeddings (frozen text encoder)")
		print(f"   ├─ {type(all_class_embeds)}")
		print(f"   ├─ {all_class_embeds.shape}")
		print(f"   ├─ {all_class_embeds.dtype}")
		print(f"   └─ {all_class_embeds.device}")

	# Optimizer setup
	ia3_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(
		params=ia3_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)
	
	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in ia3_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=T_max,
		eta_min=eta_min,
		last_epoch=-1,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**11,      # 2048 (Conservative start)
		growth_factor=1.5,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=5000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'} (cuda_cap: {cuda_capability})")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_name}_"
		f"{model_arch}_"
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

	training_losses = list()
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	train_start_time = time.time()
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
			optimizer.zero_grad(set_to_none=True)
			images, _, label_vectors = batch_data

			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
						
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
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
			torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
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
					f"\t\tBatch [{bidx + 1:04d}/{len(train_loader)}] "
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

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f">> Training epoch {epoch+1} took {time.time() - train_and_val_st_time:.2f} sec. Validating Epoch {epoch+1} ...")
		
		# Compute validation loss using the same multi-label loss function
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score   = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		
		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {mode}-FT: Training - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f}) Validation: {current_val_loss:.6f}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')
		
		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break

		# # Cache stats
		# if hasattr(train_loader.dataset, 'get_cache_stats'):
		# 	cache_stats = train_loader.dataset.get_cache_stats()
		# 	if cache_stats is not None:
		# 		print(f"Train Cache: {cache_stats}")

		# if hasattr(validation_loader.dataset, 'get_cache_stats'):
		# 	cache_stats = validation_loader.dataset.get_cache_stats()
		# 	if cache_stats is not None:
		# 		print(f"Validation Cache: {cache_stats}")

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")
	
	print(f"[{mode}] Total Training Time: {time.time() - train_start_time:.1f} sec")

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params=None,
		temperature=temperature,
		topk_values=topk_values,
		verbose=verbose,
	)

	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	final_tiered_i2t = evaluation_results["tiered_i2t"]
	final_tiered_t2i = evaluation_results["tiered_t2i"]
	model_source = evaluation_results["model_loaded_from"]

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	if verbose:
		print(f"{'='*50}")
		print(f"{mode.upper()} Final evaluation from: {model_source}")
		print(f"  Model: {model_arch}")
		print(f"  CLIP frozen params: {sum(p.numel() for p in model.parameters()):,}")
		print(f"  Epochs trained: {actual_trained_epochs}")
		print(f"  Best val loss: {early_stopping.get_best_score():.6f} @ Epoch {early_stopping.get_best_epoch()+1}")
		print(f"  Best model: {mdl_fpth}")
		print("\n>> Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print("\n>> Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}"
	)
	
	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"hp_evol": os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
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
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return final_tiered_i2t, final_tiered_t2i

def vera_finetune_multi_label(
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
	temperature: float = 0.07,
	quantization_bits: int = 8,
	quantized: bool = False,
	verbose: bool = True,
):
	"""
	VeRA fine-tuning for multi-label CLIP classification.
	
	Key differences from single-label VeRA version:
	1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
	2. Handles bidirectional multi-label targets (I2T and T2I)
	3. Pre-encodes class embeddings for efficiency
	4. Uses multi-label specific loss computation
	5. Proper multi-label evaluation metrics
	
	Key differences from LoRA multi-label version:
	1. Uses VeRA adaptation (frozen random matrices + trainable scaling vectors)
	2. Significantly fewer trainable parameters than LoRA
	3. Shared random matrices across all layers
	
	Args:
		model: CLIP model to fine-tune with VeRA
		train_loader: Training DataLoader (must provide multi-label vectors)
		validation_loader: Validation DataLoader  
		num_epochs: Number of training epochs
		print_every: Print loss every N batches
		learning_rate: Learning rate for VeRA parameters
		weight_decay: Weight decay for regularization
		device: Training device (cuda/cpu)
		results_dir: Directory to save results
		lora_rank: VeRA rank parameter (for compatibility)
		lora_alpha: VeRA alpha parameter (for compatibility, not used in VeRA)
		lora_dropout: VeRA dropout parameter
		patience: Early stopping patience
		min_delta: Minimum change for improvement
		cumulative_delta: Cumulative delta for early stopping
		minimum_epochs: Minimum epochs before early stopping
		volatility_threshold: Volatility threshold for early stopping
		slope_threshold: Slope threshold for early stopping
		pairwise_imp_threshold: Pairwise improvement threshold
		topk_values: K values for evaluation metrics
		loss_weights: Optional weights for I2T and T2I losses
		temperature: Temperature scaling for similarities
		quantization_bits: Quantization bits (4 or 8)
		quantized: Whether to use quantized base weights
		verbose: Print detailed information
	"""
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	# Inspect the model for dropout layers and validate for VeRA
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	# Check for non-zero dropout in the base model
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during VeRA fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for VeRA practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during VeRA fine-tuning "
			"which gives you more control over VeRA-specific regularization without affecting the base model.\n"
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
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	# Get dataset information
	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"\n{mode.upper()} [Multi-Label]")
		print(f"   ├─ Rank: {lora_rank}")
		print(f"   ├─ Alpha: {lora_alpha} (not used in VeRA)")
		print(f"   ├─ Dropout: {lora_dropout}")
		print(f"   ├─ Model      : {model_name} {model_arch}")
		print(f"   ├─ Dataset    : {dataset_name}  classes: {num_classes}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Learning rate: {learning_rate}  Weight decay: {weight_decay}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss Weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	# Apply VeRA to the model
	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		target_text_modules=[], # Frozen text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	)
	
	model.to(device)
	get_parameters_info(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="log",
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	val_freq = diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min()}, {pos_weight.max()}]")
		print(f"   ├─ number of samples: {N}")
		print(f"   ├─ number of classes: {num_classes}")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,} / {num_classes:,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(
		reduction='none',
	)
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	model.eval()
	all_class_embeds = []
	text_batch_size = validation_loader.batch_size
	print(f"\nPre-encoding {num_classes} class texts in batch_size: {text_batch_size}")
	with torch.no_grad():
		with torch.amp.autocast(
			device_type=device.type, 
			enabled=torch.cuda.is_available(),
			dtype=amp_dtype,
		):
			for i in range(0, num_classes, text_batch_size):
				batch_tokens = clip.tokenize(class_names[i:i+text_batch_size]).to(device)
				embeds = model.encode_text(batch_tokens)
				embeds = torch.nn.functional.normalize(embeds, dim=-1)
				all_class_embeds.append(embeds.cpu())

				del batch_tokens, embeds
				torch.cuda.empty_cache()
	
	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device).detach()
	
	if verbose:
		print(f"All {num_classes} classes Embeddings (frozen text encoder)")
		print(f"   ├─ {type(all_class_embeds)}")
		print(f"   ├─ {all_class_embeds.shape}")
		print(f"   ├─ {all_class_embeds.dtype}")
		print(f"   └─ {all_class_embeds.device}")

	# Optimizer setup
	vera_params = [p for p in model.parameters() if p.requires_grad]

	optimizer = torch.optim.AdamW(
		params=vera_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in vera_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=T_max,
		eta_min=eta_min,
		last_epoch=-1,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**11,      # 2048 (Conservative start)
		growth_factor=1.5,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=5000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'} (cuda_cap: {cuda_capability})")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_name}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
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

	training_losses = list()
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	train_start_time = time.time()
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
			optimizer.zero_grad(set_to_none=True)

			images, _, label_vectors = batch_data  # Ignore tokenized_labels, use pre-encoded

			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
						
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
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
			torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
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
					f"\t\tBatch [{bidx + 1:04d}/{len(train_loader)}] "
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

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f">> Training epoch {epoch+1} took {time.time() - train_and_val_st_time:.2f} sec. Validating Epoch {epoch+1} ...")
		
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
			},
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score   = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		
		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {mode.upper()}-FT: Training - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f}) Validation: {current_val_loss:.6f}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')
		# print(f"   ├─ [VeRA SCALING VECTORS]:")
		# for name, module in model.named_modules():
		# 	if hasattr(module, 'lambda_b') and hasattr(module, 'lambda_d'):
		# 		lb = module.lambda_b.data
		# 		ld = module.lambda_d.data
		# 		print(f"\t{name:<60s}λ_b: mean={lb.mean():.6f} std={lb.std():.6f} max={lb.abs().max():.6f} λ_d: mean={ld.mean():.6f} std={ld.std():.6f} max={ld.abs().max():.6f}")

		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")
	
	print(f"[VeRA SCALING VECTORS — Post-Training Summary]")
	lambda_b_stats = []
	lambda_d_stats = []
	for name, module in model.named_modules():
		if hasattr(module, 'lambda_b') and hasattr(module, 'lambda_d'):
			lb = module.lambda_b.data.cpu()
			ld = module.lambda_d.data.cpu()
			lambda_b_stats.append(lb)
			lambda_d_stats.append(ld)
			print(
				f"  {name:<50s} "
				f"λ_b: mean={lb.mean():.6f} std={lb.std():.6f} max={lb.abs().max():.6f} | "
				f"λ_d: mean={ld.mean():.6f} std={ld.std():.6f}"
			)

	# Global summary
	glb_lb = torch.cat(lambda_b_stats)
	glb_ld = torch.cat(lambda_d_stats)

	lb_mean_abs = glb_lb.abs().mean().item()
	lb_std = glb_lb.std().item()
	ld_mean_delta = (glb_ld - 1.0).abs().mean().item()

	print(f"\n[GLOBAL SUMMARY — VeRA SCALING VECTORS]")
	print(f"  ▶ Gating Signal (λ_b): Mean |λ_b| = {lb_mean_abs} std={lb_std}")
	print(f"  ▶ Refine Signal (λ_d): Mean Δλ_d  = {ld_mean_delta} std={glb_ld.std()}")
	print(f"  ▶ λ_b — max={glb_lb.abs().max():.6f} nonzero={(glb_lb.abs() > 1e-6).sum().item()} / {glb_lb.numel()} ({100*(glb_lb.abs() > 1e-6).sum().item()/glb_lb.numel():.1f}%)")
	print(f"  ▶ λ_d — max={glb_ld.abs().max():.6f} delta_from_init={ld_mean_delta}")
	print(f"\n[DIAGNOSIS]")
	if lb_mean_abs < 1e-7:
		print("  ❌ CRITICAL: λ_b is effectively zero. The VeRA path is CLOSED. Check gradients/LR.")
	elif lb_mean_abs < 1e-4:
		print("  ⚠️ WARNING: λ_b is very weak. Model is barely using the adapted path.")
	else:
		print("  ✅ SUCCESS: λ_b is active. The model is successfully 'opening the gate' to VeRA.")
			
	if ld_mean_delta < 1e-7:
		print("  ⚠️ NOTE: λ_d hasn't moved from 1.0. Rank-space refinement is inactive.")
	elif ld_mean_delta > 0.5:
		print("  ⚠️ NOTE: λ_d has shifted significantly (>0.5). High adaptation in rank-space.")
	else:
		print("  ✅ SUCCESS: λ_d is refining the rank-space directions.")
	
	print(f"\n[{mode.upper()}] Total Training Elapsed Time: {time.time() - train_start_time:.1f} sec")

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params={
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout,
		},
		temperature=temperature,
		topk_values=topk_values,
		verbose=verbose,
	)

	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	final_tiered_i2t = evaluation_results["tiered_i2t"]
	final_tiered_t2i = evaluation_results["tiered_t2i"]
	model_source = evaluation_results["model_loaded_from"]

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	if verbose:
		print(f"{'='*50}")
		print(f"{mode.upper()} Final evaluation from: {model_source}")
		print(f"  Model: {model_arch}")
		print(f"  CLIP frozen params: {sum(p.numel() for p in model.parameters()):,}")
		print(f"  Epochs trained: {actual_trained_epochs}")
		print(f"  Best val loss: {early_stopping.get_best_score():.6f} @ Epoch {early_stopping.get_best_epoch()+1}")
		print(f"  Best model: {mdl_fpth}")
		print("\n>> Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print("\n>> Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)
	
	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"hp_evol": os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
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
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return final_tiered_i2t, final_tiered_t2i

def clip_adapter_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: DataLoader,
	validation_loader: DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	clip_adapter_method: str,  # "clip_adapter_v", "clip_adapter_t", "clip_adapter_vt"
	bottleneck_dim: int = 256,
	activation: str = "relu",
	patience: int = 7,
	min_delta: float = 1e-4,
	cumulative_delta: float = 1e-3,
	minimum_epochs: int = 10,
	volatility_threshold: float = 0.02,
	slope_threshold: float = 1e-3,
	pairwise_imp_threshold: float = 0.01,
	topk_values: List[int] = [1, 5, 10, 15, 20],
	temperature: float = 0.07,
	loss_weights: Dict[str, float] = None,
	verbose: bool = True,
):
	"""
	Fine-tunes a CLIP model using CLIP-Adapter for multi-label datasets.

	Three variants controlled by clip_adapter_method:
		- "clip_adapter_v"  : adapter on vision encoder only  → text encoder fully frozen
													→ all_class_embeds pre-encoded once, reused every epoch
		- "clip_adapter_t"  : adapter on text encoder only    → vision encoder fully frozen
													→ all_class_embeds re-encoded every epoch (adapter changes)
		- "clip_adapter_vt" : adapters on both encoders       → same as clip_adapter_t
													→ all_class_embeds re-encoded every epoch

	Loss:
		- criterion_i2t: BCEWithLogitsLoss with pos_weight  [B, C] → active_mask applied
		- criterion_t2i: BCEWithLogitsLoss plain             [C, B] → active_mask applied
	"""

	# Validate method
	valid_methods = ("clip_adapter_v", "clip_adapter_t", "clip_adapter_vt")
	assert clip_adapter_method in valid_methods, (
		f"clip_adapter_method must be one of {valid_methods}, got '{clip_adapter_method}'"
	)
	text_adapter_active = clip_adapter_method in ("clip_adapter_t", "clip_adapter_vt")

	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
	num_classes = len(class_names)

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	# Dropout check (warning only — CLIP-Adapter tolerates dropout) 
	non_zero_dropouts = [
		(name, module.p)
		for name, module in model.named_modules()
		if isinstance(module, torch.nn.Dropout) and module.p > 0
	]
	if non_zero_dropouts and verbose:
		dropout_info = ", ".join([f"{n}: p={p}" for n, p in non_zero_dropouts])
		print(f"[WARNING] Non-zero dropout in base model: {dropout_info}")

	if verbose:
		print(f"{mode.upper()} [Multi-Label] variant: {clip_adapter_method} bottleneck={bottleneck_dim} act={activation}")
		print(f"   ├─ Dataset    : {dataset_name}  classes: {num_classes}")
		print(f"   ├─ Model      : {model_name} {model_arch}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		print(f"   ├─ Text adapter active: {text_adapter_active}")
	
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_mem  = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {gpu_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	# Early stopping
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

	# Inject adapter
	model = get_adapter_peft_clip(
		clip_model=model,
		method=clip_adapter_method,
		bottleneck_dim=bottleneck_dim,
		activation=activation,
		verbose=verbose,
	).to(device)

	get_parameters_info(model=model, mode=clip_adapter_method)

	if verbose:
		trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
		frozen    = [(n, p.numel()) for n, p in model.named_parameters() if not p.requires_grad]
		print(f"Trainable tensors: {len(trainable)} | Frozen tensors: {len(frozen)}")
		for name, numel in trainable[:10]:
			print(f"  {name}: {numel:,}")
		if len(trainable) > 10:
			print(f"  ... and {len(trainable)-10} more")
		print(f"Total trainable : {sum(n for _,n in trainable):,}")
		print(f"Total frozen    : {sum(n for _,n in frozen):,}")

	# Loss masks
	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="log",
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N           = masks["N"]
	train_freq  = masks["train_freq"]

	val_freq = diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	#Criteria
	criterion_i2t = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
	criterion_t2i = torch.nn.BCEWithLogitsLoss(reduction='none')
	if verbose:
		print(f"\n[I2T] BCEWithLogitsLoss  pos_weight range: [{pos_weight.min()}, {pos_weight.max()}]")
		print(f"   ├─ N={N}  classes={num_classes}  active={active_mask.sum().item():,}")
		print(f"[T2I] BCEWithLogitsLoss  no pos_weight")

	# Optimizer
	adapter_params = [p for p in model.parameters() if p.requires_grad]
	if not adapter_params:
		raise ValueError("No trainable parameters found. Check get_adapter_peft_clip.")

	optimizer = torch.optim.AdamW(
		params=adapter_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)
	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in adapter_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=T_max,
		eta_min=eta_min,
		last_epoch=-1,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**11,      # 2048 (Conservative start)
		growth_factor=1.5,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=5000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'} (cuda_cap: {cuda_capability})")
		print()

	# Checkpoint path
	mdl_fpth = os.path.join(
		results_dir,
		f"{clip_adapter_method}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ieps_{num_epochs}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"cbd_{bottleneck_dim}_act_{activation}_temp_{temperature}_"
		f"mep_{minimum_epochs}_pat_{patience}_mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_pit_{pairwise_imp_threshold:.1e}.pth"
	)

	# Pre-encode class texts
	# For clip_adapter_v: text encoder is fully frozen → encode once, reuse forever.
	# For clip_adapter_t / clip_adapter_vt: text adapter changes every step →
	#   we keep the raw tokens and re-encode at the start of each epoch.
	text_batch_size = validation_loader.batch_size
	all_class_tokens = clip.tokenize(class_names).to(device)  # always kept

	def encode_class_texts() -> torch.Tensor:
		"""Encode all class names through the current text encoder (with adapter if active)."""
		model.eval()
		embeds_list = []
		with torch.no_grad():
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				for i in range(0, num_classes, text_batch_size):
					batch_tokens = all_class_tokens[i:i+text_batch_size]
					e = model.encode_text(batch_tokens)
					e = torch.nn.functional.normalize(e, dim=-1)
					embeds_list.append(e.cpu())
		return torch.cat(embeds_list, dim=0).to(device).detach()

	if not text_adapter_active:
		# clip_adapter_v: encode once, frozen for entire run
		print(f"\n>> Pre-encoding {num_classes} class texts (vision-only adapter — encode once)...")
		all_class_embeds = encode_class_texts()
		if verbose:
			print(f"all_class_embeds: {all_class_embeds.shape} {all_class_embeds.dtype} {all_class_embeds.device}")
	else:
		# clip_adapter_t / clip_adapter_vt: will encode at start of each epoch
		all_class_embeds = None
		print(f"\n>> Text adapter active — class texts will be re-encoded each epoch.")

	training_losses = []
	validation_losses = []
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history  = []
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()

		# Re-encode class texts if text adapter is active (weights changed last epoch)
		if text_adapter_active:
			print(f">> Re-encoding class texts with updated text adapter (epoch {epoch+1})...")
			all_class_embeds = encode_class_texts()
			if verbose:
				print(f"   all_class_embeds: {all_class_embeds.shape} {all_class_embeds.dtype}")

		model.train()
		print(f"Epoch [{epoch+1}/{num_epochs}]")

		epoch_loss_total = 0.0
		epoch_loss_i2t   = 0.0
		epoch_loss_t2i   = 0.0
		num_batches      = 0

		for bidx, (images, _, label_vectors) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)

			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()

			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				# For clip_adapter_vt / clip_adapter_t: text adapter is in train mode,
				# so we must pass through encode_text with gradients for the text adapter.
				# For clip_adapter_v: all_class_embeds is already pre-encoded and detached.
				if text_adapter_active:
					# Re-encode with gradients so text adapter receives gradient signal
					batch_class_embeds = model.encode_text(all_class_tokens)
					batch_class_embeds = torch.nn.functional.normalize(batch_class_embeds, dim=-1)
				else:
					batch_class_embeds = all_class_embeds  # pre-encoded, detached

				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=batch_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)

			if torch.isnan(total_loss):
				print(f"Warning: NaN loss at epoch {epoch+1} batch {bidx+1}. Skipping.")
				continue

			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()

			epoch_loss_total += total_loss.item()
			epoch_loss_i2t   += loss_i2t.item()
			epoch_loss_t2i   += loss_t2i.item()
			num_batches      += 1

			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\t\tBatch [{bidx+1:04d}/{len(train_loader)}] "
					f"Total: {total_loss.item():.6f} "
					f"(I2T: {loss_i2t.item():.6f}, T2I: {loss_t2i.item():.6f})"
				)

		avg_total = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t   = epoch_loss_i2t   / num_batches if num_batches > 0 else 0.0
		avg_t2i   = epoch_loss_t2i   / num_batches if num_batches > 0 else 0.0

		training_losses.append(avg_total)
		training_losses_breakdown["total"].append(avg_total)
		training_losses_breakdown["i2t"].append(avg_i2t)
		training_losses_breakdown["t2i"].append(avg_t2i)
		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		# ── Validation ────────────────────────────────────────────────────────
		print(f">> Training epoch {epoch+1} took {time.time() - train_and_val_st_time:.2f} sec. Validating Epoch {epoch+1} ...")

		# For validation loss: always use freshly encoded class embeds
		# (for clip_adapter_t/vt, re-encode under no_grad with current adapter weights)
		val_class_embeds = encode_class_texts() if text_adapter_active else all_class_embeds

		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=val_class_embeds,
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)

		full_metrics = validation_results["full_metrics"]
		cos_sim = full_metrics.get("cosine_similarity")
		align_score = full_metrics.get("alignment_score")
		
		retrieval = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"],
		}

		full_val_loss_acc_metrics_all_epochs.append(full_metrics)
		img2txt_metrics_all_epochs.append(retrieval["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval["txt2img"])

		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {clip_adapter_method}-FT: Train: {avg_total:.6f} (I2T: {avg_i2t:.6f}, T2I: {avg_t2i:.6f})  Val: {current_val_loss:.6f}\n'
			f'   ├─ LR: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ [I2T] Accuracy: {full_metrics.get("img2txt_topk_acc")}\n'
			f'   ├─ [T2I] Accuracy: {full_metrics.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')


		print(
			f"   ├─ [I2T] mAP={retrieval['img2txt'].get('mAP',{})}  R={retrieval['img2txt'].get('Recall',{})}\n"
			f"   ├─ [T2I] mAP={retrieval['txt2img'].get('mAP',{})}  R={retrieval['txt2img'].get('Recall',{})}"
		)
		if full_metrics.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_metrics["hamming_loss"]:.4f}')
			print(f'   ├─ PartialAcc: {full_metrics["partial_acc"]:.4f}')
			print(f'   └─ F1: {full_metrics["f1_score"]:.4f}')

		# ── Training health check ────────────────────────────────────────────
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}"
			)
			break

		# # Cache stats
		# if hasattr(train_loader.dataset, 'get_cache_stats'):
		# 	cs = train_loader.dataset.get_cache_stats()
		# 	if cs: print(f"Train cache: {cs}")
		
		# if hasattr(validation_loader.dataset, 'get_cache_stats'):
		# 	cs = validation_loader.dataset.get_cache_stats()
		# 	if cs: print(f"Val cache: {cs}")

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time()-train_and_val_st_time:.1f}s")

	print(f"[{clip_adapter_method}] Total Training Elapsed Time: {time.time()-train_start_time:.1f} sec")

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params=None,
		topk_values=topk_values,
		temperature=temperature,
		verbose=verbose,
	)

	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt = evaluation_results["img2txt_metrics"]
	final_txt2img = evaluation_results["txt2img_metrics"]
	final_tiered_i2t = evaluation_results["tiered_i2t"]
	final_tiered_t2i = evaluation_results["tiered_t2i"]
	model_source = evaluation_results["model_loaded_from"]

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)

	if verbose:
		print(f"{'='*50}")
		print(f"{mode.upper()} Final evaluation from: {model_source}")
		print(f"  Method   : {clip_adapter_method}")
		print(f"  Model: {model_arch}")
		print(f"  CLIP frozen params: {sum(p.numel() for p in model.parameters()):,}")
		print(f"  Epochs trained: {actual_trained_epochs}")
		print(f"  Best val loss: {early_stopping.get_best_score():.6f} @ Epoch {early_stopping.get_best_epoch()+1}")
		print(f"  Bottleneck: {bottleneck_dim}  Activation: {activation}")
		print(f"  Learning rate: {learning_rate}  Weight decay: {weight_decay}")
		print(f"  Batch size: {train_loader.batch_size}")
		print(f"  Temperature: {temperature}")
		print(f"  Best model: {mdl_fpth}")

		print("\n>> Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print("\n>> Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")

	# Generate plots
	file_base = (
		f"{clip_adapter_method}_{model_arch}_ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_bs_{train_loader.batch_size}_"
		f"cbd_{bottleneck_dim}_act_{activation}_temp_{temperature}"
	)

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=os.path.join(results_dir, f"{file_base}_loss_breakdown.png"),
	)
	
	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=os.path.join(results_dir, f"{file_base}_retrieval_per_epoch.png"),
	)
	
	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt,
		text_to_image_metrics=final_txt2img,
		fname=os.path.join(results_dir, f"{file_base}_retrieval_best.png"),
	)
	
	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=os.path.join(results_dir, f"{file_base}_hp_evol.png"),
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=os.path.join(results_dir, f"{file_base}_losses.png"),
	)

	return final_tiered_i2t, final_tiered_t2i

def tip_adapter_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: DataLoader,
	validation_loader: DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tip_adapter_method: str,  # "tip_adapter" or "tip_adapter_f"
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	topk_values: List[int]=[1, 5, 10, 15, 20],
	initial_beta: float=1.0,
	initial_alpha: float=1.0,
	support_shots: int=16,  # Number of support samples per class
	temperature: float = 0.07,
	loss_weights: Dict[str, float]=None,
	verbose: bool=True,
):
	"""
	Fine-tunes a CLIP model using Tip-Adapter or Tip-Adapter-F technique for multi-label datasets.
	
	Tip-Adapter creates a cache from support set and adapts at test time using a key-value cache.
	Tip-Adapter-F additionally trains a linear projection layer for better adaptation.
	
	Key differences from single-label:
	- Uses BCEWithLogitsLoss for multi-label classification
	- Constructs cache with multi-label support samples
	- Computes bidirectional I2T and T2I losses
	- Pre-encodes all class embeddings for efficiency
	
	Args:
		model: Pre-trained CLIP model.
		train_loader: DataLoader for training data (multi-label format).
		validation_loader: DataLoader for validation data (multi-label format).
		num_epochs: Number of training epochs.
		print_every: Print training stats every N batches.
		learning_rate: Learning rate for the optimizer.
		weight_decay: Weight decay for the optimizer.
		device: Device to run the model on.
		results_dir: Directory to save results and checkpoints.
		tip_adapter_method: "tip_adapter" (training-free) or "tip_adapter_f" (with training).
		initial_beta: Initial temperature parameter for softmax in cache attention.
		initial_alpha: Initial scaling factor for cache adaptation.
		patience: Patience for early stopping.
		min_delta: Minimum change to qualify as an improvement.
		cumulative_delta: Cumulative change threshold for early stopping.
		minimum_epochs: Minimum epochs before early stopping.
		volatility_threshold: Threshold for validation loss volatility.
		slope_threshold: Threshold for validation loss slope.
		pairwise_imp_threshold: Threshold for pairwise improvement.
		topk_values: List of k values for Top-K accuracy.
		support_shots: Number of support samples per class for cache construction.
		temperature: Temperature scaling for similarity computation.
		loss_weights: Weights for I2T and T2I losses (default: {"i2t": 0.5, "t2i": 0.5}).
		verbose: Enable detailed logging.
	"""
	
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
		
	# Check for non-zero dropout in the base model
	dropout_values = []
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))
	
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if verbose and non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		print(
			f"[Tip-Adapter Multi-Label] WARNING: Non-zero dropout detected in base model: {dropout_info}. "
			f"This might affect the frozen base model's behavior during adaptation."
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
	
	# Dataset and directory setup
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	# GET CLASS INFORMATION
	try:
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		try:
			class_names = validation_loader.dataset.dataset.classes
		except:
			class_names = train_loader.dataset.unique_labels
	
	num_classes = len(class_names)

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)
	
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	
	if verbose:
		print(f"\n{mode.upper()} [Multi-Label]")
		print(f"   ├─ Method: {tip_adapter_method}")
		print(f"   ├─ Beta[init]: {initial_beta}")
		print(f"   ├─ Alpha[init]: {initial_alpha}")
		print(f"   ├─ Dataset    : {dataset_name}  classes: {num_classes}")
		print(f"   ├─ Model      : {model_name} {model_arch}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss Weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		print(f"   ├─ Support Shots: {support_shots}")
	
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")
	
	# === SUPPORT SET CONSTRUCTION FOR MULTI-LABEL ===
	if verbose:
		print(f"\n>> Constructing support set with {support_shots} shots per class...")
	
	# For multi-label, we need to collect samples that contain each class
	class_to_samples = defaultdict(list)

	for images, _, label_vectors in train_loader:
		for img, label_vec in zip(images, label_vectors):
			active_classes = torch.where(label_vec > 0)[0].tolist()
			for class_idx in active_classes:
				if len(class_to_samples[class_idx]) < support_shots:
					class_to_samples[class_idx].append((img, label_vec))
		
		# Stop early if all classes have enough shots
		if len(class_to_samples) == num_classes and all(len(v) >= support_shots for v in class_to_samples.values()):
			break

	# Flatten the support set — pick first shot per class for the cache key
	support_images = []
	support_label_vectors = []
	missing_classes = []

	# Get placeholder shapes from any existing class
	_sample_list = next(iter(class_to_samples.values()))  # list of (img, label_vec)
	_sample_img_shape = _sample_list[0][0].shape           # img tensor shape
	_sample_lbl_shape = _sample_list[0][1].shape           # label_vec tensor shape

	for class_idx in range(num_classes):
			if class_idx in class_to_samples and len(class_to_samples[class_idx]) > 0:
					img, label_vec = class_to_samples[class_idx][0]  # ← [0] to get first shot tuple
					support_images.append(img)
					support_label_vectors.append(label_vec)
			else:
					missing_classes.append(class_idx)
					support_images.append(torch.zeros(_sample_img_shape))
					support_label_vectors.append(torch.zeros(_sample_lbl_shape))

	if missing_classes and verbose:
			print(
					f"WARNING: {len(missing_classes)} classes had no training samples "
					f"and were zero-padded: {missing_classes[:10]}{'...' if len(missing_classes) > 10 else ''}"
			)

	if verbose:
		print(f"Support set size: {len(support_images)} samples")
		print(f"Support set breakdown by class:")
		class_counts = [0] * num_classes
		for label_vec in support_label_vectors:
			active_classes = torch.where(label_vec > 0)[0].tolist()
			for class_idx in active_classes:
				class_counts[class_idx] += 1
		
		for class_idx in range(min(10, num_classes)):  # Show first 10 classes
			print(f"  Class {class_idx} ({class_names[class_idx]}): {class_counts[class_idx]} samples")
		if num_classes > 10:
			print(f"  ... and {num_classes - 10} more classes")
	
	# Convert to tensors
	support_images = torch.stack(support_images).to(device)
	support_label_vectors = torch.stack(support_label_vectors).to(device)
	
	if verbose:
		print(f"Support images shape: {support_images.shape}")
		print(f"Support label vectors shape: {support_label_vectors.shape}")
	
	# === EXTRACT FEATURES BEFORE MODIFYING MODEL ===
	if verbose:
		print("\n[Tip-Adapter] Extracting support features from original CLIP model...")
	
	with torch.no_grad():
		model.eval()
		# Extract features in batches to avoid OOM
		batch_size = train_loader.batch_size
		support_features_list = []
		
		for i in range(0, len(support_images), batch_size):
			end_idx = min(i + batch_size, len(support_images))
			batch_images = support_images[i:end_idx]
			batch_features = model.encode_image(batch_images)
			batch_features = torch.nn.functional.normalize(batch_features, dim=-1)
			support_features_list.append(batch_features)
		
		support_features = torch.cat(support_features_list, dim=0)
	
	if verbose:
		print(f"Support features shape: {support_features.shape}")
	
	# === PRE-ENCODE CLASS EMBEDDINGS ===
	if verbose:
		print("Pre-encoding class embeddings for all classes...")
	
	# Pre-encode in batches
	text_batch_size = train_loader.batch_size
	all_class_embeds = []
	model.eval()
	
	with torch.no_grad():
		with torch.amp.autocast(
			device_type=device.type, 
			enabled=torch.cuda.is_available(),
			dtype=amp_dtype,
		):
			for i in range(0, num_classes, text_batch_size):
				end_idx = min(i + text_batch_size, num_classes)

				batch_class_names = class_names[i:end_idx]

				batch_class_texts = clip.tokenize(batch_class_names).to(device)

				batch_embeds = model.encode_text(batch_class_texts)
				batch_embeds = torch.nn.functional.normalize(batch_embeds, dim=-1)

				all_class_embeds.append(batch_embeds.cpu())

				del batch_class_texts, batch_embeds
				torch.cuda.empty_cache()
	
	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device)

	if verbose:
		print(f"All {num_classes} classes Embeddings (frozen text encoder)")
		print(f"   ├─ {type(all_class_embeds)}")
		print(f"   ├─ {all_class_embeds.shape}")
		print(f"   ├─ {all_class_embeds.dtype}")
		print(f"   └─ {all_class_embeds.device}")

	model = get_adapter_peft_clip(
		clip_model=model,
		method=tip_adapter_method,
		initial_beta=initial_beta,
		initial_alpha=initial_alpha,
		verbose=verbose,
	).to(device)
	
	print("=== Full parameter inventory ===")
	for name, param in model.named_parameters():
		print(f"  {name}: numel={param.numel()} requires_grad={param.requires_grad}")
	print("=== End of parameter inventory ===")
	
	# Access the adapter module from the visual encoder
	adapter_module = getattr(model.visual, f"{tip_adapter_method.replace('-', '_')}_proj", None)
	if adapter_module is None:
		# Print available attributes to help debug
		visual_attrs = [a for a in dir(model.visual) if not a.startswith('_')]
		raise ValueError(
			f"Could not find Tip-Adapter module. "
			f"Available model.visual attributes: {visual_attrs}"
		)

	# For multi-label, we set the cache with label vectors instead of single labels
	adapter_module.set_cache(
		support_features=support_features,
		support_labels=support_label_vectors,  # Multi-label vectors
		text_features=all_class_embeds
	)
	
	if verbose:
		cache_stats = adapter_module.get_memory_footprint()
		print(f"\n[CACHE] {tip_adapter_method}:")
		print(f"  ├─ size: {cache_stats.get('cache_size', 0)} samples")
		print(f"  ├─ memory: {cache_stats.get('cache_memory_mb', 0):.4f} MB")
		print(f"  └─ Total memory: {cache_stats.get('total_memory_mb', 0):.4f} MB")
	
	# DEBUG: Check which parameters are trainable
	if verbose:
		trainable_params = []
		frozen_params = []
		for name, param in model.named_parameters():
			if param.requires_grad:
				trainable_params.append((name, param.numel()))
			else:
				frozen_params.append((name, param.numel()))
		
		print(f"\n[{tip_adapter_method} Parameters] Trainable: {len(trainable_params)} Frozen: {len(frozen_params)}")
		
		if trainable_params:
			print("Trainable parameters:")
			for name, numel in trainable_params[:10]:
				print(f"  {name}: {numel} params")
			if len(trainable_params) > 10:
				print(f"  ... and {len(trainable_params) - 10} more")
		
		total_trainable = sum(numel for _, numel in trainable_params)
		total_frozen = sum(numel for _, numel in frozen_params)
		print(f"[TOTAL] Trainable: {total_trainable:,} Frozen: {total_frozen:,}")
	
	get_parameters_info(model=model, mode=tip_adapter_method)

	masks = compute_loss_masks(
		train_loader=train_loader,
		num_classes=num_classes,
		pw_mode="log",
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	val_freq = diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min()}, {pos_weight.max()}]")
		print(f"   ├─ number of samples: {N}")
		print(f"   ├─ number of classes: {num_classes}")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,} / {num_classes:,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(
		reduction='none',
	)
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	# For training-free Tip-Adapter, we may only optimize beta and alpha
	# For Tip-Adapter-F, we also optimize the linear projection
	trainable_parameters = [p for p in model.parameters() if p.requires_grad]
	
	if not trainable_parameters:
		if tip_adapter_method == "tip_adapter":
			# Training-free version - no training needed
			if verbose:
				print("[Tip-Adapter] Training-free mode - skipping optimization setup")
			num_epochs = 0  # No training epochs needed
		else:
			raise ValueError("No trainable parameters found in the model. Check your setup.")
	
	# Setup optimizer only if we have trainable parameters
	if trainable_parameters and num_epochs > 0:
		optimizer = torch.optim.AdamW(
			params=trainable_parameters,
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)
		if verbose:
			print(f"\n{optimizer.__class__.__name__} optimizer")
			print(f"  ├─ Params: {sum(p.numel() for p in trainable_parameters):,}")
			print(f"  ├─ LR: {learning_rate}")
			print(f"  ├─ Betas: {optimizer.defaults['betas']}")
			print(f"  ├─ Eps: {optimizer.defaults['eps']}")
			print(f"  └─ Weight Decay: {weight_decay}")

		# Scheduler
		# approximate T_max: N epochs * minimum_epochs
		estimated_epochs = 2 * minimum_epochs
		T_max = estimated_epochs * len(train_loader)
		ANNEALING_RATIO = 1e-2 # 1% of initial LR
		eta_min = learning_rate * ANNEALING_RATIO
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer=optimizer,
			T_max=T_max,
			eta_min=eta_min,
			last_epoch=-1,
		)
		
		if verbose:
			print(f"\n{scheduler.__class__.__name__}")
			print(f"  ├─ minimum_epochs = {minimum_epochs}")
			print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
			print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
			print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR)")
	else:
		optimizer = None
		scheduler = None
		if verbose:
			print("\n[Tip-Adapter] No optimizer needed (training-free mode)")
	
	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**11,      # 2048 (Conservative start)
		growth_factor=1.5,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=5000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'} (cuda_cap: {cuda_capability})")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{tip_adapter_method}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"init_alpha_{initial_alpha}_"
		f"init_beta_{initial_beta}_"
		f"shots_{support_shots}_"
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
	
	training_losses = list()
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history = []
	alphas = []
	betas = []
	train_start_time = time.time()
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	
	# If training-free, skip training loop
	if num_epochs == 0 or not trainable_parameters:
		if verbose:
			print(f"\n[{tip_adapter_method}] Training-free mode - proceeding directly to evaluation")
		
		evaluation_results = evaluate_best_model(
			model=model,
			validation_loader=validation_loader,
			active_mask=active_mask,
			head_mask=head_mask,
			rare_mask=rare_mask,
			early_stopping=None,
			checkpoint_path=None,
			finetune_strategy=mode,
			device=device,
			cache_dir=results_dir,
			topk_values=topk_values,
			temperature=temperature,
			verbose=verbose,
		)
		
		final_metrics_full = evaluation_results["full_metrics"]
		final_img2txt_metrics = evaluation_results["img2txt_metrics"]
		final_txt2img_metrics = evaluation_results["txt2img_metrics"]
		final_tiered_i2t = evaluation_results["tiered_i2t"]
		final_tiered_t2i = evaluation_results["tiered_t2i"]		

		if verbose:
			print("\n>> Tiered I2T Retrieval")
			for tier, m in final_tiered_i2t.items():
				print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
			print("\n>> Tiered T2I Retrieval")
			for tier, m in final_tiered_t2i.items():
				print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
			
		
		# Generate best model plot only
		file_base_name = (
			f"{tip_adapter_method}_"
			f"{model_arch}_"
			f"training_free_"
			f"bs_{train_loader.batch_size}_"
			f"beta_{initial_beta}_"
			f"alpha_{initial_alpha}_"
			f"shots_{support_shots}_"
			f"temp_{temperature}"
		)
		
		viz.plot_retrieval_metrics_best_model(
			dataset_name=dataset_name,
			image_to_text_metrics=final_img2txt_metrics,
			text_to_image_metrics=final_txt2img_metrics,
			fname=os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		)
		
		return final_metrics_full, final_img2txt_metrics, final_txt2img_metrics, mdl_fpth
	
	# Training loop (for Tip-Adapter-F or trainable beta/alpha)
	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0

		for bidx, (images, _, label_vectors) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
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
			torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
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
					f"\t\tBatch [{bidx + 1:04d}/{len(train_loader)}] "
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

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f">> Training epoch {epoch+1} took {time.time() - train_and_val_st_time:.2f} sec. Validating Epoch {epoch+1} ...")
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score   = full_val_loss_acc_metrics_per_epoch.get("alignment_score")
		
		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		
		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {tip_adapter_method.upper()}-FT: Train - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f}) Val: {current_val_loss:.6f}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')
		
		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		print(f"   ├─ α: {adapter_module.alpha.item():.5f}, β: {adapter_module.beta.item():.5f}")
		alphas.append(adapter_module.alpha.item())
		betas.append(adapter_module.beta.item())

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break


		# Early stopping check
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break
		
		# # Cache stats
		# if hasattr(train_loader.dataset, 'get_cache_stats'):
		# 	cache_stats = train_loader.dataset.get_cache_stats()
		# 	if cache_stats is not None:
		# 		print(f"Train Cache: {cache_stats}")
		
		# if hasattr(validation_loader.dataset, 'get_cache_stats'):
		# 	cache_stats = validation_loader.dataset.get_cache_stats()
		# 	if cache_stats is not None:
		# 		print(f"Validation Cache: {cache_stats}")
		
		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")
	
	print(f"[{tip_adapter_method}] Total Training Elapsed Time: {time.time() - train_start_time:.1f} sec")
	
	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth if num_epochs > 0 else None,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		lora_params=None,
		temperature=temperature,
		verbose=verbose,
	)
	
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	final_tiered_i2t = evaluation_results["tiered_i2t"]
	final_tiered_t2i = evaluation_results["tiered_t2i"]
	model_source = evaluation_results["model_loaded_from"]

	actual_trained_epochs = len(training_losses) if training_losses else 0
	if num_epochs > 0:
		mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)

	if verbose:
		print(f"{'='*50}")
		print(f"{mode.upper()} Final evaluation from: {model_source}")
		print(f"  ├─ Model: {model_arch}")
		print(f"  ├─ Method: {tip_adapter_method}")
		print(f"  ├─ Beta[init]: {initial_beta}")
		print(f"  ├─ Alpha[init]: {initial_alpha}")
		print(f"  ├─ Epochs trained: {actual_trained_epochs}")
		print(f"  ├─ Support Shots: {support_shots}")
		print(f"  └─ Best model saved to: {mdl_fpth if num_epochs > 0 else 'N/A (training-free)'}")	
		print("\n>> Tiered I2T Retrieval")
		for tier, m in final_tiered_i2t.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print("\n>> Tiered T2I Retrieval")
		for tier, m in final_tiered_t2i.items():
			print(f"  {tier:8s} mAP@10={m['mAP'].get('10',0):.4f}  R@10={m['Recall'].get('10',0):.4f}")
		print(f"{'='*50}")

	# Generate plots
	file_base_name = (
		f"{tip_adapter_method}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"beta_{initial_beta}_"
		f"alpha_{initial_alpha}_"
		f"shots_{support_shots}_"
		f"temp_{temperature}"
	)
	
	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"loss_breakdown": os.path.join(results_dir, f"{file_base_name}_loss_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"hp_evol": os.path.join(results_dir, f"{file_base_name}_hp_evol.png"),
		"alpha_beta_evol": os.path.join(results_dir, f"{file_base_name}_alpha_beta_evol.png"),
	}
	
	if actual_trained_epochs > 0:
		# Plot loss breakdown (I2T vs T2I)
		viz.plot_multilabel_loss_breakdown(
			training_losses_breakdown=training_losses_breakdown,
			filepath=plot_paths["loss_breakdown"],
		)
		
		viz.plot_retrieval_metrics_per_epoch(
			dataset_name=dataset_name,
			image_to_text_metrics_list=img2txt_metrics_all_epochs,
			text_to_image_metrics_list=txt2img_metrics_all_epochs,
			fname=plot_paths["retrieval_per_epoch"],
		)
		
		if learning_rates_history and weight_decays_history:
			viz.plot_hyperparameter_evolution(
				eta_min=eta_min,
				learning_rates=learning_rates_history,
				weight_decays=weight_decays_history,
				fname=plot_paths["hp_evol"],
			)
	
	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)
	
	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	viz.plot_alpha_beta_evolution(
		alphas=alphas,
		betas=betas,
		fname=plot_paths["alpha_beta_evol"],
	)

	return final_tiered_i2t, final_tiered_t2i
