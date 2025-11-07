def get_status(
		model,
		phase,
		layers_to_unfreeze,
		cache=None,
	):
	# Compute parameter statistics
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
	total_params = trainable_params + frozen_params

	# Count unique layers based on group membership
	layer_groups = get_layer_groups(model)
	all_layers = set()
	for group, layers in layer_groups.items():
		for layer in layers:
			all_layers.add(layer)  # Use exact layer names from groups
	total_layers = len(all_layers)

	# Count unique frozen layers
	frozen_layers = 0
	unfrozen_layers = set(layers_to_unfreeze)  # Layers to unfreeze in this phase
	for layer in all_layers:
		# Check if any parameter in this layer is frozen
		is_frozen = all(not p.requires_grad for name, p in model.named_parameters() if layer in name)
		if is_frozen and layer not in unfrozen_layers:
			frozen_layers += 1

	# Update category breakdown
	category_breakdown = {}
	for group, layers in layer_groups.items():
		frozen_in_group = sum(1 for layer in layers if all(not p.requires_grad for name, p in model.named_parameters() if layer in name) and layer not in unfrozen_layers)
		total_in_group = len(layers)
		category_breakdown[group] = (frozen_in_group, total_in_group)

	# Cache results if provided
	if cache is not None:
		cache[f"phase_{phase}"] = {"trainable": trainable_params, "frozen": frozen_params}

	# Print detailed status using tabulate
	headers = ["Metric", "Value"]
	param_stats = [
		["Phase #", f"{phase}"],
		["Total Parameters", f"{total_params:,}"],
		["Trainable Parameters", f"{trainable_params:,} ({trainable_params/total_params*100:.3f}%)"],
		["Frozen Parameters", f"{frozen_params:,} ({frozen_params/total_params*100:.3f}%)"]
	]
	layer_stats = [
		["Total Layers", total_layers],
		["Frozen Layers", f"{frozen_layers} ({frozen_layers/total_layers*100:.3f}%)"]
	]
	category_stats = [[group, f"{frozen}/{total} ({frozen/total*100:.3f}%)"] for group, (frozen, total) in category_breakdown.items()]

	print(tabulate.tabulate(param_stats, headers=headers, tablefmt="pretty", colalign=("left", "left")))
	print("\nLayer Statistics:")
	print(tabulate.tabulate(layer_stats, headers=headers, tablefmt="pretty", colalign=("left", "left")))
	print("\nLayer Category Breakdown:")
	print(tabulate.tabulate(category_stats, headers=["Category", "Frozen/Total (Percentage)"], tablefmt="pretty", colalign=("left", "left")))

def get_num_transformer_blocks(
		model: torch.nn.Module
	) -> tuple:
	if not hasattr(model, 'visual'):
		raise ValueError(f"{model.__class__.__name__} ({model.name}) lacks 'visual' attribute.")

	if not hasattr(model, 'transformer'):
		raise ValueError(f"{model.__class__.__name__} ({model.name}) lacks 'transformer' attribute.")

	# Determine model type
	is_vit = "ViT" in model.name
	is_resnet = "RN" in model.name

	# Count visual transformer blocks
	visual_blocks = 0
	if is_vit:
		if not hasattr(model.visual, 'transformer') or not hasattr(model.visual.transformer, 'resblocks'):
			raise ValueError(f"{model.__class__.__name__} ({model.name}) lacks 'visual.transformer.resblocks' attribute.")
		visual_blocks = len(model.visual.transformer.resblocks)
	elif is_resnet:
		# ResNet models use 'layer1', 'layer2', etc.
		visual_layers = [attr for attr in dir(model.visual) if attr.startswith('layer') and attr[5:].isdigit()]
		visual_blocks = len(visual_layers)
		if visual_blocks == 0:
			print(f"Model {model.name} is a ResNet but no 'visual.layerX' blocks found. Visual blocks set to 0.")
	else:
		raise ValueError(f"Unsupported architecture {model.name}. Expected ViT or ResNet.")

	# Count text transformer blocks
	text_blocks = 0
	if hasattr(model, 'transformer') and hasattr(model.transformer, 'resblocks'):
		text_blocks = len(model.transformer.resblocks)
	else:
		print(f"Model {model.name} lacks 'transformer.resblocks'. Text blocks set to 0.")

	# print(f">> {model.__class__.__name__} {model.name}: Visual Transformer blocks: {visual_blocks}, Text Transformer blocks: {text_blocks}")
	return visual_blocks, text_blocks

def get_layer_groups(model: torch.nn.Module) -> dict:
	vis_nblocks, txt_nblocks = get_num_transformer_blocks(model=model)

	# Determine model type
	is_vit = "ViT" in model.name
	is_resnet = "RN" in model.name

	# Visual transformer or CNN blocks
	visual_blocks = list()
	if is_vit and vis_nblocks > 0:
		visual_blocks = [f'visual.transformer.resblocks.{i}' for i in range(vis_nblocks)]
	elif is_resnet and vis_nblocks > 0:
		visual_blocks = [f'visual.layer{i+1}' for i in range(vis_nblocks)]
	else:
		print(f"No visual blocks defined for model {model.name}")

	# Text transformer blocks
	text_blocks = [f'transformer.resblocks.{i}' for i in range(txt_nblocks)] if txt_nblocks > 0 else []
	if txt_nblocks == 0:
		print(f"No text transformer blocks defined for model {model.name}")

	"""
		ViT architecture (patch embedding → transformer blocks → projection)
		- Frontend (Lower Layers): initial layers responsible for converting raw inputs (images or text) into a format suitable for transformer blocks.
		- Transformer Blocks (Intermediate Layers): core layers that perform feature extraction and contextualization (self-attention mechanisms).
		- Projection (Output Layers): final layers that map the transformer outputs to the shared embedding space and compute similarity scores.
	"""

	def hasattr_nested(obj, attr_string):
		attrs = attr_string.split('.')
		for attr in attrs:
			if hasattr(obj, attr):
				obj = getattr(obj, attr)
			else:
				return False
		return True

	layer_groups = {
		'visual_frontend': [
			'visual.conv1', # patch embedding (ViT) or first conv layer (ResNet)
			'visual.class_embedding' if is_vit else 'visual.bn1', # CLS token for ViT, bn1 for ResNet
			'visual.positional_embedding' if is_vit else 'visual.relu', # positional embedding for ViT, relu for ResNet
		],
		'visual_transformer': visual_blocks,
		'text_frontend': [ # Converts tokenized text into embeddings (token_embedding) then adds positional information (positional_embedding).
			'token_embedding',
			'positional_embedding',
		],
		'text_transformer': text_blocks,
		'projections': [], # Start with an empty list
		# 'projections': [
		# 	'visual.proj', # Projects visual transformer’s output (e.g., the CLS token embedding) into the shared space.
		# 	'visual.ln_post' if is_vit else 'visual.attnpool',  # ln_post for ViT, attnpool for ResNet
		# 	'text_projection', # Projects the text transformer’s output into the shared space.
		# 	'logit_scale', # learnable scalar that scales the cosine similarities between image and text embeddings during contrastive loss computation.
		# ],
	}

	# Conditionally add projection layers only if they
	# exist in the model and are supported by the architecture
	projection_candidates = {
		'visual.proj': True, # Always check for this
		'visual.ln_post': is_vit,
		'visual.attnpool': is_resnet,
		'text_projection': True,
		'logit_scale': True,
	}

	for layer_name, should_check in projection_candidates.items():
		if should_check and hasattr_nested(model, layer_name):
			layer_groups['projections'].append(layer_name)

	# Also filter frontend layers that might not exist in all models
	layer_groups['visual_frontend'] = [
		name 
		for name in layer_groups['visual_frontend'] 
		if hasattr_nested(model, name)
	]

	layer_groups['text_frontend'] = [
		name 
		for name in layer_groups['text_frontend'] 
		if hasattr_nested(model, name)
	]

	return layer_groups

def unfreeze_layers(
		model: torch.nn.Module,
		strategy: Dict[int, List[str]],
		phase: int,
		cache: Dict[int, List[str]],
	):
	print(f"Applying unfreeze strategy for Phase {phase}...")
	
	# 1. Get the layers to unfreeze at this phase
	layers_to_unfreeze = strategy[phase]

	# 2. Unfreeze the layers
	# Assumes layer names in layers_to_unfreeze are prefixes of parameter names
	# (e.g., 'visual.transformer.resblocks.0' matches 'visual.transformer.resblocks.0.attn.in_proj_weight')
	for name, param in model.named_parameters():
		param.requires_grad = False # Freeze all layers first
		if any(ly in name for ly in layers_to_unfreeze): # Unfreeze layers in the list
			param.requires_grad = True

	# 3. Cache the frozen layers
	get_status(
		model=model,
		phase=phase,
		layers_to_unfreeze=layers_to_unfreeze,
		cache=cache,
	)

def get_unfreeze_schedule(
		model: torch.nn.Module,
		num_phases: int,
		layer_groups_to_unfreeze: List[str]=['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'],
	) -> Dict[int, List[str]]:

	unfreeze_fractions = np.linspace(0, 1, num_phases).tolist()
	print(f"Getting unfreeze schedule for {model.name} for {len(unfreeze_fractions)} phases".center(120, "-"))
	print(f"Layer groups to unfreeze: {layer_groups_to_unfreeze}")
	print(f"Unfreeze fractions: {unfreeze_fractions}")
	
	
	# Validate input
	if not all(0.0 <= p <= 1.0 for p in unfreeze_fractions):
		raise ValueError("Unfreeze fractions must be between 0.0 and 1.0.")

	if not all(g in ['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'] for g in layer_groups_to_unfreeze):
		raise ValueError("Invalid layer group specified. Accepted: visual_frontend, visual_transformer, text_frontend, text_transformer, projections.")

	def create_layer_table(num_layers: int, layer_type: str) -> str:
		table_data = list()
		for i, pct in enumerate(unfreeze_fractions):
			label = f"{int(pct * 100)}%" if pct != 0.0 and pct != 1.0 else ("None" if pct == 0.0 else "All")
			table_data.append([
				i,
				label,
				f"{int(pct * num_layers)}/{num_layers}",
				f"{(pct * 100):.0f}%"
			])

		return (
			f"\n{layer_type} Transformer Layer Unfreezing Schedule:\n"
			+ tabulate.tabulate(
				table_data,
				headers=["#", "Phase Type", "Unfrozen Layers", "Unfrozen Percentage(%)"],
				tablefmt="grid"
			)
		)

	layer_groups = get_layer_groups(model=model)
	print(f"\nLayer groups:\n{json.dumps(layer_groups, indent=2)}")

	# Explicitly get all layer groups from the model definition
	projections = layer_groups.get('projections', [])
	visual_transformer = layer_groups.get('visual_transformer', [])
	text_transformer = layer_groups.get('text_transformer', [])
	visual_frontend = layer_groups.get('visual_frontend', [])
	text_frontend = layer_groups.get('text_frontend', [])
	
	total_v_layers = len(visual_transformer)
	total_t_layers = len(text_transformer)
	print(f"Total visual transformer layers: {total_v_layers}")
	print(f"Total text transformer layers: {total_t_layers}")

	if total_v_layers == 0 and total_t_layers == 0:
		raise ValueError("No transformer blocks found in visual or text encoders. Cannot create unfreezing schedule.")

	schedule = {}
	for phase, unfreeze_pct in enumerate(unfreeze_fractions):
		# Start with an empty list for this phase's layers
		layers_to_unfreeze_for_phase = list()
		
		# 1. ALWAYS include projections if they are in the target groups
		if 'projections' in layer_groups_to_unfreeze:
			layers_to_unfreeze_for_phase.extend(projections)
			
		# 2. Add transformer layers based on percentage
		if 'visual_transformer' in layer_groups_to_unfreeze:
			v_unfreeze_count = int(unfreeze_pct * total_v_layers)
			if v_unfreeze_count > 0:
				layers_to_unfreeze_for_phase.extend(visual_transformer[-v_unfreeze_count:])

		if 'text_transformer' in layer_groups_to_unfreeze:
			t_unfreeze_count = int(unfreeze_pct * total_t_layers)
			if t_unfreeze_count > 0:
				layers_to_unfreeze_for_phase.extend(text_transformer[-t_unfreeze_count:])
				
		# 3. Add frontend layers ONLY at the final phase (100%)
		if unfreeze_pct == 1.0:
			if 'visual_frontend' in layer_groups_to_unfreeze:
				layers_to_unfreeze_for_phase.extend(visual_frontend)
			if 'text_frontend' in layer_groups_to_unfreeze:
				layers_to_unfreeze_for_phase.extend(text_frontend)
		
		# Use a set to remove duplicates (e.g., if a layer name is in multiple groups)
		# and convert back to a sorted list for consistent ordering.
		schedule[phase] = sorted(list(set(layers_to_unfreeze_for_phase)))

		# --- Enhanced Logging for verification ---
		print(f"Phase {phase} (unfreeze_pct: {unfreeze_pct*100:.2f}%): {len(schedule[phase])} layers to unfreeze:")
		for i, layer in enumerate(schedule[phase]):
			print(f"\t{i:02d} {layer}")

	print(create_layer_table(total_v_layers, "Visual"))
	print(create_layer_table(total_t_layers, "Text"))

	print("-"*120)
	return schedule

def compute_embedding_drift(
		model: torch.nn.Module, 
		val_subset: torch.utils.data.DataLoader, 
		pretrained_embeds: torch.Tensor, 
		device: torch.device, 
		phase: int, 
		epoch: int,
	):
	"""
	Embedding Drift = 1 - cosine_similarity, 
	measures how far the current image embeddings 
	moved from their original, pre-trained positions. 
	0.0: no change, 
	1.0: they are now orthogonal (completely different).
	
	In summary, the ideal Embedding Drift curve:
		Starts near zero.
		Shows small, controlled increases during early-to-mid phases 
		that are inversely correlated with validation loss (drift goes up, loss goes down).
		Plateaus in later phases, indicating that the foundational knowledge is being preserved.
	"""
	model.eval()
	with torch.no_grad():
		imgs = next(iter(val_subset)) # torch.Size([batch_size, channels, height, width ])
		imgs = imgs.to(device)
		new_embeds = model.encode_image(imgs)
		new_embeds = F.normalize(new_embeds, dim=-1)
		drift = F.cosine_similarity(new_embeds, pretrained_embeds[:new_embeds.size(0)].to(device), dim=-1)
		mean_drift = 1 - drift.mean().item()
	# print(f"[DEBUG] Embedding Drift | Phase {phase} | Epoch {epoch}: {mean_drift}")
	return mean_drift

def should_transition_to_next_phase(
		current_phase: int,
		losses: List[float],
		window: int,
		epochs_in_phase: int,
		min_epochs_per_phase: int,
		num_phases: int,
		volatility_threshold: float,
		plateau_threshold: float = 1e-2,				# 1% improvement threshold
	) -> bool:
	
	# Must meet minimum epochs requirement
	if epochs_in_phase < min_epochs_per_phase:
		return False

	# Don't transition from final phase
	if current_phase >= num_phases - 1:
		return False

	# Need enough history to detect trends
	if len(losses) < window:
		print(f"<!> Insufficient loss data ({len(losses)} < {window}) for phase transition.")
		return False
	
	recent_window = losses[-window:]
	
	# --- METRIC CALCULATIONS ---
	# 1. Volatility (Coefficient of Variation)
	mean_loss = np.mean(recent_window)
	std_loss = np.std(recent_window)
	cv = (std_loss / mean_loss) * 100 if mean_loss != 0 else 0  # Volatility in %
	high_volatility = cv > volatility_threshold

	# 2. Trend Slope (using linear regression)
	slope = np.polyfit(range(len(recent_window)), recent_window, deg=1)[0] # 1st degree polynomial (line) fit (negative = improving)
	improvement_magnitude = abs(slope)
	is_plateau = improvement_magnitude < plateau_threshold

	# Additional contextual information
	is_improving = slope < 0 and improvement_magnitude > plateau_threshold
	is_worsening = slope > 0 and improvement_magnitude > plateau_threshold

	print("=" * 120)
	print(f"TRANSITION CHECK @ Phase {current_phase}")
	# print(f"{len(losses)} losses: {losses}")
	# print(f"\t>> min: {min(losses)} max: {max(losses)} range: {max(losses) - min(losses)} std: {np.std(losses)}")
	print(
		f"Improvement over last {window} losses: {improvement_magnitude} (threshold: {plateau_threshold})\n"
		f"  ├─ Trend: {'Improving' if is_improving else 'Worsening' if is_worsening else 'Stable'}\n"
		f"  └─ Plateau: {is_plateau}"
	)
	print(f"Volatility: {cv:.2f}% (high if > {volatility_threshold}%)")

	# Transition logic
	should_transition = False
	reasons = list()
	if is_plateau and not high_volatility:
		should_transition = True
		reasons.append(f"Meaningful plateau detected ({improvement_magnitude} < {plateau_threshold})")
	elif high_volatility:
		reasons.append(f"High volatility ({cv:.2f}%) - continuing current phase for stability")
	else:
		reasons.append(f"Still learning ({improvement_magnitude} improvement > threshold: {plateau_threshold})")
	
	if should_transition:
		print(f"\t>>> TRANSITION RECOMMENDED: {', '.join(reasons)}")
	else:
		print(f"\t>>> CONTINUE CURRENT PHASE: {', '.join(reasons)}")
	print("=" * 120)
	return should_transition

def handle_phase_transition(
		current_phase: int, 
		optimizer: torch.optim.Optimizer,
	) -> Tuple[int, float, float]:
	next_phase = current_phase + 1
	new_lr = optimizer.param_groups[0]['lr'] * 1.0  # 0% reduction (no change)
	new_wd = optimizer.param_groups[0]['weight_decay'] * 1.15  # 15% increase
	return next_phase, new_lr, new_wd

