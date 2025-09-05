import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")
CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
from utils import *
from historical_dataset_loader import get_single_label_dataloaders, get_multi_label_dataloaders

def create_hyperparameter_evolution_plot(training_history, results_dir, model_arch):
	epochs = range(len(training_history['train_losses']))
	phases = training_history['phases']
	transitions = training_history['phase_transitions']
	learning_rates = training_history.get('learning_rates', [])
	weight_decays = training_history.get('weight_decays', [])
	
	# Create figure with 2 subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
	
	# Phase color mapping
	max_phase = max(phases) if phases else 0
	phase_colors = plt.cm.Set3(np.linspace(0, 1, max_phase + 1))
	
	# ===== Learning Rate Evolution Plot =====
	if learning_rates:
			# Plot learning rate line
			ax1.plot(epochs[:len(learning_rates)], learning_rates, 'b-', linewidth=2, alpha=0.8)
			
			# Color code background by phases
			for phase_num in range(max_phase + 1):
					phase_epochs = [i for i, p in enumerate(phases) if p == phase_num and i < len(learning_rates)]
					if phase_epochs:
							start_epoch = min(phase_epochs)
							end_epoch = max(phase_epochs) + 1
							ax1.axvspan(start_epoch, end_epoch, alpha=0.2, color=phase_colors[phase_num])
			
			# Mark phase transitions
			for transition_epoch in transitions:
					if transition_epoch < len(learning_rates):
							ax1.axvline(x=transition_epoch, color='red', linestyle='--', linewidth=2, alpha=0.8)
							
							# Add transition annotations
							lr_at_transition = learning_rates[transition_epoch] if transition_epoch < len(learning_rates) else learning_rates[-1]
							ax1.annotate(f'Transition\nLR: {lr_at_transition:.2e}', 
												 xy=(transition_epoch, lr_at_transition), 
												 xytext=(10, 20), textcoords='offset points',
												 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
												 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
												 fontsize=9)
			
			# Formatting
			ax1.set_ylabel('Learning Rate', fontsize=12)
			ax1.set_title('Learning Rate Evolution Across Phases', fontsize=14, fontweight='bold')
			ax1.set_yscale('log')  # Log scale for better LR visualization
			ax1.grid(True, alpha=0.3)
			
			# Add phase labels at the top
			for phase_num in range(max_phase + 1):
					phase_epochs = [i for i, p in enumerate(phases) if p == phase_num and i < len(learning_rates)]
					if phase_epochs:
							mid_epoch = (min(phase_epochs) + max(phase_epochs)) / 2
							max_lr = max(learning_rates)
							ax1.text(mid_epoch, max_lr * 1.5, f'Phase {phase_num}', 
											ha='center', va='bottom', fontsize=10, fontweight='bold',
											bbox=dict(boxstyle='round,pad=0.3', facecolor=phase_colors[phase_num], alpha=0.7))
	else:
			# No learning rate data available
			ax1.text(0.5, 0.5, 'Learning Rate Data Not Available', 
							ha='center', va='center', transform=ax1.transAxes, 
							fontsize=14, style='italic', color='gray')
			ax1.set_title('Learning Rate Evolution Across Phases', fontsize=14, fontweight='bold')
	
	# ===== Weight Decay Evolution Plot =====
	if weight_decays:
			# Plot weight decay line
			ax2.plot(epochs[:len(weight_decays)], weight_decays, 'g-', linewidth=2, alpha=0.8)
			
			# Color code background by phases
			for phase_num in range(max_phase + 1):
					phase_epochs = [i for i, p in enumerate(phases) if p == phase_num and i < len(weight_decays)]
					if phase_epochs:
							start_epoch = min(phase_epochs)
							end_epoch = max(phase_epochs) + 1
							ax2.axvspan(start_epoch, end_epoch, alpha=0.2, color=phase_colors[phase_num])
			
			# Mark phase transitions
			for transition_epoch in transitions:
					if transition_epoch < len(weight_decays):
							ax2.axvline(x=transition_epoch, color='red', linestyle='--', linewidth=2, alpha=0.8)
							
							# Add transition annotations
							wd_at_transition = weight_decays[transition_epoch] if transition_epoch < len(weight_decays) else weight_decays[-1]
							ax2.annotate(f'Transition\nWD: {wd_at_transition:.2e}', 
												 xy=(transition_epoch, wd_at_transition), 
												 xytext=(10, 20), textcoords='offset points',
												 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
												 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
												 fontsize=9)
			
			# Formatting
			ax2.set_xlabel('Epoch', fontsize=12)
			ax2.set_ylabel('Weight Decay', fontsize=12)
			ax2.set_title('Weight Decay Evolution Across Phases', fontsize=14, fontweight='bold')
			ax2.grid(True, alpha=0.3)
			
			# Add phase labels at the top
			for phase_num in range(max_phase + 1):
					phase_epochs = [i for i, p in enumerate(phases) if p == phase_num and i < len(weight_decays)]
					if phase_epochs:
							mid_epoch = (min(phase_epochs) + max(phase_epochs)) / 2
							max_wd = max(weight_decays) if weight_decays else 1
							ax2.text(mid_epoch, max_wd * 1.1, f'Phase {phase_num}', 
											ha='center', va='bottom', fontsize=10, fontweight='bold',
											bbox=dict(boxstyle='round,pad=0.3', facecolor=phase_colors[phase_num], alpha=0.7))
	else:
			# No weight decay data available
			ax2.text(0.5, 0.5, 'Weight Decay Data Not Available', 
							ha='center', va='center', transform=ax2.transAxes, 
							fontsize=14, style='italic', color='gray')
			ax2.set_xlabel('Epoch', fontsize=12)
			ax2.set_title('Weight Decay Evolution Across Phases', fontsize=14, fontweight='bold')
	
	# Final layout and save
	plt.tight_layout()
	
	# Save the plot
	plot_path = os.path.join(results_dir, f'hyperparameter_evolution_{model_arch}.png')
	plt.savefig(plot_path, dpi=300, bbox_inches='tight')
	plt.close()
	
	print(f"Hyperparameter evolution plot saved: {plot_path}")

def simplified_progressive_finetune(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		
		# Simplified parameters
		num_phases: int = 4,
		min_epochs_per_phase: int = 10,
		patience_factor: float = 1.5,
		transition_threshold: float = 0.001,
		
		# Optional parameters with defaults
		layer_groups_to_unfreeze: List[str] = None,
		topk_values: List[int] = None,
		verbose: bool = True,
	):
	
	if layer_groups_to_unfreeze is None:
		layer_groups_to_unfreeze = ['projections', 'visual_transformer', 'text_transformer']
	if topk_values is None:
		topk_values = [1, 5, 10]
	
	# Setup
	mode = "simplified_progressive"
	dataset_name = getattr(validation_loader.dataset, 'dataset_name', 'unknown')
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	
	print(f"{mode} | {model_arch} | {dataset_name} | {num_phases} phases".center(120, "-"))
	
	# Get unfreezing schedule
	unfreeze_schedule = get_simplified_unfreeze_schedule(
		model=model,
		num_phases=num_phases,
		layer_groups=layer_groups_to_unfreeze
	)
	
	# Simple early stopping
	base_patience = max(5, min_epochs_per_phase)
	early_stopping = SimpleEarlyStopping(
		patience=int(base_patience * patience_factor),
		min_delta=transition_threshold,
		mode='min'
	)
	
	# Training state
	current_phase = 0
	epochs_in_phase = 0
	
	# History tracking - WITH DEBUG PRINTS
	training_history = {
		'train_losses': [],
		'val_losses': [],
		'phases': [],
		'phase_transitions': [],
		'metrics_per_epoch': [],
		'learning_rates': [],
		'weight_decays': [],
		'optimizer_states': []
	}
	
	print(f"DEBUG: Initialized training_history with keys: {list(training_history.keys())}")
	
	# Model file path
	model_path = os.path.join(results_dir, f"{mode}_{model_arch}_phases_{num_phases}.pth")
	
	print(f"Training with {num_phases} phases, {min_epochs_per_phase} min epochs per phase")
	
	# Initialize optimizer and scheduler
	optimizer = None
	scheduler = None
	
	for epoch in range(num_epochs):
		epoch_start = time.time()
		
		# Apply current phase unfreezing
		if epochs_in_phase == 0:  # New phase started
			apply_phase_unfreezing(model, unfreeze_schedule, current_phase)
			
			# Create fresh optimizer for new phase
			optimizer = create_phase_optimizer(model, learning_rate, weight_decay)
			
			# Simple fixed LR scheduler per phase
			remaining_epochs = num_epochs - epoch
			scheduler = create_phase_scheduler(optimizer, remaining_epochs)
			
			if verbose:
				print(f"\n=== PHASE {current_phase} STARTED (Epoch {epoch+1}) ===")
				print_phase_status(model, unfreeze_schedule[current_phase])
		
		# Training step
		model.train()
		train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
		
		# Validation step
		val_metrics = validate_epoch(model, validation_loader, device, topk_values)
		val_loss = val_metrics['val_loss']
		
		# TRACK HYPERPARAMETERS WITH DEBUG
		if optimizer is not None:
			current_lr = optimizer.param_groups[0]['lr']
			current_wd = optimizer.param_groups[0]['weight_decay']
			print(f"DEBUG: Epoch {epoch+1} - LR: {current_lr:.2e}, WD: {current_wd:.2e}")
		else:
			current_lr = learning_rate
			current_wd = weight_decay
			print(f"DEBUG: Epoch {epoch+1} - No optimizer yet, using defaults LR: {current_lr:.2e}, WD: {current_wd:.2e}")
		
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		
		# Update history
		training_history['train_losses'].append(train_loss)
		training_history['val_losses'].append(val_loss)
		training_history['phases'].append(current_phase)
		training_history['metrics_per_epoch'].append(val_metrics)
		training_history['learning_rates'].append(current_lr)
		training_history['weight_decays'].append(current_wd)
		training_history['optimizer_states'].append({
			'epoch': epoch,
			'phase': current_phase,
			'lr': current_lr,
			'weight_decay': current_wd,
			'trainable_params': trainable_params
		})
		
		if epoch == 0:  # Debug first epoch
			print(f"DEBUG: After epoch 0, learning_rates length: {len(training_history['learning_rates'])}")
			print(f"DEBUG: After epoch 0, weight_decays length: {len(training_history['weight_decays'])}")
			print(f"DEBUG: Learning rates so far: {training_history['learning_rates']}")
			print(f"DEBUG: Weight decays so far: {training_history['weight_decays']}")
		
		epochs_in_phase += 1
		
		if verbose:
			print(
				f"Epoch {epoch+1:3d} | Phase {current_phase} | "
				f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
				f"LR: {current_lr:.2e} | WD: {current_wd:.2e} | "
				f"Params: {trainable_params:,} | "
				f"Time: {time.time()-epoch_start:.1f}s"
			)
		
		# Early stopping check
		if early_stopping.should_stop(val_loss, model, epoch):
			if verbose:
				print(f"Early stopping triggered at epoch {epoch+1}")
			break
		
		# Phase transition check
		if should_transition_to_next_phase(
			current_phase=current_phase,
			epochs_in_phase=epochs_in_phase,
			val_losses=training_history['val_losses'],
			min_epochs_per_phase=min_epochs_per_phase,
			transition_threshold=transition_threshold,
			num_phases=num_phases
		):
			training_history['phase_transitions'].append(epoch)
			current_phase += 1
			epochs_in_phase = 0
			
			if verbose:
				print(f"\n>>> TRANSITION TO PHASE {current_phase} @ Epoch {epoch+1}")
	
	# DEBUG: Check final training history
	print(f"DEBUG: Final training_history keys: {list(training_history.keys())}")
	print(f"DEBUG: Final learning_rates length: {len(training_history['learning_rates'])}")
	print(f"DEBUG: Final weight_decays length: {len(training_history['weight_decays'])}")
	
	# Save final model
	torch.save(
		{
		'model_state_dict': model.state_dict(),
		'training_history': training_history,
		'unfreeze_schedule': unfreeze_schedule,
		'final_phase': current_phase
		}, 
		model_path
	)
	
	if verbose:
		analyze_progressive_training(training_history, results_dir, model_arch)

	# This should now work
	print("DEBUG: About to call create_hyperparameter_evolution_plot...")
	create_hyperparameter_evolution_plot(training_history, results_dir, model_arch)

	return training_history

def get_simplified_unfreeze_schedule(model, num_phases, layer_groups):
		"""
		Create a simple, predictable unfreezing schedule.
		"""
		layer_groups_map = get_layer_groups(model)
		
		# Always start with projections only
		schedule = {0: layer_groups_map['projections']}
		
		if num_phases == 1:
				return schedule
		
		# Gradually add other groups
		other_groups = [g for g in layer_groups if g != 'projections']
		
		for phase in range(1, num_phases):
				layers_to_unfreeze = layer_groups_map['projections'].copy()
				
				# Add groups progressively
				progress = phase / (num_phases - 1)  # 0 to 1
				
				for group in other_groups:
						if group in layer_groups_map:
								group_layers = layer_groups_map[group]
								layers_to_add = int(len(group_layers) * progress)
								
								if group == 'visual_transformer':
										# Unfreeze from the end (closer to output)
										layers_to_unfreeze.extend(group_layers[-layers_to_add:])
								elif group == 'text_transformer':
										# Unfreeze from the end
										layers_to_unfreeze.extend(group_layers[-layers_to_add:])
								else:
										# For other groups, add all at final phase
										if phase == num_phases - 1:
												layers_to_unfreeze.extend(group_layers)
				
				schedule[phase] = list(set(layers_to_unfreeze))  # Remove duplicates
		
		return schedule

def should_transition_to_next_phase(
		current_phase, epochs_in_phase, val_losses, 
		min_epochs_per_phase, transition_threshold, num_phases
):
		"""
		Simple plateau-based transition criterion.
		"""
		# Must meet minimum epochs requirement
		if epochs_in_phase < min_epochs_per_phase:
				return False
		
		# Don't transition from final phase
		if current_phase >= num_phases - 1:
				return False
		
		# Need enough history to detect plateau
		window_size = min(5, min_epochs_per_phase)
		if len(val_losses) < window_size:
				return False
		
		# Check for plateau in validation loss
		recent_losses = val_losses[-window_size:]
		
		# Simple improvement check
		best_recent = min(recent_losses)
		current_loss = recent_losses[-1]
		
		# Transition if no significant improvement
		improvement = best_recent - current_loss
		
		is_plateau = improvement < transition_threshold
		
		print(f"\tTransition check: improvement={improvement}, threshold={transition_threshold}, plateau={is_plateau}")
		
		return is_plateau

class SimpleEarlyStopping:
		"""
		Simplified early stopping with single criterion.
		"""
		def __init__(self, patience=10, min_delta=1e-4, mode='min'):
				self.patience = patience
				self.min_delta = min_delta
				self.mode = mode
				self.best_score = None
				self.counter = 0
				self.sign = 1 if mode == 'min' else -1
		
		def should_stop(self, score, model=None, epoch=None):
				score = score * self.sign
				
				if self.best_score is None:
						self.best_score = score
						return False
				
				if score > self.best_score + self.min_delta:
						self.best_score = score
						self.counter = 0
						return False
				else:
						self.counter += 1
						return self.counter >= self.patience

def apply_phase_unfreezing(model, schedule, phase):
		"""
		Apply unfreezing for a specific phase.
		"""
		# Freeze all parameters first
		for param in model.parameters():
				param.requires_grad = False
		
		# Unfreeze specified layers
		layers_to_unfreeze = schedule.get(phase, [])
		
		for name, param in model.named_parameters():
				for layer_prefix in layers_to_unfreeze:
						if name.startswith(layer_prefix):
								param.requires_grad = True
								break

def create_phase_optimizer(model, lr, wd):
	trainable_params = [p for p in model.parameters() if p.requires_grad]
	
	if not trainable_params:
		raise ValueError("No trainable parameters found!")
	
	return torch.optim.AdamW(
		trainable_params,
		lr=lr,
		weight_decay=wd,
		betas=(0.9, 0.98),
		eps=1e-6
	)

def create_phase_scheduler(optimizer, remaining_epochs):
		"""
		Create a simple scheduler for the phase.
		"""
		return torch.optim.lr_scheduler.CosineAnnealingLR(
				optimizer,
				T_max=remaining_epochs,
				eta_min=optimizer.param_groups[0]['lr'] * 0.01
		)

def train_epoch(model, train_loader, optimizer, scheduler, device):
		"""
		Simple training epoch.
		"""
		model.train()
		total_loss = 0
		num_batches = 0
		
		criterion = torch.nn.CrossEntropyLoss()
		scaler = torch.amp.GradScaler()
		
		for images, tokenized_labels, _ in train_loader:
				images = images.to(device, non_blocking=True)
				tokenized_labels = tokenized_labels.to(device, non_blocking=True)
				
				optimizer.zero_grad()
				
				with torch.amp.autocast(device_type=device.type):
						logits_per_image, logits_per_text = model(images, tokenized_labels)
						ground_truth = torch.arange(len(images), device=device)
						
						loss_img = criterion(logits_per_image, ground_truth)
						loss_txt = criterion(logits_per_text, ground_truth)
						loss = 0.5 * (loss_img + loss_txt)
				
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				scaler.step(optimizer)
				scaler.update()
				
				scheduler.step()
				
				total_loss += loss.item()
				num_batches += 1
		
		return total_loss / num_batches if num_batches > 0 else 0.0

def validate_epoch(model, validation_loader, device, topk_values):
		"""
		Simple validation epoch.
		"""
		model.eval()
		total_loss = 0
		num_batches = 0
		
		criterion = torch.nn.CrossEntropyLoss()
		
		with torch.no_grad():
				for images, tokenized_labels, _ in validation_loader:
						images = images.to(device, non_blocking=True)
						tokenized_labels = tokenized_labels.to(device, non_blocking=True)
						
						logits_per_image, logits_per_text = model(images, tokenized_labels)
						ground_truth = torch.arange(len(images), device=device)
						
						loss_img = criterion(logits_per_image, ground_truth)
						loss_txt = criterion(logits_per_text, ground_truth)
						loss = 0.5 * (loss_img + loss_txt)
						
						total_loss += loss.item()
						num_batches += 1
		
		return {
				'val_loss': total_loss / num_batches if num_batches > 0 else 0.0
		}

def print_phase_status(model, layers_to_unfreeze):
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	
	print(
		f"Phase status: {trainable_params:,}/{total_params:,} "
		f"({100*trainable_params/total_params:.3f}%) parameters trainable"
	)
	print(f"Unfrozen layers: {len(layers_to_unfreeze)} layer groups:")
	print(f"{layers_to_unfreeze}")

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

def get_layer_groups(
		model: torch.nn.Module
	) -> dict:
	vis_nblocks, txt_nblocks = get_num_transformer_blocks(model=model)

	# Determine model type
	is_vit = "ViT" in model.name
	is_resnet = "RN" in model.name

	# Visual transformer or CNN blocks
	visual_blocks = []
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

def analyze_progressive_training(training_history, results_dir, model_arch):
	epochs = range(len(training_history['train_losses']))
	train_losses = training_history['train_losses']
	val_losses = training_history['val_losses']
	phases = training_history['phases']
	transitions = training_history['phase_transitions']
	
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
	
	# Plot losses
	ax1.plot(epochs, train_losses, label='Train Loss', alpha=0.7)
	ax1.plot(epochs, val_losses, label='Val Loss', alpha=0.7)
	
	# Mark phase transitions
	for transition_epoch in transitions:
		ax1.axvline(x=transition_epoch, color='red', linestyle='--', alpha=0.5)
	
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss')
	ax1.set_title('Progressive Fine-tuning: Loss Evolution')
	ax1.legend()
	ax1.grid(True, alpha=0.3)
	
	# Plot phases
	ax2.plot(epochs, phases, marker='o', linestyle='-', alpha=0.7)
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Phase')
	ax2.set_title('Phase Progression')
	ax2.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(os.path.join(results_dir, f'progressive_analysis_{model_arch}.png'), dpi=300)
	plt.close()
	
	# Print summary
	print("\n" + "="*60)
	print("PROGRESSIVE TRAINING ANALYSIS")
	print("="*60)
	print(f"Total epochs: {len(train_losses)}")
	print(f"Phase transitions: {len(transitions)} at epochs {transitions}")
	print(f"Final phase reached: {max(phases)}")
	
	# Analyze phase performance
	phase_performance = {}
	for phase in range(max(phases) + 1):
		phase_epochs = [i for i, p in enumerate(phases) if p == phase]
		if phase_epochs:
			phase_val_losses = [val_losses[i] for i in phase_epochs]
			phase_performance[phase] = {
				'epochs': len(phase_epochs),
				'best_val_loss': min(phase_val_losses),
				'final_val_loss': phase_val_losses[-1],
				'improvement': phase_val_losses[0] - phase_val_losses[-1] if len(phase_val_losses) > 1 else 0
			}
	
	print("\nPhase Performance:")
	for phase, perf in phase_performance.items():
		print(
			f"\tPhase {phase}: {perf['epochs']} epochs, "
			f"best_val={perf['best_val_loss']:.4f}, "
			f"improvement={perf['improvement']:.4f}"
		)
	
	print("="*60)
		
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
	parser.add_argument('--dataset_type', '-dt', type=str, choices=['single_label', 'multi_label'], default='single_label', help='Dataset type (single_label/multi_label)')
	parser.add_argument('--mode', '-m', type=str, choices=['train', 'finetune', 'pretrain'], default='finetune', help='Choose mode (train/finetune/pretrain)')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size for training')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_epochs', '-ne', type=int, default=25, help='Number of epochs for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5, help='Learning rate for training')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='Weight decay for training')
	parser.add_argument('--num_workers', '-nw', type=int, default=10, help='Number of workers for data loading')
	parser.add_argument('--dropout', '-do', type=float, default=0.0, help='Dropout probability')
	parser.add_argument('--num_phases', '-np', type=int, default=8, help='Number of phases for progressive fine-tuning')
	parser.add_argument('--min_epochs_per_phase', '-mep', type=int, default=10, help='Minimum epochs per phase')
	parser.add_argument('--patience_factor', '-pf', type=float, default=1.5, help='Patience factor for early stopping')
	parser.add_argument('--transition_threshold', '-tt', type=float, default=0.001, help='Transition threshold for phase change')
	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	args.dataset_dir = os.path.normpath(args.dataset_dir)

	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	print_args_table(args=args, parser=parser)
	print(args)
	set_seeds(seed=42)
	RESULT_DIRECTORY = os.path.join(args.dataset_dir, f"{args.dataset_type}")
	os.makedirs(RESULT_DIRECTORY, exist_ok=True)

	print(f">> CLIP {args.model_architecture} Architecture:")
	model_config = get_config(
		architecture=args.model_architecture, 
		dropout=args.dropout,
	)
	print(json.dumps(model_config, indent=4, ensure_ascii=False))
	model, _ = clip.load(
		name=args.model_architecture,
		device=args.device, 
		jit=False, # training or finetuning => jit=False
		random_weights=True if args.mode == 'train' else False, 
		dropout=args.dropout,
		download_root=get_model_directory(path=args.dataset_dir),
	)
	model = model.float() # Convert model parameters to FP32
	model.name = args.model_architecture  # Custom attribute to store model name
	model_name = model.__class__.__name__
	print(f"Loaded {model_name} {model.name} in {args.device}")
	dataset_functions = {
		'single_label': get_single_label_dataloaders,
		'multi_label': get_multi_label_dataloaders
	}
	train_loader, validation_loader = dataset_functions[args.dataset_type](
		dataset_dir=args.dataset_dir,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		input_resolution=model_config["image_resolution"],
		# cache_size=args.cache_size,
	)
	print_loader_info(loader=train_loader, batch_size=args.batch_size)
	print_loader_info(loader=validation_loader, batch_size=args.batch_size)

	print(f">> Start Progressive Fine-tuning...")
	simplified_progressive_finetune(
		model=model,
		train_loader=train_loader,
		validation_loader=validation_loader,
		num_epochs=args.num_epochs,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		device=args.device,
		results_dir=RESULT_DIRECTORY,
		num_phases=args.num_phases,
		min_epochs_per_phase=args.min_epochs_per_phase,
		patience_factor=args.patience_factor,
		transition_threshold=args.transition_threshold,
	)
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))

if __name__ == "__main__":
	main()