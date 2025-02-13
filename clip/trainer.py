from utils import *
from datasets_loader import *
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import AdamW, SGD, Adam, lr_scheduler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import copy

# train cifar100 from scratch:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -e 12 -lr 1e-4 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:3" -m "train" -md "ViT-B/32" > /media/volume/ImACCESS/trash/cifar100_train.out &

# finetune cifar100:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -e 256 -lr 1e-4 -wd 1e-3 --print_every 100 -nw 50 --device "cuda:2" -m "finetune" -md "ViT-B/32" > /media/volume/ImACCESS/trash/cifar100_ft.out &

# train imagenet from scratch:
# $ nohup python -u trainer.py -d imagenet -bs 256 -e 100 -lr 1e-5 -wd 1e-3 --print_every 2500 -nw 50 --device "cuda:1" -m "train" -md "ViT-B/32" > /media/volume/ImACCESS/trash/imagenet_train.out &

# finetune imagenet:
# $ nohup python -u trainer.py -d imagenet -bs 256 -e 100 -lr 1e-5 -wd 1e-3 --print_every 2500 -nw 50 --device "cuda:0" -m "finetune" -md "ViT-B/32" > /media/volume/ImACCESS/trash/imagenet_ft.out &

USER = os.environ.get('USER')

class EarlyStopping:
	def __init__(
			self,
			patience: int = 5, # epochs to wait before stopping the training
			min_delta: float = 1e-3, # minimum difference between new and old loss to count as improvement
			cumulative_delta: float = 0.01,
			window_size: int = 5,
			mode: str = 'min',
			min_epochs: int = 5,
			restore_best_weights: bool = True,
		):
		"""
		Args:
			patience: Number of epochs to wait before early stopping
			min_delta: Minimum change in monitored value to qualify as an improvement
			cumulative_delta: Minimum cumulative improvement over window_size epochs
			window_size: Size of the window for tracking improvement trends
			mode: 'min' for loss, 'max' for metrics like accuracy
			min_epochs: Minimum number of epochs before early stopping can trigger
			restore_best_weights: Whether to restore model to best weights when stopped
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.cumulative_delta = cumulative_delta
		self.window_size = window_size
		self.mode = mode
		self.min_epochs = min_epochs
		self.restore_best_weights = restore_best_weights
		
		self.best_score = None
		self.best_weights = None
		self.counter = 0
		self.stopped_epoch = 0
		self.value_history = []
		self.improvement_history = []
		
		self.sign = 1 if mode == 'min' else -1
	
	def is_improvement(self, current_value: float) -> bool:
		if self.best_score is None:
			return True
		improvement = (self.best_score - current_value) * self.sign
		return improvement > self.min_delta
	
	def calculate_trend(self) -> float:
		"""Calculate improvement trend over window"""
		if len(self.value_history) < self.window_size:
			return float('inf') if self.mode == 'min' else float('-inf')
		window = self.value_history[-self.window_size:]
		if self.mode == 'min':
			return sum(window[i] - window[i+1] for i in range(len(window)-1))
		return sum(window[i+1] - window[i] for i in range(len(window)-1))
	
	def should_stop(self, current_value: float, model: nn.Module, epoch: int) -> bool:
		"""
		Enhanced stopping decision based on multiple criteria
		"""
		self.value_history.append(current_value)
		# Don't stop before minimum epochs
		if epoch < self.min_epochs:
			if self.best_score is None or current_value * self.sign < self.best_score * self.sign:
				self.best_score = current_value
				self.stopped_epoch = epoch  # Update stopped_epoch when a new best score is achieved
				if self.restore_best_weights:
					self.best_weights = copy.deepcopy(model.state_dict())
			return False
		
		if self.is_improvement(current_value):
			self.best_score = current_value
			self.stopped_epoch = epoch  # Update stopped_epoch when a new best score is achieved
			if self.restore_best_weights:
				self.best_weights = copy.deepcopy(model.state_dict())
			self.counter = 0
			self.improvement_history.append(True)
		else:
			self.counter += 1
			self.improvement_history.append(False)
		
		# Calculate trend over window
		trend = self.calculate_trend()
		cumulative_improvement = abs(trend) if len(self.value_history) >= self.window_size else float('inf')
		
		# Decision logic combining multiple factors
		should_stop = False
		
		# Check primary patience criterion
		if self.counter >= self.patience:
			should_stop = True
		
		# Check if stuck in local optimum
		if len(self.improvement_history) >= self.window_size:
			recent_improvements = sum(self.improvement_history[-self.window_size:])
			if recent_improvements == 0 and cumulative_improvement < self.cumulative_delta:
				should_stop = True
		
		# If stopping, restore best weights if configured
		if should_stop and self.restore_best_weights and self.best_weights is not None:
			model.load_state_dict(self.best_weights)
			self.stopped_epoch = epoch
		
		return should_stop
	
	def get_best_score(self) -> float:
		return self.best_score
	
	def get_stopped_epoch(self) -> int:
		return self.stopped_epoch

def get_dataset(dname:str="CIFAR10"):
	dname = dname.upper()
	ddir = {
		"farid": f'/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/{dname}',
		"ubuntu": f'/media/volume/ImACCESS/WW_DATASETs/{dname}',
		"alijanif": f'/scratch/project_2004072/ImACCESS/WW_DATASETs/{dname}',
	}
	if dname == 'CIFAR100':
		train_dataset = CIFAR100(
			root=os.path.expanduser("~/.cache"), 
			train=True,
			download=True,
			transform=None
		)
		validation_dataset = CIFAR100(
			root=os.path.expanduser("~/.cache"), 
			train=False,
			download=True,
			transform=None
		)
	elif dname == 'CIFAR10':
		train_dataset = CIFAR10(
			root=os.path.expanduser("~/.cache"), 
			train=True,
			download=True,
			transform=None,
		)
		validation_dataset = CIFAR10(
			root=os.path.expanduser("~/.cache"), 
			train=False,
			download=True,
			transform=None,
		)
	elif dname == 'IMAGENET':
		train_dataset = ImageNet(
			root=ddir.get(USER),
			train=True,
			transform=None
		)
		validation_dataset = ImageNet(
			root=ddir.get(USER),
			train=False,
			transform=None
	)	
	elif dname == 'CINIC10':
		train_dataset = CINIC10(
			root=ddir.get(USER),
			train=True,
			download=True,
			transform=None
		)
		validation_dataset = CINIC10(
			root=ddir.get(USER),
			train=False,
			download=True,
			transform=None
		)
	else:
		raise ValueError(f"Invalid dataset name: {dname}. Available: [CIFAR10, cifar100, IMAGENET, CINIC10]")
	print(train_dataset)
	print(validation_dataset)
	return train_dataset, validation_dataset

def get_dataloaders(train_dataset, valid_dataset, preprocess, batch_size=32, nw=10):
	trainset = CUSTOMIZEDDATASET(
		dataset=train_dataset, 
		transformer=preprocess,
	)
	validset = CUSTOMIZEDDATASET(
		dataset=valid_dataset, 
		transformer=preprocess,
	)
	
	train_loader = DataLoader(
		dataset=trainset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=nw,
		pin_memory=True, # Move data to GPU faster if using CUDA
		persistent_workers=True if nw > 1 else False,  # Keep workers alive if memory allows
	)
	validation_loader = DataLoader(
		dataset=validset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=nw,
		pin_memory=True, # when using CUDA
	)
	return train_loader, validation_loader

def load_model(model_name:str="ViT-B/32", device:str="cuda", jit:bool=False):
	model, preprocess = clip.load(model_name, device=device, jit=jit) # training or finetuning => jit=False
	model = model.float() # Convert model parameters to FP32
	input_resolution = model.visual.input_resolution
	context_length = model.context_length
	vocab_size = model.vocab_size
	print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
	print("Input resolution:", input_resolution)
	print("Context length:", context_length)
	print("Vocab size:", vocab_size)
	return model, preprocess

def evaluate(
	model,
	validation_loader,
	criterion,
	device="cuda",
	topK_values=[1, 3, 5],
	):
	model.eval()
	total_loss = 0
	img2txt_correct = 0
	txt2img_correct = 0
	total_samples = len(validation_loader.dataset)
	img2txt_topk_accuracy = {k: 0 for k in topK_values}
	reciprocal_ranks = []
	cosine_similarities = []
	precision_list, recall_list, f1_list = [], [], []
	with torch.no_grad():
		for bidx, (images, labels) in enumerate(validation_loader):
			images, labels = images.to(device), labels.to(device)
			batch_size = images.size(0)

			# Forward pass:
			logits_per_image, logits_per_text = model(images, labels) # [batch_size, batch_size]
			print(f"logits_per_image: {logits_per_image.shape}")
			print(f"logits_per_text: {logits_per_text.shape}")
			# Ground Truth:
			correct_labels = torch.arange(start=0, end=batch_size, dtype=torch.long, device=device)

			# Validation Loss: Average of both losses
			loss_img = criterion(logits_per_image, correct_labels)
			loss_txt = criterion(logits_per_text, correct_labels)
			total_loss += 0.5 * (loss_img.item() + loss_txt.item())

			# Predictions:
			pred_lbl_per_img_idxs = torch.argmax(input=logits_per_image, dim=1) # [batch_size x 1]
			pred_img_per_lbl_idxs = torch.argmax(input=logits_per_text, dim=1) # [batch_size x 1]
			# Metrics
			img2txt_correct += (pred_lbl_per_img_idxs == correct_labels).sum().item()
			txt2img_correct += (pred_img_per_lbl_idxs == correct_labels).sum().item()

			# Top-k Accuracy
			for k in topK_values:
				effective_k = min(k, batch_size)
				# print(f"Top-{k} Accuracy => effective_k: {effective_k}")
				# print(f"logits_per_image: {logits_per_image.shape}")
				# print(f"correct_labels: {correct_labels.shape}")
				# print(f"correct_labels.unsqueeze(1): {correct_labels.unsqueeze(1).shape}")
				topk_predicted_labels_values, topk_predicted_labels_idxs = torch.topk(input=logits_per_image, k=effective_k, dim=1) # values, indices
				# print(f"topk_predicted_labels_values: {topk_predicted_labels_values.shape}")
				# print(f"topk_predicted_labels_idxs: {topk_predicted_labels_idxs.shape}")
				# print(f"topk_predicted_labels_idxs == correct_labels.unsqueeze(1): {(topk_predicted_labels_idxs == correct_labels.unsqueeze(1)).shape}")
				# print(f"(topk_predicted_labels_idxs == correct_labels.unsqueeze(1)).any(dim=1): {(topk_predicted_labels_idxs == correct_labels.unsqueeze(1)).any(dim=1).shape}")
				# print(f"(topk_predicted_labels_idxs == correct_labels.unsqueeze(1)).any(dim=1).sum().item(): {(topk_predicted_labels_idxs == correct_labels.unsqueeze(1)).any(dim=1).sum().item()}")
				img2txt_topk_accuracy[k] += (topk_predicted_labels_idxs == correct_labels.unsqueeze(1)).any(dim=1).sum().item()

			# Reciprocal Rank
			for i in range(batch_size):
				ranks = torch.argsort(logits_per_image[i], descending=True)
				rank_of_true_label = (ranks == correct_labels[i]).nonzero(as_tuple=True)[0].item() + 1
				reciprocal_ranks.append(1 / rank_of_true_label)

			# Cosine Similarity
			cos_sim = torch.nn.functional.cosine_similarity(logits_per_image, logits_per_text, dim=1).cpu().numpy()
			cosine_similarities.extend(cos_sim)

			# Precision, Recall, F1
			precision = (pred_lbl_per_img_idxs == correct_labels).sum().item() / pred_lbl_per_img_idxs.size(0)
			recall = (pred_img_per_lbl_idxs == correct_labels).sum().item() / correct_labels.size(0)
			f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
			precision_list.append(precision)
			recall_list.append(recall)
			f1_list.append(f1)

	# Compute average metrics
	print(f"Total Samples: {total_samples} | validation_loader: {len(validation_loader)} | batch_size: {batch_size} | validation_loader * batch_size: {len(validation_loader) * batch_size}")
	print(f"val_loader contains {len(validation_loader.dataset)} samples")
	avg_loss = total_loss / len(validation_loader)
	img2txt_acc = img2txt_correct / total_samples
	txt2img_acc = txt2img_correct / total_samples
	img2txt_topk_accuracy = {k: v / total_samples for k, v in img2txt_topk_accuracy.items()}
	mean_reciprocal_rank = np.mean(reciprocal_ranks) # sum(reciprocal_ranks) / len(reciprocal_ranks)
	cosine_sim_mean = np.mean(cosine_similarities)
	avg_precision = np.mean(precision_list)
	avg_recall = np.mean(recall_list)
	avg_f1 = np.mean(f1_list)
	return (
		avg_loss,
		img2txt_acc,
		txt2img_acc,
		img2txt_topk_accuracy,
		mean_reciprocal_rank,
		cosine_sim_mean,
		avg_precision,
		avg_recall,
		avg_f1,
	)

def plot_loss_accuracy(
		train_losses,
		val_losses,
		val_acc_img2txt_list,
		val_acc_txt2img_list,
		img2txt_topk_accuracy_list,
		mean_reciprocal_rank_list,
		cosine_similarity_list,
		precision_list,
		recall_list,
		f1_list,
		losses_file_path="losses.png",
		accuracy_file_path="accuracy.png",
		topk_accuracy_file_path="img2txt_topk_accuracy.png",
		mean_reciprocal_rank_file_path="mean_reciprocal_rank.png",
		cosine_similarity_file_path="cosine_similarity.png",
		precision_recall_f1_file_path="precision_recall_f1.png",
		DPI=250,
		figure_size=(10, 5),
		TOP_K_VALUES=[1, 3, 5],
	):
	num_epochs = len(train_losses)
	if num_epochs == 1:
		return
	epochs = range(1, num_epochs + 1)

	# Set xticks to be dynamically defined
	num_xticks = 10
	# Check if num_xticks is greater than num_epochs + 1
	if num_xticks > num_epochs + 1:
		num_xticks = num_epochs + 1
	xticks = np.arange(0, num_epochs + 1, (num_epochs + 1) // num_xticks)

	# Plot losses:
	plt.figure(figsize=figure_size)
	plt.plot(epochs, train_losses, color='b', label='Train', lw=1.25)
	plt.plot(epochs, val_losses, color='r', label='Validation', lw=1.25)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(os.path.splitext(os.path.basename(losses_file_path))[0], fontsize=10)
	plt.legend(ncols=2, fontsize=10, loc='best')
	plt.grid(True)
	plt.xlim(0, num_epochs + 1)
	plt.xticks(xticks, fontsize=7)
	plt.tight_layout()
	plt.savefig(losses_file_path, dpi=DPI, bbox_inches='tight')
	plt.close()

	plt.figure(figsize=figure_size)
	plt.plot(epochs, val_acc_img2txt_list, label='text retrieval per image')
	plt.plot(epochs, val_acc_txt2img_list, label='image retrieval per text')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title(os.path.splitext(os.path.basename(accuracy_file_path))[0], fontsize=10)
	plt.legend(title='[Top-1] Accuracy (Zero-Shot)', fontsize=8, title_fontsize=9, loc='best')
	plt.grid(True)
	plt.xlim(0, num_epochs + 1)
	plt.xticks(xticks, fontsize=7)
	plt.tight_layout()
	plt.savefig(accuracy_file_path, dpi=DPI, bbox_inches='tight')
	plt.close()
	
	plt.figure(figsize=figure_size)
	for k, acc in zip(TOP_K_VALUES, zip(*img2txt_topk_accuracy_list)):
		plt.plot(epochs, acc, label=f'Top-{k}')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title("Image-to-Text Top-k Accuracy")
	plt.legend(ncols=len(TOP_K_VALUES), loc='best')
	plt.grid(True)
	plt.tight_layout()
	plt.xlim(0, num_epochs + 1)
	plt.xticks(xticks, fontsize=7)
	plt.savefig(topk_accuracy_file_path, dpi=DPI, bbox_inches='tight')
	plt.close()
	
	plt.figure(figsize=figure_size)
	plt.plot(epochs, mean_reciprocal_rank_list,  label='Mean Reciprocal Rank')
	plt.xlabel('Epoch')
	plt.ylabel('MRR')
	plt.title("Mean Reciprocal Rank")
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.xlim(0, num_epochs + 1)
	plt.xticks(xticks, fontsize=7)
	plt.savefig(mean_reciprocal_rank_file_path, dpi=DPI, bbox_inches='tight')
	plt.close()
	
	plt.figure(figsize=figure_size)
	plt.plot(epochs, precision_list,  label='Precision')
	plt.plot(epochs, recall_list, label='Recall')
	plt.plot(epochs, f1_list, label='F1 Score')
	plt.xlabel('Epoch')
	plt.ylabel('Score')
	plt.title("Precision, Recall, and F1 Score")
	plt.grid(True)
	plt.legend(ncols=3, loc='best')
	plt.tight_layout()
	plt.xlim(0, num_epochs + 1)
	plt.xticks(xticks, fontsize=7)
	plt.savefig(precision_recall_f1_file_path, dpi=DPI, bbox_inches='tight')
	plt.close()
	
	plt.figure(figsize=figure_size)
	plt.plot(epochs, cosine_similarity_list,  linestyle='-', color='g', label='Cosine Similarity')
	plt.xlabel('Epoch')
	plt.ylabel('Cosine Similarity')
	plt.title("Cosine Similarity Over Epochs", fontsize=10)
	plt.grid(True)
	plt.tight_layout()
	plt.legend()
	plt.xlim(0, num_epochs + 1)
	plt.xticks(xticks, fontsize=7)
	plt.savefig(cosine_similarity_file_path, dpi=DPI, bbox_inches='tight')
	plt.close()

def count_clip_layers(model):
		"""
		Count total number of layers in CLIP model
		"""
		total_layers = 0
		unique_layers = set()
		
		# Count each named parameter's layer (get base layer name without .weight/.bias)
		for name, _ in model.named_parameters():
				# Split the name and take everything except the last part (weight/bias)
				layer_name = '.'.join(name.split('.')[:-1])
				if layer_name:  # Avoid empty strings
						unique_layers.add(layer_name)
		
		total_layers = len(unique_layers)
		
		# Detailed breakdown
		visual_transformer_blocks = len([l for l in unique_layers if 'visual.transformer.resblocks' in l])
		text_transformer_blocks = len([l for l in unique_layers if 'transformer.resblocks' in l])
		projection_layers = len([l for l in unique_layers if any(x in l for x in ['visual.proj', 'text_projection', 'visual.ln_post'])])
		frontend_layers = len([l for l in unique_layers if any(x in l for x in ['visual.conv1', 'visual.class_embedding', 'positional_embedding', 'token_embedding'])])
		
		print(f"\nCLIP Layer Statistics:")
		print(f"Total unique layers: {total_layers}")
		print(f"Visual transformer blocks: {visual_transformer_blocks}")
		print(f"Text transformer blocks: {text_transformer_blocks}")
		print(f"Projection layers: {projection_layers}")
		print(f"Frontend layers: {frontend_layers}")
		return total_layers

def get_status(model, current_phase=0, layers_to_freeze=[], total_layers=0):
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)    
	total_params = sum(p.numel() for p in model.parameters())
	trainable_percent = (trainable_params / total_params) * 100
	frozen_percent = (frozen_params / total_params) * 100
	freeze_percentage = (len(layers_to_freeze) / total_layers) * 100 if total_layers > 0 else 0
	unfreeze_percentage = 100 - freeze_percentage
	print("\n" + "="*60)
	print(f"Model Status - Phase {current_phase}".center(60))
	print("="*60)
	print("\nParameter Statistics:")
	print(f"{'Total:':<25} {total_params:,}")
	print(f"{'Trainable:':<25} {trainable_params:,} ({trainable_percent:.2f}%)")
	print(f"{'Frozen:':<25} {frozen_params:,} ({frozen_percent:.2f}%)")
	print("\nLayer Statistics:")
	print(f"{'Total:':<25} {total_layers}")
	print(f"{'Frozen:':<25} {len(layers_to_freeze)} ({freeze_percentage:.1f}%)")
	print(f"{'Trainable:':<25} {total_layers - len(layers_to_freeze)} ({unfreeze_percentage:.1f}%)")
	print("\n" + "="*60 + "\n")

def get_num_vit_blocks(model):
	if not hasattr(model, 'visual') or not hasattr(model.visual, 'transformer'):
		raise ValueError("Model structure not compatible - missing visual transformer")
	vis_transformer = model.visual.transformer
	txt_transformer = model.transformer
	return len(vis_transformer.resblocks), len(txt_transformer.resblocks)

def get_layer_groups(nv:int=12, nt:int=12):
	layer_groups = {
		'visual_frontend': [
			'visual.conv1', # patch embedding
			'visual.class_embedding', # CLS token
			'visual.positional_embedding', # positional embedding
			# 'visual.ln_pre' # 
		],
		'visual_transformer': [f'visual.transformer.resblocks.{i}' for i in range(nv)],
		'text_frontend': ['token_embedding','positional_embedding'],
		'text_transformer': [f'transformer.resblocks.{i}' for i in range(nt)],
		'projections': [
			'visual.proj', # final normalization before projection
			'visual.ln_post',
			'text_projection',
			'logit_scale', # Temperature parameter
		],
	}
	return layer_groups

def get_progressive_freeze_schedule(layer_groups:dict):
	total_v_layers = len(layer_groups['visual_transformer'])
	total_t_layers = len(layer_groups['text_transformer'])
	print(f"Total visual layers: {total_v_layers} | 80%: {int(0.8*total_v_layers)} 60%: {int(0.6*total_v_layers)} 40%: {int(0.4*total_v_layers)}")
	print(f"Total text layers: {total_t_layers} | 80%: {int(0.8*total_t_layers)} 60%: {int(0.6*total_t_layers)} 40%: {int(0.4*total_t_layers)}")
	schedule = [
		# Phase 0: Freeze all layers except the projection layers:
		layer_groups['visual_frontend'] + layer_groups['visual_transformer'] + layer_groups['text_frontend'] + layer_groups['text_transformer'],
		# Phase 1: Freeze 80% of transformer blocks:
		layer_groups['visual_frontend'] + layer_groups['visual_transformer'][:int(0.8*total_v_layers)] + layer_groups['text_frontend'] + layer_groups['text_transformer'][:int(0.8*total_t_layers)],
		# Phase 2: freeze 60% of transformer blocks:
		layer_groups['visual_frontend'] + layer_groups['visual_transformer'][:int(0.6*total_v_layers)] + layer_groups['text_frontend'] + layer_groups['text_transformer'][:int(0.6*total_t_layers)],
		# Phase 3: freeze 40% of transformer blocks:
		layer_groups['visual_frontend'] + layer_groups['visual_transformer'][:int(0.4*total_v_layers)] + layer_groups['text_frontend'] + layer_groups['text_transformer'][:int(0.4*total_t_layers)],
		# Phase 4: freeze only (visual + text) frontends
		layer_groups['visual_frontend'] + layer_groups['text_frontend']
	]
	return schedule

def freeze_(layers, model):
	for name, param in model.named_parameters():
		param.requires_grad = True # Unfreeze all layers first
		if any(ly in name for ly in layers): # Freeze layers in the list
			param.requires_grad = False

def should_transition_phase(losses:List[float], th: float=1e-4, window:int=3) -> bool:
	if len(losses) < window:
		return False # Not enough data to make a decision
	last_window_losses = losses[-window:]
	avg_loss = sum(last_window_losses) / window
	relative_change = abs(last_window_losses[-1] - avg_loss) / avg_loss # Relative change in loss
	transition_required: bool = relative_change < th
	return transition_required

def handle_phase_transition(current_phase, initial_lr, max_phases):
	if current_phase >= max_phases - 1:
		return current_phase, initial_lr * (0.1 ** current_phase)
	new_phase = current_phase + 1
	new_lr = initial_lr * (0.1 ** new_phase) # Reduce learning rate by 10x
	print(f"<!> Plateau detected! Transitioning to Phase {new_phase} with learning rate {new_lr:.1e}")
	return new_phase, new_lr

def finetune(
		model:nn.Module,
		train_loader:DataLoader,
		validation_loader:DataLoader,
		num_epochs:int=7,
		nw:int=10,
		print_every:int=150,
		model_name:str="ViT-B/32",
		learning_rate:float=1e-5,
		weight_decay:float=1e-3,
		dataset_name:str="CIFAR10",
		device:str="cuda",
		results_dir:str="results",
		window_size:int=10,
		patience:int=10,
		min_delta:float=1e-4,
		cumulative_delta:float=5e-3,
		minimum_epochs:int=20,
		TOP_K_VALUES=[1, 5, 10, 15, 20],
	):
	early_stopping = EarlyStopping(
		patience=patience,									# Wait for 10 epochs without improvement before stopping
		min_delta=min_delta,								# Consider an improvement only if the change is greater than 0.0001
		cumulative_delta=cumulative_delta,	# Cumulative improvement over the window should be greater than 0.005
		window_size=window_size,						# Consider the last 10 epochs for cumulative trend
		mode='min',													# Minimize loss
		min_epochs=minimum_epochs,					# Ensure at least 20 epochs of training
		restore_best_weights=True						# Restore model weights to the best epoch
	)
	os.makedirs(results_dir, exist_ok=True)
	mode = "finetune"
	print(f"{mode} CLIP {model_name} « {dataset_name} » {num_epochs} Epoch(s) {device} [x{nw} cores]".center(160, "-"))
	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))

	total_layers = count_clip_layers(model)
	vis_nblocks, txt_nblocks = get_num_vit_blocks(model)
	print(f"[Transformer Blocks] Vision: {vis_nblocks} | Text: {txt_nblocks}")
	layer_groups = get_layer_groups(nv=vis_nblocks, nt=txt_nblocks,)
	total_v_layers = len(layer_groups['visual_transformer'])
	total_t_layers = len(layer_groups['text_transformer'])
	print(f"[Layer Groups] Visual: {total_v_layers} | Text: {total_t_layers}")
	freeze_schedule = get_progressive_freeze_schedule(layer_groups) # progressive freezing based on validation loss plateau
	print(f"Freeze Schedule[{len(freeze_schedule)}]:\n{json.dumps(freeze_schedule, indent=2)}")
	mdl_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_clip.pth"
	)
	criterion = nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)
	training_losses, val_losses = [], []
	val_acc_img2txt_list = []
	val_acc_txt2img_list = []
	img2txt_topk_accuracy_list = []
	mean_reciprocal_rank_list = []
	cosine_similarity_list = []
	precision_list, recall_list, f1_list = [], [], []
	current_phase = 0
	plateau_threshold = min_delta # ensure parameter consistency
	initial_learning_rate = learning_rate # Store the initial value
	ft_st = time.time()
	for epoch in range(num_epochs):
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		# Adaptive Progressive Layer Freezing Schedule:
		if len(val_losses) > 1: # 2 epochs needed to compare
			should_transition = should_transition_phase(
				losses=val_losses,
				th=plateau_threshold,
				window=window_size,
			)
			if should_transition:
				print(f"Plateau detected @ Epoch: {epoch+1} Transitioning from phase: {current_phase} to next phase.")
				current_phase, learning_rate = handle_phase_transition(
					current_phase=current_phase,
					initial_lr=initial_learning_rate,
					max_phases=len(freeze_schedule)
				)
		layers_to_freeze = freeze_schedule[current_phase]
		freeze_(layers=layers_to_freeze, model=model)
		get_status(model, current_phase, layers_to_freeze, total_layers)
		# optimizer = AdamW(
		optimizer = AdamW(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate, # potentially update learning rate based on phase
			betas=(0.9, 0.98),
			eps=1e-8,
			weight_decay=weight_decay,
		)
		scheduler = lr_scheduler.OneCycleLR(
			optimizer=optimizer,
			max_lr=learning_rate,
			steps_per_epoch=len(train_loader),
			epochs=num_epochs - epoch,  # Adjust for remaining epochs
			pct_start=0.1,
			anneal_strategy='cos',
		)
		epoch_loss = 0.0
		for bidx, (images, labels) in enumerate(train_loader):
			optimizer.zero_grad() # Clear gradients from previous batch
			images, labels = images.to(device), labels.to(device) # torch.Size([b, 3, 224, 224]), torch.Size([b, 77])
			with torch.amp.autocast(device_type=device.type): # # Automatic Mixed Precision (AMP) backpropagation:
				logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)
			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # stabilize training if exploding gradients
			scaler.step(optimizer)
			scaler.update()
			scheduler.step() # Update learning rate
			if bidx%print_every==0 or bidx+1==len(train_loader):
				print(
					f"\t\tBatch [{bidx+1}/{len(train_loader)}] "
					f"Loss: {total_loss.item():.7f}",
				)
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		# print(f"Average {mode} Loss: {avg_training_loss:.7f} ")
		training_losses.append(avg_training_loss)
		(
			avg_valid_loss,
			img2txt_val_acc,
			txt2img_val_acc,
			img2txt_topk_accuracy,
			mean_reciprocal_rank, 
			cosine_sim_mean, 
			avg_precision, 
			avg_recall, 
			avg_f1
		) = evaluate(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion, 
			device=device,
			topK_values=TOP_K_VALUES,
		)
		val_losses.append(avg_valid_loss)
		val_acc_img2txt_list.append(img2txt_val_acc)
		val_acc_txt2img_list.append(txt2img_val_acc)
		img2txt_topk_accuracy_list.append([img2txt_topk_accuracy[k] for k in TOP_K_VALUES])
		mean_reciprocal_rank_list.append(mean_reciprocal_rank)
		cosine_similarity_list.append(cosine_sim_mean)
		precision_list.append(avg_precision)
		recall_list.append(avg_recall)
		f1_list.append(avg_f1)
		print(
			f'@ Epoch: {epoch+1}\n'
			f'\t[Loss] {mode}: {avg_training_loss:.7f} Valid: {avg_valid_loss:.9f}\n'
			f'\tValid Acc [text retrieval per image]: {img2txt_val_acc} '
			f'[image retrieval per text]: {txt2img_val_acc}'
		)
		# ############################## Early stopping ##############################
		if early_stopping.should_stop(avg_valid_loss, model, epoch):
			print(
				f'\nEarly stopping triggered at epoch {epoch+1}\t'
				f'Best validation loss: {early_stopping.get_best_score():.5f} @ Epoch {early_stopping.get_stopped_epoch()+1}\n'
			)
			break
		else:
			print(f"Saving best model in {mdl_fpth} for best validation loss: {avg_valid_loss:.9f}")
			torch.save(model.state_dict(), mdl_fpth)
		# ############################## Early stopping ##############################
		print("-"*170)
	print(f"{mode} Elapsed_t: {time.time()-ft_st:.1f} sec".center(160, " "))

	losses_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_losses_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	val_acc_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_val_acc_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	topk_acc_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_top_k_acc_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	mrr_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_mrr_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	cs_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_cs_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	pr_f1_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_pr_f1_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")

	plot_loss_accuracy(
		train_losses=training_losses,
		val_losses=val_losses,
		val_acc_img2txt_list=val_acc_img2txt_list,
		val_acc_txt2img_list=val_acc_txt2img_list,
		img2txt_topk_accuracy_list=img2txt_topk_accuracy_list,
		mean_reciprocal_rank_list=mean_reciprocal_rank_list,
		cosine_similarity_list=cosine_similarity_list,
		precision_list=precision_list,
		recall_list=recall_list,
		f1_list=f1_list,
		losses_file_path=losses_fpth,
		accuracy_file_path=val_acc_fpth,
		topk_accuracy_file_path=topk_acc_fpth,
		mean_reciprocal_rank_file_path=mrr_fpth,
		cosine_similarity_file_path=cs_fpth,
		precision_recall_f1_file_path=pr_f1_fpth,
		TOP_K_VALUES=TOP_K_VALUES,
	)

def train(
		model:nn.Module,
		train_loader:DataLoader,
		validation_loader:DataLoader,
		num_epochs:int=5,
		nw:int=10,
		print_every:int=150,
		model_name:str="ViT-B/32",
		learning_rate:float=1e-5,
		weight_decay:float=1e-3,
		dataset_name:str="CIFAR10",
		device:str="cuda",
		results_dir:str="results",
		window_size:int=10,
		patience:int=10,
		min_delta:float=1e-4,
		cumulative_delta:float=5e-3,
		minimum_epochs:int=20,
		TOP_K_VALUES=[1, 5, 10, 15, 20],
	):
	early_stopping = EarlyStopping(
		patience=patience,												# Wait for 10 epochs without improvement before stopping
		min_delta=min_delta,											# Consider an improvement only if the change is greater than 0.0001
		cumulative_delta=cumulative_delta,				# Cumulative improvement over the window should be greater than 0.005
		window_size=window_size,									# Consider the last 10 epochs for cumulative trend
		mode='min',																# Minimize loss
		min_epochs=minimum_epochs,			# Ensure at least 20 epochs of training
		restore_best_weights=True									# Restore model weights to the best epoch
	)
	os.makedirs(results_dir, exist_ok=True)
	mode = "train"
	print(f"{mode} CLIP {model_name} « {dataset_name} » {num_epochs} Epoch(s) {device} [x{nw} cores]".center(160, "-"))
	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))

	for name, param in model.named_parameters():
		# print(f"{name} requires_grad: {param.requires_grad}")
		param.requires_grad = True # Unfreeze all layers (train from scratch)

	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)	
	total_params = sum(p.numel() for p in model.parameters())
	trainable_percent = (trainable_params / total_params) * 100
	frozen_percent = (frozen_params / total_params) * 100
	print(
		f"[Model Parameters Statictics] Total: {total_params:,} "
		f"Trainable: {trainable_params:,} ({trainable_percent:.2f}%) "
		f"Frozen: {frozen_params:,} ({frozen_percent:.2f}%)"
		.center(160, "-")
	)
	mdl_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_clip.pth"
	)
	# optimizer = AdamW(
	optimizer = AdamW(
		params=[p for p in model.parameters() if p.requires_grad],# Only optimizes parameters that require gradients
		lr=learning_rate,
		betas=(0.9,0.98),
		eps=1e-8,
		weight_decay=weight_decay,
	)
	scheduler = lr_scheduler.OneCycleLR(
		optimizer=optimizer, 
		max_lr=learning_rate, 
		steps_per_epoch=len(train_loader), 
		epochs=num_epochs,
		pct_start=0.1, # percentage of the cycle (in number of steps) spent increasing the learning rate
		anneal_strategy='cos', # cos/linear annealing
	)
	criterion = nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)
	training_losses, val_losses = [], []
	val_acc_img2txt_list = []
	val_acc_txt2img_list = []
	img2txt_topk_accuracy_list = []
	mean_reciprocal_rank_list = []
	cosine_similarity_list = []
	precision_list, recall_list, f1_list = [], [], []
	ft_st = time.time()
	print(torch.cuda.memory_summary(device=device))
	for epoch in range(num_epochs):
		torch.cuda.empty_cache() # Clear GPU memory cache
		model.train()
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, labels) in enumerate(train_loader):
			optimizer.zero_grad() # Clear gradients from previous batch
			images, labels = images.to(device), labels.to(device) # torch.Size([b, 3, 224, 224]), torch.Size([b, 77])
			with torch.amp.autocast(device_type=device.type): # # Automatic Mixed Precision (AMP) backpropagation:
				logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)
			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # stabilize training if exploding gradients
			scaler.step(optimizer)
			scaler.update()
			scheduler.step() # Update learning rate
			if bidx%print_every==0 or bidx+1==len(train_loader):
				print(
					f"\t\tBatch [{bidx+1}/{len(train_loader)}] Loss: {total_loss.item():.7f}",
				)
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)
		(
			avg_valid_loss,
			img2txt_val_acc,
			txt2img_val_acc,
			img2txt_topk_accuracy,
			mean_reciprocal_rank,
			cosine_sim_mean,
			avg_precision,
			avg_recall,
			avg_f1
		) = evaluate(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=TOP_K_VALUES,
		)
		torch.cuda.empty_cache() # free up GPU memory
		val_losses.append(avg_valid_loss)
		val_acc_img2txt_list.append(img2txt_val_acc)
		val_acc_txt2img_list.append(txt2img_val_acc)
		img2txt_topk_accuracy_list.append([img2txt_topk_accuracy[k] for k in TOP_K_VALUES])
		mean_reciprocal_rank_list.append(mean_reciprocal_rank)
		cosine_similarity_list.append(cosine_sim_mean)
		precision_list.append(avg_precision)
		recall_list.append(avg_recall)
		f1_list.append(avg_f1)
		print(
			f'@ Epoch {epoch+1}:\n'
			f'\t[LOSS] {mode}: {avg_training_loss:.5f} Valid: {avg_valid_loss:.8f}\n'
			f'\tValid Acc [text retrieval per image]: {img2txt_val_acc} '
			f'[image retrieval per text]: {txt2img_val_acc}'
		)
		# ############################## Early stopping ##############################
		if early_stopping.should_stop(avg_valid_loss, model, epoch):
			print(
				f'\nEarly stopping triggered at epoch {epoch+1}\t'
				f'Best validation loss: {early_stopping.get_best_score():.5f} @ Epoch {early_stopping.get_stopped_epoch()+1}\n'
			)
			break
		else:
			print(f"Saving best model in {mdl_fpth} for best validation loss: {avg_valid_loss:.9f}")
			torch.save(model.state_dict(), mdl_fpth)
		# ############################## Early stopping ##############################
		print("-"*170)

	print(f"Elapsed_t: {time.time()-ft_st:.1f} sec".center(150, "-"))

	losses_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_losses_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	val_acc_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_acc_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	topk_acc_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_top_k_acc_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	mrr_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_mrr_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	cs_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_cs_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	pr_f1_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_prf1_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	plot_loss_accuracy(
		train_losses=training_losses,
		val_losses=val_losses,
		val_acc_img2txt_list=val_acc_img2txt_list,
		val_acc_txt2img_list=val_acc_txt2img_list,
		img2txt_topk_accuracy_list=img2txt_topk_accuracy_list,
		mean_reciprocal_rank_list=mean_reciprocal_rank_list,
		cosine_similarity_list=cosine_similarity_list,
		precision_list=precision_list,
		recall_list=recall_list,
		f1_list=f1_list,
		losses_file_path=losses_fpth,
		accuracy_file_path=val_acc_fpth,
		topk_accuracy_file_path=topk_acc_fpth,
		mean_reciprocal_rank_file_path=mrr_fpth,
		cosine_similarity_file_path=cs_fpth,
		precision_recall_f1_file_path=pr_f1_fpth,
		TOP_K_VALUES=TOP_K_VALUES,
	)

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Balanced Dataset")
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=10, help='Number of CPUs')
	parser.add_argument('--epochs', '-e', type=int, default=12, help='Number of epochs')
	parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='small learning rate for better convergence [def: 1e-4]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3, help='Weight decay [def: 1e-3]')
	parser.add_argument('--print_every', type=int, default=150, help='Print loss')
	parser.add_argument('--model_name', '-md', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--dataset', '-d', type=str, choices=['cifar10', 'cifar100', 'cinic10', 'imagenet'], default='cifar100', help='Choose dataset (CIFAR10/cifar100)')
	parser.add_argument('--mode', '-m', type=str, choices=['train', 'finetune'], default='finetune', help='Choose mode (train/finetune)')
	parser.add_argument('--window_size', '-ws', type=int, default=5, help='Windows size for early stopping and progressive freezing')
	parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
	parser.add_argument('--minimum_delta', '-mdelta', type=float, default=1e-4, help='Min delta for early stopping & progressive freezing [Platueau threshhold]')
	parser.add_argument('--cumulative_delta', '-cdelta', type=float, default=5e-3, help='Cumulative delta for early stopping')
	parser.add_argument('--minimum_epochs', type=int, default=20, help='Early stopping minimum epochs')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print(args)
	set_seeds()
	print(clip.available_models())
	model, preprocess = load_model(
		model_name=args.model_name,
		device=args.device,
		jit=False,
	)
	train_dataset, validation_dataset = get_dataset(dname=args.dataset)
	train_loader, validation_loader = get_dataloaders(
		train_dataset=train_dataset, 
		valid_dataset=validation_dataset, 
		preprocess=preprocess,
		batch_size=args.batch_size,
		nw=args.num_workers,
	)
	print(f"Train Loader: {len(train_loader)} batches, Validation Loader: {len(validation_loader)} batches")
	# visualize_(dataloader=train_loader, num_samples=5)
	if args.mode == 'finetune':
		finetune(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			model_name=args.model_name,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			dataset_name=args.dataset,
			device=args.device,
			results_dir=os.path.join(args.dataset, "results"),
			window_size=args.window_size, 	# early stopping & progressive unfreezing
			patience=10, 										# early stopping
			min_delta=1e-4, 								# early stopping & progressive unfreezing
			cumulative_delta=5e-3, 					# early stopping
			minimum_epochs=20, 							# early stopping
		)
	elif args.mode == 'train':
		train(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			model_name=args.model_name,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			dataset_name=args.dataset,
			device=args.device,
			results_dir=os.path.join(args.dataset, "results"),
			window_size=args.window_size, # early stopping
			patience=10,									# early stopping
			min_delta=1e-4,								# early stopping
			cumulative_delta=5e-3,				# early stopping
			minimum_epochs=20,						# early stopping
		)
	else:
		raise ValueError("Invalid mode. Choose either 'finetune' or 'train'.")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))