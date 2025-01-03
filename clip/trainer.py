from utils import *
from datasets import *
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import AdamW, SGD, Adam, lr_scheduler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.optim.lr_scheduler')

# run in pouta:
# train cifar100 from scratch:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -ne 128 -lr 5e-4 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:3" -m "train" -md "ViT-B/32" > /media/volume/ImACCESS/trash/cifar100_train.out &

# finetune cifar100:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -ne 128 -lr 5e-4 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:0" -m "finetune" -md "ViT-B/32" > /media/volume/ImACCESS/trash/cifar100_ft.out &

# train imagenet from scratch:
# $ nohup python -u trainer.py -d imagenet -bs 256 -ne 50 -lr 5e-4 -wd 1e-2 --print_every 5000 -nw 50 --device "cuda:1" -m "train" -md "ViT-B/32" > /media/volume/ImACCESS/trash/imagenet_train.out &

USER = os.environ.get('USER')

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

def evaluate(model, validation_loader, criterion, device="cuda", top_k=(1, 3, 5)):
	model.eval()
	total_loss = 0
	correct_text_description = 0
	correct_image_for_text = 0
	total_samples = 0
	top_k_accuracy = {k: 0 for k in top_k}
	reciprocal_ranks = []
	cosine_similarities = []
	precision_list, recall_list, f1_list = [], [], []
	with torch.no_grad():
		for bidx, (images, labels) in enumerate(validation_loader):
			images, labels = images.to(device), labels.to(device)
			batch_size = images.size(0)
			total_samples += batch_size
			logits_per_image, logits_per_text = model(images, labels) # Output sizes: [batch_size, batch_size]
			# Predictions and Ground Truth
			predicted_text_idxs = torch.argmax(input=logits_per_image, dim=1)
			predicted_image_idxs = torch.argmax(input=logits_per_text, dim=1)
			correct_labels = torch.arange(start=0, end=batch_size, dtype=torch.long, device=device)
			# Metrics
			correct_text_description += (predicted_text_idxs == correct_labels).sum().item()
			correct_image_for_text += (predicted_image_idxs == correct_labels).sum().item()
			# Top-k Accuracy
			for k in top_k:
				top_k_preds = torch.topk(logits_per_image, k=k, dim=1).indices
				top_k_accuracy[k] += (top_k_preds == correct_labels.unsqueeze(1)).any(dim=1).sum().item()
			# Reciprocal Rank
			for i in range(batch_size):
				ranks = torch.argsort(logits_per_image[i], descending=True)
				rank_of_true_label = (ranks == correct_labels[i]).nonzero(as_tuple=True)[0].item() + 1
				reciprocal_ranks.append(1 / rank_of_true_label)
			# Cosine Similarity
			cos_sim = torch.nn.functional.cosine_similarity(logits_per_image, logits_per_text, dim=1).cpu().numpy()
			cosine_similarities.extend(cos_sim)
			# Precision, Recall, F1
			precision = (predicted_text_idxs == correct_labels).sum().item() / predicted_text_idxs.size(0)
			recall = (predicted_image_idxs == correct_labels).sum().item() / correct_labels.size(0)
			f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
			precision_list.append(precision)
			recall_list.append(recall)
			f1_list.append(f1)
			# Validation Loss
			loss_img = criterion(logits_per_image, correct_labels)
			loss_txt = criterion(logits_per_text, correct_labels)
			total_loss += 0.5 * (loss_img.item() + loss_txt.item())
	# Compute average metrics
	avg_loss = total_loss / len(validation_loader)
	accuracy_text_description = correct_text_description / total_samples
	accuracy_image_for_text = correct_image_for_text / total_samples
	top_k_accuracy = {k: v / total_samples for k, v in top_k_accuracy.items()}
	mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)
	cosine_sim_mean = np.mean(cosine_similarities)
	avg_precision = np.mean(precision_list)
	avg_recall = np.mean(recall_list)
	avg_f1 = np.mean(f1_list)
	return (
		avg_loss,
		accuracy_text_description,
		accuracy_image_for_text,
		top_k_accuracy,
		mean_reciprocal_rank,
		cosine_sim_mean,
		avg_precision,
		avg_recall,
		avg_f1,
	)

def plot_loss_accuracy(
		train_losses,
		val_losses,
		validation_accuracy_text_description_for_each_image_list,
		validation_acc_img_per_txt_list,
		top_k_accuracy_list,
		mean_reciprocal_rank_list,
		cosine_similarity_list,
		precision_list,
		recall_list,
		f1_list,
		losses_file_path="losses.png",
		accuracy_file_path="accuracy.png",
		topk_accuracy_file_path="top_k_accuracy.png",
		mean_reciprocal_rank_file_path="mean_reciprocal_rank.png",
		cosine_similarity_file_path="cosine_similarity.png",
		precision_recall_f1_file_path="precision_recall_f1.png",
	):
	num_epochs = len(train_losses)
	if num_epochs == 1:
		return
	epochs = range(1, num_epochs + 1)
	figure_size = (9, 4)
	plt.figure(figsize=figure_size)
	plt.plot(epochs, train_losses, color='b', label='Train', lw=1.25)
	plt.plot(epochs, val_losses, color='r', label='Validation', lw=1.25)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(os.path.splitext(os.path.basename(losses_file_path))[0], fontsize=10)
	plt.legend(ncols=2, title="Loss", fontsize=9, title_fontsize=10, loc='best')
	plt.tight_layout()
	plt.grid(True)
	# Set xticks to only integer values
	plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.savefig(losses_file_path)
	plt.close()

	plt.figure(figsize=figure_size)
	plt.plot(epochs, validation_accuracy_text_description_for_each_image_list, label='text retrieval per image')
	plt.plot(epochs, validation_acc_img_per_txt_list, label='image retrieval per text')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title(os.path.splitext(os.path.basename(accuracy_file_path))[0], fontsize=10)
	plt.legend(title='Validation Accuracy', fontsize=9, title_fontsize=10, loc='best')
	plt.grid(True)
	plt.tight_layout()
	# Set xticks to only integer values
	plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.savefig(accuracy_file_path)
	plt.close()
	
	plt.figure(figsize=figure_size)
	for k, acc in zip([1, 3, 5], zip(*top_k_accuracy_list)):
		plt.plot(epochs, acc, label=f'Top-{k}')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title("Top-k Accuracy")
	plt.legend(ncols=len([1, 3, 5]), loc='best')
	plt.grid(True)
	plt.tight_layout()
	# Set xticks to only integer values
	plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.savefig(topk_accuracy_file_path)
	plt.close()
	
	plt.figure(figsize=figure_size)
	plt.plot(epochs, mean_reciprocal_rank_list,  label='Mean Reciprocal Rank')
	plt.xlabel('Epoch')
	plt.ylabel('MRR')
	plt.title("Mean Reciprocal Rank")
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	# Set xticks to only integer values
	plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.savefig(mean_reciprocal_rank_file_path)
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
	# Set xticks to only integer values
	plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.savefig(precision_recall_f1_file_path)
	plt.close()
	
	plt.figure(figsize=figure_size)
	plt.plot(epochs, cosine_similarity_list,  linestyle='-', color='g', label='Cosine Similarity')
	plt.xlabel('Epoch')
	plt.ylabel('Cosine Similarity')
	plt.title("Cosine Similarity Over Epochs", fontsize=10)
	plt.grid(True)
	plt.tight_layout()
	plt.legend()
	# Set xticks to only integer values
	plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.savefig(cosine_similarity_file_path)
	plt.close()

def print_model_stat(model):
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)	
	total_params = sum(p.numel() for p in model.parameters())
	trainable_percent = (trainable_params / total_params) * 100
	frozen_percent = (frozen_params / total_params) * 100
	print(
		f"[Model Parameters Statictics] Total: {total_params:,} "
		f"Trainable: {trainable_params:,} ({trainable_percent:.2f}%) "
		f"Frozen: {frozen_params:,} ({frozen_percent:.2f}%)"
		.center(170, " ")
	)

def get_num_vit_blocks(model):
	if not hasattr(model, 'visual') or not hasattr(model.visual, 'transformer'):
		raise ValueError("Model structure not compatible - missing visual transformer")
	vis_transformer = model.visual.transformer
	txt_transformer = model.transformer
	return len(vis_transformer.resblocks), len(txt_transformer.resblocks)

def get_layer_groups(nv:int=12, nt:int=12):
	layer_groups = {
		'visual_frontend': ['visual.conv1','visual.class_embedding','visual.positional_embedding'],
		'visual_transformer': [f'visual.transformer.resblocks.{i}' for i in range(nv)],
		'text_frontend': ['token_embedding','positional_embedding'],
		'text_transformer': [f'transformer.resblocks.{i}' for i in range(nt)],
		'projections': ['visual.ln_post','text_projection','logit_scale'],
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

def set_freeze(model, layers_to_freeze):
	for name, param in model.named_parameters():
		param.requires_grad = True # Unfreeze all layers first
		if any(layer in name for layer in layers_to_freeze): # Freeze layers in the list
			param.requires_grad = False

def should_transition_phase(val_losses, threshold, window=3):
	if len(val_losses) < window:
		return False # Not enough data to make a decision
	recent_losses = val_losses[-window:]
	avg_loss = sum(recent_losses) / window
	relative_change = abs(recent_losses[-1] - avg_loss) / avg_loss # Relative change in loss
	return relative_change < threshold

def handle_phase_transition(current_phase, initial_lr, max_phases):
	if current_phase >= max_phases - 1:
		return current_phase, initial_lr * (0.1 ** current_phase)
	new_phase = current_phase + 1
	new_lr = initial_lr * (0.1 ** new_phase)  # Reduce learning rate by 10x
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
		early_stopping_patience:int=3,
		learning_rate:float=1e-5,
		weight_decay:float=1e-3,
		dataset_name:str="CIFAR10",
		device:str="cuda",
		results_dir:str="results",
	):
	os.makedirs(results_dir, exist_ok=True)
	mode = "finetune"
	print(f"{mode} CLIP {model_name} « {dataset_name} » {num_epochs} Epoch(s) {device} [x{nw} cores]".center(160, "-"))
	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))

	vis_nblocks, txt_nblocks = get_num_vit_blocks(model)
	print(f"[Transformer Blocks] Vision: {vis_nblocks} | Text: {txt_nblocks}")
	layer_groups = get_layer_groups(nv=vis_nblocks, nt=txt_nblocks,)
	total_v_layers = len(layer_groups['visual_transformer'])
	total_t_layers = len(layer_groups['text_transformer'])
	print(f"[Layer Groups] Visual: {total_v_layers} | Text: {total_t_layers}")
	freeze_schedule = get_progressive_freeze_schedule(layer_groups) # progressive freezing based on validation loss plateau
	print(f"Freeze Schedule:\n{json.dumps(freeze_schedule, indent=2)}")
	best_loss = np.inf
	current_phase = 0
	plateau_threshold: float = 1e-4
	no_improvement_count = 0
	# patience_per_phase: int = 3
	# counter = 0
	criterion = nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)
	training_losses, validation_losses = [], []
	validation_accuracy_text_description_for_each_image_list = []
	validation_acc_img_per_txt_list = []
	top_k_accuracy_list = []
	mean_reciprocal_rank_list = []
	cosine_similarity_list = []
	precision_list, recall_list, f1_list = [], [], []
	# smoothed_val_losses = []
	initial_learning_rate = learning_rate # Store the initial value
	ft_st = time.time()
	for epoch in range(num_epochs):
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		# Check for plateau to adapt phases of progressive freezing
		# if epoch > 0 and len(validation_losses) > 1:
		# 	current_smoothed_loss = smooth_(losses=validation_losses, window=3)
		# 	smoothed_val_losses.append(current_smoothed_loss)
		# 	if len(smoothed_val_losses) > 1:
		# 		loss_diff = smoothed_val_losses[-2] - smoothed_val_losses[-1]
		# 		if loss_diff < plateau_threshold:
		# 			counter += 1
		# 			print(f"Plateau counter: {counter}/{patience_per_phase} (Smoothed loss: {current_smoothed_loss:.6f})")
		# 		else:
		# 			counter = 0
		# 			print(f"No plateau detected. Continuing current phase. (Smoothed loss: {current_smoothed_loss:.6f})")
		# 		if counter >= patience_per_phase and current_phase < len(freeze_schedule) - 1:
		# 			current_phase += 1
		# 			counter = 0
		# 			learning_rate = initial_learning_rate * (0.1 ** current_phase) # Reduce learning rate by 10x for each new phase
		# 			print(f"Plateau detected. Transitioning to Phase {current_phase} with updated LR: {learning_rate:.1e}")
		WINDOWs = 3
		if epoch > 0 and len(validation_losses) > 1:
			should_transition = should_transition_phase(
				val_losses=validation_losses,
				threshold=plateau_threshold,
				window=WINDOWs,
			)
			if should_transition:
				current_phase, learning_rate = handle_phase_transition(
					current_phase=current_phase,
					initial_lr=initial_learning_rate,
					max_phases=len(freeze_schedule)
				)
			else:
				print(f"No plateau detected! Continuing with phase: {current_phase} ...")
		layers_to_freeze = freeze_schedule[current_phase]
		set_freeze(model, layers_to_freeze)
		print(f"Phase {current_phase}: Freezing {len(layers_to_freeze)} layers.")
		print_model_stat(model)
		optimizer = AdamW(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate, # potentially update learning rate based on phase
			betas=(0.9, 0.98),
			eps=1e-6,
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
		avg_valid_loss, accuracy_text_description_for_each_image, acc_img_per_txt, top_k_accuracy, mean_reciprocal_rank, cosine_sim_mean, avg_precision, avg_recall, avg_f1 = evaluate(model, validation_loader, criterion, device=device)
		validation_losses.append(avg_valid_loss)
		validation_accuracy_text_description_for_each_image_list.append(accuracy_text_description_for_each_image)
		validation_acc_img_per_txt_list.append(acc_img_per_txt)
		top_k_accuracy_list.append([top_k_accuracy[k] for k in [1, 3, 5]])
		mean_reciprocal_rank_list.append(mean_reciprocal_rank)
		cosine_similarity_list.append(cosine_sim_mean)
		precision_list.append(avg_precision)
		recall_list.append(avg_recall)
		f1_list.append(avg_f1)
		print(
			f'@ Epoch: {epoch+1}\n'
			f'\t[Loss] {mode}: {avg_training_loss:.7f} Valid: {avg_valid_loss:.9f}\n'
			f'\tValid Acc [text retrieval per image]: {accuracy_text_description_for_each_image} '
			f'[image retrieval per text]: {acc_img_per_txt}'
		)

		############################## Early stopping ##############################
		mdl_fpth = os.path.join(
			results_dir,
			f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_clip.pth"
		)
		if avg_valid_loss < best_loss:
			best_loss = avg_valid_loss
			torch.save(model.state_dict(), mdl_fpth)
			print(f"Saving model in {mdl_fpth} for best avg loss: {best_loss:.9f}")
			no_improvement_count = 0
		else:
			no_improvement_count += 1
			if no_improvement_count >= early_stopping_patience:
				print(f"Early stopping triggered after {epoch+1} epochs due to no improvement.")
				break
		############################## Early stopping ##############################
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
		val_losses=validation_losses,
		validation_accuracy_text_description_for_each_image_list=validation_accuracy_text_description_for_each_image_list,
		validation_acc_img_per_txt_list=validation_acc_img_per_txt_list,
		top_k_accuracy_list=top_k_accuracy_list,
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
	)

def train(
		model:nn.Module,
		train_loader:DataLoader,
		validation_loader:DataLoader,
		num_epochs:int=5,
		nw:int=10,
		print_every:int=150,
		model_name:str="ViT-B/32",
		early_stopping_patience:int=3,
		learning_rate:float=1e-5,
		weight_decay:float=1e-3,
		dataset_name:str="CIFAR10",
		device:str="cuda",
		results_dir:str="results",
	):
	mode = "train"
	os.makedirs(results_dir, exist_ok=True)

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
	best_loss = np.inf
	best_accuracy = 0.0
	no_improvement_count = 0
	moving_average_loss = []
	moving_average_window = 3
	optimizer = AdamW(
		params=[p for p in model.parameters() if p.requires_grad],# Only optimizes parameters that require gradients
		lr=learning_rate,
		betas=(0.9,0.98),
		eps=1e-6,
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
	training_losses, validation_losses = [], []
	validation_accuracy_text_description_for_each_image_list = []
	validation_acc_img_per_txt_list = []
	top_k_accuracy_list = []
	mean_reciprocal_rank_list = []
	cosine_similarity_list = []
	precision_list, recall_list, f1_list = [], [], []
	ft_st = time.time()
	for epoch in range(num_epochs):
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
					f"\tBatch [{bidx+1}/{len(train_loader)}] "
					f"Loss: {total_loss.item():.7f}",
				)
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		print(f"Average {mode} Loss: {avg_training_loss:.5f} @ Epoch: {epoch+1}")
		training_losses.append(avg_training_loss)
		avg_valid_loss, accuracy_text_description_for_each_image, acc_img_per_txt, top_k_accuracy, mean_reciprocal_rank, cosine_sim_mean, avg_precision, avg_recall, avg_f1 = evaluate(model, validation_loader, criterion, device=device)
		validation_losses.append(avg_valid_loss)
		validation_accuracy_text_description_for_each_image_list.append(accuracy_text_description_for_each_image)
		validation_acc_img_per_txt_list.append(acc_img_per_txt)
		top_k_accuracy_list.append([top_k_accuracy[k] for k in [1, 3, 5]])
		mean_reciprocal_rank_list.append(mean_reciprocal_rank)
		cosine_similarity_list.append(cosine_sim_mean)
		precision_list.append(avg_precision)
		recall_list.append(avg_recall)
		f1_list.append(avg_f1)
		print(
			f'{mode} Loss: {avg_training_loss:.4f} '
			f'Validation Loss: {avg_valid_loss:.4f} '
			f'Validation Accuracy [text description for each image]: {accuracy_text_description_for_each_image:.4f} '
			f'[image for each text description]: {acc_img_per_txt:.4f}'
		)

		# ############################## Early stopping ##############################
		mdl_fpth = os.path.join(
			results_dir,
			f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_clip.pth"
		)
		# if avg_valid_loss < best_loss:
		# 	best_loss = avg_valid_loss
		# 	torch.save(model.state_dict(), mdl_fpth)
		# 	print(f"Saving model in « {mdl_fpth} » | best avg loss: {best_loss:.5f}")
		# 	no_improvement_count = 0
		# else:
		# 	no_improvement_count += 1
		# 	if no_improvement_count >= early_stopping_patience:
		# 		print(f"Early stopping triggered after {epoch+1} epochs.")
		# 		break
		moving_average_loss.append(avg_valid_loss)
		if len(moving_average_loss) > moving_average_window:
			moving_average_loss.pop(0) # Remove the oldest loss
		avg_moving_loss = sum(moving_average_loss) / len(moving_average_loss)
		if avg_valid_loss < best_loss:
			best_loss = avg_valid_loss
			torch.save(model.state_dict(), mdl_fpth)
			print(f"Saving model in « {mdl_fpth} » | best avg loss: {best_loss:.9f}")
			no_improvement_count = 0
		else:
			no_improvement_count += 1
			if no_improvement_count >= early_stopping_patience:
				if avg_moving_loss > best_loss * 1.05: # 5% of the best lossv
					print(f"Early stopping triggered after {epoch + 1} epochs.")
					break
		# ############################## Early stopping ##############################

	print(f"Elapsed_t: {time.time()-ft_st:.1f} sec".center(150, "-"))

	losses_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_losses_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs.png")
	val_acc_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_accuracy_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs.png")
	topk_acc_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_top_k_accuracy_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs.png")
	mrr_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_mean_reciprocal_rank_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs.png")
	cs_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_cosine_similarity_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs.png")
	pr_f1_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_precision_recall_f1_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs.png")
	plot_loss_accuracy(
		train_losses=training_losses,
		val_losses=validation_losses,
		validation_accuracy_text_description_for_each_image_list=validation_accuracy_text_description_for_each_image_list,
		validation_acc_img_per_txt_list=validation_acc_img_per_txt_list,
		top_k_accuracy_list=top_k_accuracy_list,
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
	)

def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for CIFAR10x Dataset")
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=18, help='Number of CPUs [def: max cpus]')
	parser.add_argument('--num_epochs', '-ne', type=int, default=7, help='Number of epochs')
	parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4, help='small learning rate for better convergence [def: 1e-3]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='Weight decay [def: 5e-4]')
	parser.add_argument('--print_every', type=int, default=150, help='Print loss')
	parser.add_argument('--model_name', '-md', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--dataset', '-d', type=str, choices=['cifar10', 'cifar100', 'cinic10', 'imagenet'], default='cifar10', help='Choose dataset (CIFAR10/cifar100)')
	parser.add_argument('--mode', '-m', type=str, choices=['train', 'finetune'], default='finetune', help='Choose mode (train/finetune)')

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
			num_epochs=args.num_epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			model_name=args.model_name,
			early_stopping_patience=3,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			dataset_name=args.dataset,
			device=args.device,
			results_dir=os.path.join(args.dataset, "results")
		)
	elif args.mode == 'train':
		train(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.num_epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			model_name=args.model_name,
			early_stopping_patience=3,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			dataset_name=args.dataset,
			device=args.device,
			results_dir=os.path.join(args.dataset, "results")
		)
	else:
		raise ValueError("Invalid mode. Choose either 'finetune' or 'train'.")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	START_EXECUTION_TIME = time.time()
	main()
	END_EXECUTION_TIME = time.time()
	print(
		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
		.center(160, " ")
	)