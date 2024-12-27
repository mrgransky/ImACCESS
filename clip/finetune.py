from utils import *

# run in pouta:
# finetune CIFAR10x dataset with given frozen layers:
# $ nohup python -u finetune.py -d CIFAR100 -bs 260 -ne 32 -lr 5e-6 -wd 1e-3 --print_every 100 -nw 25 --device "cuda:0" -md "ViT-B/32" -fl visual.conv1 visual.ln_pre > /media/volume/ImACCESS/trash/cifar100_finetune.out &

# train CIFAR100 from scratch:
# $ nohup python -u finetune.py -d CIFAR100 -bs 260 -ne 32 -lr 5e-6 -wd 1e-3 --print_every 100 -nw 25 --device "cuda:1" -md "ViT-B/32" > /media/volume/ImACCESS/trash/cifar100_train.out &

# finetune CINIC10 dataset with given frozen layers:
# $ nohup python -u finetune.py -d CINIC10 -bs 260 -ne 32 -lr 5e-6 -wd 1e-3 --print_every 100 -nw 50 --device "cuda:0" -md "ViT-B/32" -fl visual.conv1 visual.ln_pre > /media/volume/ImACCESS/trash/cinic10_finetune.out &

USER = os.environ.get('USER')

class CINIC10(Dataset):
	classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	def __init__(self, root, train=True, download=False, transform=None):
		self.root = root
		self.train = train
		self.transform = transform
		if train:
			self.data = self._load_data(os.path.join(root, 'train'))
		else:
			self.data = self._load_data(os.path.join(root, 'valid'))

	def _load_data(self, directory):
		data = []
		labels = []
		for idx, class_name in enumerate(self.classes):
			# print(f"Loading {idx} {class_name} images...")
			class_dir = os.path.join(directory, class_name)
			for file_name in os.listdir(class_dir):
				file_path = os.path.join(class_dir, file_name)
				data.append(file_path)
				labels.append(idx)
		return list(zip(data, labels))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		file_path, label = self.data[index]
		image = Image.open(file_path)
		if self.transform is not None:
			image = self.transform(image)
		return image, label

	def __repr__(self):
		split = 'Train' if self.train else 'Test'
		return (
			f'Dataset CINIC10\n' \
			f'    Number of datapoints: {len(self)}\n' \
			f'    Root location: {self.root}\n' \
			f'    Split: {split}'
		)

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
	elif dname == 'ImageNet':
		train_dataset = ImageNet(
			root=os.path.expanduser("~/.cache"),
			split='train',
			download=True,
			transform=None
		)
		validation_dataset = ImageNet(
			root=os.path.expanduser("~/.cache"),
			split='val',
			download=True,
			transform=None
		)
	elif dname == 'CINIC10':
		ddir = {
			"farid": '/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/CINIC10',
			"ubuntu": '/media/volume/ImACCESS/WW_DATASETs/CINIC10',
			"alijanif": '/scratch/project_2004072/ImACCESS/WW_DATASETs/CINIC10',
		}
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
		raise ValueError(f"Invalid dataset name: {dname}. Available: [CIFAR10, CIFAR100]")
	print(train_dataset)
	print(validation_dataset)
	return train_dataset, validation_dataset

class CUSTOMIZEDDATASET(torch.utils.data.Dataset):
	def __init__(self, dataset, transformer=None,):
		self.dataset = dataset
		self.images = [img for idx, (img,lbl) in enumerate(dataset)]
		self.labels = clip.tokenize(texts=[dataset.classes[lbl_idx] for i, (img, lbl_idx) in enumerate(dataset)])
		if transformer:
			self.transform = transformer
		else:
			self.transform = T.Compose(
				[
					T.ToTensor(),
					T.Normalize(
						(0.491, 0.482, 0.446), 
						(0.247, 0.243, 0.261)
					)
				]
			)

	def __getitem__(self, index):
		image = self.images[index]
		text = self.labels[index]
		return self.transform(image), text

	def __len__(self):
		return len(self.dataset)

def get_dataloaders(train_dataset, valid_dataset, preprocess, batch_size=32, num_workers=10):
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
		num_workers=num_workers,
		pin_memory=True, # Move data to GPU faster if using CUDA
		persistent_workers=True if num_workers > 1 else False,  # Keep workers alive if memory allows
	)
	validation_loader = DataLoader(
		dataset=validset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
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
		validation_accuracy_text_image_for_each_text_description_list,
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
	figure_size = (10, 5)
	plt.figure(figsize=figure_size)
	plt.plot(epochs, train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
	plt.plot(epochs, val_losses, marker='o', linestyle='-', color='r', label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(os.path.splitext(os.path.basename(losses_file_path))[0], fontsize=10)
	plt.legend()
	plt.tight_layout()
	plt.savefig(losses_file_path)
	plt.close()

	plt.figure(figsize=figure_size)
	plt.plot(epochs, validation_accuracy_text_description_for_each_image_list, marker='o', label='text retrieval per image')
	plt.plot(epochs, validation_accuracy_text_image_for_each_text_description_list, marker='o', label='image retrieval per text')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title(os.path.splitext(os.path.basename(accuracy_file_path))[0], fontsize=10)
	plt.legend(title='Validation Accuracy', fontsize=9, title_fontsize=10, loc='best')
	plt.tight_layout()
	plt.savefig(accuracy_file_path)
	plt.close()
	
	plt.figure(figsize=figure_size)
	for k, acc in zip([1, 3, 5], zip(*top_k_accuracy_list)):
		plt.plot(epochs, acc, marker='o', label=f'Top-{k} Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title("Top-k Accuracy")
	plt.legend()
	plt.tight_layout()
	plt.savefig(topk_accuracy_file_path)
	plt.close()
	
	plt.figure(figsize=figure_size)
	plt.plot(epochs, mean_reciprocal_rank_list, marker='o', label='Mean Reciprocal Rank')
	plt.xlabel('Epoch')
	plt.ylabel('MRR')
	plt.title("Mean Reciprocal Rank")
	plt.legend()
	plt.tight_layout()
	plt.savefig(mean_reciprocal_rank_file_path)
	plt.close()
	
	plt.figure(figsize=figure_size)
	plt.plot(epochs, precision_list, marker='o', label='Precision')
	plt.plot(epochs, recall_list, marker='o', label='Recall')
	plt.plot(epochs, f1_list, marker='o', label='F1 Score')
	plt.xlabel('Epoch')
	plt.ylabel('Score')
	plt.title("Precision, Recall, and F1 Score")
	plt.legend()
	plt.tight_layout()
	plt.savefig(precision_recall_f1_file_path)
	plt.close()
	
	plt.figure(figsize=figure_size)
	plt.plot(epochs, cosine_similarity_list, marker='o', linestyle='-', color='g', label='Cosine Similarity')
	plt.xlabel('Epoch')
	plt.ylabel('Cosine Similarity')
	plt.title("Cosine Similarity Over Epochs", fontsize=10)
	plt.tight_layout()
	plt.legend()
	plt.savefig(cosine_similarity_file_path)
	plt.close()

def finetune(
		model:nn.Module,
		train_loader:DataLoader,
		validation_loader:DataLoader,
		num_epochs:int=5,
		num_workers:int=10,
		print_every:int=150,
		model_name:str="ViT-B/32",
		early_stopping_patience:int=5,
		learning_rate:float=1e-5,
		weight_decay:float=1e-3,
		dataset_name:str="CIFAR10",
		device:str="cuda",
		freeze_layers: list = None,
		results_dir:str="results",
	):
	mode = "finetune" if freeze_layers else "train"
	freeze_layers = freeze_layers or []
	os.makedirs(results_dir, exist_ok=True)

	print(f"{mode} CLIP {model_name} « {dataset_name} » {num_epochs} Epoch(s) {device} [x{num_workers} cores]".center(160, "-"))
	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))

	for name, param in model.named_parameters():
		if name.startswith(tuple(freeze_layers)):
			param.requires_grad = False
			print(f"{name} requires_grad: {param.requires_grad} => frozen")
		else:
			param.requires_grad = True

	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)	
	total_params = sum(p.numel() for p in model.parameters())
	trainable_percent = (trainable_params / total_params) * 100
	frozen_percent = (frozen_params / total_params) * 100
	print(
		f"[Model Parameters Statictics] Total: {total_params:,} "
		f"Trainable: {trainable_params:,} ({trainable_percent:.2f}%) "
		f"Frozen Parameters: {frozen_params:,} ({frozen_percent:.2f}%)"
		.center(160, "-")
	)

	best_loss = np.inf
	no_improvement_count = 0
	optimizer = optim.AdamW(
		params=model.parameters(),
		lr=learning_rate,
		betas=(0.9,0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(
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
	validation_accuracy_text_image_for_each_text_description_list = []
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
		print(f"Average {mode.capitalize()} Loss: {avg_training_loss:.5f} @ Epoch: {epoch+1}")
		training_losses.append(avg_training_loss)
		avg_valid_loss, accuracy_text_description_for_each_image, accuracy_text_image_for_each_text_description, top_k_accuracy, mean_reciprocal_rank, cosine_sim_mean, avg_precision, avg_recall, avg_f1 = evaluate(model, validation_loader, criterion, device=device)
		validation_losses.append(avg_valid_loss)
		validation_accuracy_text_description_for_each_image_list.append(accuracy_text_description_for_each_image)
		validation_accuracy_text_image_for_each_text_description_list.append(accuracy_text_image_for_each_text_description)
		top_k_accuracy_list.append([top_k_accuracy[k] for k in [1, 3, 5]])
		mean_reciprocal_rank_list.append(mean_reciprocal_rank)
		cosine_similarity_list.append(cosine_sim_mean)
		precision_list.append(avg_precision)
		recall_list.append(avg_recall)
		f1_list.append(avg_f1)
		print(
			f'{mode.capitalize()} Loss: {avg_training_loss:.4f} '
			f'Validation Loss: {avg_valid_loss:.4f} '
			f'Validation Accuracy [text description for each image]: {accuracy_text_description_for_each_image:.4f} '
			f'[image for each text description]: {accuracy_text_image_for_each_text_description:.4f}'
		)

		############################## Early stopping ##############################
		mdl_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_clip.pth")
		if avg_valid_loss < best_loss:
			best_loss = avg_valid_loss
			torch.save(model.state_dict(), mdl_fpth)
			print(f"Saving model in {mdl_fpth} for best avg loss: {best_loss:.5f}")
			no_improvement_count = 0
		else:
			no_improvement_count += 1
			if no_improvement_count >= early_stopping_patience:
				print(f"Early stopping triggered after {epoch+1} epochs.")
				break
		############################## Early stopping ##############################

	print(f"Elapsed_t: {time.time()-ft_st:.1f} sec".center(150, "-"))

	if mode == "finetune" and freeze_layers:
		freeze_layers_str = '_'.join(freeze_layers)
		losses_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_losses_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs_freeze_{freeze_layers_str}.png")
		val_acc_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_accuracy_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs_freeze_{freeze_layers_str}.png")
		topk_acc_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_top_k_accuracy_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs_freeze_{freeze_layers_str}.png")
		mrr_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_mean_reciprocal_rank_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs_freeze_{freeze_layers_str}.png")
		cs_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_cosine_similarity_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs_freeze_{freeze_layers_str}.png")
		pr_f1_fpth = os.path.join(results_dir, f"{dataset_name}_{mode}_{re.sub('/', '', model_name)}_precision_recall_f1_ep_{len(training_losses)}_lr_{learning_rate}_wd_{weight_decay}_{train_loader.batch_size}_bs_freeze_{freeze_layers_str}.png")
	else:
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
		validation_accuracy_text_image_for_each_text_description_list=validation_accuracy_text_image_for_each_text_description_list,
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
	parser.add_argument('--num_epochs', '-ne', type=int, default=5, help='Number of epochs')
	parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='small learning rate for better convergence [def: 1e-3]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3, help='Weight decay [def: 5e-4]')
	parser.add_argument('--print_every', type=int, default=250, help='Print loss')
	parser.add_argument('--model_name', '-md', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--dataset', '-d', type=str, choices=['CIFAR10', 'CIFAR100', 'CINIC10', 'ImageNet'], default='CIFAR10', help='Choose dataset (CIFAR10/CIFAR100)')
	parser.add_argument('--freeze_layers', '-fl', nargs='+', default=[], help='Layers to freeze, no "" needed')

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
		num_workers=args.num_workers,
	)
	print(f"Train Loader: {len(train_loader)} batches, Validation Loader: {len(validation_loader)} batches")
	# visualize_(dataloader=train_loader, num_samples=5)
	# return
	finetune(
		model=model,
		train_loader=train_loader,
		validation_loader=validation_loader,
		num_epochs=args.num_epochs,
		num_workers=args.num_workers,
		print_every=args.print_every,
		model_name=args.model_name,
		early_stopping_patience=5,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		dataset_name=args.dataset,
		device=args.device,
		freeze_layers=args.freeze_layers,
		results_dir=os.path.join(args.dataset, "results")
	)

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