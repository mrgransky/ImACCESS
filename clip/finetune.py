from utils import *

# run in pouta:
# $ nohup python -u finetune.py -d CIFAR100 -bs 16 -ne 30 -lr 1e-5 -wd 1e-3 --print_every 1000 -nw 50 --device "cuda:3" -m "ViT-L/14" > /media/volume/ImACCESS/trash/finetune_cifar100_cuda3.out &

class CIFARDATASET(torch.utils.data.Dataset):
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
						(0.48145466, 0.4578275, 0.40821073), 
						(0.26862954, 0.26130258, 0.27577711)
					)
				]
			)

	def __getitem__(self, index):
		image = self.images[index]
		text = self.labels[index]
		return self.transform(image), text

	def __len__(self):
		return len(self.dataset)

def get_dataloaders(train_dataset, test_dataset, preprocess, batch_size=32, num_workers=10):
	train_dataset = CIFARDATASET(
		dataset=train_dataset, 
		transformer=preprocess,
	)
	test_dataset = CIFARDATASET(
		dataset=test_dataset, 
		transformer=preprocess,
	)
	
	train_loader = DataLoader(
		dataset=train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True, # Move data to GPU faster if using CUDA
		persistent_workers=True if num_workers > 1 else False,  # Keep workers alive if memory allows
	)
	test_loader = DataLoader(
		dataset=test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True, # when using CUDA
	)
	return train_loader, test_loader

def evaluate(model, test_loader, criterion, device:str="cuda"):
	model.eval()
	total_loss = 0
	total_correct_text_description_for_each_image = 0
	total_correct_image_for_each_text_description = 0
	with torch.no_grad():
		for batch_idx, batch in enumerate(test_loader):
			images, labels = batch
			images = images.to(device)
			labels = labels.to(device)
			logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
			_, predicted_idxs_imgs = torch.max(input=logits_per_image, dim=1, keepdim=True)
			_, predicted_idxs_txts = torch.max(input=logits_per_text, dim=1, keepdim=True)
			# Get the indices of the correct text descriptions for each image
			correct_text_description_idxs = torch.argmax(labels, dim=1)
			# Compare the predicted indexes with the correct indexes
			total_correct_text_description_for_each_image += (predicted_idxs_imgs == correct_text_description_idxs.unsqueeze(1)).sum().item()
			total_correct_image_for_each_text_description += (predicted_idxs_txts == correct_text_description_idxs.unsqueeze(1)).sum().item()
			# Compute validation loss
			ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
			loss_img = criterion(logits_per_image, ground_truth) 
			loss_txt = criterion(logits_per_text, ground_truth)
			valid_loss = 0.5 * (loss_img + loss_txt)
			total_loss += valid_loss.item()
	avg_loss = total_loss / len(test_loader)
	accuracy_text_description_for_each_image = total_correct_text_description_for_each_image / len(test_loader.dataset)
	accuracy_text_image_for_each_text_description = total_correct_image_for_each_text_description / len(test_loader.dataset)
	return avg_loss, accuracy_text_description_for_each_image, accuracy_text_image_for_each_text_description

def finetune(
		model:nn.Module,
		train_loader:DataLoader,
		test_loader:DataLoader,
		num_epochs:int=5,
		num_workers:int=10,
		print_every:int=150,
		model_name:str="ViT-B/32",
		early_stopping_patience:int=5,
		learning_rate:float=1e-5,
		weight_decay:float=1e-3,
		dataset_name:str="CIFAR10",
		device:str="cuda",
	):
	print(f"Fine-Tuning CLIP {model_name} {dataset_name} {num_epochs} Epoch(s) {device} [x{num_workers} cores]".center(160, "-"))
	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))

	for name, param in model.named_parameters():
		print(f"{name} dtype: {param.dtype}")
		if 'visual.conv1' in name or 'visual.ln_pre' in name:
			param.requires_grad = False # freeze the weights of the first layer of the model
		else:
			param.requires_grad = True # backpropagation calculate and update gradients for these parameters, thereby fine-tuning these model layers.

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
	ft_st = time.time()
	for epoch in range(num_epochs):
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		epoch_loss = 0.0
		for batch_idx, batch in enumerate(train_loader):
			optimizer.zero_grad() # Clear gradients from previous batch
			images, labels = batch # torch.Size([b, 3, 224, 224]), torch.Size([b, 77])
			images, labels = images.to(device), labels.to(device)
			# logits_per_image: similarity between image embeddings and all text embeddings in batch
			# logits_per_text: similarity between text embeddings and all image embeddings in batch

			# # Conventional backpropagation:
			# logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
			# ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
			# loss_img = criterion(logits_per_image, ground_truth) 
			# loss_txt = criterion(logits_per_text, ground_truth)
			# total_loss = 0.5 * (loss_img + loss_txt)
			# total_loss.backward()
			# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			# optimizer.step() # Update weights

			with torch.amp.autocast(device_type=device.type): # # Automatic Mixed Precision (AMP) backpropagation:
				logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)
			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step() # Update learning rate
			if batch_idx%print_every==0 or batch_idx+1==len(train_loader):
				print(
					f"\tBatch [{batch_idx+1}/{len(train_loader)}] "
					f"Loss: {total_loss.item():.7f}",
				)
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		print(f"Average Training Loss: {avg_training_loss:.5f} @ Epoch: {epoch+1}")
		training_losses.append(avg_training_loss)
		avg_valid_loss, accuracy_text_description_for_each_image, accuracy_text_image_for_each_text_description = evaluate(model, test_loader, criterion, device=device)
		validation_losses.append(avg_valid_loss)
		validation_accuracy_text_description_for_each_image_list.append(accuracy_text_description_for_each_image)
		validation_accuracy_text_image_for_each_text_description_list.append(accuracy_text_image_for_each_text_description)
		print(
			f'Training Loss: {avg_training_loss:.4f} '
			f'Validation Loss: {avg_valid_loss:.4f} '
			f'Validation Accuracy [text description for each image]: {accuracy_text_description_for_each_image:.4f} '
			f'[image for each text description]: {accuracy_text_image_for_each_text_description:.4f}'
		)

		############################## Early stopping ##############################
		mdl_fpth = f"{dataset_name}_clip_finetuned.pth"
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
	plot_loss_accuracy(
		train_losses=training_losses,
		val_losses=validation_losses,
		validation_accuracy_text_description_for_each_image_list=validation_accuracy_text_description_for_each_image_list,
		validation_accuracy_text_image_for_each_text_description_list=validation_accuracy_text_image_for_each_text_description_list,
		dataset_name=dataset_name,
		learning_rate=learning_rate,
		weight_decay=weight_decay,
		batch_size=train_loader.batch_size,
	)

def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for CIFAR10x Dataset")
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=18, help='Number of CPUs [def: max cpus]')
	parser.add_argument('--num_epochs', '-ne', type=int, default=5, help='Number of epochs')
	parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='small learning rate for better convergence [def: 1e-3]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3, help='Weight decay [def: 5e-4]')
	parser.add_argument('--print_every', type=int, default=150, help='Print loss')
	parser.add_argument('--model_name', '-m', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--dataset', '-d', type=str, choices=['CIFAR10', 'CIFAR100'], default='CIFAR10', help='Choose dataset (CIFAR10/CIFAR100)')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print(args)

	print(clip.available_models())
	model, preprocess = load_model(
		model_name=args.model_name,
		device=args.device,
		jit=False,
	)
	train_dataset, test_dataset = get_dataset(dname=args.dataset)
	train_loader, test_loader = get_dataloaders(
		train_dataset=train_dataset, 
		test_dataset=test_dataset, 
		preprocess=preprocess,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
	)

	finetune(
		model=model,
		train_loader=train_loader,
		test_loader=test_loader,
		num_epochs=args.num_epochs,
		num_workers=args.num_workers,
		print_every=args.print_every,
		model_name=args.model_name,
		device=args.device,
		early_stopping_patience=5,
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