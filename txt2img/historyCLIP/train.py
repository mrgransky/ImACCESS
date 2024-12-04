from utils import *
from models import *
from dataset_loader import HistoricalDataset

# how to run [Local]:
# $ python train.py -ddir $HOME/WS_Farid/ImACCESS/txt2img/datasets/national_archive/NATIONAL_ARCHIVE_1933-01-01_1933-01-02 -vddir $HOME/WS_Farid/ImACCESS/txt2img/datasets/europeana/EUROPEANA_1900-01-01_1970-12-31 -nep 1 -lr 5e-4 -wd 5e-2
# $ python train.py -ddir $HOME/WS_Farid/ImACCESS/txt2img/datasets/europeana/europeana_1890-01-01_1960-01-01 -nep 1

# $ nohup python -u train.py -ddir $HOME/WS_Farid/ImACCESS/txt2img/datasets/national_archive/NATIONAL_ARCHIVE_1933-01-01_1933-01-02 -nep 12 -lr 5e-4 -wd 2e-2 -ps 5 -is 160 > $PWD/logs/historyCLIP.out &

# how to run [Pouta]:
# Ensure Conda:
# $ conda activate py39
# $ python train.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1935-01-01_1950-12-31 --device "cuda:2" -nep 1 -bs 128

# With Europeana as validation set:
# $ nohup python -u train.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1935-01-01_1950-12-31 -vddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -nep 30 --device "cuda:1" -lr 1e-3 -wd 5e-2 -ps 5 -is 160 -bs 64 > /media/volume/trash/ImACCESS/historyCLIP_train_NA_val_EUROPEANA_cuda1.out &

# with splited dataset of NA:
# $ nohup python -u train.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1935-01-01_1950-12-31 -nep 30 --device "cuda:3" -lr 1e-3 -wd 5e-2 -ps 5 -is 160 -bs 64 -nw 20 > /media/volume/trash/ImACCESS/historyCLIP_train_NA_val_NA_cuda3.out &

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--validation_dataset_dir', '-vddir', default=None, help='Dataset DIR')
parser.add_argument('--topk', type=int, default=5, help='Top-K images')
parser.add_argument('--batch_size', '-bs', type=int, default=27, help='Batch Size')
parser.add_argument('--image_size', '-is', type=int, default=150, help='Image size [def: max 160 local]')
parser.add_argument('--patch_size', '-ps', type=int, default=5, help='Patch size')
parser.add_argument('--embedding_size', '-es',type=int, default=1024, help='Embedding size of Vision & Text encoder [the larger the better]')
parser.add_argument('--print_every', type=int, default=200, help='Print loss')
parser.add_argument('--num_epochs', '-nep', type=int, default=10, help='Number of epochs')
parser.add_argument('--num_workers', '-nw', type=int, default=10, help='Number of CPUs [def: max cpus]')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='small learning rate for better convergence [def: 1e-3]')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-1, help='Weight decay [def: 5e-4]')
parser.add_argument('--data_augmentation', type=bool, default=False, help='Data Augmentation')
parser.add_argument('--document_description_col', type=str, default="label", help='labels')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)

args.device = torch.device(args.device)

os.makedirs(os.path.join(args.dataset_dir, "outputs"), exist_ok=True)
outputs_dir:str = os.path.join(args.dataset_dir, "outputs",)

os.makedirs(os.path.join(args.dataset_dir, "models"), exist_ok=True)
models_dir:str = os.path.join(args.dataset_dir, "models",)

# sys.exit()

def get_val_loss(model, val_loader, ):
	print(f"Validating val_loader {type(val_loader)} with {len(val_loader.dataset)} sample(s)", end="\t")
	model.eval()
	val_loss = 0.0
	with torch.no_grad():
		for data in val_loader:
			img = data["image"].to(args.device)
			cap = data["caption"].to(args.device)
			mask = data["mask"].to(args.device)
			loss = model(img, cap, mask)
			val_loss += loss.item()
	avg_val_loss = val_loss / len(val_loader)
	print(f"Validation Loss: {avg_val_loss:.5f}")
	return avg_val_loss

def train(model, train_data_loader, val_data_loader, optimizer, scheduler, checkpoint_interval:int=5, model_dir:str="path/2/model_dir", early_stopping_patience:int=10):
	mdl_fpth:str = os.path.join(args.dataset_dir, model_dir, "model.pt")
	
	os.makedirs(os.path.join(args.dataset_dir, model_dir, "results"), exist_ok=True)
	results_dir:str = os.path.join(args.dataset_dir, model_dir, "results")

	os.makedirs(os.path.join(args.dataset_dir, model_dir, "checkpoints"), exist_ok=True)
	checkpoint_dir = os.path.join(args.dataset_dir, model_dir, "checkpoints")
	
	print(f"Training CLIP model {args.num_epochs} Epoch(s) device: {args.device} & {args.num_workers} CPU(s)".center(150, "-"))
	if torch.cuda.is_available():
		print(f"GPU: {torch.cuda.get_device_name(args.device)}".center(150, " "))
	log_gpu_memory(device=args.device)
	writer = SummaryWriter(log_dir=os.path.join(outputs_dir, "logs")) # Initialize TensorBoard writer
	
	steps = []
	lrs = []
	total_params = 0
	total_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
	print(f"Total trainable parameters (Vision + Text) Encoder: {total_params} ~ {total_params/int(1e+6):.2f} M")
	best_loss = np.inf
	no_improvement_count = 0

	start_epoch = 0
	checkpoint_fpth = os.path.join(checkpoint_dir, "checkpoint.pt")
	if os.path.exists(checkpoint_fpth):
		print(f"Found checkpoint: {checkpoint_fpth}!")
		checkpoint = torch.load(checkpoint_fpth)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		start_epoch = checkpoint['epoch'] + 1
		best_loss = checkpoint['best_loss']
		no_improvement_count = checkpoint['no_improvement_count']
		print(f"Resuming training from epoch {start_epoch}...")
	
	average_train_losses = list()
	average_val_losses = list()

	scaler = torch.amp.GradScaler(
		device=args.device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)

	training_st = time.time()
	for epoch in range(args.num_epochs):
		print(f"Epoch [{epoch+1}/{args.num_epochs}]", end="\t")
		log_gpu_memory(device=args.device)
		
		epoch_loss = 0.0  # To accumulate the loss over the epoch
		model.train()
		for batch_idx, data in enumerate(train_data_loader):
			img = data["image"].to(args.device) 
			cap = data["caption"].to(args.device)
			mask = data["mask"].to(args.device)

			optimizer.zero_grad()

			# # Conventional backpropagation:
			# loss = model(img, cap, mask)
			# loss.backward()
			# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			# optimizer.step()

			# Automatic mixed precision training
			with torch.amp.autocast(device_type=args.device.type):
				loss = model(img, cap, mask)
			
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()

			scheduler.step()
			# print(scheduler.get_last_lr())
			lrs.append(scheduler.get_last_lr()[0])
			steps.append(batch_idx)

			if batch_idx % args.print_every == 0:
				print(f"\tBatch [{batch_idx + 1}/{len(train_data_loader)}] Loss: {loss.item():.5f}", end="\t")
				log_gpu_memory(device=args.device)
				
			epoch_loss += loss.item()
			writer.add_scalar('Loss/train', loss.item(), epoch * len(train_data_loader) + batch_idx)

		avg_train_loss = epoch_loss / len(train_data_loader)
		print(f"Average Training Loss: {avg_train_loss:.5f} @ Epoch: {epoch+1}")
		average_train_losses.append(avg_train_loss)
		############################## traditional model saving ##############################
		# if avg_train_loss <= best_loss:
		# 	best_loss = avg_train_loss
		# 	torch.save(model.state_dict(), mdl_fpth)
		# 	print(f"Saving model in {mdl_fpth} for best avg loss: {best_loss:.5f}")
		############################## traditional model saving ##############################
		
		############################## Early stopping ##############################
		avg_val_loss = get_val_loss(model=model, val_loader=val_data_loader,)
		writer.add_scalar('Loss/validation', avg_val_loss, epoch)
		average_val_losses.append(avg_val_loss)

		if avg_val_loss < best_loss:
			best_loss = avg_val_loss
			torch.save(model.state_dict(), mdl_fpth)
			print(f"Saving model in {mdl_fpth} for best avg loss: {best_loss:.5f}")
			no_improvement_count = 0
		else:
			no_improvement_count += 1
			if no_improvement_count >= early_stopping_patience:
				print(f"Early stopping triggered after {epoch+1} epochs.")
				break
		############################## Early stopping ##############################
		writer.add_scalar('Average Loss/train', avg_train_loss, epoch)
		# Save checkpoint
		if (epoch + 1) % checkpoint_interval == 0:
			checkpoint = {
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': scheduler.state_dict(),
				'best_loss': best_loss,
				'no_improvement_count': no_improvement_count
			}
			# log_gpu_memory(device=args.device)
			torch.save(checkpoint, checkpoint_fpth)
			print(f"Checkpoint saved at epoch {epoch+1} : {checkpoint_fpth}")
			log_gpu_memory(device=args.device)

	print(f"Elapsed_t: {time.time()-training_st:.1f} sec".center(150, "-"))

	lrs_vs_steps_fname = (
		f'lrs_vs_steps'
		+ f'_epochs_{args.num_epochs}'
		+ f'_lr_{args.learning_rate}'
		+ f'_wd_{args.weight_decay}'
		+ f'.png'
	)
	plot_lrs_vs_steps(
		lrs=lrs, 
		steps=steps, 
		fpath=os.path.join(results_dir, lrs_vs_steps_fname),
	)

	losses_fname = (
		f'losses_train_val'
		+ f'_epochs_{args.num_epochs}'
		+ f'_lr_{args.learning_rate}'
		+ f'_wd_{args.weight_decay}'
		+ f'.png'
	)
	plot_(
		train_losses=average_train_losses,
		val_losses=average_val_losses,
		save_path=os.path.join(results_dir, losses_fname),
		lr=args.learning_rate,
		wd=args.weight_decay,
	)
	writer.close()

def main():
	###################################################### BW images ######################################################
	# img_bw_mean_fpth:str = os.path.join(args.dataset_dir, "img_bw_mean.gz")
	# img_bw_std_fpth:str = os.path.join(args.dataset_dir, "img_bw_std.gz")
	# try:
	# 	img_bw_mean, img_bw_std = load_pickle(fpath=img_bw_mean_fpth), load_pickle(fpath=img_bw_std_fpth)
	# except Exception as e:
	# 	print(f"{e}")
	# 	img_bw_mean, img_bw_std = get_mean_std_grayscale_img_multiprocessing(dir=os.path.join(args.dataset_dir, "images"))
	# 	save_pickle(pkl=img_bw_mean, fname=img_bw_mean_fpth)
	# 	save_pickle(pkl=img_bw_std, fname=img_bw_std_fpth)
	# print(f"Grayscale: Mean: {img_bw_mean} | Std: {img_bw_std}")
	###################################################### BW images ######################################################

	img_rgb_mean_fpth:str = os.path.join(args.dataset_dir, "img_rgb_mean.gz")
	img_rgb_std_fpth:str = os.path.join(args.dataset_dir, "img_rgb_std.gz")

	try:
		img_rgb_mean, img_rgb_std = load_pickle(fpath=img_rgb_mean_fpth), load_pickle(fpath=img_rgb_std_fpth) # RGB images
	except Exception as e:
		# print(f"{e}")
		##################################### Mean - Std Multiprocessing ####################################
		# img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(
		# 	dir=os.path.join(args.dataset_dir, "images"), 
		# 	num_workers=args.num_workers,
		# )
		##################################### Mean - Std Multiprocessing ####################################
		######################################## Mean - Std for loop ########################################
		img_rgb_mean, img_rgb_std = get_mean_std_rgb_img(dir=os.path.join(args.dataset_dir, "images"))
		######################################## Mean - Std for loop ########################################
		save_pickle(pkl=img_rgb_mean, fname=img_rgb_mean_fpth)
		save_pickle(pkl=img_rgb_std, fname=img_rgb_std_fpth)

	if args.validation_dataset_dir:
		validation_dataset_share = None
		print(f"Separate Train and Validation datasets")
		print(f"Validation Dataset: {args.validation_dataset_dir}")
		val_img_rgb_mean_fpth:str = os.path.join(args.validation_dataset_dir, "img_rgb_mean.gz")
		val_img_rgb_std_fpth:str = os.path.join(args.validation_dataset_dir, "img_rgb_std.gz")
		try:
			val_img_rgb_mean, val_img_rgb_std = load_pickle(fpath=val_img_rgb_mean_fpth), load_pickle(fpath=val_img_rgb_std_fpth) # RGB images
		except Exception as e:
			print(f"{e}")
			##################################### Mean - Std Multiprocessing ####################################
			# img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(
			# 	dir=os.path.join(args.dataset_dir, "images"), 
			# 	num_workers=args.num_workers,
			# )
			##################################### Mean - Std Multiprocessing ####################################

			######################################## Mean - Std for loop ########################################
			val_img_rgb_mean, val_img_rgb_std = get_mean_std_rgb_img(dir=os.path.join(args.dataset_dir, "images"))
			######################################## Mean - Std for loop ########################################

			save_pickle(pkl=val_img_rgb_mean, fname=val_img_rgb_mean_fpth)
			save_pickle(pkl=val_img_rgb_std, fname=val_img_rgb_std_fpth)
	else:
			# val = train
			validation_dataset_share = 0.3
			val_img_rgb_mean = img_rgb_mean
			val_img_rgb_std = img_rgb_std

	print(f"[Train] Mean: {img_rgb_mean} Std: {img_rgb_std}".center(180, " "))
	print(f"[Validation] Mean: {val_img_rgb_mean} Std: {val_img_rgb_std}".center(180, " "))

	print(f"Train & Validation metadata df".center(180, "-"))
	train_metadata_df, val_metadata_df = get_train_val_metadata_df(
		tddir=args.dataset_dir, 
		vddir=args.validation_dataset_dir, 
		split_pct=validation_dataset_share, 
		doc_desc="label", 
		seed=True,
	)

	LABELs_dict_fpth:str = os.path.join(args.dataset_dir, "LABELs_dict.gz")
	LABELs_list_fpth:str = os.path.join(args.dataset_dir, "LABELs_list.gz")
	try:
		LABELs_dict = load_pickle(fpath=LABELs_dict_fpth)
		LABELs_list = load_pickle(fpath=LABELs_list_fpth)
	except Exception as e:
		print(f"{e}")
		LABELs_dict, LABELs_list = get_doc_description(df=train_metadata_df, col='label') # must be "label", regardless of captions
		save_pickle(pkl=LABELs_dict, fname=LABELs_dict_fpth)
		save_pickle(pkl=LABELs_list, fname=LABELs_list_fpth)

	print(
		f"LABELs_dict {type(LABELs_dict)} {len(LABELs_dict)} "
		f"LABELs_list {type(LABELs_list)} {len(LABELs_list)}"
	)


	return
	
	print(f"Creating Train Dataloader for {len(train_metadata_df)} samples", end="\t")
	tdl_st = time.time()
	train_dataset = HistoricalDataset(
		data_frame=train_metadata_df,
		img_sz=args.image_size,
		dataset_directory=os.path.join(args.dataset_dir, "images"),
		max_seq_length=max_seq_length,
		mean=img_rgb_mean,
		std=img_rgb_std,
		txt_category=args.document_description_col,
		augment_data=False,
	)
	train_data_loader = DataLoader(
		dataset=train_dataset,
		shuffle=True,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=True,  # Move data to GPU faster if using CUDA
		persistent_workers=True if args.num_workers > 1 else False,  # Keep workers alive if memory allows
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(train_data_loader.dataset)} Elapsed_t: {time.time()-tdl_st:.5f} sec")
	get_info(dataloader=train_data_loader)

	# ###################### Visualize Samples ######################
	# visualize_samples(train_data_loader, num_samples=5)
	# sys.exit(-1)
	# ###################### Visualize Samples ######################

	print(f"Creating Validation Dataloader for {len(val_metadata_df)} samples", end="\t")
	vdl_st = time.time()
	val_dataset = HistoricalDataset(
		data_frame=val_metadata_df,
		img_sz=args.image_size,
		dataset_directory=os.path.join(args.validation_dataset_dir, "images") if args.validation_dataset_dir else os.path.join(args.dataset_dir, "images"),
		max_seq_length=max_seq_length,
		mean=val_img_rgb_mean,
		std=val_img_rgb_std,
		txt_category=args.document_description_col,
		augment_data=False,
	)
	val_data_loader = DataLoader(
		dataset=val_dataset,
		shuffle=False,
		batch_size=args.batch_size, 
		num_workers=args.num_workers,
		pin_memory=True, # when using CUDA
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(val_data_loader.dataset)} Elapsed_t: {time.time()-vdl_st:.5f} sec")
	get_info(dataloader=val_data_loader)

	print(f"Defining CLIP model...")
	model = CLIP(
		emb_dim=args.embedding_size,
		vit_layers=vit_layers,
		vit_d_model=vit_d_model,
		img_size=(args.image_size, args.image_size),
		patch_size=(args.patch_size, args.patch_size),
		n_channels=n_channels,
		vit_heads=vit_heads,
		vocab_size=vocab_size,
		max_seq_length=max_seq_length,
		text_heads=text_heads,
		text_layers=text_layers,
		text_d_model=text_d_model,
		device=args.device,
		retrieval=False,
	).to(args.device)

	print(f"Defining Optimizer...")
	optimizer = optim.AdamW(
		params=model.parameters(),
		betas=(0.9, 0.98), # Based on original CLIP paper
		eps=1e-8,
		lr=args.learning_rate,
		weight_decay=args.weight_decay, # weight decay (L2 regularization)
	)

	print(f"Defining Scheduler...")
	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer, 
		max_lr=args.learning_rate, 
		steps_per_epoch=len(train_data_loader), 
		epochs=args.num_epochs,
		pct_start=0.1, # percentage of the cycle (in number of steps) spent increasing the learning rate
		anneal_strategy='cos', # cos/linear annealing
	)

	model_fname = (
		f"model"
		+ f"_augmentation_{args.data_augmentation}"
		+ f"_ep_{args.num_epochs}"
		+ f"_train_{len(train_data_loader.dataset)}"
		+ f"_val_{len(val_data_loader.dataset)}"
		+ f"_batch_{args.batch_size}"
		+ f"_img_{args.image_size}"
		+ f"_patch_{args.patch_size}"
		+ f"_emb_{args.embedding_size}"
		+ f"_{re.sub(r':', '', str(args.device))}"
		+ f"_{optimizer.__class__.__name__}"
		+ f"_lr_{args.learning_rate}"
		+ f"_wd_{args.weight_decay}"
		# + f"_{get_args(optimizer)}"
		+ f"_{scheduler.__class__.__name__}"
		# + f"_{get_args(scheduler)}"
	)
	print(len(model_fname), model_fname)
	os.makedirs(os.path.join(args.dataset_dir, models_dir, model_fname),exist_ok=True)
	model_fpth = os.path.join(args.dataset_dir, models_dir, model_fname)

	train(
		model=model,
		train_data_loader=train_data_loader,
		val_data_loader=val_data_loader,
		optimizer=optimizer,
		scheduler=scheduler,
		checkpoint_interval=5,
		model_dir=model_fpth,
	)

	# # Construct the command as a list of arguments
	# command = [
	# 	'python', 'topk_image_retrieval.py',
	# 	'--query', args.query,
	# 	'--processed_image_path', os.path.join(outputs_dir, f"Top{args.topk}_Q_{re.sub(' ', '-', args.query)}_{args.num_epochs}_epochs.png"),
	# 	'--topk', str(args.topk),
	# 	'--dataset_dir', args.dataset_dir,
	# 	'--image_size', str(args.image_size),
	# 	'--patch_size', str(args.patch_size),
	# 	'--batch_size', str(args.batch_size),
	# 	'--embedding_size', str(args.embedding_size),
	# 	'--num_epochs', str(args.num_epochs),
	# 	'--num_workers', str(args.num_workers),
	# 	'--validation_dataset_share', str(validation_dataset_share),
	# 	'--learning_rate', str(args.learning_rate),
	# 	'--weight_decay', str(args.weight_decay),
	# 	'--document_description_col', args.document_description_col,
	# ]
	# # print("Running command:", ' '.join(command))
	# result = subprocess.run(command, capture_output=True, text=True)
	# print(f"Output:\n{result.stdout}")
	# print(f"Error:\nresult.stderr")

if __name__ == "__main__":
	main()