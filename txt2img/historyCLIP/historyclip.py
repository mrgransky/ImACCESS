from utils import *
from models import *
from dataset_loader import HistoricalDataset

# how to run [Local]:
# $ python historyclip.py --query "air base" --dataset_dir $HOME/WS_Farid/ImACCESS/txt2img/datasets/national_archive/NATIONAL_ARCHIVE_1933-01-01_1933-01-02 --num_epochs 1 --num_workers 15
# $ python historyclip.py --query "airbae" --dataset_dir $HOME/WS_Farid/ImACCESS/txt2img/datasets/europeana/europeana_1890-01-01_1960-01-01 --num_epochs 1

# $ nohup python -u historyclip.py --dataset_dir $HOME/WS_Farid/ImACCESS/txt2img/datasets/national_archive/NATIONAL_ARCHIVE_1933-01-01_1933-01-02 --num_epochs 16 --learning_rate 1e-4 --weight_decay 1e-1 --patch_size 5 --image_size 160 --num_workers 11 --batch_size 22 > $PWD/logs/historyCLIP.out &

# how to run [Pouta]:
# Ensure Conda:
# $ conda activate py39
# $ python historyclip.py --dataset_dir /media/volume/ImACCESS/NA_DATASETs/NATIONAL_ARCHIVE_1914-07-28_1945-09-02 --device "cuda:2" --num_epochs 1 --batch_size 128
# $ nohup python -u historyclip.py --dataset_dir /media/volume/ImACCESS/NA_DATASETs/NATIONAL_ARCHIVE_1914-07-28_1945-09-02 --num_epochs 30 --device "cuda:2" --learning_rate 1e-4 --weight_decay 1e-1 --patch_size 5 --image_size 160 --batch_size 128 > /media/volume/trash/ImACCESS/historyCLIP_cuda2.out &

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--topk', type=int, default=5, help='Top-K images')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--image_size', type=int, default=160, help='Image size [def: max 160 local]')
parser.add_argument('--patch_size', type=int, default=5, help='Patch size')
parser.add_argument('--embedding_size', type=int, default=1024, help='Embedding size of Vision & Text encoder [the larger the better]')
parser.add_argument('--query', type=str, default="air base", help='Query')
parser.add_argument('--print_every', type=int, default=100, help='Print loss')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(), help='Number of CPUs [def: max cpus]')
parser.add_argument('--validation_dataset_share', type=float, default=0.3, help='share of Validation set [def: 0.23]')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='small learning rate for better convergence [def: 1e-3]')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay [def: 5e-4]')
parser.add_argument('--examine_model', type=bool, default=True, help='Model Validation upon request')
parser.add_argument('--visualize', type=bool, default=False, help='Model Validation upon request')
parser.add_argument('--document_description_col', type=str, default="query", help='labels')
# parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='Device (cuda or cpu)')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()

print(args)
print(type(args.device), args.device)
print(args.device.type)
os.makedirs(os.path.join(args.dataset_dir, "outputs"), exist_ok=True)
outputs_dir:str = os.path.join(args.dataset_dir, "outputs",)
models_dir_name = (
	f"models"
	+ f"_nEpochs_{args.num_epochs}"
	+ f"_lr_{args.learning_rate}"
	+ f"_wd_{args.weight_decay}"
	+ f"_val_{args.validation_dataset_share}"
	+ f"_batch_size_{args.batch_size}"
	+ f"_image_size_{args.image_size}"
	+ f"_patch_size_{args.patch_size}"
	+ f"_embedding_size_{args.embedding_size}"
	+ f"_device_{re.sub(r':', '', str(args.device))}"
)

os.makedirs(os.path.join(args.dataset_dir, models_dir_name),exist_ok=True)
mdl_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "model.pt")
df_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "df.pkl")
train_df_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "train_df.pkl")
val_df_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "val_df.pkl")
img_lbls_dict_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "image_labels_dict.pkl")
img_lbls_list_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "image_labels_list.pkl")

img_bw_mean_fpth:str = os.path.join(args.dataset_dir, "img_bw_mean.pkl")
img_bw_std_fpth:str = os.path.join(args.dataset_dir, "img_bw_std.pkl")

img_rgb_mean_fpth:str = os.path.join(args.dataset_dir, "img_rgb_mean.pkl")
img_rgb_std_fpth:str = os.path.join(args.dataset_dir, "img_rgb_std.pkl")

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

def get_val_loss(val_df, model, mean, std):
	print(f"Validating val_dataset: {val_df.shape}", end="\t")
	model.eval()
	val_dataset = HistoricalDataset(
		data_frame=val_df,
		img_sz=args.image_size,
		dataset_directory=os.path.join(args.dataset_dir, "images"),
		max_seq_length=max_seq_length,
		mean=mean,
		std=std,
	)
	val_loader = DataLoader(
		dataset=val_dataset,
		shuffle=False,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=True,  # Move data to GPU faster if using CUDA
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)		
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

def examine_model(val_df, class_names, img_lbls_dict, model_fpth: str=f"path/to/models/clip.pt", TOP_K: int=10, mean:List[float]=[0.5, 0.5, 0.5], std:List[float]=[0.5, 0.5, 0.5]):
	print(f"Model Examination & Accuracy".center(160, "-"))
	print(f"Validating {model_fpth} in {args.device}")
	vdl_st = time.time()
	print(f"Creating Validation Dataloader for {len(val_df)} samples", end="\t\t")
	vdl_st = time.time()
	val_dataset = HistoricalDataset(
		data_frame=val_df,
		img_sz=args.image_size,
		dataset_directory=os.path.join(args.dataset_dir, "images"),
		max_seq_length=max_seq_length,
		mean=mean,
		std=std,
	)
	val_loader = DataLoader(
		dataset=val_dataset,
		shuffle=False,
		batch_size=args.batch_size, 
		num_workers=args.num_workers,
		pin_memory=True, # when using CUDA
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(val_loader.dataset)} Elapsed_t: {time.time()-vdl_st:.5f} sec")
	get_info(dataloader=val_loader)	
	model.load_state_dict(torch.load(model_fpth, map_location=args.device))

	# get_model_details(
	# 	model, 
	# 	img_size=(3, args.image_size, args.image_size),
	# 	text_size=(max_seq_length,),
	# )

	text = torch.stack(
		[
			tokenizer(text=txt, encode=True, max_seq_length=max_seq_length)[0] for txt in img_lbls_dict.values()
		]
	).to(args.device) # <class 'torch.Tensor'> torch.Size([55, 256]) 
	mask = torch.stack(
		[
			tokenizer(text=txt, encode=True, max_seq_length=max_seq_length)[1] for txt in img_lbls_dict.values()
		]
	) # <class 'torch.Tensor'> torch.Size([55, 256, 256])
	mask = mask.repeat(1, len(mask[0])).reshape(len(mask), len(mask[0]), len(mask[0])).to(args.device)
	print(f"text: {type(text)} {text.shape} ")
	print(f"mask: {type(mask)} {mask.shape} ") # 1D tensor of size max_seq_length

	correct, total = 0,0
	with torch.no_grad():
		for data in val_loader:
			images, labels = data["image"].to(args.device), data["caption"].to(args.device)
			
			image_features = model.vision_encoder(images)
			text_features = model.text_encoder(text, mask=mask)
			
			image_features /= image_features.norm(dim=-1, keepdim=True)
			text_features /= text_features.norm(dim=-1, keepdim=True)

			similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1) # (batch_size, num_captions)
			# print(type(similarity), similarity.shape, similarity)
			# Compare Predictions with Ground Truth:
			_, predicted_label_idx = torch.max(input=similarity, dim=1) 
			predicted_label = torch.stack(
				[
					tokenizer(img_lbls_dict[int(i)], encode=True, max_seq_length=max_seq_length)[0] for i in predicted_label_idx
				]
			).to(args.device) # <class 'torch.Tensor'> torch.Size([32, 256])
			# print(type(predicted_label), predicted_label.shape, )
			correct += int(sum(torch.sum((predicted_label==labels),dim=1)//len(predicted_label[0])))
			total += len(labels)
	print(f'Model Accuracy (Top1): {100 * correct // total} %'.center(160, "-"))

def train(train_df, val_df, mean:List[float]=[0.5, 0.5, 0.5], std:List[float]=[0.5, 0.5, 0.5], checkpoint_interval:int=5):
	print(f"Training CLIP model using {args.device}[{torch.cuda.get_device_name(args.device)}] & {args.num_workers} CPU(s)".center(150, "-"))
	writer = SummaryWriter(log_dir=os.path.join(outputs_dir, "logs")) # Initialize TensorBoard writer
	
	print(f"Creating Train Dataloader", end="\t")
	tdl_st = time.time()
	train_dataset = HistoricalDataset(
		data_frame=train_df,
		# captions=captions,
		img_sz=args.image_size,
		dataset_directory=os.path.join(args.dataset_dir, "images"),
		max_seq_length=max_seq_length,
		mean=mean,
		std=std,
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
	# visualize_samples(train_data_loader, num_samples=5)
	# sys.exit(-1)

	optimizer = optim.AdamW(
		params=model.parameters(),
		betas=(0.9, 0.98), # Based on original CLIP paper
		eps=1e-6,
		lr=args.learning_rate,
		weight_decay=args.weight_decay, # weight decay (L2 regularization)
	)
	steps = []
	lrs = []

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer, 
		max_lr=args.learning_rate, 
		steps_per_epoch=len(train_data_loader), 
		epochs=args.num_epochs,
		pct_start=0.1, # percentage of the cycle (in number of steps) spent increasing the learning rate
		anneal_strategy='cos', # cos/linear annealing
	)

	total_params = 0
	total_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
	print(f"Total trainable parameters (Vision + Text) Encoder: {total_params} ~ {total_params/int(1e+6):.2f} M")
	best_loss = np.inf
	patience = 5  # Number of epochs to wait for improvement before stopping
	no_improvement_count = 0

	# Check if there is a checkpoint to load
	start_epoch = 0
	checkpoint_dir = os.path.join(args.dataset_dir, models_dir_name, "checkpoints")
	os.makedirs(checkpoint_dir, exist_ok=True)
	checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
	if os.path.exists(checkpoint_path):
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		start_epoch = checkpoint['epoch'] + 1
		best_loss = checkpoint['best_loss']
		no_improvement_count = checkpoint['no_improvement_count']
		print(f"Resuming training from epoch {start_epoch}...")
				
	print(f"Training {args.num_epochs} Epoch(s) in {args.device}".center(100, "-"))
	training_st = time.time()
	
	average_train_losses = list()
	average_val_losses = list()

	# Add loss scaling for better numerical stability
	scaler = torch.amp.GradScaler(
		device=args.device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)

	for epoch in range(args.num_epochs):
		print(f"Epoch [{epoch+1}/{args.num_epochs}]")
		epoch_loss = 0.0  # To accumulate the loss over the epoch
		model.train()
		for batch_idx, data in enumerate(train_data_loader):
			img = data["image"].to(args.device) 
			cap = data["caption"].to(args.device)
			mask = data["mask"].to(args.device)

			optimizer.zero_grad()

			# # Convential backpropagation
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
				print(f"\tBatch [{batch_idx + 1}/{len(train_data_loader)}] Loss: {loss.item():.5f}")

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
		avg_val_loss = get_val_loss(val_df, model, mean, std)
		writer.add_scalar('Loss/validation', avg_val_loss, epoch)
		average_val_losses.append(avg_val_loss)

		if avg_val_loss < best_loss:
			best_loss = avg_val_loss
			torch.save(model.state_dict(), mdl_fpth)
			print(f"Saving model in {mdl_fpth} for best avg loss: {best_loss:.5f}")
			no_improvement_count = 0
		else:
			no_improvement_count += 1
			if no_improvement_count >= patience:
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
			torch.save(checkpoint, checkpoint_path)
			print(f"Checkpoint saved at epoch {epoch+1} : {checkpoint_path}")

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
		fpath=os.path.join(outputs_dir, lrs_vs_steps_fname),
	)

	loss_fname = (
		f'loss'
		+ f'_epochs_{args.num_epochs}'
		+ f'_lr_{args.learning_rate}'
		+ f'_wd_{args.weight_decay}'
		+ f'.png'
	)
	plot_loss(
		losses=average_train_losses, 
		num_epochs=args.num_epochs, 
		save_path=os.path.join(outputs_dir, loss_fname),
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
		num_epochs=args.num_epochs, 
		save_path=os.path.join(outputs_dir, losses_fname),
	)
	print(f"Elapsed_t: {time.time()-training_st:.5f} sec".center(150, "-"))
	writer.close()

def main():
	set_seeds()
	###################################################### BW images ######################################################
	# try:
	# 	img_bw_mean, img_bw_std = load_pickle(fpath=img_bw_mean_fpth), load_pickle(fpath=img_bw_std_fpth)
	# except Exception as e:
	# 	print(f"{e}")
	# 	img_bw_mean, img_bw_std = get_mean_std_grayscale_img_multiprocessing(dir=os.path.join(args.dataset_dir, "images"))
	# 	save_pickle(pkl=img_bw_mean, fname=img_bw_mean_fpth)
	# 	save_pickle(pkl=img_bw_std, fname=img_bw_std_fpth)
	# print(f"Grayscale: Mean: {img_bw_mean} | Std: {img_bw_std}")
	###################################################### BW images ######################################################

	try:
		img_rgb_mean, img_rgb_std = load_pickle(fpath=img_rgb_mean_fpth), load_pickle(fpath=img_rgb_std_fpth) # RGB images
	except Exception as e:
		print(f"{e}")
		img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(dir=os.path.join(args.dataset_dir, "images"), num_workers=args.num_workers)
		save_pickle(pkl=img_rgb_mean, fname=img_rgb_mean_fpth)
		save_pickle(pkl=img_rgb_std, fname=img_rgb_std_fpth)
	print(f"RGB: Mean: {img_rgb_mean} | Std: {img_rgb_std}")
	
	try:
		df = load_pickle(fpath=df_fpth)
		train_df = load_pickle(fpath=train_df_fpth)
		val_df = load_pickle(fpath=val_df_fpth)
		img_lbls_dict = load_pickle(fpath=img_lbls_dict_fpth)
		img_lbls_list = load_pickle(fpath=img_lbls_list_fpth)
	except Exception as e:
		print(f"{e}")
		df = get_dframe(
			fpth=os.path.join(args.dataset_dir, "metadata.csv"), 
			img_dir=os.path.join(args.dataset_dir, "images"), 
		)
		# Split the dataset: training and validation sets
		# TODO: Train: National Archive, Validation: Europeana
		img_lbls_dict, img_lbls_list = get_doc_description(df=df, col=args.document_description_col)
		train_df, val_df = train_test_split(
			df, 
			shuffle=True, 
			test_size=args.validation_dataset_share, # 0.05
			random_state=42,
		)

		train_df.to_csv(os.path.join(args.dataset_dir, models_dir_name, "metadata_train.csv"), index=False)
		try:
			train_df.to_excel(os.path.join(args.dataset_dir, models_dir_name, "metadata_train.xlsx"), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")

		val_df.to_csv(os.path.join(args.dataset_dir, models_dir_name, "metadata_val.csv"), index=False)
		try:
			val_df.to_excel(os.path.join(args.dataset_dir, models_dir_name, "metadata_val.xlsx"), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")

		save_pickle(pkl=df, fname=df_fpth,)
		save_pickle(pkl=train_df, fname=train_df_fpth,)
		save_pickle(pkl=val_df, fname=val_df_fpth,)
		save_pickle(pkl=img_lbls_dict, fname=img_lbls_dict_fpth,)
		save_pickle(pkl=img_lbls_list, fname=img_lbls_list_fpth,)
		query_counts_val = val_df[args.document_description_col].value_counts()
		# print(query_counts_val.tail(25))
		plt.figure(figsize=(23, 15))
		query_counts_val.plot(kind='bar', fontsize=8)
		plt.title(f'Validation Query Frequency (total: {query_counts_val.shape})')
		plt.xlabel('Query')
		plt.ylabel('Frequency')
		plt.tight_layout()
		plt.savefig(os.path.join(args.dataset_dir, "outputs", f"query_freq_{query_counts_val.shape[0]}_val.png"))

	# Print the sizes of the datasets
	print(f"df: {df.shape} train_df: {train_df.shape} val_df({args.validation_dataset_share}): {val_df.shape}")
	print(
		f"img_lbls_dict {type(img_lbls_dict)} {len(img_lbls_dict)} "
		f"img_lbls_list {type(img_lbls_list)} {len(img_lbls_list)}"
	)
	# return
	if not os.path.exists(mdl_fpth):
		train(
			train_df=train_df,
			val_df=val_df,
			mean=img_rgb_mean,
			std=img_rgb_std,
			checkpoint_interval=2,
		)

	# print(f"Creating Validation Dataloader for {len(val_df)} images", end="\t")
	# vdl_st = time.time()
	# val_dataset = HistoricalDataset(
	# 	data_frame=val_df,
	# 	captions=img_lbls_dict,
	# 	img_sz=args.image_size,
	# 	dataset_directory=os.path.join(args.dataset_dir, "images")
	# )
	# val_loader = DataLoader(
	# 	dataset=val_dataset, 
	# 	shuffle=False,
	# 	batch_size=args.batch_size, #32, # double check!!!! 
	# 	num_workers=args.num_workers,
	# 	collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	# )
	# print(f"num_samples[Total]: {len(val_loader.dataset)} Elapsed_t: {time.time()-vdl_st:.5f} sec")
	# get_info(dataloader=val_loader)

	if args.examine_model:
		examine_model(
			val_df=val_df,
			class_names=img_lbls_list,
			img_lbls_dict=img_lbls_dict,
			model_fpth=mdl_fpth,
			TOP_K=args.topk,
			mean=img_rgb_mean,
			std=img_rgb_std,
		)
				
	# Construct the command as a list of arguments
	command = [
		'python', 'topk_image_retrieval.py',
		'--query', args.query,
		'--processed_image_path', os.path.join(outputs_dir, f"Top{args.topk}_Q_{re.sub(' ', '-', args.query)}_{args.num_epochs}_epochs.png"),
		'--topk', str(args.topk),
		'--dataset_dir', args.dataset_dir,
		'--image_size', str(args.image_size),
		'--patch_size', str(args.patch_size),
		'--batch_size', str(args.batch_size),
		'--embedding_size', str(args.embedding_size),
		'--num_epochs', str(args.num_epochs),
		'--num_workers', str(args.num_workers),
		'--validation_dataset_share', str(args.validation_dataset_share),
		'--learning_rate', str(args.learning_rate),
		'--weight_decay', str(args.weight_decay),
		'--document_description_col', args.document_description_col,
	]

	# Print the command for debugging purposes
	# print("Running command:", ' '.join(command))

	# Execute the command
	result = subprocess.run(command, capture_output=True, text=True)

	# Print the output and error (if any)
	print(f"Output:\n{result.stdout}")
	# print("Error:", result.stderr)

if __name__ == "__main__":
	main()