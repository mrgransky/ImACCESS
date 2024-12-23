from utils import *
from models import *
from dataset_loader import HistoryDataset

# how to run [Local]:
# $ python finetune.py -ddir $HOME/WS_Farid/ImACCESS/txt2img/datasets/national_archive/NATIONAL_ARCHIVE_1933-01-01_1933-01-02 -vddir $HOME/WS_Farid/ImACCESS/txt2img/datasets/europeana/EUROPEANA_1900-01-01_1970-12-31 -nep 1 -lr 5e-4 -wd 5e-2
# $ python finetune.py -ddir $HOME/WS_Farid/ImACCESS/txt2img/datasets/europeana/EUROPEANA_1900-01-01_1970-12-31 -nep 1

# $ nohup python -u finetune.py -ddir $HOME/WS_Farid/ImACCESS/txt2img/datasets/national_archive/NATIONAL_ARCHIVE_1933-01-01_1933-01-02 -nep 12 -lr 5e-4 -wd 2e-2 -ps 5 -is 160 > $PWD/logs/historyCLIP.out &

# how to run [Pouta]:
# Ensure Conda:
# $ conda activate py39
# $ python finetune.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1935-01-01_1950-12-31 --device "cuda:2" -nep 1 -bs 128

# With Europeana as validation set:
# $ nohup python -u finetune.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1935-01-01_1950-12-31 -vddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -nep 13 --device "cuda:2" -lr 5e-4 -wd 5e-2 -ps 5 -is 160 -bs 64 > /media/volume/trash/ImACCESS/historyCLIP_finetune_NA_val_EUROPEANA_cuda2.out &

# with splited dataset of NA:
# $ nohup python -u finetune.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1935-01-01_1950-12-31 -nep 40 --device "cuda:1" -lr 1e-4 -wd 5e-2 -ps 5 -is 160 -bs 64 -nw 40 > /media/volume/trash/ImACCESS/historyCLIP_finetune_NA_val_NA_cuda1.out &

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--validation_dataset_dir', '-vddir', default=None, help='Dataset DIR')
parser.add_argument('--topk', type=int, default=5, help='Top-K images')
parser.add_argument('--batch_size', '-bs', type=int, default=80, help='Batch Size')
parser.add_argument('--image_size', '-is', type=int, default=150, help='Image size [def: max 160 local]')
parser.add_argument('--patch_size', '-ps', type=int, default=5, help='Patch size')
parser.add_argument('--embedding_size', '-es',type=int, default=1024, help='Embedding size of Vision & Text encoder [the larger the better]')
parser.add_argument('--print_every', type=int, default=100, help='Print loss')
parser.add_argument('--num_epochs', '-nep', type=int, default=10, help='Number of epochs')
parser.add_argument('--num_workers', '-nw', type=int, default=10, help='Number of CPUs [def: max cpus]')
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4, help='small learning rate for better convergence [def: 1e-3]')
parser.add_argument('--weight_decay', '-wd', type=float, default=5e-2, help='Weight decay [def: 5e-4]')
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

def convert_models_to_fp32(model): 
	for p in model.parameters(): 
		p.data = p.data.float()
		p.grad.data = p.grad.data.float()
				
def visualize_(dataloader, num_samples=5, ):
	for batch_idx, (batch_imgs, batch_lbls) in enumerate(dataloader):
		print(batch_idx, batch_imgs.shape, batch_lbls.shape, len(batch_imgs), len(batch_lbls)) # torch.Size([32, 3, 224, 224]) torch.Size([32])
		if batch_idx >= num_samples:
			break
		
		image = batch_imgs[batch_idx].permute(1, 2, 0).numpy() # Convert tensor to numpy array and permute dimensions
		caption_idx = batch_lbls[batch_idx]
		print(image.shape, caption_idx)
		print()
			
		# # Denormalize the image
		image = image * np.array([0.2268645167350769]) + np.array([0.6929051876068115])
		image = np.clip(image, 0, 1)  # Ensure pixel values are in [0, 1] range
		
		plt.figure(figsize=(13, 7))
		plt.imshow(image)
		plt.title(f"Caption {caption_idx.shape}\n{caption_idx}", fontsize=5)
		plt.axis('off')
		plt.show()

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

# def finetune(model, finetune_data_loader, val_data_loader, optimizer, scheduler, checkpoint_interval:int=5, model_dir:str="path/2/model_dir", early_stopping_patience:int=10)
def finetune(model, finetune_data_loader, val_data_loader, model_dir:str="path/2/model_dir"):
	mdl_fpth:str = os.path.join(args.dataset_dir, model_dir, "model.pt")
	
	os.makedirs(os.path.join(args.dataset_dir, model_dir, "results"), exist_ok=True)
	results_dir:str = os.path.join(args.dataset_dir, model_dir, "results")

	os.makedirs(os.path.join(args.dataset_dir, model_dir, "checkpoints"), exist_ok=True)
	checkpoint_dir = os.path.join(args.dataset_dir, model_dir, "checkpoints")
	
	print(f"Fine-Tuning CLIP model {args.num_epochs} Epoch(s) {args.device} & {args.num_workers} CPU(s)".center(160, "-"))
	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(args.device)}".center(160, " "))
	log_gpu_memory(device=args.device)

	total_params = 0
	total_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
	print(f"Total finetuneable parameters (Vision + Text) Encoder: {total_params} ~ {total_params/int(1e+6):.2f} M")

def main():
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
			validation_dataset_share = 0.03
			val_img_rgb_mean = img_rgb_mean
			val_img_rgb_std = img_rgb_std

	print(f"[Train] Mean: {img_rgb_mean} Std: {img_rgb_std}".center(180, " "))
	print(f"[Validation] Mean: {val_img_rgb_mean} Std: {val_img_rgb_std}".center(180, " "))

	print(f"Train & Validation metadata df".center(180, "-"))
	finetune_metadata_df, val_metadata_df = get_train_val_metadata_df(
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
		LABELs_dict, LABELs_list = get_doc_description(df=finetune_metadata_df, col='label') # must be "label", regardless of captions
		save_pickle(pkl=LABELs_dict, fname=LABELs_dict_fpth)
		save_pickle(pkl=LABELs_list, fname=LABELs_list_fpth)

	print(
		f"LABELs_dict {type(LABELs_dict)} {len(LABELs_dict)} "
		f"LABELs_list {type(LABELs_list)} {len(LABELs_list)}"
	)
	# copy to validation dataset dir if not none:
	if args.validation_dataset_dir:
		print(f">> Copying (LABELs) into {args.validation_dataset_dir}")
		# copy
		shutil.copy(LABELs_dict_fpth, args.validation_dataset_dir)
		shutil.copy(LABELs_list_fpth, args.validation_dataset_dir)
	# return
	print(f">> OpenAI original CLIP model + preprocessing...")
	mst = time.time()
	# OpenAI CLIP model and preprocessing
	clip_model, clip_preprocessor = clip.load("ViT-B/32", jit=False)
	clip_model.to(args.device)
	print(f"Elaped_t: {time.time()-mst:.2f}")

	print(f"Creating Train Dataloader for {len(finetune_metadata_df)} samples", end="\t")
	tdl_st = time.time()
	finetune_dataset = HistoryDataset(
		data_frame=finetune_metadata_df,
		dataset_directory=os.path.join(args.dataset_dir, "images"),
		mean=img_rgb_mean,
		std=img_rgb_std,
		transformer=clip_preprocessor,
	)
	finetune_data_loader = DataLoader(
		dataset=finetune_dataset,
		shuffle=True,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=True,  # Move data to GPU faster if using CUDA
		persistent_workers=True if args.num_workers > 1 else False,  # Keep workers alive if memory allows
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(finetune_data_loader.dataset)} Elapsed_t: {time.time()-tdl_st:.5f} sec")
	get_info(dataloader=finetune_data_loader)

	###################### Visualize Samples ######################
	visualize_(
		dataloader=finetune_data_loader, 
		num_samples=5,
	)
	sys.exit(-1)
	###################### Visualize Samples ######################

	print(f"Creating Validation Dataloader for {len(val_metadata_df)} samples", end="\t")
	vdl_st = time.time()
	val_dataset = HistoryDataset(
		data_frame=val_metadata_df,
		dataset_directory=os.path.join(args.validation_dataset_dir, "images") if args.validation_dataset_dir else os.path.join(args.dataset_dir, "images"),
		mean=val_img_rgb_mean,
		std=val_img_rgb_std,
		transformer=clip_preprocessor,
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

	# model_ft = CLIPFineTuner(model, num_classes).to(device)
	# print(f"Defining Optimizer...")

	# optimizer = optim.AdamW(
	# 	params=model.parameters(),
	# 	betas=(0.9, 0.98), # Based on original CLIP paper
	# 	eps=1e-8,
	# 	lr=args.learning_rate,
	# 	weight_decay=args.weight_decay, # weight decay (L2 regularization)
	# )

	# # print(f"Defining Scheduler...")
	# scheduler = torch.optim.lr_scheduler.OneCycleLR(
	# 	optimizer=optimizer, 
	# 	max_lr=args.learning_rate, 
	# 	steps_per_epoch=len(finetune_data_loader), 
	# 	epochs=args.num_epochs,
	# 	pct_start=0.1, # percentage of the cycle (in number of steps) spent increasing the learning rate
	# 	anneal_strategy='cos', # cos/linear annealing
	# )

	model_fname = (
		f"finetuned_model"
		# + f"_augmentation_{args.data_augmentation}"
		+ f"_ep_{args.num_epochs}"
		+ f"_finetune_{len(finetune_data_loader.dataset)}"
		+ f"_val_{len(val_data_loader.dataset)}"
		+ f"_batch_{args.batch_size}"
		# + f"_img_{args.image_size}"
		# + f"_patch_{args.patch_size}"
		# + f"_emb_{args.embedding_size}"
		+ f"_{re.sub(r':', '', str(args.device))}"
		# + f"_{optimizer.__class__.__name__}"
		+ f"_lr_{args.learning_rate}"
		+ f"_wd_{args.weight_decay}"
		# + f"_{get_args(optimizer)}"
		# + f"_{scheduler.__class__.__name__}"
		# + f"_{get_args(scheduler)}"
	)
	# print(len(model_fname), model_fname)
	os.makedirs(os.path.join(args.dataset_dir, models_dir, model_fname),exist_ok=True)
	model_fpth = os.path.join(args.dataset_dir, models_dir, model_fname)

	finetune(
		model=clip_model,
		finetune_data_loader=finetune_data_loader,
		val_data_loader=val_data_loader,
		# # optimizer=optimizer,
		# # scheduler=scheduler,
		# checkpoint_interval=5,
		model_dir=model_fpth,
	)

	# command = [
	# 	'python', 'evaluate.py',
	# 	'--model_path', os.path.join(args.dataset_dir, models_dir, model_fname, "model.pt"),
	# 	'--validation_dataset_dir', args.validation_dataset_dir if args.validation_dataset_dir else args.dataset_dir,
	# 	'--image_size', str(args.image_size),
	# 	'--patch_size', str(args.patch_size),
	# 	'--batch_size', str(args.batch_size),
	# 	'--embedding_size', str(args.embedding_size),
	# 	'--num_workers', str(args.num_workers),
	# 	'--device', str(args.device),
	# ]
	# # print("Running command:", ' '.join(command))
	# result = subprocess.run(command, capture_output=True, text=True)
	# print(f"Output:\n{result.stdout}")
	# print(f"Error:\nresult.stderr")


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