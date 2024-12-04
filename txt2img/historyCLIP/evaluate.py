from utils import *
from models import *
from dataset_loader import HistoricalDataset

# how to run:
# $ python evaluate.py -mdir $HOME/WS_Farid/ImACCESS/txt2img/datasets/national_archive/NATIONAL_ARCHIVE_1933-01-01_1933-01-02/models/model_augmentation_False_ep_1_train_11123_val_4767_batch_22_img_160_patch_5_emb_1024_cuda0_AdamW_lr_0.0005_wd_0.05_OneCycleLR

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--model_dir', '-mdir', type=str, required=True, help='Directory containing the model')
parser.add_argument('--num_workers', '-nw', type=int, default=multiprocessing.cpu_count(), help='Number of CPUs [def: max cpus]')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)
args.device = torch.device(args.device)

def extract_model_info(model_dir):
	# Extract the last part of the path (the model directory name)
	model_info = os.path.basename(model_dir)
	info = {}
	info['augmentation'] = re.search(r'augmentation_(\w+)', model_info).group(1) == 'True'
	info['epochs'] = int(re.search(r'ep_(\d+)', model_info).group(1))
	info['num_training_samples'] = int(re.search(r'train_(\d+)', model_info).group(1))
	info['num_val_samples'] = int(re.search(r'val_(\d+)', model_info).group(1))
	info['batch_size'] = int(re.search(r'batch_(\d+)', model_info).group(1))
	info['image_size'] = int(re.search(r'img_(\d+)', model_info).group(1))
	info['patch_size'] = int(re.search(r'patch_(\d+)', model_info).group(1))
	info['embedding_size'] = int(re.search(r'emb_(\d+)', model_info).group(1))
	info['device'] = re.search(r'(cuda\d+|cpu)', model_info).group(1)
	info['optimizer'] = re.search(r'(AdamW|SGD|Adam)', model_info).group(1)
	info['learning_rate'] = float(re.search(r'lr_(\d+\.\d+)', model_info).group(1))
	info['weight_decay'] = float(re.search(r'wd_(\d+\.\d+)', model_info).group(1))
	info['scheduler'] = re.search(r'(OneCycleLR|StepLR|CosineAnnealingLR)', model_info).group(1)
	return info

def get_dataset_dir():
	parser = argparse.ArgumentParser(description="Extract dataset directory")
	parser.add_argument('--model_dir', type=str, required=True, help='Dataset directory')
	args = parser.parse_args()
	# Split the path at the first occurrence of 'model_'
	dataset_dir = args.model_dir.split('model_')[0].rstrip('/')
	# Expand the $HOME variable if present
	dataset_dir = os.path.expanduser(dataset_dir)
	return dataset_dir

# Use the function
DATASET_DIR = get_dataset_dir()
print(f"DATASET_DIR = {DATASET_DIR}")
model_info = extract_model_info(model_dir=args.model_dir)
print(model_info)

def evaluate(model, val_loader, img_lbls_dict, model_fpth: str=f"path/to/models/historyCLIP.pt", device:str="cpu"):
	print(f"Model examination & accuracy Validation: {type(val_loader)} {len(val_loader.dataset)} sample(s)".center(160, "-"))
	validate_start_time = time.time()
	model.load_state_dict(torch.load(model_fpth, map_location=device))
	# get_model_details(
	# 	model, 
	# 	img_size=(3, args.image_size, args.image_size),
	# 	text_size=(max_seq_length,),
	# )
	text = torch.stack(
		[
			tokenizer(text=txt, encode=True, max_seq_length=max_seq_length)[0] for txt in img_lbls_dict.values()
		]
	).to(device) # <class 'torch.Tensor'> torch.Size([55, 256]) 
	mask = torch.stack(
		[
			tokenizer(text=txt, encode=True, max_seq_length=max_seq_length)[1] for txt in img_lbls_dict.values()
		]
	) # <class 'torch.Tensor'> torch.Size([55, 256, 256])
	mask = mask.repeat(1, len(mask[0])).reshape(len(mask), len(mask[0]), len(mask[0])).to(device)
	print(f"text: {type(text)} {text.shape} ")
	print(f"mask: {type(mask)} {mask.shape} ") # 1D tensor of size max_seq_length

	correct, total = 0,0
	with torch.no_grad():
		for data in val_loader:
			images, labels = data["image"].to(device), data["caption"].to(device)
			
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
			).to(device) # <class 'torch.Tensor'> torch.Size([32, 256])
			# print(type(predicted_label), predicted_label.shape, )
			correct += int(sum(torch.sum((predicted_label==labels),dim=1)//len(predicted_label[0])))
			total += len(labels)
	acc = correct / total
	print(f'Model Accuracy (Top-1 label): {acc:.3f} ({100 * correct // total} %) Elapsed_t: {time.time()-validate_start_time:.1f} sec'.center(160, "-"))

def main():
	img_rgb_mean_fpth:str = os.path.join(DATASET_DIR, "img_rgb_mean.pkl")
	img_rgb_std_fpth:str = os.path.join(DATASET_DIR, "img_rgb_std.pkl")
	try:
		img_rgb_mean, img_rgb_std = load_pickle(fpath=img_rgb_mean_fpth), load_pickle(fpath=img_rgb_std_fpth) # RGB images
	except Exception as e:
		print(f"{e}")
		return

	print(f"IMAGE Mean: {img_rgb_mean} | Std: {img_rgb_std}")
	
	img_lbls_dict_fpth:str = os.path.join(DATASET_DIR, "image_labels_dict.gz")
	# img_lbls_list_fpth:str = os.path.join(DATASET_DIR, "image_labels_list.gz")
	try:
		img_lbls_dict = load_pickle(fpath=img_lbls_dict_fpth)
		# img_lbls_list = load_pickle(fpath=img_lbls_list_fpth)
	except Exception as e:
		print(f"{e}")
		return

	print(
		f"img_lbls_dict {type(img_lbls_dict)} {len(img_lbls_dict)} "
		# f"img_lbls_list {type(img_lbls_list)} {len(img_lbls_list)}"
	)

	val_metadata_df_fpth:str = os.path.join(DATASET_DIR, "val_metadata_df.gz")
	try:
		val_metadata_df = load_pickle(fpath=val_metadata_df_fpth)
	except Exception as e:
		print(f"{e}")
		return

	print(f"Creating Validation Dataloader for {len(val_metadata_df)} samples", end="\t")
	vdl_st = time.time()
	val_dataset = HistoricalDataset(
		data_frame=val_metadata_df,
		img_sz=model_info["image_size"],
		dataset_directory=os.path.join(DATASET_DIR, "images"),
		max_seq_length=max_seq_length,
		mean=img_rgb_mean,
		std=img_rgb_std,
		augment_data=False,
	)
	val_data_loader = DataLoader(
		dataset=val_dataset,
		shuffle=False,
		batch_size=model_info["batch_size"], 
		num_workers=args.num_workers,
		pin_memory=True, # when using CUDA
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(val_data_loader.dataset)} Elapsed_t: {time.time()-vdl_st:.5f} sec")
	get_info(dataloader=val_data_loader)
	model = CLIP(
		emb_dim=model_info["embedding_size"],
		vit_layers=vit_layers,
		vit_d_model=vit_d_model,
		img_size=(model_info["image_size"], model_info["image_size"]),
		patch_size=(model_info["patch_size"], model_info["patch_size"]),
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


	evaluate(
		model=model,
		val_loader=val_data_loader,
		img_lbls_dict=img_lbls_dict,
		model_fpth=os.path.join(args.model_dir, "model.pt"),
		device=args.device,
	)

if __name__ == "__main__":
	main()	