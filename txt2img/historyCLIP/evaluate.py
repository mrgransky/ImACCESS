from utils import *
from models import *
from dataset_loader import HistoricalDataset
from natsort import natsorted
import glob

# how to run [Pouta]:
# With Europeana as validation set:
# $ python evaluate.py -mpth /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1935-01-01_1950-12-31/models/model_augmentation_False_ep_30_train_58361_val_25013_batch_64_img_160_patch_5_emb_1024_cuda3_AdamW_lr_0.001_wd_0.05_OneCycleLR/model.pt -vddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -ps 5 -is 160 -bs 64

# with splited dataset of NA:
# $ nohup python -u evaluate.py -mpth /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1935-01-01_1950-12-31/models/model_augmentation_False_ep_30_train_58361_val_25013_batch_64_img_160_patch_5_emb_1024_cuda3_AdamW_lr_0.001_wd_0.05_OneCycleLR/model.pt -vddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1935-01-01_1950-12-31 -ps 5 -is 160 -bs 64 > /media/volume/trash/ImACCESS/evak.out &

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--model_path', '-mpth', type=str, required=True, help='CLIP model path')
parser.add_argument('--validation_dataset_dir', '-vddir', type=str, required=True, help='Vaidation Dataset Directory')
parser.add_argument('--num_workers', '-nw', type=int, default=multiprocessing.cpu_count(), help='Number of CPUs [def: max cpus]')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--batch_size', '-bs', type=int, default=27, help='Batch Size')
parser.add_argument('--image_size', '-is', type=int, default=150, help='Image size [def: max 160 local]')
parser.add_argument('--patch_size', '-ps', type=int, default=5, help='Patch size')
parser.add_argument('--embedding_size', '-es',type=int, default=1024, help='Embedding size of Vision & Text encoder [the larger the better]')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)
args.device = torch.device(args.device)

def evaluate(model, val_loader, img_lbls_dict, model_fpth: str=f"path/to/models/historyCLIP.pt", device:str="cpu"):
	print(f"Model examination & accuracy Validation: {type(val_loader)} {len(val_loader.dataset)} sample(s)".center(160, "-"))
	validate_start_time = time.time()
	model.load_state_dict(torch.load(model_fpth, map_location=device))

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
			print(type(predicted_label_idx), predicted_label_idx.shape, predicted_label_idx)
			print("#"*180)
			print(img_lbls_dict)
			# print(f"{json.dumps(img_lbls_dict, indent=2, ensure_ascii=False)}")
			print("#"*200)
			for i in predicted_label_idx:
				print(f"{i} {type(i)} int: {int(i)}")
				print(img_lbls_dict.get(int(i)))
				print(
					tokenizer(
						# text=img_lbls_dict[int(i)],
						text=img_lbls_dict.get(int(i)),
						encode=True,
						max_seq_length=max_seq_length
					)[0]
				)
			print("-"*180)
			predicted_label = torch.stack(
				[
					tokenizer(
						text=img_lbls_dict[int(i)],
						encode=True,
						max_seq_length=max_seq_length
					)[0]
					for i in predicted_label_idx
				]
			).to(device) # <class 'torch.Tensor'> torch.Size([32, 256])
			# print(type(predicted_label), predicted_label.shape, )
			correct += int(sum(torch.sum((predicted_label==labels),dim=1)//len(predicted_label[0])))
			total += len(labels)
	acc = correct / total
	print(f'Model Accuracy (Top-1 label): {acc:.3f} ({100 * correct // total} %) Elapsed_t: {time.time()-validate_start_time:.1f} sec'.center(160, "-"))

def main():
	img_rgb_mean_fpth:str = os.path.join(args.validation_dataset_dir, "img_rgb_mean.gz")
	img_rgb_std_fpth:str = os.path.join(args.validation_dataset_dir, "img_rgb_std.gz")
	try:
		img_rgb_mean, img_rgb_std = load_pickle(fpath=img_rgb_mean_fpth), load_pickle(fpath=img_rgb_std_fpth) # RGB images
	except Exception as e:
		print(f"{e}")
		return

	print(f"IMAGE Mean: {img_rgb_mean} | Std: {img_rgb_std}".center(180, " "))
	
	LABELs_dict_fpth:str = os.path.join(args.validation_dataset_dir, "LABELs_dict.gz")
	LABELs_list_fpth:str = os.path.join(args.validation_dataset_dir, "LABELs_list.gz")
	try:
		LABELs_dict = load_pickle(fpath=LABELs_dict_fpth)
		LABELs_list = load_pickle(fpath=LABELs_list_fpth)
	except Exception as e:
		print(f"{e}")
		return
	print(
		f"LABELs_dict {type(LABELs_dict)} {len(LABELs_dict)} "
		f"LABELs_list {type(LABELs_list)} {len(LABELs_list)}"
	)
	print(f"{json.dumps(LABELs_dict, indent=2, ensure_ascii=False)}")
	val_metadata_df_fpth:str = natsorted( glob.glob( args.validation_dataset_dir+'/'+'*_val_df.gz' ) )[0]
	try:
		val_metadata_df = load_pickle(fpath=val_metadata_df_fpth)
	except Exception as e:
		print(f"{e}")
		return

	print(f"Creating Validation Dataloader for {len(val_metadata_df)} samples", end="\t")
	vdl_st = time.time()
	val_dataset = HistoricalDataset(
		data_frame=val_metadata_df,
		img_sz=args.image_size,
		dataset_directory=os.path.join(args.validation_dataset_dir, "images"),
		max_seq_length=max_seq_length,
		mean=img_rgb_mean,
		std=img_rgb_std,
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

	evaluate(
		model=model,
		val_loader=val_data_loader,
		img_lbls_dict=LABELs_dict,
		model_fpth=args.model_path,
		device=args.device,
	)

if __name__ == "__main__":
	main()	