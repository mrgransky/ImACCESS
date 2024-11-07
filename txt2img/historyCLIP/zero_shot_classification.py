from utils import *
from models import *
from dataset_loader import HistoricalDataset

# how to run:
# $ python zero_shot_classification.py --dataset_dir $HOME/WS_Farid/ImACCESS/txt2img/datasets/national_archive/NATIONAL_ARCHIVE_1933-01-01_1933-01-02 --num_epochs 1 --batch_size 22

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--processed_image_path', type=str, default="my_img.png", help='Path to resulted image with topk images')
parser.add_argument('--dataset_dir', type=str, default="myntradataset", help='Dataset DIR')
parser.add_argument('--image_size', type=int, default=160, help='Image size')
parser.add_argument('--patch_size', type=int, default=5, help='Patch size')
parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(), help='Number of CPUs [def: max cpus]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--embedding_size', type=int, default=1024, help='Embedding size of Vision & Text encoder [the larger the better]')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--validation_dataset_share', type=float, default=0.3, help='share of Validation set')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay [def: 1e-4]')
parser.add_argument('--visualize', type=bool, default=False, help='Model Validation upon request')

args, unknown = parser.parse_known_args()
# print(args)

# TODO: investigation required!
# if USER == "ubuntu":
# 	args.dataset_dir = ddir

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
val_df_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "val_df.pkl")
img_lbls_dict_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "image_labels_dict.pkl")
img_lbls_list_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "image_labels_list.pkl")

df = load_pickle(fpath=df_fpth)
val_df = load_pickle(fpath=val_df_fpth)
img_lbls_dict = load_pickle(fpath=img_lbls_dict_fpth)
img_lbls_list = load_pickle(fpath=img_lbls_list_fpth)

print(f"Creating Validation Dataloader for {len(val_df)} images", end="\t")
vdl_st = time.time()
val_dataset = HistoricalDataset(
	data_frame=val_df,
	img_sz=args.image_size,
	dataset_directory=os.path.join(args.dataset_dir, "images"),
	max_seq_length=max_seq_length,
)
val_loader = DataLoader(
	dataset=val_dataset, 
	shuffle=False,
	batch_size=args.batch_size,
	num_workers=args.num_workers,
	collate_fn=custom_collate_fn,  # Use custom collate function to handle None values
)
print(f"num_samples[Total]: {len(val_loader.dataset)} Elapsed_t: {time.time()-vdl_st:.5f} sec")
get_info(dataloader=val_loader)

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

def zero_shot_clasification(vidx:int=110, model_fpth:str=f"path/to/models/clip.pt", topk_labels:int=5):
	# vidx = random.randint(0, len(val_df))
	print(f"Zero-shot Classification for vidx: {vidx}")
	model.load_state_dict(torch.load(mdl_fpth, map_location=args.device))
	class_names:List[str] = img_lbls_list

	text = torch.stack([tokenizer(x)[0] for x in class_names]).to(args.device)
	mask = torch.stack([tokenizer(x)[1] for x in class_names])
	mask = mask.repeat(1, len(mask[0])).reshape(len(mask), len(mask[0]), len(mask[0])).to(args.device)

	img = val_dataset[vidx]["image"][None,:]

	if args.visualize:
		plt.imshow(img[0].permute(1, 2, 0)  ,cmap="gray")
		plt.title(tokenizer(val_dataset[vidx]["caption"], encode=False, mask=val_dataset[vidx]["mask"][0], max_seq_length=max_seq_length)[0])
		plt.show()

	img = img.to(args.device)
	with torch.no_grad():
		image_features = model.vision_encoder(img)
		text_features = model.text_encoder(text, mask=mask)

	image_features /= image_features.norm(dim=-1, keepdim=True)
	text_features /= text_features.norm(dim=-1, keepdim=True)
	similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
	values, indices = similarity[0].topk(topk_labels)
	print(
		f'\nTop-{topk_labels} predicted label(s) for IMG: {vidx}: '
		f'Decoded text: {tokenizer(val_dataset[vidx]["caption"], encode=False, mask=val_dataset[vidx]["mask"][0], max_seq_length=max_seq_length)}:\n')
	print(f"{'vIdx':<15}{'Predicted Label':<40}{'Probablity':<30}")
	print("-"*70)
	for value, index in zip(values, indices):
		print(f"{index:<15}{class_names[int(index)]:<40}{value.item():.5f}")

def main():
	zero_shot_clasification(vidx=random.randint(0, len(val_df)))

if __name__ == "__main__":
	main()