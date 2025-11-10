from utils import *

from models import *
from dataset_loader import MyntraDataset

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--query', type=str, default="bags", help='Query')
parser.add_argument('--processed_image_path', type=str, default="my_img.png", help='Path to resulted image with topk images')
parser.add_argument('--dataset_dir', type=str, default="myntradataset", help='Dataset DIR')
parser.add_argument('--topk', type=int, default=5, help='Top-K images')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--validation_dataset_share', type=float, default=0.23, help='share of Validation set')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--product_description_col', type=str, default="subCategory", help='caption col ["articleType", "subCategory", "customized_caption"]')
args, unknown = parser.parse_known_args()
print(args)

if USER == "ubuntu":
	args.dataset_dir = ddir
# models_dir_name = f""

models_dir_name = (
	f"models_"
	+ f"nEpochs_{args.num_epochs}_"
	+ f"batchSZ_{args.batch_size}_"
	+ f"lr_{args.learning_rate}_"
	+ f"val_{args.validation_dataset_share}_"
	+ f"descriptions_{args.product_description_col}"
)

os.makedirs(os.path.join(args.dataset_dir, models_dir_name),exist_ok=True)
mdl_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "model.pt")
df_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "df.pkl")
val_df_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "val_df.pkl")
img_lbls_dict_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "image_labels_dict.pkl")

df = load_pickle(fpath=df_fpth)
val_df = load_pickle(fpath=val_df_fpth)
img_lbls_dict = load_pickle(fpath=img_lbls_dict_fpth)

print(f"Creating Validation Dataloader for {len(val_df)} images", end="\t")
vdl_st = time.time()
val_dataset = MyntraDataset(
	data_frame=val_df,
	captions=img_lbls_dict,
	img_sz=80,
	dataset_directory=os.path.join(args.dataset_dir, "images")
)
val_loader = DataLoader(
	dataset=val_dataset, 
	shuffle=False,
	batch_size=args.batch_size, #32, # double check!!!! 
	num_workers=nw,
	collate_fn=custom_collate_fn  # Use custom collate function to handle None values
)
print(f"num_samples[Total]: {len(val_loader.dataset)} Elapsed_t: {time.time()-vdl_st:.5f} sec")
# get_info(dataloader=val_loader)

model = CLIP(
	emb_dim,
	vit_layers, 
	vit_d_model, 
	img_size,
	patch_size,
	n_channels,
	vit_heads,
	vocab_size,
	max_seq_length,
	text_heads,
	text_layers,
	text_d_model,
	retrieval=False,
).to(device)

retrieval_model = CLIP(
	emb_dim, 
	vit_layers, 
	vit_d_model, 
	img_size,patch_size,
	n_channels,
	vit_heads,
	vocab_size,
	max_seq_length,
	text_heads,
	text_layers,
	text_d_model,
	retrieval=True
).to(device)
print(f"Loading retrieval model...", end="\t")
rm_st = time.time()
retrieval_model.load_state_dict(torch.load(mdl_fpth, map_location=device))
print(f"Elapsed_t: {time.time()-rm_st:.5f} sec")

def img_retrieval(query:str="bags", model_fpth: str=mdl_fpth, TOP_K: int=args.topk, resulted_IMGname: str="topk_img.png"):
	print(f"Top-{TOP_K} Image Retrieval | Query: {query} | user: {USER}".center(100, "-"))
	args.processed_image_path = resulted_IMGname
	print(f"val_df: {val_df.shape} | {val_df['subCategory'].value_counts().shape} / {df['subCategory'].value_counts().shape}")
	if query not in val_df['subCategory'].value_counts():
		print(f"Query: {query} Not Found! Search something else! from the list:")
		print(val_df['subCategory'].value_counts())
		return
	
	query_counts = val_df['subCategory'].value_counts()
	print(query_counts.tail(25))
	plt.figure(figsize=(18, 12))
	query_counts.plot(kind='bar', fontsize=8)
	plt.title(f'Query Frequency (total: {query_counts.shape})')
	plt.xlabel('Query')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig("query_freq.png")

	# Step 1: Encode the text query using your tokenizer and TextEncoder
	query_text, query_mask = tokenizer(query)
	print(type(query_text), type(query_mask))
	query_text = query_text.unsqueeze(0).to(device) # Add batch dimension
	query_mask = query_mask.unsqueeze(0).to(device)

	with torch.no_grad():
		query_features = retrieval_model.text_encoder(query_text, mask=query_mask)
		query_features /= query_features.norm(dim=-1, keepdim=True)

	# Step 2: Encode all images in the dataset and store features
	image_features_list = []
	val_images_paths = []
	val_images_descriptions = []

	with torch.no_grad():
		for batch in val_loader:
			# print(batch)
			images = batch["image"].to(device)
			features = retrieval_model.vision_encoder(images)
			features /= features.norm(dim=-1, keepdim=True)			
			image_features_list.append(features)
			# print(type(batch["image_filepath"]), type(batch.get("caption")))
			val_images_paths.extend(batch["image_filepath"])  # Assuming batch contains image paths or IDs
			val_images_descriptions.extend(batch.get("caption"))
	
	print(f"val_images_paths {type(val_images_paths)} {len(val_images_paths)}")
	print(f"val_images_descriptions: {type(val_images_descriptions)} {len(val_images_descriptions)}")
	# print(val_images_descriptions) # Tensor [0. 10. 11. 19, 2727. ...]

	# Concatenate all image features
	image_features = torch.cat(image_features_list, dim=0)

	# Step 3: Compute similarity using the CLIP model's logic
	# In your CLIP model, this is done using logits and temperature scaling
	similarities = (query_features @ image_features.T) * torch.exp(model.temperature)

	# Apply softmax to the similarities if needed
	similarities = similarities.softmax(dim=-1)
	print(type(similarities), similarities.shape)
	print(similarities)

	# Retrieve topK matches
	top_values, top_indices = similarities.topk(TOP_K)
	print(type(top_values), type(top_indices))
	print(top_values.shape, top_indices.shape, TOP_K)
	print(top_values)
	print(top_indices)

	# Step 4: Retrieve and display (or save) top N images:
	print(f"Saving Top-{TOP_K} / {len(val_loader.dataset)} [val] | Query: {query} in {args.processed_image_path}")
	fig, axes = plt.subplots(1, TOP_K, figsize=(18, 4))  # Adjust figsize as needed
	for ax, value, index in zip(axes, top_values[0], top_indices[0]):
		img_path = val_images_paths[index]
		img_fname = get_img_name_without_suffix(fpth=img_path)
		img_GT = df.loc[df['id'] == img_fname, 'subCategory'].values
		print(f"vidx: {index} | Similarity: {100 * value.item():.6f}% | {img_path} | GT: {img_GT}")
		img = Image.open(img_path).convert("RGB")
		img_title = f"vidx_{index}_sim_{100 * value.item():.2f}%\nGT: {img_GT}"
		ax.set_title(img_title, fontsize=9)
		ax.axis('off')
		ax.imshow(img)
	plt.tight_layout()
	plt.savefig(resulted_IMGname)
	
def main():
	img_retrieval(query=args.query, resulted_IMGname=args.processed_image_path)

if __name__ == "__main__":
	main()