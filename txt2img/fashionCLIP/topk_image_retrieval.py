from utils import *

from models import *
from dataset_loader import MyntraDataset

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--query', type=str, default="bags", help='Query')
parser.add_argument('--processed_image_path', type=str, default="topkIMG.png", help='Path to resulted image with topk images')
# args_IMG_retrieval = parser.parse_args()
args_IMG_retrieval, unknown = parser.parse_known_args()
print(args_IMG_retrieval)

def img_retrieval(df, val_df, val_loader, query:str="bags", model_fpth: str=f"path/to/models/clip.pt", TOP_K: int=10, resulted_IMGname: str="topk_img.png"):
	print(f"Top-{TOP_K} Image Retrieval for Query: {query}".center(100, "-"))
	args_IMG_retrieval.processed_image_path = resulted_IMGname
	print(f"Saving topK resulted image in: {args_IMG_retrieval.processed_image_path}")
	print(f"val_df: {val_df.shape} | {val_df['subCategory'].value_counts().shape} / {df['subCategory'].value_counts().shape}")
	if query not in val_df['subCategory'].value_counts():
		print(f"Query: {query} Not Found! Search something else! from the list:")
		print(val_df['subCategory'].value_counts())
		return

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
	
	retrieval_model.load_state_dict(torch.load(model_fpth, map_location=device))

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
	print(f"Top-{TOP_K} images from Validation: ({len(val_loader.dataset)}) | Query: '{query}':\n")
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
	img_retrieval()

if __name__ == "__main__":
	main()