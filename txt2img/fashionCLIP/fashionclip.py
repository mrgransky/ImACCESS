from utils import *
from models import *
from dataset_loader import MyntraDataset
from topk_image_retrieval import img_retrieval

# how to run [Local]:
# $ python fashionclip.py --query tie
# $ python fashionclip.py --query tie --dataset_dir myntradataset --num_epochs 1
# $ nohup python -u fashionclip.py --num_epochs 100 > $HOME/datasets/trash/logs/fashionclip.out & 

# how to run [Pouta]:
# $ python fashionclip.py --dataset_dir /media/volume/ImACCESS/myntradataset --num_epochs 1
# $ nohup python -u --dataset_dir /media/volume/ImACCESS/myntradataset --num_epochs 3 --query "topwear" > /media/volume/ImACCESS/trash/logs/fashionclip.out & 

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--query', type=str, default="bags", help='Query')
parser.add_argument('--topk', type=int, default=5, help='Top-K images')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--validation_dataset_share', type=float, default=0.23, help='share of Validation set')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--validate', type=bool, default=True, help='Model Validation upon request')
parser.add_argument('--product_description_col', type=str, default="subCategory", help='caption col ["articleType", "subCategory", "customized_caption"]')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)

os.makedirs(os.path.join(args.dataset_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(args.dataset_dir, "outputs"), exist_ok=True)

mdl_fpth:str = os.path.join(
	args.dataset_dir, 
	"models",
	f"fashionclip_nEpochs_{args.num_epochs}_"
	f"batchSZ_{args.batch_size}_"
	f"lr_{args.learning_rate}_"
	f"val_{args.validation_dataset_share}_"
	f"descriptions_{args.product_description_col}.pt",
)
outputs_dir:str = os.path.join(
	args.dataset_dir, 
	"outputs",
)

def validate(val_df, class_names, CAPTIONSs, model_fpth: str=f"path/to/models/clip.pt", TOP_K: int=10):
	print(f"Validation {model_fpth} using {device}".center(100, "-"))
	vdl_st = time.time()
	print(f"Creating Validation Dataloader for {len(val_df)} samples", end="\t")
	vdl_st = time.time()
	val_dataset = MyntraDataset(
		data_frame=val_df,
		captions=CAPTIONSs,
		img_sz=80,
		dataset_directory=os.path.join(args.dataset_dir, "images")
	)
	val_loader = DataLoader(
		dataset=val_dataset, 
		shuffle=False,
		batch_size=args.batch_size, 
		num_workers=nw,
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(val_loader.dataset)} Elapsed_t: {time.time()-vdl_st:.5f} sec")
	# get_info(dataloader=val_loader)

	# Loading Best Model
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
	
	model.load_state_dict(torch.load(model_fpth, map_location=device))
	text = torch.stack([tokenizer(x)[0] for x in val_dataset.captions.values()]).to(device)
	mask = torch.stack([tokenizer(x)[1] for x in val_dataset.captions.values()])
	mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)
	correct, total = 0,0

	with torch.no_grad():
		for data in val_loader:				
			images, labels = data["image"].to(device), data["caption"].to(device)
			image_features = model.vision_encoder(images)
			text_features = model.text_encoder(text, mask=mask)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			text_features /= text_features.norm(dim=-1, keepdim=True)
			similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
			_, indices = torch.max(similarity,1)
			pred = torch.stack([tokenizer(val_dataset.captions[int(i)])[0] for i in indices]).to(device)
			correct += int(sum(torch.sum((pred==labels),dim=1)//len(pred[0])))
			total += len(labels)

	print(f'\nModel Accuracy: {100 * correct // total} %')
	text = torch.stack([tokenizer(x)[0] for x in class_names]).to(device)
	mask = torch.stack([tokenizer(x)[1] for x in class_names])
	mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)
	idx = 19
	# idx = random.randint(0, len(val_df))
	img = val_dataset[idx]["image"][None,:]

	if visualize:
		plt.imshow(img[0].permute(1, 2, 0)  ,cmap="gray")
		plt.title(tokenizer(val_dataset[idx]["caption"], encode=False, mask=val_dataset[idx]["mask"][0])[0])
		plt.show()

	img = img.to(device)
	with torch.no_grad():
		image_features = model.vision_encoder(img)
		text_features = model.text_encoder(text, mask=mask)

	image_features /= image_features.norm(dim=-1, keepdim=True)
	text_features /= text_features.norm(dim=-1, keepdim=True)
	similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
	values, indices = similarity[0].topk(TOP_K)

	# Print the result
	print(f'\nTop-{TOP_K} Prediction(s) for IMG: {idx} {tokenizer(val_dataset[idx]["caption"], encode=False, mask=val_dataset[idx]["mask"][0])}:\n')
	for value, index in zip(values, indices):
		print(f"index: {index}: {class_names[int(index)]:>30s}: {100 * value.item():.3f}%")
	print(f"Elapsed_t: {time.time()-vdl_st:.2f} sec")

def fine_tune(train_df,captions):
	print(f"Fine-tuning in {torch.cuda.get_device_name(device)} using {nw} CPU(s)".center(150, "-"))
	print(f"Creating Train Dataloader", end="\t")
	tdl_st = time.time()
	train_dataset = MyntraDataset(
		data_frame=train_df,
		captions=captions, 
		img_sz=80,
		dataset_directory=os.path.join(args.dataset_dir, "images")
	)
	train_loader = DataLoader(
		dataset=train_dataset,
		shuffle=True,
		batch_size=args.batch_size,
		num_workers=nw,
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(train_loader.dataset)} Elapsed_t: {time.time()-tdl_st:.5f} sec")
	# get_info(dataloader=train_loader)
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
	optimizer = optim.AdamW(
		params=model.parameters(), 
		lr=args.learning_rate, 
		weight_decay=wd, # weight decay (L2 regularization)
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
	total_params = 0
	total_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
	print(f"Total trainable parameters: {total_params} ~ {total_params/int(1e+6):.2f} M")
	best_loss = np.inf
	print(f"Training {args.num_epochs} Epoch(s) in {device}".center(100, "-"))
	training_st = time.time()
	average_losses = list()
	for epoch in range(args.num_epochs):
		print(f"Epoch [{epoch+1}/{args.num_epochs}]")
		epoch_loss = 0.0  # To accumulate the loss over the epoch
		for batch_idx, data in enumerate(train_loader):
			img = data["image"].to(device) 
			cap = data["caption"].to(device)
			mask = data["mask"].to(device)
			optimizer.zero_grad()
			loss = model(img, cap, mask)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			if batch_idx % 200 == 0:
				print(f"\tBatch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.5f}")
			epoch_loss += loss.item()
		avg_loss = epoch_loss / len(train_loader)
		scheduler.step(avg_loss)
		print(f"Average Loss: {avg_loss:.5f} @ Epoch: {epoch+1}")
		average_losses.append(avg_loss)
		if avg_loss <= best_loss:
			best_loss = avg_loss
			torch.save(model.state_dict(), mdl_fpth)
			print(f"Saving model in {mdl_fpth} for best avg loss: {best_loss:.5f}")
	print(f"Elapsed_t: {time.time()-training_st:.5f} sec")
	loss_fname = (
		f'loss'
		+ f'epochs_{args.num_epochs}_'
		+ f'lr_{args.learning_rate}'
		+ f'.png'
	)
	plot_loss(
		losses=average_losses, 
		num_epochs=args.num_epochs, 
		save_path=os.path.join(outputs_dir, loss_fname),)

def main():
	set_seeds()
	df = get_dframe(
		fpth=os.path.join(args.dataset_dir, "styles.csv"), 
		img_dir=os.path.join(args.dataset_dir, "images"), 
	)
	# sys.exit()

	# Split the dataset into training and validation sets
	train_df, val_df = train_test_split(
		df, 
		shuffle=True, 
		test_size=args.validation_dataset_share, # 0.05
		random_state=42,
	)

	# Print the sizes of the datasets
	print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

	captions, class_names = get_product_description(df=df, col=args.product_description_col)
	# sys.exit()

	if not os.path.exists(mdl_fpth):
		fine_tune(train_df=train_df, captions=captions)

	print(f"Creating Validation Dataloader for {len(val_df)} images", end="\t")
	vdl_st = time.time()
	val_dataset = MyntraDataset(
		data_frame=val_df,
		captions=captions,
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
	# return

	if args.validate:
		validate(
			val_df=val_df,
			class_names=class_names,
			CAPTIONSs=captions,
			model_fpth=mdl_fpth,
			TOP_K=args.topk,
		)

	img_retrieval(
		df=df,
		val_df=val_df,
		val_loader=val_loader,
		query=args.query,
		model_fpth=mdl_fpth,
		TOP_K=args.topk,
		resulted_IMGname=os.path.join(outputs_dir, f"Top_{args.topk}_imgs_Q_{re.sub(' ', '-', args.query)}_{args.num_epochs}_epochs.png"),
	)

if __name__ == "__main__":
	main()