from utils import *
from models import *
from dataset_loader import HistoricalDataset

# how to run [Local]:
# $ python historyclip.py --query sailboat --dataset_dir $HOME/WS_Farid/ImACCESS/txt2img/datasets/national_archive/NATIONAL_ARCHIVE_1933-01-01_1933-01-02 --num_epochs 20
# $ python historyclip.py --query sailboat --dataset_dir $HOME/WS_Farid/ImACCESS/txt2img/datasets/europeana/europeana_1890-01-01_1960-01-01 --num_epochs 20

# $ nohup python -u historyclip.py --dataset_dir $HOME/WS_Farid/ImACCESS/txt2img/datasets/national_archive/NATIONAL_ARCHIVE_1933-01-01_1933-01-02 --num_epochs 20 --patch_size 5 --image_size 170 >> $PWD/logs/historyclip.out & 

# how to run [Pouta]:
# $ python historyclip.py --dataset_dir /media/volume/ImACCESS/national_archive --num_epochs 1
# $ nohup python -u historyclip.py --dataset_dir /media/volume/ImACCESS/NA_DATASETs/NATIONAL_ARCHIVE_1914-07-28_1945-09-02 --num_epochs 20 --patch_size 5 --image_size 170 --query "dam construction" >> /media/volume/trash/ImACCESS/historyclip.out &

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--topk', type=int, default=5, help='Top-K images')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--image_size', type=int, default=150, help='Image size')
parser.add_argument('--patch_size', type=int, default=5, help='Patch size')
parser.add_argument('--query', type=str, default="aircraft", help='Query')
parser.add_argument('--print_every', type=int, default=100, help='Print loss')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--validation_dataset_share', type=float, default=0.3, help='share of Validation set [def: 0.23]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='small learning rate for better convergence [def: 1e-3]')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay [def: 1e-4]')
parser.add_argument('--validate', type=bool, default=True, help='Model Validation upon request')
parser.add_argument('--visualize', type=bool, default=False, help='Model Validation upon request')
parser.add_argument('--document_description_col', type=str, default="query", help='labels')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
os.makedirs(os.path.join(args.dataset_dir, "outputs"), exist_ok=True)
outputs_dir:str = os.path.join(args.dataset_dir, "outputs",)
wd = 5e-4  # Stronger regularization to prevent overfitting
models_dir_name = (
	f"models"
	+ f"_nEpochs_{args.num_epochs}"
	+ f"_lr_{args.learning_rate}"
	+ f"_val_{args.validation_dataset_share}"
	+ f"_descriptions_{args.document_description_col}"
	+ f"_batch_size_{args.batch_size}"
	+ f"_image_size_{args.image_size}"
	+ f"_patch_size_{args.patch_size}"
	+ f"_wd_{args.weight_decay}"
)
os.makedirs(os.path.join(args.dataset_dir, models_dir_name),exist_ok=True)
mdl_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "model.pt")
df_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "df.pkl")
train_df_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "train_df.pkl")
val_df_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "val_df.pkl")
img_lbls_dict_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "image_labels_dict.pkl")
img_lbls_list_fpth:str = os.path.join(args.dataset_dir, models_dir_name, "img_labels_list.pkl")

def validate(val_df, class_names, CAPTIONSs, model_fpth: str=f"path/to/models/clip.pt", TOP_K: int=10):
	print(f"Validating {model_fpth} in {device}".center(160, "-"))
	vdl_st = time.time()
	print(f"Creating Validation Dataloader for {len(val_df)} samples", end="\t")
	vdl_st = time.time()
	val_dataset = HistoricalDataset(
		data_frame=val_df,
		captions=CAPTIONSs,
		img_sz=args.image_size,
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
		(args.image_size, args.image_size),
		(args.patch_size, args.patch_size),
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
	idx = 1101
	# idx = random.randint(0, len(val_df))
	img = val_dataset[idx]["image"][None,:]

	if args.visualize:
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

	query_counts = val_df['query'].value_counts()
	# print(query_counts.tail(25))
	plt.figure(figsize=(23, 15))
	query_counts.plot(kind='bar', fontsize=8)
	plt.title(f'Validation Query Frequency (total: {query_counts.shape})')
	plt.xlabel('Query')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(args.dataset_dir, "outputs", f"query_freq_{query_counts.shape[0]}_val.png"))

def fine_tune(train_df, captions):
	print(f"Fine-tuning using {device} in {torch.cuda.get_device_name(device)} using {nw} CPU(s)".center(150, "-"))
	print(f"Creating Train Dataloader", end="\t")
	tdl_st = time.time()
	train_dataset = HistoricalDataset(
		data_frame=train_df,
		captions=captions, 
		img_sz=args.image_size,
		dataset_directory=os.path.join(args.dataset_dir, "images")
	)
	train_data_loader = DataLoader(
		dataset=train_dataset,
		shuffle=True,
		batch_size=args.batch_size,
		num_workers=nw,
		pin_memory=True,  # Move data to GPU faster if using CUDA
    persistent_workers=True if nw > 1 else False,  # Keep workers alive if memory allows
		collate_fn=custom_collate_fn  # Use custom collate function to handle None values
	)
	print(f"num_samples[Total]: {len(train_data_loader.dataset)} Elapsed_t: {time.time()-tdl_st:.5f} sec")
	get_info(dataloader=train_data_loader)
	model = CLIP(
		emb_dim, 
		vit_layers,
		vit_d_model,
		(args.image_size, args.image_size),
		(args.patch_size, args.patch_size),
		n_channels,
		vit_heads,
		vocab_size,
		max_seq_length,
		text_heads,
		text_layers,
		text_d_model,
		device=device,
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
		for batch_idx, data in enumerate(train_data_loader):
			img = data["image"].to(device) 
			cap = data["caption"].to(device)
			mask = data["mask"].to(device)
			optimizer.zero_grad()
			loss = model(img, cap, mask)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			if batch_idx % args.print_every == 0:
				print(f"\tBatch [{batch_idx + 1}/{len(train_data_loader)}] Loss: {loss.item():.5f}")
			epoch_loss += loss.item()
		avg_loss = epoch_loss / len(train_data_loader)
		scheduler.step(avg_loss)
		print(f"Average Loss: {avg_loss:.5f} @ Epoch: {epoch+1}")
		average_losses.append(avg_loss)
		if avg_loss <= best_loss:
			best_loss = avg_loss
			torch.save(model.state_dict(), mdl_fpth)
			print(f"Saving model in {mdl_fpth} for best avg loss: {best_loss:.5f}")
	print(f"Elapsed_t: {time.time()-training_st:.5f} sec".center(150, "-"))
	loss_fname = (
		f'loss'
		+ f'_epochs_{args.num_epochs}'
		+ f'_lr_{args.learning_rate}'
		+ f'_wd_{args.weight_decay}'
		+ f'.png'
	)
	plot_loss(
		losses=average_losses, 
		num_epochs=args.num_epochs, 
		save_path=os.path.join(outputs_dir, loss_fname),
	)

def main():
	set_seeds()
	
	try:
		# load
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
		train_df, val_df = train_test_split(
			df, 
			shuffle=True, 
			test_size=args.validation_dataset_share, # 0.05
			random_state=42,
		)
		img_lbls_dict, img_lbls_list = get_doc_description(df=df, col=args.document_description_col)
		save_pickle(pkl=df, fname=df_fpth,)
		save_pickle(pkl=train_df, fname=train_df_fpth,)
		save_pickle(pkl=val_df, fname=val_df_fpth,)
		save_pickle(pkl=img_lbls_dict, fname=img_lbls_dict_fpth,)
		save_pickle(pkl=img_lbls_list, fname=img_lbls_list_fpth,)
	
	# Print the sizes of the datasets
	print(f"df: {df.shape} train_df: {train_df.shape} val_df({args.validation_dataset_share}): {val_df.shape}")
	print(f"img_lbls_dict {type(img_lbls_dict)} {len(img_lbls_dict)}")
	print(f"img_lbls_list {type(img_lbls_list)} {len(img_lbls_list)}")
	# return
	if not os.path.exists(mdl_fpth):
		fine_tune(train_df=train_df, captions=img_lbls_dict)

	print(f"Creating Validation Dataloader for {len(val_df)} images", end="\t")
	vdl_st = time.time()
	val_dataset = HistoricalDataset(
		data_frame=val_df,
		captions=img_lbls_dict,
		img_sz=args.image_size,
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
	get_info(dataloader=val_loader)

	if args.validate:
		validate(
			val_df=val_df,
			class_names=img_lbls_list,
			CAPTIONSs=img_lbls_dict,
			model_fpth=mdl_fpth,
			TOP_K=args.topk,
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
		'--num_epochs', str(args.num_epochs),
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