from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# local:
# $ python history_clip_evaluate.py -ddir /home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31 -ss "kfold-stratified_sampling" -k 1

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--query_image', '-qi', type=str, default="/home/farid/WS_Farid/ImACCESS/TEST_IMGs/5968_115463.jpg", help='image path for zero shot classification')
parser.add_argument('--query_label', '-ql', type=str, default="aircraft", help='image path for zero shot classification')
parser.add_argument('--topK', '-k', type=int, default=5, help='TopK results')
parser.add_argument('--kfolds', '-kf', type=int, default=3, help='kfolds for stratified sampling')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch size')
parser.add_argument('--model_name', '-md', type=str, default="ViT-B/32", help='CLIP model name')
parser.add_argument('--visualize', '-v', action='store_true', help='visualize the dataset')
parser.add_argument('--sampling_strategy', '-ss', type=str, default="simple_random_sampling", choices=["simple_random_sampling", "kfold-stratified_sampling"], help='Sampling strategy')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)
args.device = torch.device(args.device)
OUTPUT_DIRECTORY = os.path.join(args.dataset_dir, "outputs")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

def load_model(model_name:str="ViT-B/32", device:str="cuda:0"):
	model, preprocess = clip.load(model_name, device=device)
	model = model.float()
	input_resolution = model.visual.input_resolution
	context_length = model.context_length
	vocab_size = model.vocab_size
	print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
	print("Input resolution:", input_resolution)
	print("Context length:", context_length)
	print("Vocab size:", vocab_size)
	return model, preprocess

def get_dataset(
	ddir: str = "path/2/dataset_dir",
	sampling_strategy: str = "simple_random_sampling", # "simple_random_sampling" or "kfold-stratified_sampling"
	kfolds:int=5,  # Number of folds for K-Fold
	force_regenerate:bool=False, # Force regenerate K-Fold splits
	):
	if sampling_strategy not in ["simple_random_sampling", "kfold-stratified_sampling"]:
		raise ValueError("Invalid sampling_strategy. Choose 'simple_random_sampling' or 'kfold-stratified_sampling'.")

	print(f"Loading dataset {ddir} ...")
	metadata_fpth = os.path.join(ddir, "metadata.csv")
	df = pd.read_csv(filepath_or_buffer=metadata_fpth, on_bad_lines='skip')
	print(f"FULL Dataset (df) shape: {df.shape}")
	if sampling_strategy == "simple_random_sampling":
		print(f"Simple Random Sampling...")
		metadata_train_fpth = os.path.join(ddir, "metadata_train.csv")
		metadata_val_fpth = os.path.join(ddir, "metadata_val.csv")
		# Load training and validation datasets
		df_train = pd.read_csv(filepath_or_buffer=metadata_train_fpth, on_bad_lines='skip')
		df_val = pd.read_csv(filepath_or_buffer=metadata_val_fpth, on_bad_lines='skip')
		# Generate label mappings for simple sampling
		labels_train = list(set(df_train["label"].tolist()))
		label_dict_train = {lbl: idx for idx, lbl in enumerate(labels_train)}
		df_train['label_int'] = df_train['label'].map(label_dict_train)
		labels_val = list(set(df_val["label"].tolist()))
		label_dict_val = {lbl: idx for idx, lbl in enumerate(labels_val)}
		df_val['label_int'] = df_val['label'].map(label_dict_val)
		return df_train, df_val
	elif sampling_strategy == "kfold-stratified_sampling":
		if kfolds < 2:
			raise ValueError("kfolds must be at least 2.")
		fold_dir = os.path.join(ddir, sampling_strategy)
		if os.path.exists(fold_dir) and not force_regenerate:
			print(f"K-Fold splits already exist in {fold_dir}. Loading existing splits...")
			folds = []
			for fold in range(1, kfolds + 1):
				train_fpth = os.path.join(fold_dir, f"fold_{fold}", "metadata_train.csv")
				val_fpth = os.path.join(fold_dir, f"fold_{fold}", "metadata_val.csv")
				df_train = pd.read_csv(train_fpth)
				df_val = pd.read_csv(val_fpth)
				folds.append((df_train, df_val))
			return folds
		print(f"K-Fold Stratified sampling with K={kfolds} folds...")
		if "label" not in df.columns:
			raise ValueError("The dataset must have a 'label' column for stratified sampling.")
		# Exclude labels that occur only once
		label_counts = df["label"].value_counts()
		labels_to_drop = label_counts[label_counts == 1].index
		df = df[~df["label"].isin(labels_to_drop)]
		if df.empty:
			raise ValueError("No valid labels for stratified sampling (after removing labels with one occurrence).")
		# Generate label mappings
		labels = list(set(df["label"].tolist()))
		label_dict = {lbl: idx for idx, lbl in enumerate(labels)}
		df["label_int"] = df["label"].map(label_dict)
		# Create stratified K-Fold splits
		skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)
		folds = []
		for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
			fold_dir = os.path.join(ddir, sampling_strategy, f"fold_{fold + 1}")
			os.makedirs(fold_dir, exist_ok=True)
			train_fpth = os.path.join(fold_dir, "metadata_train.csv")
			val_fpth = os.path.join(fold_dir, "metadata_val.csv")
			df_train = df.iloc[train_idx].copy()
			df_val = df.iloc[val_idx].copy()
			df_train["label_int"] = df_train["label"].map(label_dict)
			df_val["label_int"] = df_val["label"].map(label_dict)
			df_train.to_csv(train_fpth, index=False)
			df_val.to_csv(val_fpth, index=False)
			folds.append((df_train, df_val))
		print(f"K(={kfolds})-Fold splits saved in {ddir}")
		return folds
	else:
		raise ValueError("Invalid sampling_strategy. Use 'simple_random_sampling' or 'kfold-stratified_sampling'.")

def get_linear_prob_zero_shot_accuracy(
	dataset_dir,
	train_dataset,
	validation_dataset,
	model,
	preprocess,
	batch_size:int=1024,
	device:str="cuda:0",
	train_image_features_file:str="train_image_features.gz",
	val_image_features_file:str="validation_image_features.gz",
	):
	print(f"Getting training features")
	train_dataset_images_id = train_dataset["id"].tolist()
	train_dataset_images_path = train_dataset["img_path"].tolist()
	train_dataset_labels = train_dataset["label"].tolist() # ['naval training', 'medical service', 'medical service', 'naval forces', 'naval forces', ...]
	train_dataset_labels_int = torch.tensor(train_dataset["label_int"].tolist()) # torch[3, 17, 4, 9, ...]
	torch.cuda.empty_cache() # Clear CUDA cache
	t0 = time.time()
	if not os.path.exists(train_image_features_file):
		train_dataset_images_features = []
		for i in range(0, len(train_dataset_images_path), batch_size):
			batch_train_images_path = [train_dataset_images_path[j] for j in range(i, min(i + batch_size, len(train_dataset_images_path)))]
			batch_train_tensors = torch.stack([preprocess(Image.open(img_path)).to(device) for img_path in batch_train_images_path])
			with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
				train_image_features = model.encode_image(batch_train_tensors)
				train_image_features /= train_image_features.norm(dim=-1, keepdim=True)
			train_dataset_images_features.append(train_image_features)
			if i % 50 == 0:
				torch.cuda.empty_cache()  # Clear CUDA cache
		train_dataset_images_features = torch.cat(train_dataset_images_features, dim=0)
		save_pickle(pkl=train_dataset_images_features, fname=train_image_features_file)
	else:
		train_dataset_images_features = load_pickle(fpath=train_image_features_file)
	print(f"Elapsed_t: {time.time()-t0:.2f} sec")

	print(f"Getting validation features")
	val_dataset_images_id = validation_dataset["id"].tolist()
	val_dataset_images_path = validation_dataset["img_path"].tolist()
	val_dataset_labels = validation_dataset["label"].tolist() # ['naval training', 'medical service', 'medical service', 'naval forces', 'naval forces', ...]
	val_dataset_labels_int = torch.tensor(validation_dataset["label_int"].tolist()) # torch[3, 17, 4, 9, ...]
	t0 = time.time()
	if not os.path.exists(val_image_features_file):
		val_dataset_images_features = []
		for i in range(0, len(val_dataset_images_path), batch_size):
			batch_val_images_path = [val_dataset_images_path[j] for j in range(i, min(i + batch_size, len(val_dataset_images_path)))]
			batch_val_tensors = torch.stack([preprocess(Image.open(img_path)).to(device) for img_path in batch_val_images_path])
			with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
				val_image_features = model.encode_image(batch_val_tensors)
				val_image_features /= val_image_features.norm(dim=-1, keepdim=True)
			val_dataset_images_features.append(val_image_features)
			torch.cuda.empty_cache() # Clear CUDA cache
		val_dataset_images_features = torch.cat(val_dataset_images_features, dim=0)
		save_pickle(pkl=val_dataset_images_features, fname=val_image_features_file)
	else:
		val_dataset_images_features = load_pickle(fpath=val_image_features_file)
	print(f"Elapsed_t: {time.time()-t0:.2f} sec")

	# Perform logistic regression
	train_dataset_images_features = train_dataset_images_features.detach().cpu().numpy()
	train_dataset_labels_int = train_dataset_labels_int.cpu().numpy()

	val_dataset_images_features = val_dataset_images_features.detach().cpu().numpy()
	val_dataset_labels_int = val_dataset_labels_int.cpu().numpy()

	print(type(train_dataset_images_features), train_dataset_images_features.shape)
	print(type(train_dataset_labels_int), train_dataset_labels_int.shape)

	# Perform logistic regression
	t0 = time.time()
	solver = 'saga' # 'saga' is faster for large datasets
	print(f"Training the logistic regression classifier with {solver} solver")
	classifier = LogisticRegression(
		random_state=0,
		C=0.316,
		max_iter=1000,
		verbose=1,
		solver=solver, # 'saga' is faster for large datasets
		n_jobs=-1, # to utilize all cores
	)

	classifier.fit(train_dataset_images_features, train_dataset_labels_int)

	# Evaluate using the logistic regression classifier
	predictions = classifier.predict(val_dataset_images_features)
	linear_probe_accuracy = np.mean((val_dataset_labels_int == predictions).astype(float))# * 100
	print(f"Linear Probe Accuracy = {linear_probe_accuracy:.3f} | Elapsed_t: {time.time()-t0:.2f} sec")
	################################## Zero Shot Classifier ##################################
	t0 = time.time()
	# Get unique labels for validation dataset:
	val_labels = list(set(validation_dataset["label"].tolist()))
	text_inputs = clip.tokenize(texts=val_labels).to(device=device)
	with torch.no_grad():
		text_features = model.encode_text(text_inputs)
	# Normalize the features of the validation dataset (numpy):
	val_dataset_images_features = val_dataset_images_features / np.linalg.norm(val_dataset_images_features, axis=1, keepdims=True)
	# Normalize text_features(tensor):
	text_features = text_features / text_features.norm(dim=-1, keepdim=True)
	# Convert test_features to a PyTorch tensor
	val_dataset_images_features = torch.from_numpy(val_dataset_images_features).to(device)
	# Calculate the similarity scores
	similarity_scores = (100.0 * val_dataset_images_features @ text_features.T).softmax(dim=-1)
	# Get the predicted class indices
	predicted_class_indices = np.argmax(similarity_scores.cpu().numpy(), axis=1)
	print(type(predicted_class_indices), predicted_class_indices.shape)
	zero_shot_accuracy = np.mean((val_dataset_labels_int == predicted_class_indices).astype(float))# * 100
	print(f"Zero-shot Accuracy = {zero_shot_accuracy:.3f} | Elapsed_t: {time.time()-t0:.2f} sec")
	################################## Zero Shot Classifier ##################################
	return linear_probe_accuracy, zero_shot_accuracy

def get_image_to_texts(dataset, model, preprocess, img_path, topk:int=5, device:str="cuda:0"):
	print(f"[Image-to-text(s)] Zero-Shot Image Classification of image: {img_path}".center(200, " "))
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	print(len(labels), type(labels))
	if topk > len(labels):
		print(f"ERROR: requested Top-{topk} labeling is greater than number of labels({len(labels)}) => EXIT...")
		return
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	
	# img = Image.open(img_path) # only from path like strings: /home/farid/WS_Farid/ImACCESS/TEST_IMGs/5968_115463.jpg

	try:
		img = Image.open(img_path)
	except FileNotFoundError:
		try:
			response = requests.get(img_path)
			response.raise_for_status()
			img = Image.open(BytesIO(response.content))
		except requests.exceptions.RequestException as e:
			print(f"ERROR: failed to load image from {img_path} => {e}")
			return

	image_tensor = preprocess(img).unsqueeze(0).to(device) # <class 'torch.Tensor'> torch.Size([1, 3, 224, 224])

	with torch.no_grad():
		image_features = model.encode_image(image_tensor)
		labels_features = model.encode_text(tokenized_labels_tensor)

	image_features /= image_features.norm(dim=-1, keepdim=True)
	labels_features /= labels_features.norm(dim=-1, keepdim=True)

	similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
	topk_pred_probs, topk_pred_labels_idx = similarities.topk(topk, dim=-1)
	print(topk_pred_probs)
	print(topk_pred_labels_idx)
	print(f"Top-{topk} predicted labels: {[labels[i] for i in topk_pred_labels_idx.cpu().numpy().flatten()]}")
	print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))
	img_hash = hashlib.sha256(img_path.encode()).hexdigest()[:8]
	plt.figure(figsize=(12, 8))
	plt.imshow(img)
	plt.axis('off')
	plt.title(f'Top-{topk} predicted labels: {[labels[i] for i in topk_pred_labels_idx.cpu().numpy().flatten()]}', fontsize=15)
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(OUTPUT_DIRECTORY, f'Img2Txt_Top{topk}_LBLs_IMG_{img_hash}_dataset_{os.path.basename(args.dataset_dir)}.png'),
		dpi=250,
		bbox_inches='tight',
	)
	plt.close()

def get_image_to_texts_precision_at_(
	dataset,
	model,
	preprocess,
	K:int=5,
	device:str="cuda:0",
	):
	print(f"Image-to-Text Retrival [Classification] {device} CLIP [performance metrics: Precision@{K}]".center(160, " "))
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	print(len(labels), type(labels))
	if K > len(labels):
		print(f"ERROR: requested Top-{K} labeling is greater than number of labels({len(labels)}) => EXIT...")
		return

	dataset_images_id = dataset["id"].tolist()
	dataset_images_path = dataset["img_path"].tolist()
	dataset_labels = dataset["label"].tolist() # ['naval training', 'medical service', 'medical service', 'naval forces', 'naval forces', ...]
	dataset_labels_int = dataset["label_int"].tolist() # [3, 17, 4, 9, ...]
	print(len(dataset_images_id), len(dataset_labels))
	
	print(f"[1] Encode Labels")
	t1 = time.time()
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	with torch.no_grad():
		labels_features = model.encode_text(tokenized_labels_tensor)
		labels_features /= labels_features.norm(dim=-1, keepdim=True)
	print(f"Elapsed_t: {time.time()-t1:.3f} sec")

	print(f"[2] Encode Images")
	predicted_labels = []
	true_labels = []
	t2 = time.time()
	torch.cuda.empty_cache() # Clear CUDA cache
	with torch.no_grad():
		for i, (img_pth, gt_lbl) in enumerate(zip(dataset_images_path, dataset_labels_int)): #img: <class 'PIL.Image.Image'>
			# print(i, img_pth, gt_lbl)
			img_raw = Image.open(img_pth)
			img_tensor = preprocess(img_raw).unsqueeze(0).to(device)
			image_features = model.encode_image(img_tensor)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
			_, topk_labels_idx = similarities.topk(K, dim=-1)
			predicted_labels.append(topk_labels_idx.cpu().numpy().flatten())
			true_labels.append(gt_lbl)
			if i % 100 == 0:
				torch.cuda.empty_cache() # Clear CUDA cache
	print(f"Elapsed_t: {time.time()-t2:.3f} sec")
	print(len(predicted_labels), len(true_labels))
	print(type(predicted_labels[0]), predicted_labels[0].shape,)
	print(predicted_labels[:10])
	print(true_labels[:10])

	print(f"[3] Calculate Precision@{K}")
	pred_st = time.time()
	prec_at_k = 0
	for ilbl, vlbl in enumerate(true_labels):
		preds = predicted_labels[ilbl] # <class 'numpy.ndarray'>
		if vlbl in preds:
			prec_at_k += 1
	avg_prec_at_k = prec_at_k/len(true_labels)
	print(f"[OWN] Precision@{K}: {prec_at_k} | {avg_prec_at_k:.3f} Elapsed_t: {time.time()-pred_st:.4f} sec")
	##################################################################################################

	# pred_st = time.time()
	# # Calculate Precision at K
	# prec_at_k = sum(1 for i, v in enumerate(true_labels) if v in predicted_labels[i])
	# avg_prec_at_k = prec_at_k / len(true_labels)
	
	# # Calculate Recall at K
	# recall_at_k = sum(1 / K for i, v in enumerate(true_labels) if v in predicted_labels[i])
	# avg_recall_at_k = recall_at_k / len(true_labels)
	
	# print(
	# 	f"top-{K} Precision: {prec_at_k} | {avg_prec_at_k} "
	# 	f"Recall: {recall_at_k} | {avg_recall_at_k} "
	# 	f"Elapsed_t: {time.time()-pred_st:.2f} sec"
	# )
	print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))
	return avg_prec_at_k

def get_text_to_images(
	dataset,
	model,
	preprocess,
	query:str="cat",
	topk:int=5,
	batch_size:int=64,
	device:str="cuda:0",
	):
	print(f"Top-{topk} Image Retrieval {device} CLIP Query: « {query} »".center(160, " "))
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	print(len(labels), type(labels))

	dataset_images_id = dataset["id"].tolist()
	dataset_images_path = dataset["img_path"].tolist()
	dataset_labels = dataset["label"].tolist() # ['naval training', 'medical service', 'medical service', 'naval forces', 'naval forces', ...]
	dataset_labels_int = dataset["label_int"].tolist() # [3, 17, 4, 9, ...]
	print(len(dataset_images_id), len(dataset_labels))
	
	tokenized_query_tensor = clip.tokenize(texts=query).to(device)#<class 'torch.Tensor'> torch.Size([1, 77])
	query_features = model.encode_text(tokenized_query_tensor) # <class 'torch.Tensor'> torch.Size([1, 512])
	query_features /= query_features.norm(dim=-1, keepdim=True)
	
	image_features_file = os.path.join(args.dataset_dir, 'outputs', 'validation_image_features.gz')
	if not os.path.exists(image_features_file):
		dataset_images_features = []
		for i in range(0, len(dataset_images_path), batch_size):
			batch_images_path = [dataset_images_path[j] for j in range(i, min(i + batch_size, len(dataset_images_path)))]
			batch_tensors = torch.stack([preprocess(Image.open(img_path)).to(device) for img_path in batch_images_path])
			with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
				image_features = model.encode_image(batch_tensors)
				image_features /= image_features.norm(dim=-1, keepdim=True)
			dataset_images_features.append(image_features)
			torch.cuda.empty_cache()  # Clear CUDA cache
		dataset_images_features = torch.cat(dataset_images_features, dim=0)
		save_pickle(pkl=dataset_images_features, fname=image_features_file)
	else:
		dataset_images_features = load_pickle(fpath=image_features_file)
		dataset_images_features = dataset_images_features.to(device)

	similarities = (100.0 * query_features @ dataset_images_features.T).softmax(dim=-1)

	# Get the top-k most similar images
	topk_probs, topk_indices = similarities.topk(topk, dim=-1)
	print(topk_probs)
	print(topk_indices)

	# Retrieve the top-k images
	topk_pred_image_paths = [dataset_images_path[topk_indices.squeeze().item()]] if topk==1 else [dataset_images_path[idx] for idx in topk_indices.squeeze().cpu().numpy()] # [<PIL.Image.Image image mode=RGB size=32x32 at 0x7C16A47D8F40>, ...]
	topk_pred_images = [Image.open(dataset_images_path[topk_indices.squeeze().item()])] if topk==1 else [Image.open(dataset_images_path[idx]) for idx in topk_indices.squeeze().cpu().numpy()] # [<PIL.Image.Image image mode=RGB size=32x32 at 0x7C16A47D8F40>, ...]
	topk_pred_labels_idxs = [dataset_labels_int[topk_indices.squeeze().item()]] if topk==1 else [dataset_labels_int[idx] for idx in topk_indices.squeeze().cpu().numpy()] # [3, 3, 8, 8, 3]

	topk_probs = topk_probs.squeeze().cpu().detach().numpy()
	print(topk_pred_image_paths)
	# print(topk_pred_images)
	print(topk_probs, topk_pred_labels_idxs)
	# print(len(labels), labels)

	# Save the top-k images in a single file
	fig, axes = plt.subplots(1, topk, figsize=(5*topk, 9))
	if topk == 1:
		axes = [axes]  # Convert to list of axes
	fig.suptitle(f"Text-To-Image(s) [Top-{topk}] Rerieval\nQuery: « {query} »\n{os.path.basename(args.dataset_dir)}", fontsize=11)
	for i, (img, ax) in enumerate(zip(topk_pred_images, axes)):
		ax.imshow(img)
		ax.axis('off')
		if topk == 1:
			ax.set_title(f"Top-1\nprob: {topk_probs:.4f}\nGT: {labels[topk_pred_labels_idxs[0]]}", fontsize=9)
		else:
			ax.set_title(f"Top-{i+1}\nprob: {topk_probs[i]:.4f}\nGT: {labels[topk_pred_labels_idxs[i]]}", fontsize=9)
				
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(OUTPUT_DIRECTORY, f"Txt2Img_Top{topk}_IMGs_dataset_{os.path.basename(args.dataset_dir)}_query_{re.sub(' ', '_', query)}.png"),
		dpi=250,
		bbox_inches='tight',
	)
	plt.close()
	print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))

def get_text_to_images_precision_recall_at_(
	dataset,
	model,
	preprocess,
	K:int=5,
	batch_size:int=64,
	device:str="cuda:0",
	image_features_file = 'validation_image_features.gz',
	):
	print(f"Text-to-Image Retrieval {device} CLIP batch_size: {batch_size} [performance metrics: Precision@{K}]".center(160, " "))
	torch.cuda.empty_cache()  # Clear CUDA cache
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	print(f"Labels {type(labels)}: {len(labels)}")

	dataset_images_id = dataset["id"].tolist()
	dataset_images_path = dataset["img_path"].tolist()
	dataset_labels = dataset["label"].tolist()
	dataset_labels_int = dataset["label_int"].tolist()
	print(len(dataset_images_id), len(dataset_labels))
	print(f"[1] Encode Labels")
	t1 = time.time()
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
	with torch.no_grad():
		tokenized_labels_features = model.encode_text(tokenized_labels_tensor)
		tokenized_labels_features /= tokenized_labels_features.norm(dim=-1, keepdim=True)
	print(f"Elapsed_t: {time.time()-t1:.3f} sec")

	print(f"[2] Encode Images")
	t2 = time.time()
	# image_features_file = os.path.join(args.dataset_dir, 'outputs', 'validation_image_features.gz')
	if not os.path.exists(image_features_file):
		print(f"Encoding {len(dataset_images_path)} images, might take a while...")
		dataset_images_features = []
		for i in range(0, len(dataset_images_path), batch_size):
			batch_images_path = [dataset_images_path[j] for j in range(i, min(i + batch_size, len(dataset_images_path)))]
			batch_tensors = torch.stack([preprocess(Image.open(img_path)).to(device) for img_path in batch_images_path])
			with torch.no_grad():
				image_features = model.encode_image(batch_tensors)
				image_features /= image_features.norm(dim=-1, keepdim=True)
			dataset_images_features.append(image_features)
			torch.cuda.empty_cache()
		dataset_images_features = torch.cat(dataset_images_features, dim=0)
		save_pickle(pkl=dataset_images_features, fname=image_features_file)
	else:
		dataset_images_features = load_pickle(fpath=image_features_file)
		dataset_images_features = dataset_images_features.to(device)
	print(f"Elapsed_t: {time.time()-t2:.3f} sec")

	print(f"[3] Calculate Precision@{K}")
	t3 = time.time()
	prec_at_k = []
	recall_at_k = []
	for i, label_features in enumerate(tokenized_labels_features):
		label_features = label_features.to(device)
		sim = label_features @ dataset_images_features.T # compute similarity between the label and all images
		_, indices = sim.topk(len(dataset_images_features), dim=-1) # retrieve all images for each label
		relevant_images_for_lbl_i = [idx for idx, lbl in enumerate(dataset_labels_int) if lbl == i] # retrieve all images with same label
		retrieved_topK_relevant_images = [idx for idx in indices.squeeze().cpu().numpy()[:K] if idx in relevant_images_for_lbl_i] # retrieve topK relevant images in the top-K retrieved images
		prec_at_k.append(len(retrieved_topK_relevant_images) / K)
		recall_at_k.append(len(retrieved_topK_relevant_images) / len(relevant_images_for_lbl_i))
		if i % 100 == 0:
			torch.cuda.empty_cache() # clear CUDA cache
	avg_prec_at_k = sum(prec_at_k) / len(labels)
	avg_recall_at_k = sum(recall_at_k) / len(labels)
	print(f"Precision@{K}: {avg_prec_at_k:.3f} {np.mean(prec_at_k)}")
	print(f"Recall@{K}: {avg_recall_at_k:.3f} {np.mean(recall_at_k)}")
	print(f"Elapsed_t: {time.time()-t3:.3f} sec")
	print(f"Total Elapsed_t: {time.time() - t0:.2f} sec".center(160, "-"))
	return avg_prec_at_k, avg_recall_at_k

def get_map_at_k(
	dataset,
	model,
	preprocess,
	K:int=10,
	batch_size:int=512,
	device:str="cuda:0",
	image_features_file = 'validation_image_features.gz',
	):
	"""  
	Calculate mean average precision@K (mAP@K) for image-to-texts and text-to-images retrieval.  
	:param dataset: Dataset containing images and their corresponding labels.  
	:param model: CLIP model instance.  
	:param preprocess: Preprocessing function for images.  
	:param K: Top-K value for precision and recall calculation.  
	:param batch_size: Batch size for processing images.  
	:param device: Device (GPU or CPU) for computations.  
	:return: mAP@K values for image-to-texts and text-to-images retrieval tasks.  
	"""  
	print(f"Calculating mAP@{K} for Image-to-Texts and Text-to-Images Retrieval {device}".center(160, " "))  
	t0 = time.time()
	# image_features_file = os.path.join(args.dataset_dir, 'outputs', 'validation_image_features.gz')
	if not os.path.exists(image_features_file):  
		dataset_images_features = []  
		for i in range(0, len(dataset["img_path"]), batch_size):  
			batch_images_path = [dataset["img_path"][j] for j in range(i, min(i + batch_size, len(dataset["img_path"])))]  
			batch_tensors = torch.stack([preprocess(Image.open(img_path)).to(device) for img_path in batch_images_path])  
			with torch.no_grad():  
				image_features = model.encode_image(batch_tensors)  
				image_features /= image_features.norm(dim=-1, keepdim=True)  
			dataset_images_features.append(image_features)  
			del batch_tensors
			if i % 50 == 0:
				torch.cuda.empty_cache()  
		dataset_images_features = torch.cat(dataset_images_features, dim=0)  
		save_pickle(pkl=dataset_images_features, fname=image_features_file)  
	else:  
		dataset_images_features = load_pickle(fpath=image_features_file) 
		dataset_images_features = dataset_images_features.to(device)

	# Image-to-Texts Retrieval
	img_to_txt_precisions = []
	tokenized_labels_tensor = clip.tokenize(texts=list(set(dataset["label"]))).to(device)
	with torch.no_grad():
		labels_features = model.encode_text(tokenized_labels_tensor)  
		labels_features /= labels_features.norm(dim=-1, keepdim=True)  

	for i, (img_path, gt_lbl) in enumerate(zip(dataset["img_path"], dataset["label_int"])):  
		with torch.no_grad():  
			img_raw = Image.open(img_path)  
			img_tensor = preprocess(img_raw).unsqueeze(0).to(device)  
			image_features = model.encode_image(img_tensor)  
			image_features /= image_features.norm(dim=-1, keepdim=True)  
		similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)  
		_, topk_labels_idx = similarities.topk(len(labels_features), dim=-1)  
		rank = torch.where(topk_labels_idx == gt_lbl)[1].item() + 1
		precision = 1 / rank if rank <= K else 0  
		img_to_txt_precisions.append(precision)  
		del img_tensor, image_features
		torch.cuda.empty_cache()  
	img_to_txt_map_at_k = np.mean(img_to_txt_precisions)  

	# Text-to-Images Retrieval  
	txt_to_img_precisions = []
	with torch.no_grad():
		tokenized_labels_features = model.encode_text(tokenized_labels_tensor)  
		tokenized_labels_features /= tokenized_labels_features.norm(dim=-1, keepdim=True)  
	for i, label_features in enumerate(tokenized_labels_features):
		sim = label_features @ dataset_images_features.T  
		_, indices = sim.topk(len(dataset_images_features), dim=-1)  
		relevant_images_for_lbl_i = [idx for idx, lbl in enumerate(dataset["label_int"]) if lbl == i]  
		retrieved_topK_relevant_images = [idx for idx in indices.squeeze().cpu().numpy()[:K] if idx in relevant_images_for_lbl_i]  
		precisions = []  
		for j, idx in enumerate(indices.squeeze().cpu().numpy()):  
			if idx in relevant_images_for_lbl_i:  
				precisions.append(len(retrieved_topK_relevant_images) / (j + 1))  
		if precisions:  
			txt_to_img_precisions.append(np.mean(precisions))  
		else:  
			txt_to_img_precisions.append(0)  
	txt_to_img_map_at_k = np.mean(txt_to_img_precisions)  
	print(f"mAP@{K} Image-to-Texts: {img_to_txt_map_at_k:.3f} | Text-to-Images: {txt_to_img_map_at_k:.3f}")  
	print(f"Elapsed_t: {time.time() - t0:.2f} sec".center(160, "-"))  
	return img_to_txt_map_at_k, txt_to_img_map_at_k

def get_image_to_images(
	dataset,
	query_image_path,
	model, preprocess,
	topk:int,
	batch_size:int,
	device,
	):
	print(f"Image-to-Image(s) Retrieval {query_image_path}".center(200, " "))
	t0 = time.time()

	# Image Embedding for the query image
	try:
		qimage = Image.open(query_image_path)
	except FileNotFoundError:
		try:
			response = requests.get(query_image_path)
			response.raise_for_status()
			qimage = Image.open(BytesIO(response.content))
		except requests.exceptions.RequestException as e:
			print(f"ERROR: failed to load image from {query_image_path} => {e}")
			return

	print(f"Obtaining image embeddings for dataset of size: {dataset.shape}, please wait...")
	# Images embeddings for the dataset:
	image_features_file = os.path.join(args.dataset_dir, 'outputs', 'validation_image_features.gz')  
	if not os.path.exists(image_features_file):  
		dataset_images_features = []
		for i in range(0, len(dataset["img_path"]), batch_size):
			batch_images_path = [dataset["img_path"][j] for j in range(i, min(i + batch_size, len(dataset["img_path"])))]  
			batch_tensors = torch.stack([preprocess(Image.open(img_path)).to(device) for img_path in batch_images_path])  
			with torch.no_grad():  
				image_features = model.encode_image(batch_tensors)  
				image_features /= image_features.norm(dim=-1, keepdim=True)  
			dataset_images_features.append(image_features)  
			del batch_tensors
			torch.cuda.empty_cache()  
		dataset_images_features = torch.cat(dataset_images_features, dim=0)  
		save_pickle(pkl=dataset_images_features, fname=image_features_file)  
	else:  
		dataset_images_features = load_pickle(fpath=image_features_file)  
	print(type(dataset_images_features), len(dataset_images_features), dataset_images_features[0].shape, type(dataset_images_features[0]))

	query_image = preprocess(qimage).unsqueeze(0).to(device)

	# Encode the query image
	with torch.no_grad():
		query_image_feature = model.encode_image(query_image)
		query_image_feature = query_image_feature / query_image_feature.norm(dim=-1, keepdim=True)
	
	query_image_feature = query_image_feature.cpu().detach().numpy()
	print(query_image_feature.shape, type(query_image_feature))

	# Calculate similarity between the query image and all images in the dataset
	dataset_images_features = dataset_images_features.cpu().detach().numpy()
	similarities = cosine_similarity(query_image_feature, dataset_images_features)
	topk_indices = np.argsort(similarities[0])[-topk:][::-1] # Get indices of top-k most similar images
	topk_similarities = similarities[0][topk_indices]	# Get the similarity scores of top-k most similar images
	topk_image_paths = [dataset["img_path"][idx] for idx in topk_indices] # Get the paths of top-k most similar images
	topk_labels = [dataset["label"][idx] for idx in topk_indices] # Get the labels of top-k most similar images
	print(f"Top-{topk} similar images to {query_image_path}:")
	for i, (path, label, similarity) in enumerate(zip(topk_image_paths, topk_labels, topk_similarities)):
		print(f"{i+1}. Image Path: {path}, Label: {label}, Similarity: {similarity:.4f}")

	# Create a plot of 2 rows and topK columns
	fig, axes = plt.subplots(2, topk, figsize=(5 * topk, 8))
	# Calculate the middle index for the query image
	if topk % 2 == 0:  # Even number of columns
		middle_index = topk // 2 - 1  # Place query image in the middle-left column
	else:  # Odd number of columns
		middle_index = topk // 2  # Place query image in the median column
	# Plot the query image in the first row, middle column
	axes[0, middle_index].imshow(qimage)
	axes[0, middle_index].axis('off')
	axes[0, middle_index].set_title("Query Image", fontsize=10)
	# Hide the rest of the first row (only one image in the first row)
	for j in range(topk):
		if j != middle_index:
			axes[0, j].axis('off')
	# Plot the top-k similar images in the second row
	for i, (path, label, similarity) in enumerate(zip(topk_image_paths, topk_labels, topk_similarities)):
		img = Image.open(path)
		axes[1, i].imshow(img)
		axes[1, i].axis('off')
		axes[1, i].set_title(f"Top-{i+1}\nSimilarity: {similarity:.3f}", fontsize=12)

	# Set the plot title
	plt.suptitle(f"Image-to-Image(s) [Top-{topk}] Retrieval\n{os.path.basename(args.dataset_dir)}", fontsize=12)
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(OUTPUT_DIRECTORY, f"Img2Img_Top{topk}_IMGs_{os.path.basename(args.dataset_dir)}.png"),
		dpi=250,
		bbox_inches='tight',
	)
	plt.close()
	print(f"Elapsed_t: {time.time() - t0:.2f} sec".center(160, "-"))
	return topk_image_paths, topk_labels, topk_similarities

def run_evaluation(
	model,
	preprocess,
	train_dataset,
	val_dataset,
	train_image_features_file,
	val_image_features_file,
	):
	print(f"Running Evaluation for {os.path.basename(args.dataset_dir)}".center(160, " "))
	# Dictionary to store the metrics for this fold
	fold_metrics = {}

	if args.visualize:
		get_image_to_texts(
				dataset=val_dataset,
				model=model,
				preprocess=preprocess,
				img_path=args.query_image,
				topk=args.topK,
				device=args.device,
		)
		get_text_to_images(
				dataset=val_dataset,
				model=model,
				preprocess=preprocess,
				query=args.query_label,
				topk=args.topK,
				batch_size=args.batch_size,
				device=args.device,
		)
		get_image_to_images(
				dataset=val_dataset,
				query_image_path=args.query_image,
				model=model,
				preprocess=preprocess,
				topk=args.topK,
				batch_size=args.batch_size,
				device=args.device,
		)
	
	if args.topK == 1:
		linear_probe_accuracy, zero_shot_accuracy = get_linear_prob_zero_shot_accuracy(
			dataset_dir=args.dataset_dir,
			train_dataset=train_dataset,
			validation_dataset=val_dataset,
			model=model,
			preprocess=preprocess,
			batch_size=args.batch_size,
			device=args.device,
			train_image_features_file=train_image_features_file,
			val_image_features_file=val_image_features_file,
		)
		fold_metrics["linear_probe_accuracy"] = linear_probe_accuracy
		fold_metrics["zero_shot_accuracy"] = zero_shot_accuracy

	img_to_txt_precision = get_image_to_texts_precision_at_(
		dataset=val_dataset,
		model=model,
		preprocess=preprocess,
		K=args.topK,
		device=args.device,
	)
	fold_metrics["img_to_txt_precision"] = img_to_txt_precision

	txt_to_img_precision, txt_to_img_recall = get_text_to_images_precision_recall_at_(
		dataset=val_dataset,
		model=model,
		preprocess=preprocess,
		K=args.topK,
		batch_size=args.batch_size,
		device=args.device,
		image_features_file=val_image_features_file,
	)
	fold_metrics["txt_to_img_precision"] = txt_to_img_precision
	fold_metrics["txt_to_img_recall"] = txt_to_img_recall

	img_to_txt_map, txt_to_img_map = get_map_at_k(
		dataset=val_dataset,
		model=model,
		preprocess=preprocess,
		K=args.topK,
		batch_size=args.batch_size,
		device=args.device,
		image_features_file=val_image_features_file,
	)
	fold_metrics["img_to_txt_map"] = img_to_txt_map
	fold_metrics["txt_to_img_map"] = txt_to_img_map

	return fold_metrics

def simple_random_sampling(model, preprocess):
	print(f"{'Simple Random Sampling':^150}")
	train_dataset, val_dataset = get_dataset(
		ddir=args.dataset_dir, 
		sampling_strategy=args.sampling_strategy,
	)
	print(f"Train: {train_dataset.shape}, Validation: {val_dataset.shape}")
	train_image_features_file = os.path.join(args.dataset_dir, args.sampling_strategy, 'train_image_features.gz')
	val_image_features_file = os.path.join(args.dataset_dir, args.sampling_strategy, 'validation_image_features.gz')
	run_evaluation(
		model=model,
		preprocess=preprocess,
		train_dataset=train_dataset,
		val_dataset=val_dataset,
		train_image_features_file=train_image_features_file,
		val_image_features_file=val_image_features_file,
	)

def k_fold_stratified_sampling(model, preprocess, kfolds:int=5):
	print(f'K(={kfolds})-Fold Stratified Sampling'.center(150, "-"))

	# 1. Data Structure to Store Metrics from Each Fold
	metrics = {
		"linear_probe_accuracy": [],
		"zero_shot_accuracy": [],
		"img_to_txt_precision": [],
		"txt_to_img_precision": [],
		"txt_to_img_recall": [],
		"img_to_txt_map": [],
		"txt_to_img_map": [],
	}
	print("Checking and preparing dataset folds...")
	folded_dataset = get_dataset(
		ddir=args.dataset_dir,
		sampling_strategy=args.sampling_strategy,
		kfolds=kfolds,
	)
	# for fidx in range(kfolds):
	for fidx, (df_train, df_val) in enumerate(folded_dataset):
		t3 = time.time()
		train_dataset = df_train
		val_dataset = df_val
		print(f"Fold {fidx + 1}/{kfolds}: Train: {train_dataset.shape}, Validation: {val_dataset.shape}")
		train_image_features_file = os.path.join(args.dataset_dir, args.sampling_strategy, f"fold_{fidx + 1}", 'train_image_features.gz')
		val_image_features_file = os.path.join(args.dataset_dir, args.sampling_strategy, f"fold_{fidx + 1}", 'validation_image_features.gz')
		# 2. Call run_evaluation and Get Results
		fold_metrics = run_evaluation(
			model=model,
			preprocess=preprocess,
			train_dataset=train_dataset,
			val_dataset=val_dataset,
			train_image_features_file=train_image_features_file,
			val_image_features_file=val_image_features_file,
		)
		# 3. Store Metrics for the Current Fold
		for metric_name, metric_value in fold_metrics.items():
			metrics[metric_name].append(metric_value)
		print(f"Fold {fidx + 1}/{kfolds} evaluation completed, Elapsed time: {time.time()-t3:.1f} sec")
	# 4. Calculate and Print Average Metrics
	print("K-Fold evaluation completed. Calculating average metrics...")
	for metric_name, metric_values in metrics.items():
		if len(metric_values) == 0:
			continue
		avg_metric = np.mean(metric_values)
		print(
			f"{metric_name} "
			f"{metric_values} "
			f"Min: {np.min(metric_values):.3f} "
			f"Max: {np.max(metric_values):.3f} "
			f"mean: {avg_metric:.3f}"
		)
	print("-" * 50)

@measure_execution_time
def main():
	print(clip.available_models())
	model, preprocess = load_model(model_name=args.model_name, device=args.device,)
	if args.sampling_strategy == "simple_random_sampling":
		simple_random_sampling(model=model, preprocess=preprocess)
	elif args.sampling_strategy == "kfold-stratified_sampling":
		k_fold_stratified_sampling(model=model, preprocess=preprocess, kfolds=args.kfolds)
	else:
		raise ValueError(f"Unknown sampling strategy: {args.sampling_strategy}")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))