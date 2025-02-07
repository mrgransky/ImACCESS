from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# local:
# $ python history_clip_evaluate.py -ddir /home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31 -s "kfold_stratified" -k 1

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--query_image', '-qi', type=str, default="/home/farid/WS_Farid/ImACCESS/TEST_IMGs/5968_115463.jpg", help='image path for zero shot classification')
parser.add_argument('--query_label', '-ql', type=str, default="aircraft", help='image path for zero shot classification')
parser.add_argument('--topK', '-k', type=int, default=5, help='TopK results')
parser.add_argument('--kfolds', '-kf', type=int, default=3, help='kfolds for stratified sampling')
parser.add_argument('--seed', type=int, default=42, help='Reproducibility in KFold Stratified Sampling')
parser.add_argument('--batch_size', '-bs', type=int, default=256, help='batch size')
parser.add_argument('--model_name', '-md', type=str, default="ViT-B/32", help='CLIP model name')
parser.add_argument('--visualize', '-v', action='store_true', help='visualize the dataset')
parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)
args.device = torch.device(args.device)
OUTPUT_DIRECTORY = os.path.join(args.dataset_dir, "outputs")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

def load_model(
	model_name:str="ViT-B/32", 
	device:str="cuda:0",
	):
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
	sampling: str = "stratified_random", # "stratified_random" or "kfold_stratified"
	kfolds:int=5,  # Number of folds for K-Fold
	force_regenerate:bool=False, # Force regenerate K-Fold splits
	seed:int=42, # Seed for random sampling
	):
	if sampling not in ["stratified_random", "kfold_stratified"]:
		raise ValueError("Invalid sampling. Choose 'stratified_random' or 'kfold_stratified'.")

	print(f"Loading dataset {ddir} ...")
	metadata_fpth = os.path.join(ddir, "metadata.csv")
	# df = pd.read_csv(filepath_or_buffer=metadata_fpth, on_bad_lines='skip', low_memory=False)
	dtypes = {
		'doc_id': str,
		'id': str,
		'label': str,
		'title': str,
		'description': str,
		'img_url': str,
		'label_title_description': str,
		'raw_doc_date': str,  # Adjust based on actual data
		'doc_year': float,      # Adjust based on actual data
		'doc_url': str,
		'img_path': str,
		'doc_date': str,      # Adjust based on actual data
		'dataset': str,
		'date': str,          # Adjust based on actual data
	}
	df = pd.read_csv(
		filepath_or_buffer=metadata_fpth, 
		on_bad_lines='skip', 
		dtype=dtypes, 
		low_memory=False,
	)
	print(list(df.columns))
	print(df.head(10))
	print(f"FULL Dataset (df) shape: {df.shape}")
	if sampling == "stratified_random":
		print(f"Simple Random Sampling...")
		metadata_train_fpth = os.path.join(ddir, "metadata_train.csv")
		metadata_val_fpth = os.path.join(ddir, "metadata_val.csv")
		# Load training and validation datasets
		df_train = pd.read_csv(filepath_or_buffer=metadata_train_fpth, on_bad_lines='skip')
		df_val = pd.read_csv(filepath_or_buffer=metadata_val_fpth, on_bad_lines='skip')
		# Generate label mappings for simple sampling
		labels_train = list(set(df_train["label"].tolist()))
		labels_train = sorted(labels_train)
		label_dict_train = {lbl: idx for idx, lbl in enumerate(labels_train)}
		df_train['label_int'] = df_train['label'].map(label_dict_train)
		labels_val = list(set(df_val["label"].tolist()))
		labels_val = sorted(labels_val)
		label_dict_val = {lbl: idx for idx, lbl in enumerate(labels_val)}
		df_val['label_int'] = df_val['label'].map(label_dict_val)
		return df_train, df_val
	elif sampling == "kfold_stratified":
		if kfolds < 2:
			raise ValueError("kfolds must be at least 2.")
		fold_dir = os.path.join(ddir, sampling)
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
		labels = list(set(df["label"].tolist())) # Get unique labels
		labels = sorted(labels) # Get sorted unique labels
		label_dict = {lbl: idx for idx, lbl in enumerate(labels)}
		df["label_int"] = df["label"].map(label_dict)
		# Create stratified K-Fold splits
		folding_method = StratifiedKFold(
			n_splits=kfolds,
			shuffle=True,
			random_state=seed,
		)
		folds = []
		for fold, (train_idx, val_idx) in enumerate(folding_method.split(df, df["label"])):
			fold_dir = os.path.join(ddir, sampling, f"fold_{fold + 1}")
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
		print(f"K(={kfolds})-Fold splits saved successfully in {ddir}")
		print("*"*100)
		return folds
	else:
		raise ValueError("Invalid sampling. Use 'stratified_random' or 'kfold_stratified'.")

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
	seed:int=42,
	):
	print(f"Linear Probe & Zero Shot Classifier {device} CLIP".center(160, " "))
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
		random_state=seed,
		C=0.316, # TODO: hyperparameter tuning to find optimal value: implementing grid search or Bayesian optimization
		max_iter=1000,
		tol=1e-4, # tolerance for stopping criteria
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
	val_labels = sorted(val_labels)
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

def get_image_to_texts(
	dataset,
	model,
	preprocess,
	img_path,
	topk:int=5,
	device:str="cuda:0",
	):
	print(f"[Image-to-text(s)] Zero-Shot Image Classification of image: {img_path}".center(200, " "))
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	labels = sorted(labels) # Get sorted unique labels
	print(len(labels), type(labels))
	print(labels)
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
	labels = sorted(labels) # Get sorted unique labels
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

def get_image_to_texts_recall_at_k(
	dataset,
	model,
	preprocess,
	K:int=5,
	device:str="cuda:0",
	):
	"""
	For each image query:
		- Compute the image embedding.
		- Compare it with all label embeddings.
		- Retrieve the top-K labels.
		- Recall for this image is 1 if the ground-truth label is among the Top-K, else 0.
	Return the average (mean) Recall@K over all image queries.
	"""
	print(f"Image-to-Texts Recall@{K} Evaluation on {device}".center(150, " "))
	labels = list(set(dataset["label"].tolist()))
	labels = sorted(labels) # Get sorted unique labels
	if K > len(labels):
		print(f"ERROR: requested Top-{K} labeling is greater than number of labels({len(labels)}) => EXIT...")
		return
	t0 = time.time()
	print(f"Encoding {len(labels)} labels", end="\t")
	# Encode labels using CLIP.
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
	with torch.no_grad():
		labels_features = model.encode_text(tokenized_labels_tensor)
		labels_features /= labels_features.norm(dim=-1, keepdim=True)
	print(f"Elapsed_t: {time.time()-t0:.3f} sec")

	# Iterate over each image.
	dataset_images_path = dataset["img_path"].tolist()
	dataset_labels_int = dataset["label_int"].tolist()  # Ground-truth label index for each image.	
	recall_values = []
	for i, (img_path, gt_lbl_int) in enumerate(zip(dataset_images_path, dataset_labels_int)):
		img = Image.open(img_path).convert("RGB")
		img_tensor = preprocess(img).unsqueeze(0).to(device)
		with torch.no_grad():
			image_feature = model.encode_image(img_tensor)
			image_feature /= image_feature.norm(dim=-1, keepdim=True)
		
		# Compute similarity between image and all label embeddings.
		similarities = (100.0 * image_feature @ labels_features.T).softmax(dim=-1)
		_, topk_labels_idx = similarities.topk(K, dim=-1)
		topk_labels_idx = topk_labels_idx.cpu().numpy().flatten()
		
		# Single-label setting:
		# recall is 1 if the ground-truth label is among top-K, else 0.
		recall = 1 if gt_lbl_int in topk_labels_idx else 0
		recall_values.append(recall)
	
	mean_recall = np.mean(recall_values)
	print(f"Mean Recall@{K} for Image-to-Texts: {mean_recall:.3f}")
	print(f"Elapsed Time: {time.time()-t0:.2f} sec".center(150, "-"))
	return mean_recall

def get_text_to_images_recall_at_k(
	dataset, 
	model, 
	preprocess, 
	K: int = 5, 
	batch_size: int = 64, 
	device: str = "cuda:0", 
	image_features_file: str = "validation_image_features.gz"
	):
	"""
	For each text query (i.e. each label):
		- Compute the text embedding.
		- Compute similarities with all image embeddings.
		- Retrieve Top-K images.
		- Compute recall for that text query as:
				Recall@K = (# of images with that label in Top-K) / (Total number of images with that label)
	Return the average (mean) Recall@K over all text queries.
	"""
	print(f"Text-to-Images Recall@{K} Evaluation on {device} with batch size {batch_size}")
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	labels = sorted(labels) # Sort labels alphabetically.
	# Encode text labels.
	print(f"Encoding {len(labels)} labels", end="\t")
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
	with torch.no_grad():
		labels_features = model.encode_text(tokenized_labels_tensor)
		labels_features /= labels_features.norm(dim=-1, keepdim=True)
	print(f"Elapsed_t: {time.time()-t0:.3f} sec")
	
	# Encode image features.
	dataset_images_path = dataset["img_path"].tolist()
	if not os.path.exists(image_features_file):
		image_features_list = []
		for i in range(0, len(dataset_images_path), batch_size):
			batch_paths = dataset_images_path[i: i+batch_size]
			batch_imgs = [preprocess(Image.open(path).convert("RGB")).to(device) for path in batch_paths]
			batch_tensor = torch.stack(batch_imgs)
			with torch.no_grad():
				batch_features = model.encode_image(batch_tensor)
				batch_features /= batch_features.norm(dim=-1, keepdim=True)
			image_features_list.append(batch_features)
		dataset_images_features = torch.cat(image_features_list, dim=0)
		save_pickle(pkl=dataset_images_features, fname=image_features_file)
	else:
		dataset_images_features = load_pickle(fpath=image_features_file).to(device)
		dataset_images_features = dataset_images_features.to(device)
	
	recall_per_label = []
	# For each text query (each label), compute recall.
	for label_idx, label_feature in enumerate(labels_features):
		label_feature = label_feature.to(device)
		with torch.no_grad():
			sim = label_feature @ dataset_images_features.T  # Similarity with all images.
		_, topk_indices = sim.topk(K, dim=-1)
		topk_indices = topk_indices.cpu().numpy().flatten()
		
		# Find all image indices that have the ground-truth label corresponding to this text.
		# dataset["label_int"] should hold the label index for each image.
		relevant_images = [idx for idx, lbl in enumerate(dataset["label_int"].tolist()) if lbl == label_idx]
		
		if len(relevant_images) == 0:
			recall = 0
		else:
			num_relevant_retrieved = len(set(topk_indices) & set(relevant_images))# intersection of {1, 3, 4, 5} & {1, 2, 3, 6}: {1, 3} => len(intersection) = 2
			recall = num_relevant_retrieved / len(relevant_images)
		recall_per_label.append(recall)
	mean_recall = np.mean(recall_per_label)
	print(f"Mean Recall@{K} for Text-to-Images: {mean_recall:.3f}")
	print(f"Elapsed Time: {time.time()-t0:.2f} sec".center(150, "-"))
	return mean_recall

def get_image_to_texts_mp_at_k(
	dataset,
	model,
	preprocess,
	K:int=5,
	device:str="cuda:0",
	predicted_label_distribution_file:str="img2txt_label_distribution.png"
	):
	print(f"Image-to-Text Retrival [Classification] {device} CLIP".center(160, " "))
	labels = list(set(dataset["label"].tolist()))
	labels = sorted(labels) # Get sorted unique labels
	print(f"[performance metrics (macro-averaging): mP@{K} over all {len(labels)} labels]".center(160, " "))
	t0 = time.time()
	print(len(labels), type(labels))
	if K > len(labels):
		print(f"ERROR: requested Top-{K} labeling is greater than number of labels({len(labels)}) => EXIT...")
		return

	dataset_images_id = dataset["id"].tolist()
	dataset_images_path = dataset["img_path"].tolist()
	dataset_labels = dataset["label"].tolist() # ['naval training', 'medical service', 'medical service', 'naval forces', 'naval forces', ...]
	dataset_labels_int = dataset["label_int"].tolist() # [3, 17, 4, 9, ...]
	print(len(dataset_images_id), len(dataset_labels))
	
	print(f"[1] Encode {len(labels)} Labels", end="\t")
	t1 = time.time()
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	with torch.no_grad():
		labels_features = model.encode_text(tokenized_labels_tensor)
		labels_features /= labels_features.norm(dim=-1, keepdim=True)
	print(f"Elapsed_t: {time.time()-t1:.3f} sec")

	print(f"[2] Encode {len(dataset_images_path)} Images")
	predicted_labels = []
	true_labels = []
	t2 = time.time()
	torch.cuda.empty_cache() # Clear CUDA cache
	with torch.no_grad():
		for i, (img_pth, gt_lbl) in enumerate(zip(dataset_images_path, dataset_labels_int)): #img: <class 'PIL.Image.Image'>
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
	print(f"Predicted labels ({len(predicted_labels)}) [only few]: {predicted_labels[:10]}")
	print(f"True labels({len(true_labels)}) [only few]: {true_labels[:10]}")

	# [3] Calculate overall mP@K over all image queries
	# For a single-label case, each query's Precision@K is 1 if the correct label is among the top K, else 0.
	print(f"[3] Calculate mP@{K} over all {len(true_labels)} labels")
	p_at_k = 0
	for ilbl, vlbl in enumerate(true_labels):
		preds = predicted_labels[ilbl] # <class 'numpy.ndarray'>
		if vlbl in preds:
			p_at_k += 1
	mean_p_at_k_over_all_labels = p_at_k/len(true_labels)
	print(f"[OWN] mP@{K} over all {len(true_labels)} labels: {p_at_k} | {mean_p_at_k_over_all_labels:.3f}")
	##################################################################################################

	# [4] Compute per-label Precision@K: 
	# Group image queries by their true label and average the hit (1/0) per label.
	unique_labels = {}
	for lbl, lbl_int in zip(dataset["label"].tolist(), dataset["label_int"].tolist()):
		unique_labels[lbl_int] = lbl

	per_label_results = defaultdict(list)
	for gt_lbl, preds in zip(true_labels, predicted_labels):
		hit = 1 if gt_lbl in preds else 0
		per_label_results[gt_lbl].append(hit)
	print(per_label_results)
	# Create a list of tuples: (label_name, mP@K for that label)
	label_p_at_k = []
	for label_int in per_label_results.keys():
		print(f"label_int: {label_int}")
		print(per_label_results[label_int])
		avg_hit = np.mean(per_label_results[label_int])
		label_name = unique_labels.get(label_int, f"Unknown({label_int})")
		label_p_at_k.append((label_name, avg_hit))
		print(f"Label '{label_name}' (id {label_int}): P@{K} = {avg_hit:.3f} over {len(per_label_results[label_int])} queries")
	
	# [5] Plot Bar Chart of Per-Label P@K values
	plt.figure(figsize=(11, 6))
	# Extract label names and their corresponding P@K values
	label_names, precision_values = zip(*label_p_at_k)
	plt.bar(label_names, precision_values, color='skyblue', edgecolor='black')
	plt.xlabel("Labels")
	plt.ylabel(f"P@{K}")
	plt.title(
		f"Image-to-Text Per-label Retrieval\n"
		f"mP@{K} over {len(labels)} labels: {mean_p_at_k_over_all_labels:.3f}", 
		fontsize=10,
	)
	plt.xticks(rotation=90, fontsize=8)
	plt.grid(axis='y', alpha=0.6)
	plt.tight_layout()
	plt.savefig(predicted_label_distribution_file, dpi=200)
	
	print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))
	return mean_p_at_k_over_all_labels

def get_image_to_texts_map_at_k(
	dataset,
	model,
	preprocess,
	K: int = 5,
	device: str = "cuda:0",
	):
	"""
	1) Encodes the textual labels using CLIP.
	2) Process images one by one:
			Extracts image features using CLIP.
			Computes cosine similarity between image features and label features.
			Retrieves the Top-K most similar labels.
			Computes AP@K for each image.
			Computes Recall@K for each image.
	3) Aggregates AP@K values over all images to return mAP@K.
	"""
	print(f"Image-to-Text Retrieval Top-{K} {device} CLIP".center(160, " "))
	print(f"[Evaluation metrics: mean average precision@K (mAP@{K})]".center(160, " "))
	t0 = time.time()
	
	labels = list(set(dataset["label"].tolist()))
	labels = sorted(labels) # Get sorted unique labels
	print(f"Number of unique labels: {len(labels)}")
	if K > len(labels):
		print(f"ERROR: requested Top-{K} is greater than the number of labels ({len(labels)}) => EXIT...")
		return

	print(f"[1] Encode {len(labels)} Labels", end="\t")
	t1 = time.time()
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
	with torch.no_grad():
		labels_features = model.encode_text(tokenized_labels_tensor)
		labels_features /= labels_features.norm(dim=-1, keepdim=True)	
	print(f"Elapsed_t: {time.time()-t1:.3f} sec")

	dataset_images_path = dataset["img_path"].tolist()
	dataset_labels_int = dataset["label_int"].tolist()  # Ground-truth label indices
	print(f"[2] Encode {len(dataset_images_path)} Images & Compute mAP@{K} & Recall@{K}")
	ap_values = []
	for i, (img_pth, gt_lbl) in enumerate(zip(dataset_images_path, dataset_labels_int)):
			img_raw = Image.open(img_pth)
			img_tensor = preprocess(img_raw).unsqueeze(0).to(device)
			with torch.no_grad():
				# Compute image features
				image_features = model.encode_image(img_tensor)
				image_features /= image_features.norm(dim=-1, keepdim=True)
			
			# Compute similarity between the image and all label embeddings.
			similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
			
			# Retrieve top-K predictions
			_, topk_labels_idx = similarities.topk(K, dim=-1)
			preds = topk_labels_idx.cpu().numpy().flatten()
			
			# Compute AP@K
			relevant_count = 0
			precision_sum = 0.0			
			for rank, pred in enumerate(preds, start=1):
				if pred == gt_lbl:  # Relevant prediction
					relevant_count += 1
					precision_sum += relevant_count / rank  # Precision at this rank
			ap_at_k = precision_sum / relevant_count if relevant_count > 0 else 0.0
			ap_values.append(ap_at_k)
						
			if i % 100 == 0:
				torch.cuda.empty_cache()
	
	mAP_at_k = np.mean(ap_values)  # Compute mean Average Precision@K
	
	print(f"[Image-to-Text Retrival] mAP@{K}: {mAP_at_k:.3f}")
	print(f"Total Elapsed_t: {time.time()-t0:.2f} sec".center(160, "-"))
	
	return mAP_at_k

def get_text_to_images_map_at_k(
	dataset,
	model,
	preprocess,
	K=5,
	batch_size=64,
	device="cuda:0",
	image_features_file="validation_image_features.gz"
	):
	print(f"Text-to-Image Retrieval {device} CLIP batch_size: {batch_size}".center(160, " "))
	print(f"[Evaluation Metrics: mean average precision@K (mAP@{K})]".center(160, " "))
	torch.cuda.empty_cache()
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	labels = sorted(labels) # Sort labels alphabetically
	dataset_images_path = dataset["img_path"].tolist()
	all_labels_int = dataset["label_int"].tolist()

	print(f"[1] Encode Labels", end="\t")
	t1 = time.time()
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
	with torch.no_grad():
		tokenized_labels_features = model.encode_text(tokenized_labels_tensor)
		tokenized_labels_features /= tokenized_labels_features.norm(dim=-1, keepdim=True)
	print(f"Elapsed_t: {time.time()-t1:.3f} sec")

	print(f"[2] Encode Images")
	if not os.path.exists(image_features_file):
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
		dataset_images_features = load_pickle(fpath=image_features_file).to(device)
		dataset_images_features = dataset_images_features.to(device)
	
	print(f"[3] Calculate mAP@{K}")
	ap_values = []
	for label_idx, label_feature in enumerate(tokenized_labels_features):
		label_feature = label_feature.to(device)
		with torch.no_grad():
			sim = label_feature @ dataset_images_features.T # Similarity with all images.
		_, topk_indices = sim.topk(K, dim=-1)
		topk_indices = topk_indices.cpu().numpy().flatten()		
		relevant_images = [idx for idx, lbl in enumerate(all_labels_int) if lbl == label_idx] # Retrieve all relevant images.
		if len(relevant_images) == 0:
			print(f">> Warning << No relevant items found for label {label_idx} => Skipping AP@K calculation.")
			ap = 0.0
		else:
			num_relevant_so_far = 0
			precision_sum = 0.0
			for j, img_idx in enumerate(topk_indices, start=1):
				if img_idx in relevant_images:
					num_relevant_so_far += 1
					precision_at_j = num_relevant_so_far / j
					precision_sum += precision_at_j
			# Normalize by min(|R|, K)
			# if there are more than K relevant images, we cap the maximum AP at 1.
			ap = precision_sum / min(len(relevant_images), K)
		ap_values.append(ap)
	mAP_at_k = np.mean(ap_values)
	print(f"[Text-to-Image Retrieval] mAP@{K}: {mAP_at_k:.3f}")
	print(f"Total Elapsed_t: {time.time() - t0:.2f} sec".center(160, "-"))
	return mAP_at_k

def get_text_to_images_mp_at_k(
	dataset,
	model,
	preprocess,
	K:int=5,
	batch_size:int=64,
	device:str="cuda:0",
	image_features_file = 'validation_image_features.gz',
	predicted_label_distribution_file = 'validation_histogram_distribution_p_at_k.png',
	):
	print(f"Text-to-Image Retrieval {device} CLIP batch_size: {batch_size}".center(160, " "))
	print(f"[performance metrics(macro-averaging): mP@{K} over all labels, mean Recall@{K} over all labels".center(160, " "))
	torch.cuda.empty_cache()  # Clear CUDA cache
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	labels = sorted(labels) # Sort labels alphabetically
	print(f"Labels {type(labels)}: {len(labels)}")
	print(labels)

	dataset_images_id = dataset["id"].tolist()
	dataset_images_path = dataset["img_path"].tolist()
	dataset_labels = dataset["label"].tolist()
	dataset_labels_int = dataset["label_int"].tolist()
	print(len(dataset_images_id), len(dataset_labels))
	print(f"[1] Encode Labels", end="\t")
	t1 = time.time()
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
	with torch.no_grad():
		tokenized_labels_features = model.encode_text(tokenized_labels_tensor)
		tokenized_labels_features /= tokenized_labels_features.norm(dim=-1, keepdim=True)
	print(f"Elapsed_t: {time.time()-t1:.3f} seconds for {tokenized_labels_tensor.shape} labels")

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

	print(f"[3] Calculate mean Precision@{K} and mean Recall@{K} over all {len(tokenized_labels_features)} labels...")
	p_at_k = []
	recall_at_k = []
	label_p_at_k = []
	for i, label_features in enumerate(tokenized_labels_features):
		label_features = label_features.to(device)
		sim = label_features @ dataset_images_features.T # compute similarity between the label and all images
		_, indices = sim.topk(len(dataset_images_features), dim=-1) # retrieve all images for each label
		relevant_images_for_lbl_i = [idx for idx, lbl in enumerate(dataset_labels_int) if lbl == i] # retrieve all relevant images [with same label as GT]
		retrieved_topK_relevant_images = [idx for idx in indices.squeeze().cpu().numpy()[:K] if idx in relevant_images_for_lbl_i] # retrieve topK relevant images in the top-K retrieved images
		precision_value = len(retrieved_topK_relevant_images) / K
		p_at_k.append(precision_value)
		label_p_at_k.append((labels[i], precision_value))
		if len(relevant_images_for_lbl_i) == 0:
			recall_at_k.append(0)
		else:
			recall_at_k.append(len(retrieved_topK_relevant_images) / len(relevant_images_for_lbl_i))
		if i % 100 == 0:
			torch.cuda.empty_cache() # clear CUDA cache
	print(p_at_k)
	mean_p_at_k_over_all_labels = sum(p_at_k) / len(tokenized_labels_features) # np.mean(p_at_k)
	mean_recall_at_k_over_labels = sum(recall_at_k) / len(tokenized_labels_features) # np.mean(recall_at_k)
	print(f"mP@{K} over all {len(tokenized_labels_features)} labels: {mean_p_at_k_over_all_labels:.3f}")
	print(f"Total Elapsed_t: {time.time() - t0:.2f} sec".center(160, "-"))
	# [4] Plot Bar Chart of Per-Label P@K values
	plt.figure(figsize=(11, 6))
	# Extract label names and their corresponding P@K values
	label_names, precision_values = zip(*label_p_at_k)
	plt.bar(label_names, precision_values, color='skyblue', edgecolor='black')
	plt.xlabel("Labels")
	plt.ylabel(f"P@{K}")
	plt.title(
		f"Text-To-Image(s) Per-label Rerieval Predition\n"
		f"mP@{K} over {len(labels)} labels: {mean_p_at_k_over_all_labels:.3f}",
		fontsize=10,
	)
	plt.xticks(rotation=90, fontsize=8)
	plt.grid(axis='y', alpha=0.6)
	plt.tight_layout()
	plt.savefig(predicted_label_distribution_file, dpi=200)
	return mean_p_at_k_over_all_labels

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
	txt2img_val_pred_lbl_p_at_k_file,
	img2txt_val_pred_lbl_p_at_k_file,
	topk:int=5,
	seed:int=42,
	):
	print(f"Running Evaluation for {os.path.basename(args.dataset_dir)}".center(160, " "))
	print("*"*160)
	# Dictionary to store the metrics for this fold
	metrics = {}

	if args.visualize:
		get_image_to_texts(
				dataset=val_dataset,
				model=model,
				preprocess=preprocess,
				img_path=args.query_image,
				topk=topk,
				device=args.device,
		)
		get_text_to_images(
				dataset=val_dataset,
				model=model,
				preprocess=preprocess,
				query=args.query_label,
				topk=topk,
				batch_size=args.batch_size,
				device=args.device,
		)
		get_image_to_images(
				dataset=val_dataset,
				query_image_path=args.query_image,
				model=model,
				preprocess=preprocess,
				topk=topk,
				batch_size=args.batch_size,
				device=args.device,
		)
	
	if args.topK == 1:
		metrics["linear_probe_accuracy"], metrics["zero_shot_accuracy"] = get_linear_prob_zero_shot_accuracy(
			dataset_dir=args.dataset_dir,
			train_dataset=train_dataset,
			validation_dataset=val_dataset,
			model=model,
			preprocess=preprocess,
			batch_size=args.batch_size,
			device=args.device,
			train_image_features_file=train_image_features_file,
			val_image_features_file=val_image_features_file,
			seed=seed,
		)

	metrics["img2txt_mP_at_k"] = get_image_to_texts_mp_at_k(
		dataset=val_dataset,
		model=model,
		preprocess=preprocess,
		K=topk,
		device=args.device,
		predicted_label_distribution_file=img2txt_val_pred_lbl_p_at_k_file,
	)

	metrics["img2txt_mAP_at_k"] = get_image_to_texts_map_at_k(
		dataset=val_dataset,
		model=model,
		preprocess=preprocess,
		K=topk,
		device=args.device,
	)

	metrics["img2txt_recall_at_k"] = get_image_to_texts_recall_at_k(
		dataset=val_dataset,
		model=model,
		preprocess=preprocess,
		K=topk,
		device=args.device,
	)

	metrics["txt2img_mP_at_k"] = get_text_to_images_mp_at_k(
		dataset=val_dataset,
		model=model,
		preprocess=preprocess,
		K=topk,
		batch_size=args.batch_size,
		device=args.device,
		image_features_file=val_image_features_file,
		predicted_label_distribution_file=txt2img_val_pred_lbl_p_at_k_file,
	)

	metrics["txt2img_mAP_at_k"] = get_text_to_images_map_at_k(
		dataset=val_dataset,
		model=model,
		preprocess=preprocess,
		K=topk,
		batch_size=args.batch_size,
		device=args.device,
		image_features_file=val_image_features_file,
	)

	metrics["txt2img_recall_at_k"] = get_text_to_images_recall_at_k(
		dataset=val_dataset,
		model=model,
		preprocess=preprocess,
		K=topk,
		batch_size=args.batch_size,
		device=args.device,
		image_features_file=val_image_features_file,
	)

	print(json.dumps(metrics, indent=4, ensure_ascii=False))

	return metrics

def stratified_random_sampling(
	model,
	preprocess,
	topk:int=5,
	seed:int=42,
	):
	print(f"{'Simple Random Sampling':^150}")
	train_dataset, val_dataset = get_dataset(
		ddir=args.dataset_dir, 
		sampling=args.sampling,
		seed=seed,
	)
	print(f"Train: {train_dataset.shape}, Validation: {val_dataset.shape}")
	train_image_features_file = os.path.join(args.dataset_dir, args.sampling, 'train_image_features.gz')
	val_image_features_file = os.path.join(args.dataset_dir, args.sampling, 'validation_image_features.gz')
	txt2img_val_pred_lbl_p_at_k_file = os.path.join(args.dataset_dir, args.sampling, f'txt2img_val_per_label_prediction_p_at_{topk}.png')
	img2txt_val_pred_lbl_p_at_k_file = os.path.join(args.dataset_dir, args.sampling, f'img2txt_val_per_label_prediction_p_at_{topk}.png')
	
	os.makedirs(os.path.join(args.dataset_dir, args.sampling), exist_ok=True)
	run_evaluation(
		model=model,
		preprocess=preprocess,
		train_dataset=train_dataset,
		val_dataset=val_dataset,
		train_image_features_file=train_image_features_file,
		val_image_features_file=val_image_features_file,
		txt2img_val_pred_lbl_p_at_k_file=txt2img_val_pred_lbl_p_at_k_file,
		img2txt_val_pred_lbl_p_at_k_file=img2txt_val_pred_lbl_p_at_k_file,
		topk=topk,
		seed=seed,
	)

def k_fold_stratified_sampling(
	model,
	preprocess,
	kfolds:int=3,
	topk:int=5,
	seed:int=42,
	):
	print(f'K(={kfolds})-Fold Stratified Sampling'.center(150, "-"))
	t_start = time.time()

	# 1. Data Structure to Store Metrics from Each Fold
	metrics = {
		# Classification Metrics
		"linear_probe_accuracy": [],
		"zero_shot_accuracy": [],
		# Retrieval Metrics: Image-to-Texts
		"img2txt_mP_at_k": [],
		"img2txt_mAP_at_k": [],
		"img2txt_recall_at_k": [],
		# Retrieval Metrics: Text-to-Images
		"txt2img_mP_at_k": [],
		"txt2img_mAP_at_k": [],
		"txt2img_recall_at_k": [],
	}
	print("Checking and preparing dataset folds...")
	folded_datasets = get_dataset(
		ddir=args.dataset_dir,
		sampling=args.sampling,
		kfolds=kfolds,
		seed=seed,
	)
	for fidx, (df_train, df_val) in enumerate(folded_datasets):
		t3 = time.time()
		train_dataset = df_train
		val_dataset = df_val
		print(f"Fold {fidx + 1}/{kfolds}: Train: {train_dataset.shape}, Validation: {val_dataset.shape}")
		train_image_features_file = os.path.join(args.dataset_dir, args.sampling, f"fold_{fidx + 1}", 'train_image_features.gz')
		val_image_features_file = os.path.join(args.dataset_dir, args.sampling, f"fold_{fidx + 1}", 'validation_image_features.gz')
		txt2img_val_pred_lbl_p_at_k_file = os.path.join(args.dataset_dir, args.sampling, f"fold_{fidx + 1}", f'txt2img_val_per_label_prediction_p_at_{topk}.png')
		img2txt_val_pred_lbl_p_at_k_file = os.path.join(args.dataset_dir, args.sampling, f"fold_{fidx + 1}", f'img2txt_val_per_label_prediction_p_at_{topk}.png')

		# 2. Get Results
		folded_results = run_evaluation(
			model=model,
			preprocess=preprocess,
			train_dataset=train_dataset,
			val_dataset=val_dataset,
			train_image_features_file=train_image_features_file,
			val_image_features_file=val_image_features_file,
			txt2img_val_pred_lbl_p_at_k_file=txt2img_val_pred_lbl_p_at_k_file,
			img2txt_val_pred_lbl_p_at_k_file=img2txt_val_pred_lbl_p_at_k_file,
			topk=topk,
			seed=seed,
		)
		# 3. Store Metrics for the Current Fold
		for metric_name, metric_value in folded_results.items():
			# metrics[metric_name].append(metric_value)
			if metric_value is not None:
				metrics[metric_name].append(metric_value)
		print(f"Fold {fidx + 1}/{kfolds} evaluation completed, Elapsed time: {time.time()-t3:.1f} sec")

	print(f"K({kfolds})-Fold evaluation completed, Elapsed time: {time.time()-t_start:.1f} sec")

	# 4. Calculate and Print Average Metrics
	print(f"Calculating average metrics for mP@K, mAP@K and Recall@K (K={args.topK}) over all {kfolds} folds".center(150, "-"))
	for metric_name, metric_values in metrics.items():
		if len(metric_values) == 0:
			print(f"{metric_name}: No valid values to average!")
			continue
		avg_metric = np.mean(metric_values)
		print(
			f"{metric_name}: {len(metric_values)} folds | "
			f"{metric_values} "
			f"Min: {np.min(metric_values):.3f} "
			f"Max: {np.max(metric_values):.3f} "
			f"mean: {avg_metric:.3f}"
		)
		print()

@measure_execution_time
def main():	
	set_seeds(seed=args.seed, debug=True)
	print(clip.available_models())
	model, preprocess = load_model(model_name=args.model_name, device=args.device,)
	if args.sampling == "stratified_random":
		stratified_random_sampling(
			model=model,
			preprocess=preprocess,
			topk=args.topK,
			seed=args.seed,
		)
	elif args.sampling == "kfold_stratified":
		k_fold_stratified_sampling(
			model=model,
			preprocess=preprocess,
			kfolds=args.kfolds,
			topk=args.topK,
			seed=args.seed,
		)
	else:
		raise ValueError(f"Unknown sampling strategy: {args.sampling}")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))