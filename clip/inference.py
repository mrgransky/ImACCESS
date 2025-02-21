from utils import *
from datasets_loader import *

parser = argparse.ArgumentParser(description="Evaluate CLIP for different datasets")
parser.add_argument('--query_image', '-qi', type=str, default="/home/farid/WS_Farid/ImACCESS/TEST_IMGs/dog.jpeg", help='image path for zero shot classification')
parser.add_argument('--query_label', '-ql', type=str, default="airplane", help='image path for zero shot classification')
parser.add_argument('--topK', '-k', type=int, default=5, help='TopK results')
parser.add_argument('--batch_size', '-bs', type=int, default=256, help='batch size')
parser.add_argument('--dataset', '-d', type=str, choices=['cifar10', 'cifar100', 'cinic10', 'imagenet'], default='cifar10', help='dataset (CIFAR10/cifar100)')
parser.add_argument('--model_name', '-md', type=str, default="ViT-B/32", help='CLIP model name')
parser.add_argument('--device', '-dv', type=str, default="cuda:0", help='device')
parser.add_argument('--num_workers', '-nw', type=int, default=12, help='number of workers')
parser.add_argument('--visualize', '-v', action='store_true', help='visualize the dataset')
args, unknown = parser.parse_known_args()
print(args)

# $ nohup python -u inference.py -d imagenet -k 1 -bs 512 -nw 4 -dv "cuda:3" > /media/volume/ImACCESS/trash/prec_at_k.out &
device = torch.device(args.device)
USER = os.environ.get('USER')
OUTPUT_DIRECTORY = os.path.join(args.dataset, "outputs")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

def get_features(
	dataset,
	model,
	batch_size:int=1024,
	device:str="cuda:0",
	nw:int=8,
	):
	all_features = []
	all_labels = []
	dataloader = DataLoader(
		dataset=dataset,
		batch_size=batch_size,
		num_workers=nw,
		pin_memory=True, # Move data to GPU faster if using CUDA
		persistent_workers=(nw > 1),
		prefetch_factor=2, # Number of batches loaded in advance by each worker
		shuffle=False,  # Shuffle is not necessary during feature extraction
	) # <class 'torch.Tensor'> torch.Size([b, 512]), <class 'torch.Tensor'> torch.Size([b])
	model.eval()
	torch.cuda.empty_cache() # Clear CUDA cache before starting the loop
	with torch.no_grad():
		for i, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
			images = images.to(device, non_blocking=True)  # non_blocking for potential faster async transfers
			features = model.encode_image(images).cpu() # <class 'torch.Tensor'> torch.Size([b, 512])
			all_features.append(features)
			all_labels.append(labels)
			if (i+1) % 50 == 0:
				torch.cuda.empty_cache() # Clear CUDA cache after each batch

	all_features = torch.cat(all_features).numpy()
	all_labels = torch.cat(all_labels).numpy()
	return all_features, all_labels

def get_image_to_texts_linear_prob_and_zero_shot_accuracy(
		train_dataset,
		validation_dataset,
		model,
		batch_size:int=1024,
		device:str="cuda:0",
		num_workers:int=8,
		train_features_fname="train_features.gz", # Path to save the training features
		train_labels_fname="train_labels.gz", # Path to save the training labels
		validation_features_fname="validation_features.gz", # Path to save the validation features
		validation_labels_fname="validation_labels.gz", # Path to save the validation labels
	):
	print(f"Getting training features and labels")
	t0 = time.time()
	try:
		# load the features from the disk
		train_features = load_pickle(fpath=train_features_fname)
		train_labels = load_pickle(fpath=train_labels_fname)
	except Exception as e:
		print(f"Error: {e}")
		torch.cuda.empty_cache() # Clear CUDA cache
		train_features, train_labels = get_features(
			dataset=train_dataset,
			model=model,
			batch_size=batch_size,
			device=device,
			nw=num_workers,
		) # <class 'numpy.ndarray'> (num_samples, 512), <class 'numpy.ndarray'> (num_samples,)
		save_pickle(pkl=train_features, fname=train_features_fname)
		save_pickle(pkl=train_labels, fname=train_labels_fname)
	print(f"Elapsed_t: {time.time()-t0:.2f} sec")

	print(f"Getting validation features and labels")
	t0 = time.time()
	try:
		# load the features from the disk
		val_features = load_pickle(fpath=validation_features_fname)
		val_labels = load_pickle(fpath=validation_labels_fname)
	except Exception as e:
		print(f"Error: {e}")
		torch.cuda.empty_cache() # Clear CUDA cache
		val_features, val_labels = get_features(
			dataset=validation_dataset,
			model=model,
			batch_size=batch_size,
			device=device,
			nw=num_workers,
		) # <class 'numpy.ndarray'> (num_samples, 512), <class 'numpy.ndarray'> (num_samples,)
		save_pickle(pkl=val_features, fname=validation_features_fname)
		save_pickle(pkl=val_labels, fname=validation_labels_fname)
	print(f"Elapsed_t: {time.time()-t0:.2f} sec")

	# Perform logistic regression
	t0 = time.time()
	solver = 'saga' # 'saga' is faster for large datasets
	print(f"Training the logistic regression classifier with {solver} solver")
	classifier = LogisticRegression(
		random_state=42,
		C=0.316,
		tol=1e-3, # tolerance for stopping criteria
		max_iter=1000,
		verbose=1,
		solver=solver, # 'saga' is faster for large datasets
		n_jobs=-1, # to utilize all cores
	)

	print(f"fitting the classifier")
	classifier.fit(train_features, train_labels)

	print(f"Getting the linear probe accuracy")
	# Evaluate using the logistic regression classifier
	predictions = classifier.predict(val_features)
	accuracy = np.mean((val_labels == predictions).astype(float))# * 100
	print(f"Linear Probe (Top-1) Accuracy = {accuracy:.3f} | Elapsed_t: {time.time()-t0:.2f} sec")

	################################## Zero Shot Classifier ##################################
	t0 = time.time()
	class_names = validation_dataset.classes

	# Encode the text descriptions of the classes
	# text_descriptions = [f"a photo of a {label}" for label in class_names] # Descriptive Prompts higher acccuracy.
	text_descriptions = list(class_names)

	with torch.no_grad():
		text_inputs = clip.tokenize(texts=text_descriptions).to(device=device)
		text_features = model.encode_text(text_inputs)
		# Normalize text_features(tensor):
		text_features = text_features / text_features.norm(dim=-1, keepdim=True)

	# Normalize the features of the test (numpy):
	val_features = val_features / np.linalg.norm(val_features, axis=1, keepdims=True)

	# Convert val_features to a PyTorch tensor
	val_features = torch.from_numpy(val_features).to(device, non_blocking=True)

	# Calculate the similarity scores
	similarity_scores = (100.0 * val_features @ text_features.T).softmax(dim=-1)

	# Get the predicted class indices
	predicted_class_indices = np.argmax(similarity_scores.cpu().numpy(), axis=1) 

	# Calculate the accuracy
	accuracy = np.mean((val_labels == predicted_class_indices).astype(float))# * 100
	print(f"Zero-shot (Top-1) Accuracy = {accuracy:.3f} | Elapsed_t: {time.time()-t0:.2f} sec")
	print("-"*160)

def get_image_to_texts_mp_at_k(
	dataset,
	model,
	K:int=5,
	device:str="cuda:0",
	):
	print(f"Image-to-Text Retrival [Classification] {device} {model.__class__.__name__} {model.name}".center(160, " "))
	print(f"[performance metrics: mP@{K}]".center(160, " "))
	labels = dataset.classes # <class 'list'> ['airplane', 'automobile', ...]
	if K > len(labels):
		print(f"ERROR: requested Top-{K} labeling is greater than number of labels({len(labels)}) => EXIT...")
		return
	fcn_start_time = time.time()
	print(f"Labels: {len(labels)}")
	print(f"Encoding labels...", end="\t")
	t0 = time.time()
	# Encode text labels without gradient tracking:
	with torch.no_grad(): # prevent GPU memory issues
		tokenized_labels_tensor = clip.tokenize(texts=labels).to(device, non_blocking=True) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
		labels_features = model.encode_text(tokenized_labels_tensor)
		labels_features /= labels_features.norm(dim=-1, keepdim=True)
	print(f"Done! [{time.time()-t0:.2f} sec]")
	print(f"Labels: {len(labels)} => lable_features: {labels_features.shape}")

	predicted_labels = []
	true_labels = []
	floop_st = time.time()
	with torch.no_grad():
		for i, data in enumerate(dataset):
			img_tensor, gt_lbl = data # <class 'torch.Tensor'> torch.Size([3, 224, 224]) <class 'int'>
			img_tensor = img_tensor.unsqueeze(0).to(device, non_blocking=True) # <class 'torch.Tensor'> torch.Size([1, 3, 224, 224])
			image_features = model.encode_image(img_tensor)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			similarities = image_features @ labels_features.T
			_, topk_labels_idx = similarities.topk(K, dim=-1)
			predicted_labels.append(topk_labels_idx.cpu().numpy().flatten())
			true_labels.append(gt_lbl)
	print(f"Total (for loop): {time.time()-floop_st:.3f} sec")
	print(len(predicted_labels), len(true_labels))
	print(type(predicted_labels[0]), predicted_labels[0].shape,)
	print(predicted_labels[:10])
	print(true_labels[:10])

	##################################################################################################
	pred_st = time.time()
	precision_at_k = []
	for ilbl, vlbl in enumerate(true_labels):
		preds = predicted_labels[ilbl] # <class 'numpy.ndarray'>
		correct = np.sum(preds == vlbl)  # Count how many times the true label appears in the top-K
		precision = correct / K # Precision@K for this sample
		precision_at_k.append(precision)
	mP_at_k = np.mean(precision_at_k)
	print(f"[OWN] mP@{K}: {mP_at_k} | Elapsed_t: {time.time()-pred_st:.2f} sec")
	print(f"Elapsed_t: {time.time()-fcn_start_time:.2f} sec".center(160, "-"))
	##################################################################################################

def get_text_to_images_mp_at_k(
		dataset,
		model,
		K:int=5,
		batch_size:int=1024,
		device:str="cuda:0",
	):
	torch.cuda.empty_cache()  # Clear CUDA cache
	print(f"Text-to-Image Retrieval {device} {model.__class__.__name__} {model.name} batch_size: {batch_size}".center(160, " "))
	print(f"[performance metrics: mP@{K}]".center(160, " "))
	labels = dataset.classes
	print(f"Labels: {len(labels)}")

	print(f"Encoding labels", end="\t")
	t0 = time.time()
	with torch.no_grad():
		tokenized_labels_tensor = clip.tokenize(texts=labels).to(device, non_blocking=True) # <class 'torch.Tensor'> torch.Size([num_lbls, 77])
		tokenized_labels_features = model.encode_text(tokenized_labels_tensor) # <class 'torch.Tensor'> torch.Size([num_lbls, 512])
		tokenized_labels_features /= tokenized_labels_features.norm(dim=-1, keepdim=True)
	print(f"Elapsed_t: {time.time()-t0:.2f} s | tokenized_labels_features: {tokenized_labels_features.shape}")

	# Check if the image features file exists
	image_features_file = os.path.join(OUTPUT_DIRECTORY, 'validation_image_features.gz')
	if not os.path.exists(image_features_file):
		print(f"Encoding {len(dataset)} images, might take a while...")
		t0 = time.time()
		dataset_images_features = []
		for i in range(0, len(dataset), batch_size):
			batch_tensors = torch.stack([dataset[j][0].to(device, non_blocking=True) for j in range(i, min(i + batch_size, len(dataset)))]) # <class 'torch.Tensor'> torch.Size([b, 3, 224, 224]
			# print(f"Processing batch {i // batch_size + 1}/{len(dataset) // batch_size + 1}")
			# print(f"batch_tensors: {batch_tensors.shape}") # batch_tensors: torch.Size([b, 3, 224, 224])
			with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
				image_features = model.encode_image(batch_tensors)
				image_features /= image_features.norm(dim=-1, keepdim=True)
			dataset_images_features.append(image_features)
			if i % 50 == 0:
				torch.cuda.empty_cache()  # Clear CUDA cache
		dataset_images_features = torch.cat(dataset_images_features, dim=0)
		print(f"Elapsed_t: {time.time()-t0:.2f} sec => dataset_images_features: {dataset_images_features.shape}")
		save_pickle(pkl=dataset_images_features, fname=image_features_file)
	else:
		dataset_images_features = load_pickle(image_features_file)
		dataset_images_features = dataset_images_features.to(device, non_blocking=True)  # <--- Fix: Move to current device
	###################################### Wrong approach for P@K and R@K ######################################
	# it is rather accuracy or hit rate!
	# prec_at_k = 0
	# recall_at_k = []
	# for i, label_features in enumerate(tokenized_labels_features):
	# 	sim = (100.0 * label_features @ dataset_images_features.T).softmax(dim=-1) # similarities between query and all images
	# 	topk_probs, topk_indices = sim.topk(K, dim=-1)
	# 	topk_pred_labels = [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()] # [3, 3, 8, 8, 3]
	# 	recall_at_k.append(topk_pred_labels.count(i)/K)
	# 	if i in topk_pred_labels: # just checking if the label is present
	# 		prec_at_k += 1
	# avg_prec_at_k = prec_at_k / len(tokenized_labels_features)
	# avg_recall_at_k = sum(recall_at_k) / len(labels)
	# print(f"Precision@{K}: {prec_at_k} {avg_prec_at_k}")
	# print(f"Recall@{K}: {recall_at_k} {avg_recall_at_k} {np.mean(recall_at_k)}")
	# print(labels)
	###################################### Wrong approach for P@K and R@K ######################################

	# Create an index of images in the dataset, grouped by labels
	# print("Creating an index of images in the dataset, grouped by labels...")
	# t = time.time()
	# index = {}
	# for idx, (img, lbl) in enumerate(dataset):
	# 	if lbl not in index:
	# 		index[lbl] = []
	# 	index[lbl].append(idx)
	# print(f"Index created in {time.time() - t:.2f} sec")

	print("Creating an index of images in the dataset, grouped by labels...")
	t = time.time()
	index = defaultdict(list)
	for idx, (img, lbl) in enumerate(dataset):
		index[lbl].append(idx)
	print(f"Index created in {time.time() - t:.2f} sec")

	print(f"Calculating Precision@{K} and Recall@{K}, might take a while...")
	prec_at_k = []
	t0 = time.time()
	for i, label_features in enumerate(tokenized_labels_features):
		label_features = label_features.to(device, non_blocking=True) # Ensure label_features is on the correct device
		# sim = (100.0 * label_features @ dataset_images_features.T).softmax(dim=-1) # similarities between query and all images
		sim = label_features @ dataset_images_features.T # similarities between query and all images
		topk_probs, topk_indices = sim.topk(K, dim=-1)
		topk_pred_labels = [dataset[topk_indices.squeeze().item()][1]] if K==1 else [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()]# K@1, 5,...
		relevant_retrieved_images_for_label_i = topk_pred_labels.count(i)  # counting relevant images in top-K retrieved images
		prec_at_k.append(relevant_retrieved_images_for_label_i/K)
		all_images_with_label_i = index.get(i, [])  # get the list of images with label i
		num_all_images_with_label_i = len(all_images_with_label_i)
		if i % 100 == 0:
			torch.cuda.empty_cache()  # Clear CUDA cache
	
	avg_prec_at_k = np.mean(prec_at_k) #sum(prec_at_k)/len(labels)
	print(f"mP@{K}: {avg_prec_at_k:.3f}")
	print(f"Elapsed_t: {time.time()-t0:.2f} sec")
	print("-"*160)

def plot_precision_recall_curve(
	tokenized_labels_features: torch.Tensor,
	labels: list,
	dataset,
	dataset_images_features: torch.Tensor,
	):
	print(f"Plotting PR and ROC curves for {len(labels)} labels")
	fpr_values = []
	tpr_values = []
	precision_values = []
	recall_values = []
	for i, label_features in enumerate(tokenized_labels_features):
		sim = (100.0 * label_features @ dataset_images_features.T).softmax(dim=-1)
		sim = sim.squeeze().cpu().detach().numpy()
		predicted_labels = np.argsort(-sim)
		true_labels = [1 if dataset[j][1] == i else 0 for j in range(len(dataset))]
		prec, rec, thresh = precision_recall_curve(true_labels, sim)
		fpr, tpr, _ = roc_curve(true_labels, sim)
		precision_values.append(prec)
		recall_values.append(rec)
		fpr_values.append(fpr)
		tpr_values.append(tpr)

	fig, ax = plt.subplots(1, 2, figsize=(11,7))
	for i in range(len(tokenized_labels_features)):
		ax[0].plot(recall_values[i], precision_values[i], label=f'{labels[i]}')
		ax[0].set_xlabel('Recall')
		ax[0].set_ylabel('Precision')
		ax[0].set_title('Precision-Recall')
		ax[1].plot(fpr_values[i], tpr_values[i])
		ax[1].set_xlabel('False Positive')
		ax[1].set_ylabel('True Positive')
		ax[1].set_title('ROC')

	# fig.legend(bbox_to_anchor=(0.5, 0.99), fontsize=7, ncol=len(labels), frameon=False)
	fig.tight_layout()
	plt.savefig(
		fname=os.path.join(OUTPUT_DIRECTORY, f"{args.dataset}_PR_ROC_x{len(labels)}_labels.png"),
		dpi=250,
		bbox_inches='tight',
	)
	print("-"*160)

def get_image_to_texts(
	dataset,
	model,
	preprocess,
	img_path,
	topk:int=5,
	device:str="cuda:0",
	):
	print(f"Image-to-Text Retrieval [Classification]: {img_path}".center(160, " "))
	labels = dataset.classes
	if topk > len(labels):
		print(f"ERROR: requested Top-{topk} labeling is greater than number of labels({len(labels)}) => EXIT...")
		return
	
	torch.cuda.empty_cache()  # Clear CUDA cache
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device, non_blocking=True) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	img = Image.open(img_path)
	image_tensor = preprocess(img).unsqueeze(0).to(device, non_blocking=True) # <class 'torch.Tensor'> torch.Size([1, 3, 224, 224])

	with torch.no_grad():
		image_features = model.encode_image(image_tensor)
		image_features /= image_features.norm(dim=-1, keepdim=True)
	
		labels_features = model.encode_text(tokenized_labels_tensor)
		labels_features /= labels_features.norm(dim=-1, keepdim=True)

	similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
	topk_pred_probs, topk_pred_labels_idx = similarities.topk(topk, dim=-1)
	print(topk_pred_probs)
	print(topk_pred_labels_idx)
	print(f"Top-{topk} predicted labels: {[labels[i] for i in topk_pred_labels_idx.cpu().numpy().flatten()]}")
	print("-"*160)

def get_image_to_images(
	dataset,
	model,
	preprocess,
	img_path:str="path/2/img.jpg",
	topk:int=5,
	batch_size:int=1024,
	):
	print(f"Image-to-Image(s) Retrieval: {img_path}".center(160, " "))

def get_text_to_images(
	dataset,
	model,
	query:str="cat",
	topk:int=5,
	batch_size:int=1024,
	device:str="cuda:0",
	):
	print(f"Top-{topk} Image Retrieval {device} CLIP Query: « {query} »".center(160, " "))
	labels = dataset.classes
	tokenized_query_tensor = clip.tokenize(texts=query).to(device, non_blocking=True) #<class 'torch.Tensor'> torch.Size([1, 77])
	query_features = model.encode_text(tokenized_query_tensor) # <class 'torch.Tensor'> torch.Size([1, 512])
	query_features /= query_features.norm(dim=-1, keepdim=True)
	# Encode all the images
	dataset_images_features = []
	for i in range(0, len(dataset), batch_size):
		batch_tensors = torch.stack([dataset[j][0].to(device, non_blocking=True) for j in range(i, min(i + batch_size, len(dataset)))]) # <class 'torch.Tensor'> torch.Size([b, 3, 224, 224]
		with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
			image_features = model.encode_image(batch_tensors)
			image_features /= image_features.norm(dim=-1, keepdim=True)
		dataset_images_features.append(image_features)
		torch.cuda.empty_cache() # Clear CUDA cache

	dataset_images_features = torch.cat(dataset_images_features, dim=0)

	# Compute similarities between query and all images
	similarities = (100.0 * query_features @ dataset_images_features.T).softmax(dim=-1)

	# Get the top-k most similar images
	topk_probs, topk_indices = similarities.topk(topk, dim=-1)
	
	# Retrieve the top-k images
	topk_pred_images = [dataset[topk_indices.squeeze().item()][0]] if topk==1 else [dataset[idx][0] for idx in topk_indices.squeeze().cpu().numpy()] # [<PIL.Image.Image image mode=RGB size=32x32 at 0x7C16A47D8F40>, ...]
	topk_pred_labels = [dataset[topk_indices.squeeze().item()][1]] if topk==1 else [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()] # [3, 3, 8, 8, 3]

	topk_probs = topk_probs.squeeze().cpu().detach().numpy()
	print(type(topk_pred_images), len(topk_pred_images), type(topk_pred_images[0]), topk_pred_images[0].size)
	print(topk_pred_labels, labels)
	print(topk_probs)

	# Save the top-k images in a single file
	fig, axes = plt.subplots(1, topk, figsize=(13, 6))
	if topk == 1:
		axes = [axes]  # Convert to list of axes
	fig.suptitle(f"Top-{topk} Result(s)\nQuery: « {query} »", fontsize=10)
	for i, (img, ax) in enumerate(zip(topk_pred_images, axes)):
		# Check if the image shape needs to be transposed
		if len(img.shape) == 3 and img.shape[0] == 3:  # If shape is (3, height, width)
			img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, 3)
		elif len(img.shape) == 3 and img.shape[2] == 3:  # If shape is already (height, width, 3)
			pass  # No need to transpose
		else:
			raise ValueError(f"Invalid image shape: {img.shape}. Expected shape is (height, width, 3) or (3, height, width).")

		# Normalize the image data to the range [0, 1]
		img_min = torch.min(img)
		img_max = torch.max(img)
		img = (img - img_min) / (img_max - img_min)

		ax.imshow(img)
		ax.axis('off')
		# ax.set_title(f"Top-{i+1}\nprob: {topk_probs[i]:.8f}\nGT: {labels[topk_pred_labels[i]]}", fontsize=9)
		if topk == 1:
			ax.set_title(f"Top-1\nprob: {topk_probs:.8f}\nGT: {labels[topk_pred_labels[0]]}", fontsize=9)
		else:
			ax.set_title(f"Top-{i+1}\nprob: {topk_probs[i]:.8f}\nGT: {labels[topk_pred_labels[i]]}", fontsize=9)
	plt.tight_layout()
	plt.savefig(
		fname=f"top{topk}_IMGs_query_{re.sub(' ', '_', query)}.png",
		dpi=250,
		bbox_inches='tight',
	)
	print("-"*160)

@measure_execution_time
def main():
	print(clip.available_models())

	model, preprocess = clip.load(args.model_name, device=args.device) # training or finetuning => jit=False
	model = model.float() # Convert model parameters to FP32
	model.name = args.model_name  # Custom attribute to store model name

	train_dataset, valid_dataset = get_dataset(
		dname=args.dataset,
		# transorm=preprocess, # from CLIP
		USER=USER,
	)
	print(f"Train dataset: {len(train_dataset)} | Validation dataset: {len(valid_dataset)}")

	if USER=="farid" and args.visualize:
		get_text_to_images(
			dataset=valid_dataset,
			model=model,
			query=args.query_label,
			topk=args.topK,
			batch_size=args.batch_size,
		)
		get_image_to_texts(
			dataset=valid_dataset,
			model=model,
			preprocess=preprocess,
			img_path=args.query_image,
			topk=args.topK,
		)
		for q in ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']:
			get_text_to_images(
				dataset=valid_dataset,
				model=model,
				query=q,
				topk=args.topK,
				batch_size=args.batch_size,
			)

	if args.topK == 1: # only Top-1 is used for zero-shot accuracy
		get_image_to_texts_linear_prob_and_zero_shot_accuracy(
			train_dataset=train_dataset,
			validation_dataset=valid_dataset,
			model=model,
			batch_size=args.batch_size,
			device=device,
			num_workers=args.num_workers,
			train_features_fname=os.path.join(OUTPUT_DIRECTORY, "train_features.gz"),
			train_labels_fname=os.path.join(OUTPUT_DIRECTORY, "train_labels.gz"),
			validation_features_fname=os.path.join(OUTPUT_DIRECTORY, "validation_features.gz"),
			validation_labels_fname=os.path.join(OUTPUT_DIRECTORY, "validation_labels.gz"),
		)

	get_text_to_images_mp_at_k(
		dataset=valid_dataset,
		model=model,
		K=args.topK,
		batch_size=args.batch_size,
		device=device,
	)

	get_image_to_texts_mp_at_k(
		dataset=valid_dataset,
		model=model,
		K=args.topK,
		device=device,
	)

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))