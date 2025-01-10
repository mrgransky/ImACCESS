from utils import *

# local:
# $ python evaluate_clip.py -ddir /home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--query_image', '-qi', type=str, default="/home/farid/WS_Farid/ImACCESS/TEST_IMGs/5968_115463.jpg", help='image path for zero shot classification')
parser.add_argument('--query_label', '-ql', type=str, default="naval forces", help='image path for zero shot classification')
parser.add_argument('--topK', '-k', type=int, default=5, help='TopK results')
parser.add_argument('--batch_size', '-bs', type=int, default=1024, help='TopK results')
parser.add_argument('--model_name', '-md', type=str, default="ViT-B/32", help='CLIP model name')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)

args.device = torch.device(args.device)

USER = os.getenv('USER')
print(f"USER: {USER} device: {args.device}")

os.makedirs(os.path.join(args.dataset_dir, "outputs"), exist_ok=True)
outputs_dir:str = os.path.join(args.dataset_dir, "outputs",)

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

def get_dataset(ddir:str="path/2/dataset_dir", sliced:bool=False):
	metadata_fpth = os.path.join(ddir, "metadata.csv")
	img_dir = os.path.join(ddir, "images")

	df = pd.read_csv(
		filepath_or_buffer=metadata_fpth,
		on_bad_lines='skip',
	)

	if sliced:
		df = df.iloc[:1500]

	labels = list(set(df["label"].tolist()))

	# Create a dictionary that maps each label to its index
	label_dict = {label: index for index, label in enumerate(labels)}

	# Map the labels to their indices
	df['label_int'] = df['label'].map(label_dict)

	return df

def get_image_to_texts(dataset, model, preprocess, img_path, topk:int=5):
	print(f"[Image-to-text(s)] Zero-Shot Image Classification of image: {img_path}".center(160, " "))
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	print(len(labels), type(labels))
	if topk > len(labels):
		print(f"ERROR: requested Top-{topk} labeling is greater than number of labels({len(labels)}) => EXIT...")
		return
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(args.device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	
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

	image_tensor = preprocess(img).unsqueeze(0).to(args.device) # <class 'torch.Tensor'> torch.Size([1, 3, 224, 224])

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
	plt.figure(figsize=(15, 10))
	plt.imshow(img)
	plt.axis('off')
	plt.title(f'Top-{topk} predicted labels: {[labels[i] for i in topk_pred_labels_idx.cpu().numpy().flatten()]}', fontsize=10)
	plt.tight_layout()
	plt.savefig(os.path.join(outputs_dir, f'Img2Txt_Top{topk}_LBLs_IMG_{img_hash}_dataset_{os.path.basename(args.dataset_dir)}.png'))
	plt.close()

def get_image_to_texts_precision_at_(dataset, model, preprocess, K:int=5):
	print(f"Zero-Shot Image Classification {args.device} CLIP [performance metrics: Precision@{K}]".center(160, " "))
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
	
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(args.device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	labels_features = model.encode_text(tokenized_labels_tensor)
	labels_features /= labels_features.norm(dim=-1, keepdim=True)

	predicted_labels = []
	true_labels = []
	floop_st = time.time()

	with torch.no_grad():
		for i, (img_pth, gt_lbl) in enumerate(zip(dataset_images_path, dataset_labels_int)): #img: <class 'PIL.Image.Image'>
			# print(i, img_pth, gt_lbl)
			img_raw = Image.open(img_pth)
			img_tensor = preprocess(img_raw).unsqueeze(0).to(args.device)
			image_features = model.encode_image(img_tensor)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
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

def get_text_to_images(dataset, model, preprocess, query:str="cat", topk:int=5, batch_size:int=1500):
	print(f"Top-{topk} Image Retrieval {args.device} CLIP Query: « {query} »".center(160, " "))
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	print(len(labels), type(labels))

	dataset_images_id = dataset["id"].tolist()
	dataset_images_path = dataset["img_path"].tolist()
	dataset_labels = dataset["label"].tolist() # ['naval training', 'medical service', 'medical service', 'naval forces', 'naval forces', ...]
	dataset_labels_int = dataset["label_int"].tolist() # [3, 17, 4, 9, ...]
	print(len(dataset_images_id), len(dataset_labels))
	
	tokenized_query_tensor = clip.tokenize(texts=query).to(args.device)#<class 'torch.Tensor'> torch.Size([1, 77])
	query_features = model.encode_text(tokenized_query_tensor) # <class 'torch.Tensor'> torch.Size([1, 512])
	query_features /= query_features.norm(dim=-1, keepdim=True)
	
	image_features_file = os.path.join(args.dataset_dir, 'outputs', 'image_features.gz')
	if not os.path.exists(image_features_file):
		dataset_images_features = []
		for i in range(0, len(dataset_images_path), batch_size):
			batch_images_path = [dataset_images_path[j] for j in range(i, min(i + batch_size, len(dataset_images_path)))]
			batch_tensors = torch.stack([preprocess(Image.open(img_path)).to(args.device) for img_path in batch_images_path])
			with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
				image_features = model.encode_image(batch_tensors)
				image_features /= image_features.norm(dim=-1, keepdim=True)
			dataset_images_features.append(image_features)
			torch.cuda.empty_cache()  # Clear CUDA cache
		dataset_images_features = torch.cat(dataset_images_features, dim=0)
		save_pickle(pkl=dataset_images_features, fname=image_features_file)
	else:
		dataset_images_features = load_pickle(fpath=image_features_file)

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
	print(len(labels), labels)

	# Save the top-k images in a single file
	fig, axes = plt.subplots(1, topk, figsize=(16, 8))
	if topk == 1:
		axes = [axes]  # Convert to list of axes
	fig.suptitle(f"Top-{topk} Image(s) Query: {query}\nSource Dataset: {os.path.basename(args.dataset_dir)}", fontsize=11)
	for i, (img, ax) in enumerate(zip(topk_pred_images, axes)):
		ax.imshow(img)
		ax.axis('off')
		if topk == 1:
			ax.set_title(f"Top-1\nprob: {topk_probs:.4f}\nGT: {labels[topk_pred_labels_idxs[0]]}", fontsize=9)
		else:
			ax.set_title(f"Top-{i+1}\nprob: {topk_probs[i]:.4f}\nGT: {labels[topk_pred_labels_idxs[i]]}", fontsize=9)
				
	plt.tight_layout()
	plt.savefig(os.path.join(outputs_dir, f"Txt2Img_Top{topk}_IMGs_dataset_{os.path.basename(args.dataset_dir)}_query_{re.sub(' ', '_', query)}.png"))
	print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))

def get_text_to_images_precision_recall_at_(dataset, model, preprocess, K: int = 5, batch_size: int = 1024):
	print(f"Image Retrieval {args.device} CLIP [performance metrics: Precision@{K}]".center(160, " "))
	t0 = time.time()
	labels = list(set(dataset["label"].tolist()))
	print(len(labels), type(labels))
	dataset_images_id = dataset["id"].tolist()
	dataset_images_path = dataset["img_path"].tolist()
	dataset_labels = dataset["label"].tolist()
	dataset_labels_int = dataset["label_int"].tolist()
	print(len(dataset_images_id), len(dataset_labels))
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(args.device)
	tokenized_labels_features = model.encode_text(tokenized_labels_tensor)
	tokenized_labels_features /= tokenized_labels_features.norm(dim=-1, keepdim=True)
	image_features_file = os.path.join(args.dataset_dir, 'outputs', 'image_features.gz')
	if not os.path.exists(image_features_file):
		dataset_images_features = []
		for i in range(0, len(dataset_images_path), batch_size):
			batch_images_path = [dataset_images_path[j] for j in range(i, min(i + batch_size, len(dataset_images_path)))]
			batch_tensors = torch.stack([preprocess(Image.open(img_path)).to(args.device) for img_path in batch_images_path])
			with torch.no_grad():
				image_features = model.encode_image(batch_tensors)
				image_features /= image_features.norm(dim=-1, keepdim=True)
			dataset_images_features.append(image_features)
			torch.cuda.empty_cache()
		dataset_images_features = torch.cat(dataset_images_features, dim=0)
		save_pickle(pkl=dataset_images_features, fname=image_features_file)
	else:
		dataset_images_features = load_pickle(fpath=image_features_file)
	prec_at_k = []
	recall_at_k = []
	for i, label_features in enumerate(tokenized_labels_features):
		sim = label_features @ dataset_images_features.T # compute similarity between the label and all images
		_, indices = sim.topk(len(dataset_images_features), dim=-1) # retrieve all images for each label
		relevant_images_for_lbl_i = [idx for idx, lbl in enumerate(dataset_labels_int) if lbl == i] # retrieve all images with same label
		retrieved_topK_relevant_images = [idx for idx in indices.squeeze().cpu().numpy()[:K] if idx in relevant_images_for_lbl_i] # retrieve topK relevant images in the top-K retrieved images
		prec_at_k.append(len(retrieved_topK_relevant_images) / K)
		recall_at_k.append(len(retrieved_topK_relevant_images) / len(relevant_images_for_lbl_i))
	avg_prec_at_k = sum(prec_at_k) / len(labels)
	avg_recall_at_k = sum(recall_at_k) / len(labels)
	print(f"Precision@{K}: {avg_prec_at_k:.3f} {np.mean(prec_at_k)}")
	print(f"Recall@{K}: {avg_recall_at_k:.3f} {np.mean(recall_at_k)}")
	print(f"Elapsed_t: {time.time() - t0:.2f} sec".center(160, "-"))

def get_map_at_k(dataset, model, preprocess, K: int, batch_size: int, device):
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
	image_features_file = os.path.join(args.dataset_dir, 'outputs', 'image_features.gz')  
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

	# Image-to-Texts Retrieval
	img_to_txt_precisions = []  
	tokenized_labels_tensor = clip.tokenize(texts=list(set(dataset["label"]))).to(device)  
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
	tokenized_labels_tensor = clip.tokenize(texts=list(set(dataset["label"]))).to(device)  
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

def get_image_to_images(dataset, query_image_path, model, preprocess, topk: int, batch_size: int, device):
	print(f"Image-to-Image(s) Retrieval {query_image_path}".center(160, " "))
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
	image_features_file = os.path.join(args.dataset_dir, 'outputs', 'image_features.gz')  
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
	fig, axes = plt.subplots(2, topk, figsize=(5 * topk, 10))
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
	plt.suptitle(f"Image-to-Image Retrieval\nTop-{topk} Image(s) Source Dataset:{os.path.basename(args.dataset_dir)}", fontsize=16)
	# Save the plot
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(outputs_dir, f"IMG2IMG_Top{topk}_IMGs_{os.path.basename(args.dataset_dir)}.png"),
		dpi=250,
		bbox_inches='tight',
	)
	plt.close()
	print(f"Elapsed_t: {time.time() - t0:.2f} sec".center(160, "-"))
	return topk_image_paths, topk_labels, topk_similarities

@measure_execution_time
def main():
	print(clip.available_models())
	model, preprocess = load_model(model_name=args.model_name, device=args.device)

	dataset = get_dataset(
		ddir=args.dataset_dir,
		sliced=False,
	)
	print(type(dataset), dataset.shape)
	print(list(dataset.columns))
	print(dataset.head(10))
	return
	get_image_to_images(
		dataset=dataset,
		query_image_path=args.query_image,
		model=model,
		preprocess=preprocess,
		topk=args.topK,
		batch_size=args.batch_size,
		device=args.device,
	)

	# img_to_txt_map_at_k, txt_to_img_map_at_k = get_map_at_k(
	# 	dataset=dataset,
	# 	model=model,
	# 	preprocess=preprocess,
	# 	K=args.topK,
	# 	batch_size=args.batch_size,
	# 	device=args.device
	# )

	if USER == "farid":
		get_image_to_texts(
			dataset=dataset,
			model=model,
			preprocess=preprocess,
			img_path=args.query_image,
			topk=args.topK,
		)

	# get_image_to_texts_precision_at_(
	# 	dataset=dataset,
	# 	model=model,
	# 	preprocess=preprocess,
	# 	K=args.topK,
	# )

	if USER == "farid":
		get_text_to_images(
			dataset=dataset,
			model=model,
			preprocess=preprocess,
			query=args.query_label,
			topk=args.topK,
			batch_size=args.batch_size,
		)

	# get_text_to_images_precision_recall_at_(
	# 	dataset=dataset,
	# 	model=model,
	# 	preprocess=preprocess,		
	# 	K=args.topK,
	# 	batch_size=args.batch_size,
	# )

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))