from datasets_loader import *
from utils import *

parser = argparse.ArgumentParser(description="Evaluate CLIP for CIFAR10x")
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--query_image', '-qi', type=str, default="/home/farid/WS_Farid/ImACCESS/TEST_IMGs/dog.jpeg", help='image path for zero shot classification')
parser.add_argument('--query_label', '-ql', type=str, default="airplane", help='image path for zero shot classification')
parser.add_argument('--topK', '-k', type=int, default=5, help='TopK results')
parser.add_argument('--batch_size', '-bs', type=int, default=256, help='batch size')
parser.add_argument('--dataset', '-d', type=str, choices=['cifar10', 'cifar100', 'cinic10', 'imagenet'], default='cifar10', help='Choose dataset (CIFAR10/cifar100)')

args, unknown = parser.parse_known_args()
print(args)

args.device = torch.device(args.device)

# $ nohup python -u inference.py > /media/volume/ImACCESS/trash/prec_at_K.out &

USER = os.getenv('USER')
print(f"USER: {USER} device: {args.device}")

def load_model():
	model, preprocess = clip.load("ViT-B/32", device=args.device)
	model = model.float()
	input_resolution = model.visual.input_resolution
	context_length = model.context_length
	vocab_size = model.vocab_size
	print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
	print("Input resolution:", input_resolution)
	print("Context length:", context_length)
	print("Vocab size:", vocab_size)
	return model, preprocess

# def get_dataset(dname:str="CIFAR10"):
# 	if dname == 'CIFAR10':
# 			dataset = CIFAR10(
# 					root=os.path.expanduser("~/.cache"), 
# 					transform=None,
# 					download=True,
# 					train=False,  # split Test
# 			)
# 	elif dname == 'CIFAR100':
# 			dataset = CIFAR100(
# 					root=os.path.expanduser("~/.cache"), 
# 					transform=None,
# 					download=True,
# 					train=False,  # split Test
# 			)
# 	else:
# 			raise ValueError(f"Invalid dataset name: {dname}. Supported datasets are 'CIFAR10' and 'CIFAR100'.")
# 	print(dataset)
# 	return dataset

def get_dataset(dname:str="CIFAR10"):
	dname = dname.upper()
	ddir = {
		"farid": f'/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/{dname}',
		"ubuntu": f'/media/volume/ImACCESS/WW_DATASETs/{dname}',
		"alijanif": f'/scratch/project_2004072/ImACCESS/WW_DATASETs/{dname}',
	}
	if dname == 'CIFAR100':
		train_dataset = CIFAR100(
			root=os.path.expanduser("~/.cache"), 
			train=True,
			download=True,
			transform=None
		)
		validation_dataset = CIFAR100(
			root=os.path.expanduser("~/.cache"), 
			train=False,
			download=True,
			transform=None
		)
	elif dname == 'CIFAR10':
		train_dataset = CIFAR10(
			root=os.path.expanduser("~/.cache"), 
			train=True,
			download=True,
			transform=None,
		)
		validation_dataset = CIFAR10(
			root=os.path.expanduser("~/.cache"), 
			train=False,
			download=True,
			transform=None,
		)
	elif dname == 'IMAGENET':
		train_dataset = ImageNet(
			root=ddir.get(USER),
			train=True,
			transform=None
		)
		validation_dataset = ImageNet(
			root=ddir.get(USER),
			train=False,
			transform=None
	)	
	elif dname == 'CINIC10':
		train_dataset = CINIC10(
			root=ddir.get(USER),
			train=True,
			download=True,
			transform=None
		)
		validation_dataset = CINIC10(
			root=ddir.get(USER),
			train=False,
			download=True,
			transform=None
		)
	else:
		raise ValueError(f"Invalid dataset name: {dname}. Available: [CIFAR10, cifar100, IMAGENET, CINIC10]")
	print(train_dataset)
	print(validation_dataset)
	return train_dataset, validation_dataset

def get_features(dataset, model, batch_size:int=1024, device:str="cuda:0", nw:int=8):
	all_features = []
	all_labels = []
	with torch.no_grad():
		for images, labels in tqdm(
				DataLoader(
					dataset=dataset,
					batch_size=batch_size,
					num_workers=nw,
					pin_memory=True, # Move data to GPU faster if using CUDA
					persistent_workers=True if nw > 1 else False,  # Keep workers alive if memory allows
				)
			):
			features = model.encode_image(images.to(device))
			all_features.append(features)
			all_labels.append(labels)
	return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def get_linear_prob_zero_shot_accuracy(train_dataset, validation_dataset, model, preprocess, batch_size:int=1024, device:str="cuda:0"):
	# Load the dataset
	root = os.path.expanduser("~/.cache")
	# train = CIFAR10(root, download=True, train=True, transform=preprocess)
	# test = CIFAR10(root, download=True, train=False, transform=preprocess)
	train = CIFAR100(root, download=True, train=True, transform=preprocess)
	test = CIFAR100(root, download=True, train=False, transform=preprocess)

	print(train)
	print("-"*25)
	print(test)
	# Calculate the image features
	print(f"Getting training features", end="\t")
	t0 = time.time()
	train_features, train_labels = get_features(
		dataset=train,
		model=model,
		batch_size=batch_size,
		device=device,
	)
	print(f"Elapsed_t: {time.time()-t0:.2f} sec")

	print(f"Getting test features", end="\t")
	t0 = time.time()
	test_features, test_labels = get_features(
		dataset=test,
		model=model,
		batch_size=batch_size,
		device=device,
	)
	print(f"Elapsed_t: {time.time()-t0:.2f} sec")

	# Perform logistic regression
	classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
	classifier.fit(train_features, train_labels)

	# Evaluate using the logistic regression classifier
	predictions = classifier.predict(test_features)
	accuracy = np.mean((test_labels == predictions).astype(float))# * 100.
	print(f"Linear Probe Accuracy = {accuracy:.3f}")

	################################## Zero Shot Classifier ##################################
	# Get the class names
	class_names = test.classes

	# Encode the text descriptions of the classes
	text_descriptions = [f"a photo of a {label}" for label in class_names]
	text_inputs = torch.cat([clip.tokenize(desc) for desc in text_descriptions]).to(device)
	with torch.no_grad():
		text_features = model.encode_text(text_inputs)

	# Normalize the features
	test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
	text_features = text_features / text_features.norm(dim=-1, keepdim=True)

	# Convert test_features to a PyTorch tensor
	test_features = torch.from_numpy(test_features).to(device)

	# Calculate the similarity scores
	similarity_scores = (100.0 * test_features @ text_features.T).softmax(dim=-1)

	# Get the predicted class indices
	predicted_class_indices = np.argmax(similarity_scores.cpu().numpy(), axis=1)

	# Calculate the accuracy
	accuracy = np.mean((test_labels == predicted_class_indices).astype(float))# * 100.
	print(f"Zero-shot Accuracy = {accuracy:.3f}")
	################################## Zero Shot Classifier ##################################

	return accuracy

def get_image_to_texts(dataset, model, preprocess, img_path, topk:int=5):
	print(f"Zero-Shot Image Classification: {img_path}".center(160, " "))
	labels = dataset.classes
	if topk > len(labels):
		print(f"ERROR: requested Top-{topk} labeling is greater than number of labels({len(labels)}) => EXIT...")
		return
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(args.device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	img = Image.open(img_path)
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
	print("-"*160)

def get_image_to_texts_precision_at_(dataset, model, preprocess, K:int=5):
	print(f"Zero-Shot Image Classification {args.device} CLIP [performance metrics: Precision@{K}]".center(160, " "))
	labels = dataset.classes # <class 'list'> ['airplane', 'automobile', ...]
	if K > len(labels):
		print(f"ERROR: requested Top-{K} labeling is greater than number of labels({len(labels)}) => EXIT...")
		return
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(args.device) # torch.Size([num_lbls, context_length]) # ex) 10 x 77
	labels_features = model.encode_text(tokenized_labels_tensor)
	labels_features /= labels_features.norm(dim=-1, keepdim=True)

	predicted_labels = []
	true_labels = []
	floop_st = time.time()

	with torch.no_grad():
		for i, (img_raw, gt_lbl) in enumerate(dataset): #img: <class 'PIL.Image.Image'>
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
	print(f"[OWN] Precision@{K}: {prec_at_k} | {avg_prec_at_k:.3f} Elapsed_t: {time.time()-pred_st:.2f} sec")
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
	print("-"*160)

def get_text_to_images(dataset, model, preprocess, query:str="cat", topk:int=5, batch_size:int=1024):
	print(f"Top-{topk} Image Retrieval {args.device} CLIP Query: « {query} »".center(160, " "))
	labels = dataset.classes
	tokenized_query_tensor = clip.tokenize(texts=query).to(args.device) #<class 'torch.Tensor'> torch.Size([1, 77])
	query_features = model.encode_text(tokenized_query_tensor) # <class 'torch.Tensor'> torch.Size([1, 512])
	query_features /= query_features.norm(dim=-1, keepdim=True)
	# Encode all the images
	all_image_features = []
	for i in range(0, len(dataset), batch_size):
		batch_images = [dataset[j][0] for j in range(i, min(i + batch_size, len(dataset)))]
		batch_tensors = torch.stack([preprocess(img).to(args.device) for img in batch_images])
		with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
			image_features = model.encode_image(batch_tensors)
			image_features /= image_features.norm(dim=-1, keepdim=True)
		all_image_features.append(image_features)
		torch.cuda.empty_cache() # Clear CUDA cache

	all_image_features = torch.cat(all_image_features, dim=0)

	# Compute similarities between query and all images
	similarities = (100.0 * query_features @ all_image_features.T).softmax(dim=-1)

	# Get the top-k most similar images
	topk_probs, topk_indices = similarities.topk(topk, dim=-1)
	
	# Retrieve the top-k images
	topk_pred_images = [dataset[topk_indices.squeeze().item()][0]] if topk==1 else [dataset[idx][0] for idx in topk_indices.squeeze().cpu().numpy()] # [<PIL.Image.Image image mode=RGB size=32x32 at 0x7C16A47D8F40>, ...]
	topk_pred_labels = [dataset[topk_indices.squeeze().item()][1]] if topk==1 else [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()] # [3, 3, 8, 8, 3]

	topk_probs = topk_probs.squeeze().cpu().detach().numpy()
	print(topk_pred_images)
	print(topk_pred_labels, labels)
	print(topk_probs)

	# Save the top-k images in a single file
	fig, axes = plt.subplots(1, topk, figsize=(16, 8))
	if topk == 1:
		axes = [axes]  # Convert to list of axes
	fig.suptitle(f"Top-{topk} Query: {query}", fontsize=11)
	for i, (img, ax) in enumerate(zip(topk_pred_images, axes)):
		ax.imshow(img)
		ax.axis('off')
		# ax.set_title(f"Top-{i+1}\nprob: {topk_probs[i]:.8f}\nGT: {labels[topk_pred_labels[i]]}", fontsize=9)
		if topk == 1:
			ax.set_title(f"Top-1\nprob: {topk_probs:.8f}\nGT: {labels[topk_pred_labels[0]]}", fontsize=9)
		else:
			ax.set_title(f"Top-{i+1}\nprob: {topk_probs[i]:.8f}\nGT: {labels[topk_pred_labels[i]]}", fontsize=9)
				
	plt.tight_layout()
	plt.savefig(f"top{topk}_IMGs_query_{re.sub(' ', '_', query)}.png")
	print("-"*160)

def get_text_to_images_precision_recall_at_(dataset, model, preprocess, K:int=5, batch_size:int=1024):
	torch.cuda.empty_cache()  # Clear CUDA cache
	print(f"Image Retrieval {args.device} CLIP [performance metrics: Precision@{K}]".center(160, " "))
	labels = dataset.classes
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(args.device)#<class 'torch.Tensor'> torch.Size([num_lbls, 77])
	tokenized_labels_features = model.encode_text(tokenized_labels_tensor) # <class 'torch.Tensor'> torch.Size([num_lbls, 512])
	tokenized_labels_features /= tokenized_labels_features.norm(dim=-1, keepdim=True)
	# Encode all the images
	all_image_features = []
	for i in range(0, len(dataset), batch_size):
		batch_images = [dataset[j][0] for j in range(i, min(i + batch_size, len(dataset)))]
		batch_tensors = torch.stack([preprocess(img).to(args.device) for img in batch_images])
		with torch.no_grad(): # prevent PyTorch from computing gradients, can consume significant memory
			image_features = model.encode_image(batch_tensors)
			image_features /= image_features.norm(dim=-1, keepdim=True)
		all_image_features.append(image_features)
		torch.cuda.empty_cache()  # Clear CUDA cache

	all_image_features = torch.cat(all_image_features, dim=0)

	###################################### Wrong approach for P@K and R@K ######################################
	# it is rather accuracy or hit rate!
	# prec_at_k = 0
	# recall_at_k = []
	# for i, label_features in enumerate(tokenized_labels_features):
	# 	sim = (100.0 * label_features @ all_image_features.T).softmax(dim=-1) # similarities between query and all images
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

	prec_at_k = []
	recall_at_k = []
	for i, label_features in enumerate(tokenized_labels_features):
		sim = (100.0 * label_features @ all_image_features.T).softmax(dim=-1) # similarities between query and all images
		topk_probs, topk_indices = sim.topk(K, dim=-1)
		# topk_pred_labels = [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()] # [3, 3, 8, 8, 3] # only K@(>1)
		topk_pred_labels = [dataset[topk_indices.squeeze().item()][1]] if K==1 else [dataset[idx][1] for idx in topk_indices.squeeze().cpu().numpy()]# K@1, 5, ...
		relevant_retrieved_images_for_label_i = topk_pred_labels.count(i)  # counting relevant images in top-K retrieved images
		prec_at_k.append(relevant_retrieved_images_for_label_i/K)
		all_images_with_label_i = [idx for idx, (img, lbl) in enumerate(dataset) if lbl == i]
		num_all_images_with_label_i = len(all_images_with_label_i)
		recall_at_k.append(relevant_retrieved_images_for_label_i/num_all_images_with_label_i)

	avg_prec_at_k = sum(prec_at_k)/len(labels)
	avg_recall_at_k = sum(recall_at_k) / len(labels)
	# print(f"Precision@{K}: {prec_at_k} {avg_prec_at_k} {np.mean(prec_at_k)}")
	# print(f"Recall@{K}: {recall_at_k} {avg_recall_at_k} {np.mean(recall_at_k)}")
	print(f"Precision@{K}: {avg_prec_at_k} {np.mean(prec_at_k)}")
	print(f"Recall@{K}: {avg_recall_at_k} {np.mean(recall_at_k)}")
	print(labels)

	fpr_values = []
	tpr_values = []
	precision_values = []
	recall_values = []
	for i, label_features in enumerate(tokenized_labels_features):
		sim = (100.0 * label_features @ all_image_features.T).softmax(dim=-1)
		sim = sim.squeeze().cpu().detach().numpy()
		predicted_labels = np.argsort(-sim)
		true_labels = [1 if dataset[j][1] == i else 0 for j in range(len(dataset))]
		prec, rec, thresh = precision_recall_curve(true_labels, sim)
		fpr, tpr, _ = roc_curve(true_labels, sim)
		precision_values.append(prec)
		recall_values.append(rec)
		fpr_values.append(fpr)
		tpr_values.append(tpr)

	fig, ax = plt.subplots(1, 2, figsize=(22,11))
	for i in range(len(tokenized_labels_features)):
		ax[0].plot(recall_values[i], precision_values[i], label=f'{labels[i]}')
		ax[0].set_xlabel('Recall')
		ax[0].set_ylabel('Precision')
		ax[0].set_title('Precision-Recall')
		ax[1].plot(fpr_values[i], tpr_values[i])
		ax[1].set_xlabel('False Positive')
		ax[1].set_ylabel('True Positive')
		ax[1].set_title('ROC')

	fig.legend(bbox_to_anchor=(0.5, 0.99), fontsize=7, ncol=len(labels), frameon=False)
	# fig.tight_layout()
	plt.savefig(f"{args.dataset}_PR_ROC_x{len(labels)}_labels.png")

	print("-"*160)

def get_image_to_images(dataset, model, preprocess, img_path:str="path/2/img.jpg", topk:int=5, batch_size:int=1024):
	print(f"Image-to-Image(s) Retrieval: {img_path}".center(160, " "))

def main():
	print(clip.available_models())
	model, preprocess = load_model()
	# dataset = get_dataset(dname=args.dataset)
	train_dataset, valid_dataset = get_dataset(dname=args.dataset)

	if USER == "farid":
		get_image_to_texts(
			dataset=valid_dataset,
			model=model,
			preprocess=preprocess,
			img_path=args.query_image,
			topk=args.topK,
		)

	get_linear_prob_zero_shot_accuracy(
		train_dataset=train_dataset,
		validation_dataset=valid_dataset,
		model=model,
		preprocess=preprocess,
		batch_size=args.batch_size,
		device=args.device,
	)

	get_image_to_texts_precision_at_(
		dataset=valid_dataset,
		model=model,
		preprocess=preprocess,
		K=args.topK,
	)

	# if USER == "farid":
	# 	get_text_to_images(
	# 		dataset=valid_dataset,
	# 		model=model,
	# 		preprocess=preprocess,
	# 		query=args.query_label,
	# 		topk=args.topK,
	# 		batch_size=args.batch_size,
	# 	)

	# if USER == "farid":
	# 	for q in ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']:
	# 		get_text_to_images(
	# 			dataset=valid_dataset,
	# 			model=model,
	# 			preprocess=preprocess,
	# 			query=q,
	# 			topk=args.topK,
	# 			batch_size=args.batch_size,
	# 		)

	# get_text_to_images_precision_recall_at_(
	# 	dataset=valid_dataset,
	# 	model=model,
	# 	preprocess=preprocess,
	# 	K=args.topK,
	# 	batch_size=args.batch_size,
	# )

if __name__ == "__main__":
	main()