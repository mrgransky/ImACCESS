from utils import *
from evals import get_validation_metrics
import visualize as viz

def pretrain_multi_label():
	pass

def extract_features(
	loader: DataLoader,
	model: torch.nn.Module,
	device: torch.device,
	desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Extract L2-normalized image features + integer labels from a DataLoader.
	"""
	model.eval()
	all_features = []
	all_labels = []
	with torch.no_grad():
		for batch in tqdm(loader, desc=desc):
			# Single-label loaders return (images, tokenized_text, labels_int)
			# We only need images and labels_int
			if len(batch) == 3:
				images, _, labels = batch
			else:
				images, labels = batch  # fallback
			images = images.to(device, non_blocking=True)
			
			with torch.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(), 
				dtype=torch.float16 if device.type == "cuda" else torch.float32
			):
				feats = model.encode_image(images)
				feats = feats / feats.norm(dim=-1, keepdim=True) # CLIP already normalizes, but we force it to be safe
			
			all_features.append(feats.cpu())
			all_labels.append(labels.cpu())
		features = torch.cat(all_features, dim=0).numpy()
		labels = torch.cat(all_labels, dim=0).numpy()

	return features, labels

def pretrain_single_label(
		model: torch.nn.Module,
		train_loader: Optional[DataLoader],
		validation_loader: DataLoader,
		device: torch.device,
		results_dir: str,
		cache_dir: str=None,
		topk_values: List=[1, 3, 5],
		verbose:bool=True,
		embeddings_cache=None,
	):
	model_name = model.__class__.__name__
	model_arch = re.sub(r"[/@]", "_", model.name)
	if cache_dir is None:
		cache_dir = results_dir
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except:
		dataset_name = validation_loader.dataset.dataset_name

	if verbose:
		print(f"Pretrain Evaluation {dataset_name} {model_name} - {model_arch} {device}".center(170, "-"))

	# 1. Zero-shot retrieval metrics
	criterion = torch.nn.CrossEntropyLoss()

	validation_results = get_validation_metrics(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		device=device,
		topK_values=topk_values,
		cache_dir=cache_dir,
		verbose=verbose,
		embeddings_cache=embeddings_cache,
		is_training=False,
		model_hash=get_model_hash(model),
	)
	if verbose:
		print(json.dumps(validation_results, indent=2, ensure_ascii=False))

	retrieval_metrics = {
		"img2txt": validation_results["img2txt_metrics"],
		"txt2img": validation_results["txt2img_metrics"]
	}
	img2txt_metrics = retrieval_metrics["img2txt"]
	txt2img_metrics = retrieval_metrics["txt2img"]

	retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{dataset_name}_pretrained_{model_name}_{model_arch}_retrieval_metrics_img2txt_txt2img.png")
	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=img2txt_metrics,
		text_to_image_metrics=txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
		best_model_name=f"Pretrained {model_name} {model_arch}",
	)

	# 2. Linear Probe evaluation
	linear_probe_accuracy: Optional[float] = None
	if train_loader is not None:
		print("\n>>> Starting Linear Probe evaluation on frozen pre-trained CLIP features <<<\n")
		train_features, train_labels = extract_features(train_loader, model, device, desc="Train features")
		val_features,   val_labels   = extract_features(validation_loader, model, device, desc="Val features")
		print(
			f"Train samples: {len(train_labels):,}, Feature: {train_features.shape}, "
			f"Val samples: {len(val_labels):,}, Feature: {val_features.shape}"
		)
		# Standard CLIP linear probe: sweep C in logspace
		Cs = np.logspace(-4, 4, 9)   # 0.0001, 0.001, ..., 10000.0
		best_acc = 0.0
		best_C = None
		for C in Cs:
			clf = LogisticRegression(
				C=C,
				max_iter=2000,
				random_state=42,
				solver="saga",
				multi_class="multinomial",
				tol=1e-3,
				verbose=1 if verbose else 0,
				n_jobs=-1,
				warm_start=False,
			)
			clf.fit(train_features, train_labels)

			if verbose:
				print("Evaluating on validation set...")

			predictions = clf.predict(val_features)
			accuracy = np.mean((val_labels == predictions).astype(float))

			if verbose:
				print(f"Linear Probe Accuracy: {accuracy:.4f}")

			acc = clf.score(val_features, val_labels)
			print(f"C = {C:<15}Accuracy = {acc:.4f}")
			print("-"*100)
			if acc > best_acc:
				best_acc = acc
				best_C = C
		print(f"\nBest Linear Probe Accuracy = {best_acc:.4f} (C = {best_C})\n")
		linear_probe_accuracy = best_acc
	else:
		if verbose:
			print("train_loader not provided → skipping linear probe evaluation")

	return img2txt_metrics, txt2img_metrics, linear_probe_accuracy
