from utils import *

def pretrain_multilabel():
	pass

def pretrain(
		model: torch.nn.Module,
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

	validation_results = get_validation_metrics(
		model=model,
		validation_loader=validation_loader,
		criterion=torch.nn.CrossEntropyLoss(),
		device=device,
		topK_values=topk_values,
		cache_dir=cache_dir,
		verbose=True,
		embeddings_cache=embeddings_cache,
		is_training=False,
		model_hash=get_model_hash(model),
	)
	# in_batch_metrics = validation_results["in_batch_metrics"]
	# full_metrics = validation_results["full_metrics"]
	retrieval_metrics = {
		"img2txt": validation_results["img2txt_metrics"],
		"txt2img": validation_results["txt2img_metrics"]
	}
	img2txt_metrics = retrieval_metrics["img2txt"]
	txt2img_metrics = retrieval_metrics["txt2img"]

	if verbose:
		print("Image to Text Metrics: ")
		print(json.dumps(img2txt_metrics, indent=2, ensure_ascii=False))
		print("Text to Image Metrics: ")
		print(json.dumps(txt2img_metrics, indent=2, ensure_ascii=False))

	retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{dataset_name}_pretrained_{model_name}_{model_arch}_retrieval_metrics_img2txt_txt2img.png")
	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=img2txt_metrics,
		text_to_image_metrics=txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
		best_model_name=f"Pretrained {model_name} {model_arch}",
	)

	return img2txt_metrics, txt2img_metrics