from utils import *
from dataset_loader import get_dataloaders
from trainer import train, pretrain, full_finetune, lora_finetune, progressive_unfreeze_finetune

# train cifar100 from scratch:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -e 250 -lr 1e-4 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:3" -m train -a "ViT-B/32" -do 0.1 > /media/volume/ImACCESS/trash/cifar100_train.out &

# finetune cifar100:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -e 250 -lr 1e-4 -wd 1e-3 --print_every 100 -nw 50 --device "cuda:2" -m finetune -a "ViT-B/32" > /media/volume/ImACCESS/trash/cifar100_ft.out &

# finetune cifar100 with lora:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -e 250 -lr 1e-4 -wd 1e-3 --print_every 100 -nw 50 --device "cuda:2" -m finetune -fts "lora"  -a "ViT-B/32" --lora > /media/volume/ImACCESS/trash/cifar100_ft_lora.out &

# finetune cifar100 with progressive unfreezing:
# $ nohup python -u trainer.py -d cifar100 -bs 128 -e 250 -lr 1e-4 -wd 1e-2 --print_every 200 -nw 50 --device "cuda:2" -m finetune -fts progressive -a "ViT-B/32"  > /media/volume/ImACCESS/trash/cifar100_ft_progressive.out &

# finetune svhn with progressive unfreezing:
# $ nohup python -u trainer.py -d svhn -bs 512 -e 250 -lr 1e-5 -wd 1e-1 --print_every 50 -nw 50 --device "cuda:0" -m finetune -fts progressive -a "ViT-B/32" > /media/volume/ImACCESS/trash/svhn_ft_progreessive.out &

# finetune imagenet [full]:
# $ nohup python -u trainer.py -d imagenet -bs 256 -e 250 -lr 1e-4 -wd 1e-2 --print_every 2500 -nw 50 --device "cuda:0" -m finetune -a "ViT-B/32" > /media/volume/ImACCESS/trash/imagenet_ft.out &

# finetune imagenet with progressive unfreezing:
# $ nohup python -u trainer.py -d imagenet -bs 256 -e 250 -lr 1e-4 -wd 1e-2 --print_every 2500 -nw 50 --device "cuda:0" -m finetune -fts progressive -a "ViT-B/32" > /media/volume/ImACCESS/trash/imagenet_prog_unfreeze_ft.out &

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Balanced Dataset. Note: 'train' mode always initializes with random weights, while 'pretrain' and 'finetune' use pre-trained OpenAI weights.")
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=10, help='Number of CPUs')
	parser.add_argument('--epochs', '-e', type=int, default=12, help='Number of epochs')
	parser.add_argument('--batch_size', '-bs', type=int, default=8, help='Batch size for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='small learning rate for better convergence [def: 1e-4]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='Weight decay [def: 1e-3]')
	parser.add_argument('--print_every', type=int, default=250, help='Print every [def: 250]')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP Architecture (ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px)')
	parser.add_argument('--dataset', '-d', type=str, choices=['cifar10', 'cifar100', 'cinic10', 'imagenet', 'svhn'], default='cifar100', help='Choose dataset (CIFAR10/cifar100)')
	parser.add_argument('--mode', '-m', type=str, choices=['pretrain', 'train', 'finetune'], default='pretrain', help='Choose mode (pretrain/train/finetune)')
	parser.add_argument('--finetune_strategy', '-fts', type=str, choices=['full', 'lora', 'progressive'], default='full', help='Fine-tuning strategy (full/lora/progressive) when mode is finetune')
	parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank (used if finetune_strategy=lora)')
	parser.add_argument('--lora_alpha', type=float, default=16.0, help='LoRA alpha (used if finetune_strategy=lora)')
	parser.add_argument('--lora_dropout', type=float, default=0.0, help='Regularizes trainable LoRA parameters, [primary focus of fine-tuning]')
	parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
	parser.add_argument('--minimum_delta', '-mdelta', type=float, default=1e-4, help='Min delta for early stopping & progressive freezing [Platueau threshhold]')
	parser.add_argument('--cumulative_delta', '-cdelta', type=float, default=5e-3, help='Cumulative delta for early stopping')
	parser.add_argument('--minimum_epochs', type=int, default=15, help='Early stopping minimum epochs')
	parser.add_argument('--topK_values', '-k', type=int, nargs='+', default=[1, 5, 10, 15, 20], help='Top K values for retrieval metrics')
	parser.add_argument('--dropout', '-do', type=float, default=0.0, help='Dropout rate for the base model')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print(type(args.device), args.device, torch.cuda.device_count(), args.device.index)

	print_args_table(args=args, parser=parser)
	set_seeds()
	print(clip.available_models()) # List all available CLIP models
	# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	print(f">> CLIP Model Architecture: {args.model_architecture}...")
	model_config = get_config(architecture=args.model_architecture, dropout=args.dropout,)

	print(json.dumps(model_config, indent=4, ensure_ascii=False))
	model, preprocess = clip.load(
		name=args.model_architecture,
		device=args.device, 
		jit=False, # training or finetuning => jit=False
		random_weights=True if args.mode == 'train' else False, 
		dropout=args.dropout,
	)
	model = model.float() # Convert model parameters to FP32
	model.name = args.model_architecture  # Custom attribute to store model name
	print(f"Model: {model.__class__.__name__} loaded with {model.name} architecture on {args.device} device")
	# print(model.visual.conv1.weight[0, 0, 0])  # Random value (not zeros or pretrained values)
	# print(f"embed_dim: {model.text_projection.size(0)}, transformer_width: {model.text_projection.size(1)}")

	train_loader, validation_loader = get_dataloaders(
		dataset_name=args.dataset,
		batch_size=args.batch_size,
		nw=args.num_workers,
		USER=os.environ.get('USER'),
		input_resolution=model_config["image_resolution"],
		preprocess=None,# preprocess,
	)
	print_loader_info(loader=train_loader, batch_size=args.batch_size)
	print_loader_info(loader=validation_loader, batch_size=args.batch_size)

	# visualize_(dataloader=train_loader, num_samples=5)

	if args.mode == 'finetune':
		if args.finetune_strategy == 'full':
			full_finetune(
				model=model,
				train_loader=train_loader,
				validation_loader=validation_loader,
				num_epochs=args.epochs,
				nw=args.num_workers,
				print_every=args.print_every,
				learning_rate=args.learning_rate,
				weight_decay=args.weight_decay,
				device=args.device,
				results_dir=os.path.join(args.dataset, "results"),
				patience=10, 										# early stopping
				min_delta=1e-4, 								# early stopping & progressive unfreezing
				cumulative_delta=5e-3, 					# early stopping
				minimum_epochs=20, 							# early stopping
				topk_values=args.topK_values,
			)
		elif args.finetune_strategy == 'lora':
			lora_finetune(
				model=model,
				train_loader=train_loader,
				validation_loader=validation_loader,
				num_epochs=args.epochs,
				nw=args.num_workers,
				print_every=args.print_every,
				learning_rate=args.learning_rate,
				weight_decay=args.weight_decay,
				device=args.device,
				results_dir=os.path.join(args.dataset, "results"),
				lora_rank=args.lora_rank,
				lora_alpha=args.lora_alpha,
				lora_dropout=args.lora_dropout,
				patience=args.patience,
				min_delta=args.minimum_delta,
				cumulative_delta=args.cumulative_delta,
				minimum_epochs=args.minimum_epochs,
				topk_values=args.topK_values,
			)
		elif args.finetune_strategy == 'progressive':
			progressive_unfreeze_finetune(
				model=model,
				train_loader=train_loader,
				validation_loader=validation_loader,
				num_epochs=args.epochs,
				nw=args.num_workers,
				print_every=args.print_every,
				initial_learning_rate=args.learning_rate,
				weight_decay=args.weight_decay,
				device=args.device,
				results_dir=os.path.join(args.dataset, "results"),
				patience=10,									# early stopping
				min_delta=1e-4,								# early stopping
				cumulative_delta=5e-3,				# early stopping and progressive unfreezing
				minimum_epochs=20,						# early stopping
				top_k_values=args.topK_values,
			)
		else:
			raise ValueError(f"Invalid mode: {args.mode}")
	elif args.mode == 'train':
		train(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			device=args.device,
			results_dir=os.path.join(args.dataset, "results"),
			patience=10,									# early stopping
			min_delta=1e-4,								# early stopping
			cumulative_delta=5e-3,				# early stopping
			minimum_epochs=20,						# early stopping
			topk_values=args.topK_values,
		)
	elif args.mode == "pretrain":
		all_img2txt_metrics = {}
		all_txt2img_metrics = {}
		all_vit_encoders = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
		all_available_clip_encoders = clip.available_models() # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
		for model_arch in all_vit_encoders:
			print(f"Evaluating pre-trained {model_arch}")
			model_config = get_config(architecture=model_arch, dropout=args.dropout,)
			print(json.dumps(model_config, indent=4, ensure_ascii=False))

			model, preprocess = clip.load(
				name=model_arch,
				device=args.device,
				random_weights=False,
				dropout=args.dropout,
			)
			model = model.float()
			model.name = model_arch  # Custom attribute to store model name
			print(f"{model.__class__.__name__} - {model_arch} loaded successfully")
			train_loader, validation_loader = get_dataloaders(
				dataset_name=args.dataset,
				batch_size=args.batch_size,
				nw=args.num_workers,
				USER=os.environ.get('USER'),
				input_resolution=model_config["image_resolution"],
				preprocess=None,#preprocess,
			)
			print_loader_info(loader=train_loader, batch_size=args.batch_size)
			print_loader_info(loader=validation_loader, batch_size=args.batch_size)

			img2txt_metrics, txt2img_metrics = pretrain(
				model=model,
				validation_loader=validation_loader,
				results_dir=os.path.join(args.dataset, "results"),
				device=args.device,
				topk_values=args.topK_values,
			)
			all_img2txt_metrics[model_arch] = img2txt_metrics
			all_txt2img_metrics[model_arch] = txt2img_metrics
			del model  # Clean up memory
			torch.cuda.empty_cache()
		# Pass all metrics to the new visualization function
		plot_all_pretrain_metrics(
			dataset_name=args.dataset,
			img2txt_metrics_dict=all_img2txt_metrics,
			txt2img_metrics_dict=all_txt2img_metrics,
			results_dir=os.path.join(args.dataset, "results"),
			topK_values=args.topK_values,
		)
	else:
		raise ValueError("Invalid mode. Choose either 'finetune' or 'train'.")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))
