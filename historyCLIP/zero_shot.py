import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

for p in sys.path:
	print(p)

from utils import *
import clip

# how to run:
# i2t:  $ python zero_shot.py -i /path/to/query.jpg
# t2i:  $ python zero_shot.py -t "damaged aircraft" -r /home/farid/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/images
# i2i:  $ python zero_shot.py -i /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/images/mcs_36.jpg -r /home/farid/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/images

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

def collect_image_paths(paths_or_dirs: List[str]) -> List[str]:
	"""Resolve a mixed list of file paths and/or directories into a flat list of image paths."""
	image_paths = []
	for p in paths_or_dirs:
		if os.path.isdir(p):
			for fname in sorted(os.listdir(p)):
				if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
					image_paths.append(os.path.join(p, fname))
		elif os.path.isfile(p):
			image_paths.append(p)
		else:
			print(f"  ⚠️ Path not found, skipping: {p}")
	return image_paths

def get_or_compute_image_embeddings(
	model: torch.nn.Module,
	preprocess,
	device: Union[str, torch.device],
	image_paths: List[str],
	architecture: str,
	cache_dir: Optional[str] = None,
	batch_size: int = 32,
) -> Tuple[torch.Tensor, List[str]]:
	"""
	Return (embeddings [N, D], valid_paths [N]) in the same order as *image_paths*.
	* First call  → encodes every image, writes a .pt cache file.
	* Later calls → loads the cache, encodes only new / modified images,
									merges, and overwrites the cache.
	The cache is keyed by **absolute path + file mtime**, so renaming,
	moving, or editing an image automatically triggers re-encoding.
	"""
	if not image_paths:
		return torch.empty(0, model.embed_dim), []
	
	# ---- resolve paths to absolute so the cache is location-independent ----
	abs_paths = [os.path.abspath(p) for p in image_paths]
	if cache_dir is None:
		cache_dir = os.path.dirname(image_paths[0]).replace('images', 'outputs')

	# ---- create cache dir if it doesn't exist ----
	os.makedirs(cache_dir, exist_ok=True)
	safe_arch = architecture.replace('/', '_').replace('@', '_')
	cache_path = os.path.join(cache_dir, f'{safe_arch}_embeddings.pt')
	
	# ---- 1. Load existing cache (if any) ----
	cached_paths_set = {}          # abs_path → index in cached tensor
	cached_embeddings = None       # [M, D]
	cached_mtimes = {}             # abs_path → mtime at encode time
	if os.path.isfile(cache_path):
		try:
			blob = torch.load(cache_path, map_location='cpu', weights_only=False)
		except Exception as e:
			print(f"  ⚠️ Could not load cache ({e}). Recomputing all.")
			blob = None
		if blob is not None:
			if blob.get('architecture') != architecture:
				print(
					f"  ⚠️ Cache was built with '{blob.get('architecture')}' "
					f"but model is '{architecture}'. Recomputing all."
				)
			else:
				paths_list  = blob['paths']
				mtimes_list = blob['mtimes']
				cached_embeddings = blob['embeddings']          # [M, D] fp16 on cpu
				cached_paths_set  = {p: i for i, p in enumerate(paths_list)}
				cached_mtimes     = {p: m for p, m in zip(paths_list, mtimes_list)}
				print(f"  ✅ Loaded cache: {len(paths_list)} embeddings from {cache_path}")
	
	# ---- 2. Partition into hit / miss ----
	hit_indices  = []              # position in abs_paths → index in cached tensor
	miss_indices = []              # position in abs_paths → needs encoding
	for i, ap in enumerate(abs_paths):
		if ap in cached_paths_set:
			try:
				current_mtime = os.path.getmtime(ap)
			except OSError:
				miss_indices.append(i)
				continue
			if abs(cached_mtimes[ap] - current_mtime) < 1.0:   # mtime match
				hit_indices.append(i)
			else:
				miss_indices.append(i)                          # file was modified
		else:
			miss_indices.append(i)
	print(f"  Cache hit: {len(hit_indices)}  |  miss: {len(miss_indices)}")
	
	# ---- 3. Encode only the misses ----
	new_embeds_cpu = None
	new_valid_map  = {}            # position in miss_list → row in new_embeds_cpu
	if miss_indices:
		miss_paths = [abs_paths[i] for i in miss_indices]
		print(f"  Encoding {len(miss_paths)} images …")
		new_embeds_dev, valid_paths = _encode_images_batched(
			model, 
			preprocess, 
			device, 
			miss_paths, 
			batch_size,
		)
		new_embeds_cpu = new_embeds_dev.cpu().half() # fp16 to save disk
		# Build a map: abs_path → row in new_embeds_cpu
		valid_set = {os.path.abspath(p): row for row, p in enumerate(valid_paths)}
		for j, i in enumerate(miss_indices):
			ap = abs_paths[i]
			if ap in valid_set:
				new_valid_map[i] = valid_set[ap]
	
	# ---- 4. Assemble the output tensor in input order ----
	D = model.embed_dim
	out = torch.zeros(len(abs_paths), D, dtype=torch.float16)
	valid_mask = [False] * len(abs_paths)
	# fill from cache
	if cached_embeddings is not None:
		for i in hit_indices:
			out[i] = cached_embeddings[cached_paths_set[abs_paths[i]]]
			valid_mask[i] = True
	# fill from freshly computed
	if new_embeds_cpu is not None:
		for i, row in new_valid_map.items():
			out[i] = new_embeds_cpu[row]
			valid_mask[i] = True
	valid_paths_out = [abs_paths[i] for i in range(len(abs_paths)) if valid_mask[i]]
	out = out[valid_mask].float().to(device)
	
	# ---- 5. Persist the merged cache ----
	all_paths_to_save  = []
	all_mtimes_to_save = []
	all_embeds_to_save = []
	# keep old entries that are still on disk
	if cached_embeddings is not None:
		for p, idx in cached_paths_set.items():
			if os.path.isfile(p):
				all_paths_to_save.append(p)
				all_mtimes_to_save.append(cached_mtimes[p])
				all_embeds_to_save.append(cached_embeddings[idx])
	
	# add / overwrite with new entries
	if new_embeds_cpu is not None:
		for i, row in new_valid_map.items():
			ap = abs_paths[i]
			# remove stale entry for this path if it existed
			if ap in all_paths_to_save:
				old_idx = all_paths_to_save.index(ap)
				all_paths_to_save.pop(old_idx)
				all_mtimes_to_save.pop(old_idx)
				all_embeds_to_save.pop(old_idx)
			try:
				mtime = os.path.getmtime(ap)
			except OSError:
				mtime = 0.0
			all_paths_to_save.append(ap)
			all_mtimes_to_save.append(mtime)
			all_embeds_to_save.append(new_embeds_cpu[row])

	if all_embeds_to_save:
		merged = torch.stack(all_embeds_to_save, dim=0) # [K, D] fp16
		torch.save({
				'architecture': architecture,
				'paths':        all_paths_to_save,
				'mtimes':       all_mtimes_to_save,
				'embeddings':   merged,
			},
			cache_path
		)
		print(
			f"  💾 Cache saved: {len(all_paths_to_save)} embeddings → {cache_path} "
			f"({os.path.getsize(cache_path) / 1e6:.1f} MB)"
		)
	
	return out, valid_paths_out

def _encode_images_batched(
	model: torch.nn.Module,
	preprocess,
	device: Union[str, torch.device],
	image_paths: List[str],
	batch_size: int,
) -> Tuple[torch.Tensor, List[str]]:
	all_embeds = []
	valid_paths = []

	for start in range(0, len(image_paths), batch_size):
		batch_paths = image_paths[start : start + batch_size]
		batch_tensors = []
		batch_valid = []

		for p in batch_paths:
			try:
				img = Image.open(p).convert('RGB')
				batch_tensors.append(preprocess(img))
				batch_valid.append(p)
			except Exception as e:
				print(f"  ⚠️ Skipping unreadable image {p}: {e}")

		if not batch_tensors:
			continue

		batch = torch.stack(batch_tensors).to(device)

		with torch.no_grad(), torch.amp.autocast(
			device_type=device.type,
			dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
			enabled=torch.cuda.is_available(),
		):
			embeds = model.encode_image(batch)
			embeds = torch.nn.functional.normalize(embeds, dim=-1)

		all_embeds.append(embeds.float())   # back to fp32 for similarity math
		valid_paths.extend(batch_valid)

	if not all_embeds:
		return torch.empty(0, model.embed_dim, device=device), []

	return torch.cat(all_embeds, dim=0), valid_paths

def t2i(
	model: torch.nn.Module,
	preprocess,
	device: Union[str, torch.device],
	query_text: str,
	reference_image_paths: List[str],
	architecture: str,
	top_k: int = 5,
	batch_size: int = 32,
	cache_dir: Optional[str] = None,
):
	print(f"\n[t2i] Query text: \"{query_text}\"")
	print(f"[t2i] Reference images: {len(reference_image_paths)}")
	
	# 1. Encode the text query
	text_tokens = clip.tokenize([query_text]).to(device)
	print(f"TOKENS: {text_tokens.shape}")
	with torch.no_grad(), torch.amp.autocast(
		device_type=device.type,
		dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
		enabled=torch.cuda.is_available(),
	):
		text_embed = model.encode_text(text_tokens)
		text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
	print(f"[EMBEDDING] text {text_embed.shape}")
	
	# 2. Image embeddings (cached)
	image_embeds, valid_paths = get_or_compute_image_embeddings(
		model, preprocess, device,
		reference_image_paths,
		architecture=architecture,
		cache_dir=cache_dir,
		batch_size=batch_size,
	)
	print(f"[EMBEDDING] image {image_embeds.shape}")
	if image_embeds.shape[0] == 0:
		print("  ⚠️ No valid reference images. Aborting.")
		return
	
	# 3. Similarity → top-k
	similarity = (100.0 * text_embed.float() @ image_embeds.float().T).softmax(dim=-1)
	probs = similarity[0]
	k = min(top_k, len(valid_paths))
	top_values, top_indices = torch.topk(probs, k)
	
	print(f"\n{'='*70}")
	print(f"Text-to-Image Retrieval  (top {k} of {len(valid_paths)})")
	print(f"{'='*70}")
	for rank, (score, idx) in enumerate(zip(top_values, top_indices), 1):
		idx = idx.item()
		fname = os.path.basename(valid_paths[idx])
		bar = "█" * int(score.item() * 40)
		print(f"  {rank:>3}. {score:.6f}  {bar:<20}  {fname}")
		print(f"       {valid_paths[idx]}")

def i2i(
	model: torch.nn.Module,
	preprocess,
	device: Union[str, torch.device],
	query_image_path: str,
	reference_image_paths: List[str],
	architecture: str,
	top_k: int = 5,
	batch_size: int = 32,
	cache_dir: Optional[str] = None,
):
	print(f"\n[i2i] Query image: {query_image_path}")
	print(f"[i2i] Reference images: {len(reference_image_paths)}")
	
	# 1. Encode query image
	query_img = Image.open(query_image_path).convert('RGB')
	query_input = preprocess(query_img).unsqueeze(0).to(device)
	with torch.no_grad(), torch.amp.autocast(
		device_type=device.type,
		dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
		enabled=torch.cuda.is_available(),
	):
		query_embed = model.encode_image(query_input)
		query_embed = torch.nn.functional.normalize(query_embed, dim=-1)
	print(f"  query embedding: {query_embed.shape}")
	
	# 2. Reference embeddings (cached)
	image_embeds, valid_paths = get_or_compute_image_embeddings(
		model, preprocess, device,
		reference_image_paths,
		architecture=architecture,
		cache_dir=cache_dir,
		batch_size=batch_size,
	)
	print(f"  image embeddings: {image_embeds.shape}")
	if image_embeds.shape[0] == 0:
		print("  ⚠️ No valid reference images. Aborting.")
		return
	
	# 3. Similarity → top-k (exclude the query itself)
	similarity = (100.0 * query_embed.float() @ image_embeds.float().T).softmax(dim=-1)
	probs = similarity[0]
	query_abs = os.path.abspath(query_image_path)
	mask = torch.ones(len(valid_paths), dtype=torch.bool, device=device)
	for j, p in enumerate(valid_paths):
		if os.path.abspath(p) == query_abs:
			mask[j] = False
			print(f"  ℹ️ Excluded query image (index {j})")
	masked_probs = probs.clone()
	masked_probs[~mask] = -1.0
	k = min(top_k, int(mask.sum().item()))
	if k == 0:
		print("  ⚠️ No remaining images after exclusion. Aborting.")
		return
	top_values, top_indices = torch.topk(masked_probs, k)
	
	print(f"\n{'='*70}")
	print(f"Image-to-Image Retrieval  (top {k} of {int(mask.sum().item())})")
	print(f"{'='*70}")
	for rank, (score, idx) in enumerate(zip(top_values, top_indices), 1):
		idx = idx.item()
		fname = os.path.basename(valid_paths[idx])
		bar = "█" * int(score.item() * 40)
		print(f"  {rank:>3}. {score:.6f}  {bar:<20}  {fname}")
		print(f"       {valid_paths[idx]}")

def i2t(
	model: torch.nn.Module,
	preprocess,
	device: Union[str, torch.device],
	query_image_path: str,
	labels: list,
):
	image = Image.open(query_image_path).convert('RGB')
	print(f"{query_image_path} {type(image)} {image.size}")
	image_input = preprocess(image).unsqueeze(0).to(device)

	with torch.no_grad(), torch.amp.autocast(
		device_type=device.type,
		dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
		enabled=torch.cuda.is_available(),
	):
		image_embed = model.encode_image(image_input)
		image_embed = torch.nn.functional.normalize(image_embed, dim=-1)

	print(f"image embedding: {type(image_embed)} {image_embed.shape}")

	text_tokens = clip.tokenize(labels).to(device)

	with torch.no_grad(), torch.amp.autocast(
		device_type=device.type,
		dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
		enabled=torch.cuda.is_available(),
	):
		text_embed = model.encode_text(text_tokens)
		text_embed = torch.nn.functional.normalize(text_embed, dim=-1)

	print(f"text embedding: {type(text_embed)} {text_embed.shape}")

	similarity = (100.0 * image_embed.float() @ text_embed.float().T).softmax(dim=-1)
	probs = similarity[0]
	best_idx = torch.argmax(probs).item()

	print("\n" + "=" * 40)
	print("Zero-Shot Classification Results:")
	print("=" * 40)
	for i, (label, score) in enumerate(zip(labels, probs)):
		marker = " ✅ (Best Match)" if i == best_idx else ""
		print(f"{label:<30}{score:.6f}{marker}")

def retrieval(
	architecture: str,
	device: Union[str, torch.device],
	query_image: Optional[str] = None,
	query_text: Optional[str] = None,
	labels: List[str] = ["tank", "aircraft", "soldiers", "naval ship", "damaged aircraft", "fuselage"],
	reference_images: Optional[List[str]] = None,
	top_k: int = 5,
	batch_size: int = 32,
	cache_dir: Optional[str] = None,
):
	print(f">> CLIP Model Architecture: {architecture}...")
	model_config = clip.get_config(
		architecture=architecture,
		dropout=0,
	)
	print(json.dumps(model_config, indent=4, ensure_ascii=False))
	model, preprocess = clip.load(
		name=architecture,
		device=device,
		jit=False,
		random_weights=False,
		dropout=0,
		download_root=cache_directory.get(USER),
	)
	model.name = architecture
	model_name = model.__class__.__name__
	print(f"Loaded {model_name} {model.name} in {device}")

	if query_text and reference_images:
		t2i(
			model, 
			preprocess, 
			device, 
			query_text, 
			reference_images,
			architecture=architecture, 
			top_k=top_k,
			batch_size=batch_size, 
			cache_dir=cache_dir
		)
	elif query_image and reference_images:
		i2i(
			model, 
			preprocess, 
			device, 
			query_image, 
			reference_images,
			architecture=architecture, 
			top_k=top_k,
			batch_size=batch_size, 
			cache_dir=cache_dir
		)
	elif query_image:
		i2t(model, preprocess, device, query_image, labels)
	else:
		raise ValueError("Provide query_image, query_text+reference_images, or query_image+reference_images")

def main():
	parser = argparse.ArgumentParser(description="Zero-Shot CLIP Retrieval (i2t / t2i / i2i)")
	parser.add_argument('--query_image', '-i', type=str, default=None, help='Path to the query image')
	parser.add_argument('--query_text', '-t', type=str, default=None, help='Text query for text-to-image retrieval')
	parser.add_argument('--reference_images', '-r', type=str, nargs='+', default=None, help='Reference image paths and/or directories for t2i / i2i retrieval')
	parser.add_argument('--labels', '-l', type=str, nargs='+', default=["tank", "aircraft", "soldiers", "naval ship", "damaged aircraft", "fuselage"], help='Class labels for i2t zero-shot classification')
	parser.add_argument('--top_k', '-k', type=int, default=3, help='Number of top results to display')
	parser.add_argument('--batch_size', '-bs', type=int, default=32, help='Batch size for encoding reference images')
	parser.add_argument('--architecture', '-a', type=str, default="ViT-B/32", help='CLIP architecture')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--cache_dir', type=str, default=None, help='Directory for embedding cache (default: .clip_cache next to images)')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print(args)

	# Resolve reference image paths from files / directories
	reference_paths = None
	if args.reference_images:
		reference_paths = collect_image_paths(args.reference_images)
		print(f"Collected {len(reference_paths)} reference images")

	retrieval(
		architecture=args.architecture,
		device=args.device,
		query_image=args.query_image,
		query_text=args.query_text,
		labels=args.labels,
		reference_images=reference_paths,
		top_k=args.top_k,
		batch_size=args.batch_size,
		cache_dir=args.cache_dir,
	)

if __name__ == "__main__":
	main()