
# def get_visual_based_annotation(
# 		csv_file: str,
# 		st_model_name: str,
# 		vlm_model_name: str,
# 		batch_size: int,
# 		device: str,
# 		num_workers: int,
# 		verbose: bool,
# 		metadata_fpth: str,
# ) -> List[List[str]]:
# 		print(f"Semi-Supervised label extraction from image data (using VLM) batch_size: {batch_size}".center(160, "-"))
# 		start_time = time.time()
# 		dataset_dir = os.path.dirname(csv_file)

# 		if verbose:
# 			print(f"Loading metadata from {csv_file}...")
# 		dtypes = {
# 			'doc_id': str, 'id': str, 'label': str, 'title': str,
# 			'description': str, 'img_url': str, 'enriched_document_description': str,
# 			'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
# 			'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
# 		}
# 		df = pd.read_csv(
# 			filepath_or_buffer=csv_file,
# 			on_bad_lines='skip',
# 			dtype=dtypes,
# 			low_memory=False,
# 		)

# 		if verbose:
# 			print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")

# 		# Derive doc_year if missing
# 		if 'doc_year' not in df.columns:
# 			print("Warning: 'doc_year' column missing. Deriving from 'doc_date' or 'raw_doc_date'...")
# 			if 'doc_date' in df.columns:
# 				df['doc_year'] = pd.to_datetime(df['doc_date'], errors='coerce').dt.year
# 			elif 'raw_doc_date' in df.columns:
# 				df['doc_year'] = pd.to_datetime(df['raw_doc_date'], errors='coerce').dt.year
# 			else:
# 				print("No date columns available. Setting 'doc_year' to None.")
# 				df['doc_year'] = None

# 		df['content'] = df['enriched_document_description'].fillna('').astype(str)
# 		image_paths = df['img_path'].tolist()
# 		text_descriptions = df['content'].tolist()
# 		doc_years = df.get('doc_year', [None] * len(df)).tolist()

# 		# Checkpoint handling with validation
# 		checkpoint_path = os.path.join(dataset_dir, "visual_annotation_checkpoint.pkl")
# 		start_idx = 0
# 		all_labels = []
# 		scene_labels = []
# 		activity_labels = []
# 		csv_hash = hashlib.md5(open(csv_file, 'rb').read()).hexdigest()

# 		if os.path.exists(checkpoint_path):
# 			if verbose:
# 				print(f"Found checkpoint file. Loading...")
# 			try:
# 				with open(checkpoint_path, 'rb') as f:
# 					checkpoint = pickle.load(f)
# 					if checkpoint['csv_hash'] == csv_hash and checkpoint['image_paths'] == image_paths:
# 						start_idx = checkpoint['next_idx']
# 						all_labels = checkpoint['all_labels']
# 						scene_labels = checkpoint['scene_labels']
# 						activity_labels = checkpoint['activity_labels']
# 						if verbose:
# 							print(f"Resuming from index {start_idx}/{len(image_paths)}")
# 					else:
# 						print("Checkpoint invalid due to CSV or image path mismatch. Starting from beginning.")
# 			except Exception as e:
# 				print(f"Error loading checkpoint: {e}. Starting from beginning.")

# 		# Load models with local caching
# 		if verbose:
# 			print(f"Loading sentence-transformer model: {st_model_name}...")
# 		sent_model = SentenceTransformer(
# 			model_name_or_path=st_model_name, 
# 			device=device,
# 		)

# 		if verbose:
# 			print(f"Loading VLM model: {vlm_model_name} for image labeling...")
# 		model = AutoModel.from_pretrained(
# 			pretrained_model_name_or_path=vlm_model_name,
# 			device_map=device, 
# 			# torch_dtype=torch.float16
# 		).eval()
# 		processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=vlm_model_name)

# 		# Category type map and thresholds
# 		category_type_map = {}
# 		for cat in object_categories:
# 			category_type_map[cat] = "object"
# 		for cat in scene_categories:
# 			category_type_map[cat] = "scene"
# 		for cat in activity_categories:
# 			category_type_map[cat] = "activity"

# 		base_thresholds = {
# 			"object": 0.45,
# 			"scene": 0.3,
# 			"era": 0.05,
# 			"activity": 0.25
# 		}

# 		# Pre-compute category prompt embeddings
# 		if verbose:
# 			print("Pre-computing category prompt embeddings...")
# 		object_prompts = [f"a photo of {cat}" for cat in object_categories]
# 		scene_prompts = [f"a photo of {cat}" for cat in scene_categories]
# 		activity_prompts = [f"a photo of {cat}" for cat in activity_categories]
		
# 		with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
# 			object_prompt_embeds = sent_model.encode(object_prompts, device=device, convert_to_tensor=True, show_progress_bar=verbose)
# 			scene_prompt_embeds = sent_model.encode(scene_prompts, device=device, convert_to_tensor=True, show_progress_bar=verbose)
# 			activity_prompt_embeds = sent_model.encode(activity_prompts, device=device, convert_to_tensor=True, show_progress_bar=verbose)

# 		combined_labels = [[] for _ in range(len(image_paths))]
# 		total_failed_images = []

# 		for category_idx, (categories, prompt_embeds, category_type, labels_list) in enumerate([
# 			(object_categories, object_prompt_embeds, "object", all_labels),
# 			(scene_categories, scene_prompt_embeds, "scene", scene_labels),
# 			(activity_categories, activity_prompt_embeds, "activity", activity_labels)
# 		]):
# 			if verbose:
# 				print(f"Processing {category_type} categories ({len(categories)} categories)...")
# 			if len(labels_list) >= len(image_paths):
# 				if verbose:
# 					print(f"Skipping {category_type} categories - already processed")
# 				continue
# 			if torch.cuda.is_available():
# 				mem_limit = get_device_properties(device).total_memory * 0.9
# 				if memory_allocated(device) > mem_limit:
# 					sub_batch_size = max(8, batch_size // 2)
# 				else:
# 					sub_batch_size = batch_size
# 			if verbose:
# 				print(f"batch_size: {batch_size} sub_batch_size: {sub_batch_size}")
# 			batches = [
# 				(
# 					image_paths[i:i+batch_size],
# 					text_descriptions[i:i+batch_size],
# 					list(range(i, min(i+batch_size, len(image_paths)))),
# 					categories,
# 					prompt_embeds,
# 					category_type
# 				)
# 				for i in range(start_idx, len(image_paths), batch_size)
# 			]
# 			if verbose:
# 				print(f"Processing {len(batches)} batches sequentially...")
# 			results = []
# 			t0 = time.time()
# 			for batch in batches:
# 				batch_paths, batch_descriptions, batch_indices, categories, prompt_embeds, category_type = batch
# 				batch_results, batch_scores = process_category_batch(
# 					batch_paths=batch_paths,
# 					batch_descriptions=batch_descriptions,
# 					batch_indices=batch_indices,
# 					df=df,
# 					categories=categories,
# 					prompt_embeds=prompt_embeds,
# 					category_type=category_type,
# 					sent_model=sent_model,
# 					processor=processor,
# 					model=model,
# 					device=device,
# 					verbose=verbose,
# 					base_thresholds=base_thresholds,
# 					sub_batch_size=sub_batch_size
# 				)
# 				results.append((batch_results, batch_scores))
# 			if verbose:
# 				print(f"Elapsed_t: {time.time()-t0:.1f} sec")

# 			# Collect results
# 			for batch_idx, (batch_results, batch_scores) in enumerate(results):
# 				batch_start = start_idx + batch_idx * batch_size
# 				batch_end = min(batch_start + batch_size, len(image_paths))
# 				batch_paths = batches[batch_idx][0]  # Get batch_paths for this batch
# 				while len(labels_list) < batch_end:
# 					labels_list.append([])
# 				for idx, (results, scores) in zip(range(batch_start, batch_end), zip(batch_results, batch_scores)):
# 					if len(results) != len(scores):
# 						print(f"Error: Mismatched results ({len(results)}) and scores ({len(scores)}) at index {idx}")
# 						scores = [0.0] * len(results)
# 					labels_list[idx].extend(results)
# 					total_failed_images.extend(
# 						[
# 							batch_paths[j] for j in range(len(batch_results))
# 							if not batch_results[j] and batch_paths[j] not in total_failed_images
# 						]
# 					)

# 				# Save checkpoint
# 				if (batch_idx % 500 == 0 and batch_idx > 0) or batch_idx == len(batches) - 1:
# 					checkpoint = {
# 						'next_idx': batch_end,
# 						'all_labels': all_labels,
# 						'scene_labels': scene_labels,
# 						'activity_labels': activity_labels,
# 						'csv_hash': csv_hash,
# 						'image_paths': image_paths
# 					}
# 					with open(checkpoint_path, 'wb') as f:
# 						pickle.dump(checkpoint, f)
# 					if verbose:
# 						print(f"Checkpoint saved at index {batch_end}")

# 			start_idx = 0

# 		# Remove checkpoint
# 		if os.path.exists(checkpoint_path):
# 			os.remove(checkpoint_path)

# 		# Combine and post-process labels
# 		for i in range(len(image_paths)):
# 			image_labels = []
# 			image_scores = []
# 			if i < len(all_labels):
# 				image_labels.extend(all_labels[i])
# 				image_scores.extend([base_thresholds.get("object", 0.35)] * len(all_labels[i]))
# 			if i < len(scene_labels):
# 				image_labels.extend(scene_labels[i])
# 				image_scores.extend([base_thresholds.get("scene", 0.2)] * len(scene_labels[i]))
# 			if i < len(activity_labels):
# 				image_labels.extend(activity_labels[i])
# 				image_scores.extend([base_thresholds.get("activity", 0.26)] * len(activity_labels[i]))
# 			processed_labels = post_process_labels(
# 				labels=image_labels,
# 				text_description=text_descriptions[i],
# 				sent_model=sent_model,
# 				doc_year=doc_years[i],
# 				vlm_scores=image_scores,
# 				max_labels=10,
# 				similarity_threshold=0.8
# 			)
# 			combined_labels[i] = sorted(set(processed_labels))

# 		# Report failed images
# 		if verbose and total_failed_images:
# 			print(f"Total failed images: {len(total_failed_images)}")

# 		# Save results
# 		df['visual_based_labels'] = combined_labels
# 		df.to_csv(metadata_fpth, index=False)
# 		if verbose:
# 			total_labels = sum(len(labels) for labels in combined_labels)
# 			print(f"Vision-based annotation completed in {time.time() - start_time:.2f} seconds")
# 			print(f"Generated {total_labels} labels for {len(image_paths)} images")
# 			print(f"Average labels per image: {total_labels/len(image_paths):.2f}")
# 		print(f"Visual-based annotation Elapsed time: {time.time() - start_time:.2f} sec".center(160, " "))
# 		return combined_labels

from PIL import Image
import os

#Open the image
image_fpath = "/home/farid/datasets/WW_DATASETs/NATIONAL_ARCHIVE_1933-01-01_1933-01-05/images/7018223.jpg"
image = Image.open(image_fpath).convert("RGB")
# image_fname = image.filename
print(os.path.basename(image_fpath), image.size, image.mode)
#Define the thumbnail size as a tuple (width, height)
thumbnail_size = (1000, 1000)

#Create a thumbnail
image.thumbnail(thumbnail_size, resample=Image.Resampling.LANCZOS )
image.save(
  fp=os.path.basename(image_fpath).replace(".jpg", "_thumbnail.jpg"), 
  format="JPEG", 
  quality=100, 
  optimize=True, progressive=True
)
image.show()