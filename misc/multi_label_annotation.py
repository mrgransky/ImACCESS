from utils import *

# how to run[Pouta]:
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata.csv -d "cuda:0" -nw 50 -tbs 512 -vbs 512 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_EUROPEANA.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31/metadata.csv -d "cuda:1" -nw 50 -tbs 256 -vbs 512 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_NA.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata.csv -d "cuda:2" -nw 50 -tbs 512 -vbs 512 -vth 0.3 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_WWII.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/metadata.csv -d "cuda:3" -nw 50 -tbs 512 -vbs 512 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_HISTORY_X4.out &

# Make language detection deterministic
DetectorFactory.seed = 42

# Custom stopwords and metadata patterns
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS.union(
	{
		"bildetekst", "photo", "image", "archive", "arkivreferanse", "caption", "following", "below", "above",
		"copyright", "description", "riksarkivet", "ntbs", "ra", "pa", "bildetekst", "century", "twentieth",
		"showing", "shown", "shows", "depicts", "depicting", "pictured", "picture", "pinterest",
		"copy", "version", "view", "looking", "seen", "visible", "illustration",
		"photograph", "photography", "photo", "image", "img", "photographer",
		"sent", "received", "taken", "made", "created", "produced", "found",
		"across", "opposite", "near", "under", "over", "inside", "outside",
		"collection", "collections", "number", "abbreviation", "abbreviations",
		"folder", "box", "file", "document", "page", "index", "label", "code", "icon", "type", "unknown", "unknow",
		"folder icon", "box icon", "file icon", "document icon", "page icon", "index icon", "label icon", "code icon",
		"used", "southern", "built", "album", "album cover",
		"original", "information", "item", "http", "www", "jpg", "000",
		"jpeg", "png", "gif", "bmp", "tiff", "tif", "ico", "svg", "webp", "heic", "heif", "raw", "cr2", "nef", "orf", "arw", "dng", "nrw", "k25", "kdc", "rw2", "raf", "mrw", "pef", "sr2", "srf",
	}
)

METADATA_PATTERNS = [
	r'bildetekst \w+',        # Matches 'bildetekst german' and similar
	r'kunststoff \w+',        # Matches photography material descriptions
	r'arkivreferanse \w+',    # Archive references
	r'\w+ pa \d+',            # Reference codes like 'ra pa-1209'
	r'donated \w+',           # Donation information
	r'information received',  # Source information
	r'\w+ association',       # Organization suffixes without context
]

FastText_Language_Identification = "lid.176.bin"
if FastText_Language_Identification not in os.listdir():
	url = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{FastText_Language_Identification}"
	urllib.request.urlretrieve(url, FastText_Language_Identification)

# Semantic categories for organization
SEMANTIC_CATEGORIES = {
	'military': ['military', 'army', 'navy', 'air force', 'soldier', 'officer', 'troop', 'regiment', 'division', 'corps', 'battalion', 'brigade'],
	'political': ['government', 'parliament', 'president', 'prime minister', 'minister', 'official', 'politician', 'leader'],
	'event': ['war', 'battle', 'attack', 'invasion', 'liberation', 'occupation', 'revolution', 'protest', 'march', 'ceremony'],
	'location': ['city', 'town', 'village', 'country', 'region', 'territory', 'front', 'border', 'base', 'camp'],
	'vehicle': ['tank', 'aircraft', 'plane', 'ship', 'submarine', 'boat', 'truck', 'car', 'jeep', 'vehicle'],
	'weapon': ['gun', 'rifle', 'cannon', 'artillery', 'weapon', 'bomb', 'missile', 'ammunition'],
}

# Entity types for NER
RELEVANT_ENTITY_TYPES = {
	'PERSON',
	'ORG',
	'GPE',
	'LOC',
	'NORP',
	'FAC',
	'EVENT',
	'WORK_OF_ART',
	'PRODUCT',
	'DATE',
	'TIME',
}

def density_based_parameters(embeddings, n_neighbors=15):
		"""Determine parameters based on local density"""
		# Compute nearest neighbors distances
		nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
		distances, _ = nbrs.kneighbors(embeddings)
		
		# Estimate density
		kth_distances = distances[:, -1]
		density = 1.0 / (kth_distances + 1e-8)
		
		# Use percentiles of density to determine parameters
		min_cluster_size = int(np.percentile(density, 75) * len(embeddings) / 100)
		min_samples = int(np.percentile(density, 50) * len(embeddings) / 100)
		
		# Ensure reasonable bounds
		min_cluster_size = max(10, min(min_cluster_size, len(embeddings)//10))
		min_samples = max(5, min(min_samples, min_cluster_size//2))
		
		return min_cluster_size, min_samples

def find_knee_point(embeddings, max_size=300):
		"""Find knee point in k-distance graph"""
		# Sample a subset for efficiency
		sample_size = min(5000, len(embeddings))
		sample = embeddings[np.random.choice(len(embeddings), sample_size, replace=False)]
		
		# Compute k-nearest neighbors distances
		n_neighbors = min(15, sample_size-1)
		knn = NearestNeighbors(n_neighbors=n_neighbors)
		knn.fit(sample)
		distances, _ = knn.kneighbors(sample)
		
		# Sort distances
		k_distances = np.sort(distances[:, -1])[::-1]
		
		# Find knee point
		kneedle = KneeLocator(
				range(len(k_distances)),
				k_distances,
				curve='convex',
				direction='decreasing'
		)
		
		return max(2, int(kneedle.knee * len(embeddings) / sample_size))

def find_optimal_min_cluster_size(embeddings, dataset_size, max_clusters=50):
		"""Find optimal min_cluster_size using silhouette analysis"""
		candidate_sizes = [
				int(np.sqrt(dataset_size)),  # Current approach
				int(np.log(dataset_size)**2),  # Log-based
				100, 150, 200, 250,  # Common values
				int(dataset_size**0.4),  # Alternative power law
		]
		
		best_score = -1
		best_size = candidate_sizes[0]
		
		for size in sorted(set(candidate_sizes)):
				if size < 2:
						continue
						
				# Cap at reasonable max clusters
				if dataset_size / size > max_clusters:
						continue
						
				clusterer = hdbscan.HDBSCAN(
						min_cluster_size=size,
						min_samples=max(1, int(size/2)),
						cluster_selection_method='leaf',
						metric='euclidean'
				)
				labels = clusterer.fit_predict(embeddings)
				
				# Only calculate silhouette score if we have multiple clusters
				n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
				if n_clusters > 1:
						sample_size = min(10000, len(embeddings))
						sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
						score = silhouette_score(embeddings[sample_indices], labels[sample_indices])
						
						if score > best_score:
								best_score = score
								best_size = size
		
		return best_size

def get_hdbscan_parameters(
		embeddings, 
		use_static=False, 
		minimum_cap=10,
		percentage=None,
	):
	print(f"get hdbscan parameters for embeddings {embeddings.shape}...".center(160, "-"))
	if use_static:
			return 100, 10
	num_samples, num_embeddings = embeddings.shape
	# Method 1: Silhouette-based
	silhouette_size = find_optimal_min_cluster_size(embeddings, num_samples)
	
	# Method 2: Knee point detection
	knee_size = find_knee_point(embeddings)
	
	# Method 3: Density-based
	density_size, density_samples = density_based_parameters(embeddings)
	
	# Dynamically set percentage based on dataset size
	if num_samples < 5000:
			percentage = 0.005  # 0.5% for small datasets
	elif num_samples < 50000:
			percentage = 0.0007  # 0.07% for medium datasets (increased from 0.0005)
	else:
			percentage = 0.0003  # 0.03% for large datasets
	
	percentage_size = max(minimum_cap, int(num_samples * percentage))
	# Combine results (for logging)
	suggestions = [
			silhouette_size,
			knee_size,
			density_size,
			percentage_size,
			int(np.sqrt(num_samples) // 2),
			int(np.sqrt(num_samples) / 2),
			int(np.log(num_samples)**2),
			int(np.log(num_samples)**3),
			50, 70, 80, 100, 120,
	]
	print(f"Suggestions: {suggestions}")
	
	# Remove outliers (for logging)
	q25, q75 = np.percentile(suggestions, [25, 75])
	iqr = q75 - q25
	print(f"q25: {q25}, q75: {q75}, iqr: {iqr}")
	filtered = [x for x in suggestions if (x >= q25 - 2*iqr) and (x <= q75 + 2*iqr)]
	print(f"Filtered: {filtered} median: {np.median(filtered)} min: {min(filtered)} max: {max(filtered)} mean: {np.mean(filtered)}")
	
	# Use percentage-based size directly
	min_cluster_size = percentage_size
	
	# Adjust min_samples to scale with min_cluster_size
	min_samples = max(3, int(min_cluster_size * 0.1))  # 10% of min_cluster_size, min 3
	
	# Log expected number of clusters
	expected_clusters = num_samples / min_cluster_size
	print(f"Expected number of clusters: {expected_clusters:.0f}")
	
	return min_cluster_size, min_samples

def is_likely_english_term(term):
	"""Check if a term is likely English or a proper noun"""
	if not term or len(term) < 3:
		return False
					
	# Common characters in European languages but not in English
	non_english_chars = 'äöüßñéèêëìíîïçåæœø'
	
	# Check for non-English characters
	if any(char in non_english_chars for char in term.lower()):
		return False
					
	# Allow terms that might be proper nouns (start with uppercase)
	if term[0].isupper():
		return True

	# Check if term is in an English dictionary
	common_english_words = set(nltk.corpus.words.words())
	return term.lower() in common_english_words

def extract_semantic_topics(
		sent_model: SentenceTransformer,
		ft_model: fasttext.FastText._FastText,
		texts: List[str],
		dataset_dir: str,
		num_workers: int,
		enable_visualizations: bool = True,
	) -> Tuple[List[List[str]], Set[str]]:

	# Generate embeddings
	kw_model = KeyBERT(model=sent_model)
	dataset_size = len(texts)
	sentence_transformer_name = sent_model._first_module().auto_model.config._name_or_path.replace(r"/", "_") # 'sentence-transformers_all-mpnet-base-v2'
	emb_fpth = os.path.join(dataset_dir, f'{sentence_transformer_name}_embeddings_{dataset_size}_samples.gz')
	t0 = time.time()
	try:
		embeddings = load_pickle(fpath=emb_fpth)
	except Exception as e:
		print(e)
		print(f"Generating Text embeddings for {len(texts)} texts [might take a while]...")
		embeddings = sent_model.encode(texts, show_progress_bar=True)
		save_pickle(pkl=embeddings, fname=emb_fpth)

	print(f"Raw Embeddings: {embeddings.shape} generated in {time.time() - t0:.2f} sec")
	small_dataset_sample_size = min(5000, dataset_size)
	t0 = time.time()
	if dataset_size < small_dataset_sample_size:
		print("Dataset is small, using KMeans for clustering...")
		kmeans = KMeans(n_clusters=min(10, max(2, int(np.sqrt(dataset_size)))), random_state=42)
		labels = kmeans.fit_predict(embeddings)
	else:
		print(f"Dataset is large: {dataset_size} samples => HDBSCAN clustering...")
		min_cluster_size, min_samples = get_hdbscan_parameters(
			embeddings=embeddings,
			use_static=False,
		)
		# Check for high noise in a sample of the data
		print(f"Checking for high noise in a sample of the data...")
		sample_size = small_dataset_sample_size
		sample_indices = np.random.choice(dataset_size, sample_size, replace=False)
		sample_embeddings = embeddings[sample_indices]
		sample_clusterer = hdbscan.HDBSCAN(
			min_cluster_size=min_cluster_size,
			min_samples=min_samples,
			cluster_selection_method='eom',
			metric='euclidean',
		)
		sample_labels = sample_clusterer.fit_predict(sample_embeddings)
		noise_ratio = np.sum(sample_labels == -1) / sample_size
		print(f"Initial noise ratio ({sample_size} samples): {noise_ratio:.2%}")
		if noise_ratio > 0.3:  # Threshold for applying UMAP
			print(f"High noise detected ({noise_ratio:.2%}), applying UMAP preprocessing...")
			# UMAP embedding:
			print(f"Reducing embeddings: {embeddings.shape} to 50D for clustering using UMAP...")
			umap_reducer = umap.UMAP(
				n_neighbors=5,
				min_dist=0.1,
				densmap=True,
				spread=1.0,
				n_components=50, 
				random_state=42, 
				metric='cosine',
				n_jobs=num_workers,
			)
			embeddings = umap_reducer.fit_transform(embeddings)
			print(f"UMAP embedding {embeddings.shape} generated in {time.time() - t0:.2f} sec")
		print(f"Clustering embeddings {embeddings.shape} into topics with HDBSCAN...")
		cluster_selection_method = 'eom' if dataset_size < 50000 else 'leaf'
		print(f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, cluster_selection_method(dataset_size: {dataset_size}): {cluster_selection_method}")
		clusterer = hdbscan.HDBSCAN(
			alpha=1.0,
			min_cluster_size=min_cluster_size,
			min_samples=min_samples,
			algorithm='best',
			metric='euclidean',
			cluster_selection_method=cluster_selection_method,
			core_dist_n_jobs=num_workers,
		)
		labels = clusterer.fit_predict(embeddings)

	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise points (-1)
	print(f">>>> Found {n_clusters} clusters (excluding noise points) in {time.time() - t0:.2f} sec")
	print(f"Cluster Noise (-1) contains: {np.sum(labels == -1)} samples, [{np.sum(labels == -1) / len(labels) * 100:.2f}%]")

	# Visualization 1: Cluster Distribution Bar Plot (Before Merging)
	if enable_visualizations:
		topic_counts = Counter(labels)
		plt.figure(figsize=(16, 7))
		bars = plt.bar(range(len(topic_counts)), [topic_counts[i] for i in sorted(topic_counts.keys())], color='#3785e6', label='Before Merging')
		plt.title('Sample Distribution Across Clusters (Before Merging) [Noise included]')
		plt.xlabel('Cluster ID')
		plt.ylabel('Number of Documents')
		plt.xticks(range(len(topic_counts)), labels=sorted(topic_counts.keys()), fontsize=8, va='center', ha='center', rotation=90)
		for bar in bars:
			yval = bar.get_height()
			plt.text(bar.get_x() + bar.get_width()/2, yval + 0.7, int(yval), ha='center', va='bottom', fontsize=7, color='#16171a')
		plt.legend()
		plt.savefig(os.path.join(dataset_dir, f'topic_distribution_before_merging_{n_clusters}_clusters.png'), bbox_inches='tight')
		plt.close()

	# # Visualization 2: Interactive UMAP Scatter Plot with Plotly
	# if enable_visualizations:
	# 	print(f"UMAP reducing embeddings: {embeddings.shape}")
	# 	umap_reducer = umap.UMAP(
	# 		n_neighbors=15,
	# 		min_dist=0.1,
	# 		densmap=True,
	# 		spread=1.0,
	# 		n_components=2, 
	# 		random_state=42, 
	# 		metric='cosine',
	# 	)
	# 	emb_umap = umap_reducer.fit_transform(embeddings)

	# 	centroids = np.zeros((n_clusters, 2))
	# 	for i in range(n_clusters):
	# 		cluster_points = emb_umap[labels == i]
	# 		if len(cluster_points) > 0:
	# 			centroids[i] = np.mean(cluster_points, axis=0)
	# 	distances = np.array([np.linalg.norm(emb_umap[i] - centroids[labels[i]]) if labels[i] != -1 else 0 for i in range(len(texts))])
	# 	outliers = distances > (np.mean(distances[distances > 0]) + 2 * np.std(distances[distances > 0])) if distances[distances > 0].size > 0 else np.zeros(len(texts), dtype=bool)
	# 	df_plot = pd.DataFrame({
	# 		'UMAP1': emb_umap[:, 0],
	# 		'UMAP2': emb_umap[:, 1],
	# 		'Cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels],
	# 		'Text': [text[:100] + '...' if len(text) > 100 else text for text in texts],
	# 		'Distance_to_Centroid': distances,
	# 		'Outlier': ['Yes' if o else 'No' for o in outliers]
	# 	})
	# 	fig = px.scatter(
	# 		df_plot,
	# 		x='UMAP1',
	# 		y='UMAP2',
	# 		color='Cluster',
	# 		symbol='Outlier',
	# 		hover_data=['Text', 'Distance_to_Centroid'],
	# 		title=f'Interactive UMAP Visualization of Text Embeddings for {dataset_size} Texts into {n_clusters} Cluster'
	# 	)
	# 	fig.add_trace(go.Scatter(
	# 		x=centroids[:, 0],
	# 		y=centroids[:, 1],
	# 		mode='markers+text',
	# 		marker=dict(size=15, symbol='x', color='#000000'),
	# 		text=[f'Centroid {i}' for i in range(n_clusters)],
	# 		textposition='top center',
	# 		name='Centroids'
	# 	))
	# 	fig.write_html(os.path.join(dataset_dir, 'umap_cluster_visualization_interactive.html'))

	# Collect phrases for each cluster
	print("Extracting keywords for each cluster using KeyBERT...")
	
	t0 = time.time()
	cluster_phrases = defaultdict(Counter)
	cluster_text_counts = defaultdict(int)
	phrase_filter_log = {'total_phrases': 0, 'stopword_filtered': 0, 'length_filtered': 0}
	for i, (text, label) in tqdm(enumerate(zip(texts, labels)), total=len(texts), desc="Keywords Extraction Progress"):
		if label == -1:  # Skip noise points
			continue
		phrases = kw_model.extract_keywords(
			text,
			keyphrase_ngram_range=(1, 3),
			stop_words="english",
			top_n=15 if len(text.split()) > 100 else 5,
			diversity=0.7,
		)
		# print(phrases)
		phrase_filter_log['total_phrases'] += len(phrases)
		# Filter phrases
		valid_phrases = []
		for phrase, _ in phrases:  # Ignore KeyBERT scores for now
			words = phrase.split()
			stopword_count = sum(1 for word in words if word in CUSTOM_STOPWORDS)
			# Relaxed stopword filter: allow <=70% stopwords
			if stopword_count / len(words) > 0.7:
				phrase_filter_log['stopword_filtered'] += 1
				continue
			valid_phrases.append(phrase)
		
		# Normalize phrases to reduce repetition
		normalized_phrases = []
		seen_phrases = set()
		for phrase in valid_phrases:
			# Remove consecutive duplicate words
			words = phrase.split()
			normalized = " ".join(word for i, word in enumerate(words) if i == 0 or word != words[i-1])
			# Ensure normalized phrase is unique and meets length requirement
			if len(normalized.split()) >= 1 and normalized not in seen_phrases:
				normalized_phrases.append(normalized)
				seen_phrases.add(normalized)
			else:
				phrase_filter_log['length_filtered'] += 1
		
		if not normalized_phrases:
			print(f"Text {i} has no valid phrases: {text}")
		else:
			cluster_phrases[label].update(normalized_phrases)
		cluster_text_counts[label] += 1
	print(f"Phrase collection done in {time.time() - t0:.2f} sec")

	# Visualization 3: Phrase Retention Histogram
	if enable_visualizations:
		print("Phrase Retention Histogram")
		plt.figure(figsize=(12, 6))
		plt.bar(
			['Total Phrases', 'Stopword Filtered', 'Length Filtered'], 
			[phrase_filter_log['total_phrases'], 
			phrase_filter_log['stopword_filtered'], 
			phrase_filter_log['length_filtered']],
			color=['#005dcf', '#df7272', '#8dfc8d']
		)
		for i, val in enumerate([phrase_filter_log['total_phrases'], phrase_filter_log['stopword_filtered'], phrase_filter_log['length_filtered']]):
			plt.text(i, val + 0.5, str(val), ha='center', va='bottom')
		plt.title('Phrase Retention After Filtering')
		plt.ylabel('Number of Phrases')
		plt.savefig(os.path.join(dataset_dir, 'phrase_retention_histogram.png'), bbox_inches='tight')
		plt.close()
	
	# Extract initial topics with diversity scoring
	topics_before_merging = []
	term_counts_per_cluster = []
	for label, counter in cluster_phrases.items():
		if not counter:
			print(f"Warning: Topic {label} has no phrases.")
		# Score phrases with diversity bonus
		phrase_scores = []
		top_k_words = max(10, len(counter) // 15)
		# print(f"Topic {label}: {len(counter)} phrases, Selecting Top-{top_k_words}")
		seen_words = set()
		for phrase, count in counter.items():
			words = set(phrase.split())
			diversity_bonus = sum(1 for word in words if word not in seen_words)
			score = count * (1 + 0.1 * len(words) + 0.7 * diversity_bonus)
			phrase_scores.append((phrase, count, score))
			seen_words.update(words)
		phrase_scores.sort(key=lambda x: x[2], reverse=True)
		selected_phrases = []
		seen_words = set()
		for phrase, count, score in phrase_scores[:top_k_words * 2]:
			words = set(phrase.split())
			if any(words.issubset(set(p.split())) and counter[p] > count * 2 for p in selected_phrases):
				continue
			selected_phrases.append(phrase)
			seen_words.update(words)
		topics_before_merging.append(selected_phrases[:top_k_words])
		term_counts_per_cluster.append(len(counter))
	if not any(topics_before_merging):
		print("Error: No valid phrases found in any topics.")
		return [], set()
	
	# Calculate topic similarities
	print("Calculating topic similarities for merging [cosine similarity]...")
	similarity_matrix = np.zeros((len(topics_before_merging), len(topics_before_merging)))
	word_to_embedding = {}
	all_words = list(set(word for topic in topics_before_merging for word in topic if word))
	if all_words:
		word_embeddings = sent_model.encode(all_words, show_progress_bar=True)
		print(f"Word embeddings shape: {word_embeddings.shape}")
		for i, word in enumerate(all_words):
			word_to_embedding[word] = word_embeddings[i]
	else:
		print("Warning: No words available for topic embeddings.")

	topic_embeddings = []
	for topic in topics_before_merging:
		topic_embs = [word_to_embedding[word] for word in topic if word in word_to_embedding]
		topic_emb = np.mean(topic_embs, axis=0) if topic_embs else np.zeros(word_embeddings.shape[1])
		topic_embeddings.append(topic_emb)
	for i in range(len(topics_before_merging)):
		for j in range(i + 1, len(topics_before_merging)):
			sim = util.cos_sim([topic_embeddings[i]], [topic_embeddings[j]])[0][0].item()
			similarity_matrix[i, j] = sim
			similarity_matrix[j, i] = sim

	# Visualization 4: Phrase Co-Occurrence Network for Each Topic [before merging]
	if enable_visualizations:
		print(f"Generating co-occurrence networks for {len(topics_before_merging)} topics [before merging]...")
		for label, topic_phrases in enumerate(topics_before_merging):
			if not topic_phrases:
				# print(f"Skipping Topic {label}: No phrases available.")
				continue
			# Select top 10 phrases by frequency to reduce clutter
			counter = cluster_phrases[label]
			phrase_freq = {phrase: counter[phrase] for phrase in topic_phrases if phrase in counter}
			top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]
			top_phrases = [phrase for phrase, _ in top_phrases]
			if not top_phrases:
				# print(f"Skipping Topic {label}: No phrases after filtering.")
				continue
			cluster_texts = [texts[i] for i, l in enumerate(labels) if l == label and is_english(texts[i], ft_model)]
			if not cluster_texts:
				# print(f"Skipping Topic {label}: No valid texts for co-occurrence.")
				continue
			phrase_set = set(top_phrases)
			cooc_matrix = defaultdict(int)
			for text in cluster_texts:
				phrases = kw_model.extract_keywords(
					text,
					keyphrase_ngram_range=(1, 3),
					stop_words="english",
					top_n=15 if len(text.split()) > 100 else 5,
					diversity=0.7,
				)
				valid_phrases = []
				seen_phrases = set()
				for phrase, _ in phrases:
					words = phrase.split()
					stopword_count = sum(1 for w in words if w in CUSTOM_STOPWORDS)
					if stopword_count / len(words) > 0.6 or len(words) < 2:
						continue
					normalized = " ".join(word for i, word in enumerate(words) if i == 0 or word != words[i-1])
					if len(normalized.split()) >= 2 and normalized not in seen_phrases:
						valid_phrases.append(normalized)
						seen_phrases.add(normalized)
				text_phrases = set(valid_phrases).intersection(phrase_set)
				for p1 in text_phrases:
					for p2 in text_phrases:
						if p1 < p2:
							cooc_matrix[(p1, p2)] += 1
			G = nx.Graph()
			for (p1, p2), count in cooc_matrix.items():
				if count >= 2: # An edge exists between two phrases if they appear together in the same text at least twice
					G.add_edge(p1, p2, weight=count)
			for phrase in top_phrases:
				if phrase not in G:
					G.add_node(phrase)
			plt.figure(figsize=(12, 8))
			pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjusted k for better spacing
			print(f"Position:\n{pos}")
			edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
			print(f"Edge weights:\n{edge_weights}")
			nx.draw_networkx_edges(
				G=G, 
				pos=pos,
				width=[w * 0.1 for w in edge_weights],
				alpha=0.65,
				edge_color='#303030b7'
			)
			nx.draw_networkx_nodes(
				G, 
				pos,
				node_size=100,
				node_color='#1070b1',
				alpha=0.9,
			)
			nx.draw_networkx_labels(
				G, 
				pos,
				font_size=5,
				alpha=0.95,
				verticalalignment='baseline',
				horizontalalignment='left'
			)
			plt.title(f'Phrase Co-Occurrence Network for Topic {label} [before merging] ({len(G.nodes())} nodes, {len(G.edges())} edges)')
			plt.axis('off')
			plt.savefig(os.path.join(dataset_dir, f'cooccurrence_network_topic_{label}_before_merging.png'), bbox_inches='tight', dpi=300)
			plt.close()

	# Visualization 5: Top-K Phrases Bar Plot for Each Topic [before merging]
	if enable_visualizations:
		for label, topic_phrases in enumerate(topics_before_merging):
			if not topic_phrases:
				# print(f"Skipping Topic {label}: No phrases available.")
				continue
			counter = cluster_phrases[label]
			# print(f"Topic {label}: {len(counter)} phrases, Selecting Top-{len(topic_phrases)}")
			phrase_freq = {phrase: counter[phrase] for phrase in topic_phrases if phrase in counter}
			top_k_phrases_per_topic_before_merging = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]
			if not top_k_phrases_per_topic_before_merging:
				continue
			phrases, frequencies = zip(*top_k_phrases_per_topic_before_merging)
			plt.figure(figsize=(14, 6))
			sns.barplot(x=frequencies, y=phrases, palette='Blues_r')
			plt.title(f'Top {len(top_k_phrases_per_topic_before_merging)} Phrases {len(phrase_freq)} in Topic {label} [before merging]')
			plt.xlabel('Frequency')
			plt.ylabel('Phrase')
			plt.savefig(os.path.join(dataset_dir, f'topK_phrases_topic_{label}_before_merging.png'), bbox_inches='tight')
			plt.close()

	# Visualization 6: Topic Similarity Heatmap (before merging)
	if enable_visualizations:
		plt.figure(figsize=(17, 12))
		sns.heatmap(
			data=similarity_matrix, 
			# annot=True, 
			cmap='YlOrRd', 
			vmin=0, 
			vmax=1, 
			square=True,
		)
		plt.title(f'Topic Similarity Matrix (Cosine Similarity) [before merging]')
		plt.xlabel('Topic ID')
		plt.ylabel('Topic ID')
		plt.savefig(os.path.join(dataset_dir, f'topic_similarity_heatmap_before_merging.png'), bbox_inches='tight')
		plt.close()
	
	# Visualization 7: Dendrogram of Topic Similarities (before merging)
	if enable_visualizations:
		sim_values = similarity_matrix[np.triu_indices(len(topics_before_merging), k=1)]
		if sim_values.size > 0:
			mean_sim = np.mean(sim_values)
			min_sim = np.min(sim_values)
			max_sim = np.max(sim_values)
			print(f"Similarity matrix stats: Mean={mean_sim:.3f}, Min={min_sim:.3f}, Max={max_sim:.3f}")
			merge_threshold = np.percentile(sim_values, 75) + 0.10
			print(f"Dynamic merge threshold (75th percentile): {merge_threshold:.4f}")
			# Convert similarity → distance
			dist_matrix = 1.0 - similarity_matrix
			dist_matrix = np.clip(dist_matrix, 0, 2)  # Ensure all distances are in [0, 2]
			# Zero out diagonal to make it a valid distance matrix
			np.fill_diagonal(dist_matrix, 0.0)
			from scipy.spatial.distance import squareform
			condensed_dist = squareform(dist_matrix)  # Converts to condensed form for linkage
			from scipy.cluster.hierarchy import linkage, dendrogram
			plt.figure(figsize=(17, 10))
			linkage_matrix = linkage(condensed_dist, method='average')
			dendrogram(
				linkage_matrix,
				labels=[f'Topic {i}' for i in range(len(topics_before_merging))],
				color_threshold=1 - merge_threshold
			)
			plt.title('Dendrogram of Topic Similarities [before merging]')
			plt.xlabel('Topics')
			plt.ylabel('Distance (1 - Cosine Similarity)')
			plt.axhline(y=1 - merge_threshold, color='red', linestyle='--', label=f'Merge Threshold ({merge_threshold:.4f})')
			plt.legend()
			plt.xticks(rotation=90, fontsize=8)
			plt.savefig(os.path.join(dataset_dir, f'similarity_dendrogram_before_merging_thresh_{merge_threshold:.4f}.png'), bbox_inches='tight')
			plt.close()
		else:
			print("Similarity matrix is empty.")

	# # Visualization 8: UMAP with Top Phrases
	# if enable_visualizations:
	# 	# Verify outliers definition (HDBSCAN noise points)
	# 	outliers = labels == -1  # Noise points from HDBSCAN
	# 	print(f"Noise points (outliers) in UMAP plot: {np.sum(outliers)}/{len(texts)} texts [{np.sum(outliers) / len(texts) * 100:.2f}%]")
		
	# 	# Get unique clusters (excluding noise)
	# 	unique_clusters = np.unique(labels[~outliers])
	# 	print(f">> {len(unique_clusters)} Unique Clusters (excluding noise) [{np.sum(outliers) / len(texts) * 100:.2f}% noise]")
	# 	if len(unique_clusters) > 0:
	# 		# Calculate centroids in 2D UMAP space
	# 		centroids = np.zeros((n_clusters, 2))
	# 		for i in range(n_clusters):
	# 			cluster_points = emb_umap[labels == i]
	# 			if len(cluster_points) > 0:
	# 				centroids[i] = np.mean(cluster_points, axis=0)
	# 		print(f"Centroids shape: {centroids.shape}")
			
	# 		# Assign outliers to nearest cluster based on distance
	# 		outlier_assignments = np.full(emb_umap.shape[0], -1)
	# 		if np.sum(outliers) > 0:
	# 			# Compute distances from outlier points to centroids
	# 			outlier_indices = np.where(outliers)[0]
	# 			outlier_points = emb_umap[outlier_indices]
	# 			distances = np.linalg.norm(outlier_points[:, np.newaxis] - centroids, axis=2)
	# 			# Assign each outlier to the nearest cluster
	# 			nearest_clusters = unique_clusters[np.argmin(distances, axis=1)]
	# 			outlier_assignments[outlier_indices] = nearest_clusters
			
	# 		plt.figure(figsize=(18, 10))
	# 		# Map cluster labels to colors from the 'tab20' palette
	# 		tab20_cmap = plt.cm.get_cmap('tab20')
	# 		cluster_colors = tab20_cmap(np.linspace(0, 1, len(unique_clusters)))
	# 		# Create a mapping of cluster labels to colors
	# 		cluster_color_map = {cluster: color for cluster, color in zip(unique_clusters, cluster_colors)}
			
	# 		# Plot inliers as empty circles with edge colors matching their cluster
	# 		for cluster in unique_clusters:
	# 			cluster_mask = labels == cluster
	# 			plt.scatter(
	# 				emb_umap[cluster_mask, 0],
	# 				emb_umap[cluster_mask, 1],
	# 				facecolors='none',
	# 				edgecolors=cluster_color_map[cluster],
	# 				marker='o',
	# 				s=30,
	# 				linewidths=1.1,
	# 				alpha=0.98,
	# 				label=None,
	# 				zorder=2,
	# 			)
			
	# 		# Plot outliers with same color as nearest cluster, less transparency
	# 		if np.sum(outliers) > 0:
	# 			plt.scatter(
	# 				emb_umap[outliers, 0],
	# 				emb_umap[outliers, 1],
	# 				# facecolors='none',
	# 				facecolors=[cluster_color_map[cluster] for cluster in outlier_assignments[outliers]],
	# 				marker='^',
	# 				s=15,
	# 				linewidths=1.0,
	# 				alpha=0.7,
	# 				label=None,
	# 				zorder=1,
	# 			)
	# 		# Plot centroids with colors matching their clusters
	# 		for i, cluster in enumerate(unique_clusters):
	# 			plt.scatter(
	# 				centroids[cluster, 0],
	# 				centroids[cluster, 1],
	# 				c=[cluster_color_map[cluster]],
	# 				marker='x',
	# 				s=300,
	# 				linewidths=3.5,
	# 				alpha=0.75,
	# 				zorder=3,
	# 			)
			
	# 		plt.title(f'2D UMAP Visualization of Text Embeddings with Top Phrases for {len(unique_clusters)} Clusters')
	# 		plt.xlabel('UMAP 1')
	# 		plt.ylabel('UMAP 2')
	# 		ax = plt.gca()
	# 		if ax.legend_ is not None:
	# 			ax.legend_.remove()
	# 		plt.savefig(os.path.join(dataset_dir, 'umap_cluster_visualization_with_phrases.png'), bbox_inches='tight')
	# 		plt.close()
	# 	else:
	# 		print("No unique clusters found, skipping UMAP visualization with top phrases...")

	# Visualization 9: Cluster Size vs. Term Count Plot
	if enable_visualizations:
		print("Cluster Size vs. Term Count Plot")
		print(f"Cluster Text Counts(before merging) {len(cluster_text_counts)}: {cluster_text_counts}")
		print(f"Term Counts per Cluster(before merging): {len(term_counts_per_cluster)}: {term_counts_per_cluster}")
		# Align clusters with valid phrases
		valid_clusters = [i for i in range(n_clusters) if i in cluster_text_counts and i < len(term_counts_per_cluster)]
		if valid_clusters:
			plt.figure(figsize=(19, 13))
			sns.scatterplot(
				x=[cluster_text_counts[i] for i in valid_clusters],
				y=[term_counts_per_cluster[i] for i in valid_clusters],
				hue=[i for i in valid_clusters],
				palette='tab20',
				size=[term_counts_per_cluster[i] for i in valid_clusters],
				sizes=(50, 500),
				legend=False,
			)
			for i in valid_clusters:
				plt.text(
					cluster_text_counts[i], 
					term_counts_per_cluster[i] + 5,
					f'Topic {i}',
					ha='center', 
					va='bottom',
					fontsize=10,
				)
			plt.title('Cluster Size vs. Number of Unique Terms')
			plt.xlabel('Number of Documents in Cluster')
			plt.ylabel('Number of Unique Terms')
			ax = plt.gca()
			if ax.legend_ is not None:
				ax.legend_.remove()
			# plt.legend(title='Cluster', bbox_to_anchor=(1.01, 1), loc='upper left')
			plt.savefig(os.path.join(dataset_dir, 'cluster_size_vs_term_count.png'), bbox_inches='tight')
			plt.close()
		else:
			print("No valid clusters found, skipping Cluster Size vs. Term Count Plot...")

	# Visualization 10: Interactive Topic Similarity Network
	if enable_visualizations:
		G = nx.Graph()
		for i in range(len(topics_before_merging)):
				G.add_node(i, label=f'Topic {i}')
		edge_weights = []
		for i in range(len(topics_before_merging)):
				for j in range(i + 1, len(topics_before_merging)):
						sim = similarity_matrix[i, j]
						if sim > 0.5:
								G.add_edge(i, j, weight=sim)
								edge_weights.append(sim)
		pos = nx.spring_layout(G)
		edge_x = []
		edge_y = []
		edge_text = []
		for edge in G.edges():
				x0, y0 = pos[edge[0]]
				x1, y1 = pos[edge[1]]
				edge_x.extend([x0, x1, None])
				edge_y.extend([y0, y1, None])
				weight = G[edge[0]][edge[1]]['weight']
				edge_text.extend([f'Similarity: {weight:.2f}', None, None])
		node_x = [pos[i][0] for i in G.nodes()]
		node_y = [pos[i][1] for i in G.nodes()]
		node_text = [f'Topic {i}' for i in G.nodes()]
		fig = go.Figure()
		fig.add_trace(go.Scatter(
				x=edge_x, y=edge_y,
				mode='lines',
				line=dict(width=2, color='gray'),
				hoverinfo='text',
				text=edge_text,
				showlegend=False
		))
		fig.add_trace(go.Scatter(
				x=node_x, y=node_y,
				mode='markers+text',
				text=node_text,
				textposition='top center',
				marker=dict(size=20, color='lightblue'),
				hoverinfo='text',
				name='Topics'
		))
		fig.update_layout(
				title='Interactive Topic Similarity Network (Edges for Similarity > 0.5)',
				showlegend=True,
				hovermode='closest',
				margin=dict(b=20, l=5, r=5, t=40),
				xaxis=dict(showgrid=False, zeroline=False),
				yaxis=dict(showgrid=False, zeroline=False)
		)
		fig.write_html(os.path.join(dataset_dir, 'topic_similarity_network_interactive.html'))

	# Merge similar topics
	print(f"\n>> Merging similar topics with dynamic merging threshold {merge_threshold:.4f}")
	merged_topics = []
	used_indices = set()
	min_topics = max(1, len(topics_before_merging) // 2)
	max_merges = len(topics_before_merging) - min_topics
	merge_count = 0
	for i in range(len(topics_before_merging)):
		if i in used_indices:
			continue
		merged_words = set(topics_before_merging[i])
		merged_indices = {i}
		used_indices.add(i)
		for j in range(len(topics_before_merging)):
			if j in used_indices or i == j or merge_count >= max_merges:
				continue
			if similarity_matrix[i, j] > merge_threshold:
				merged_words.update(topics_before_merging[j])
				merged_indices.add(j)
				used_indices.add(j)
				merge_count += 1
		
		# Aggregate frequencies and prioritize diversity
		counter = Counter()
		for topic_idx in merged_indices:
			for phrase in topics_before_merging[topic_idx]:
				for orig_counter in cluster_phrases.values():
					if phrase in orig_counter:
						counter[phrase] += orig_counter[phrase]
		
		# Score phrases with diversity bonus
		phrase_scores = []
		seen_words = set()
		for phrase, count in counter.items():
			words = set(phrase.split())
			diversity_bonus = sum(1 for word in words if word not in seen_words)
			score = count * (1 + 0.2 * diversity_bonus)
			phrase_scores.append((phrase, count, score))
			seen_words.update(words)
		phrase_scores.sort(key=lambda x: x[2], reverse=True)
		print(f"Topic {i} contains {len(counter)} phrases before merging")
		sorted_phrases = [
			phrase 
			for phrase, _, _ in phrase_scores[:top_k_words]
		]
		print(f"\tTopic {i} contains {len(sorted_phrases)} phrases after merging")
		if sorted_phrases:
			merged_topics.append(sorted_phrases)
	print(f"Topics Reduced from {len(topics_before_merging)} to {len(merged_topics)} topics after merging with meriging threshold {merge_threshold}")

	# Visualization 11: Topic Diversity Plot (after merging)
	if enable_visualizations:
		unique_terms_per_topic = [len(set(topic)) for topic in merged_topics]
		plt.figure(figsize=(17, 7))
		plt.bar(range(len(merged_topics)), unique_terms_per_topic, color='#3785e6')
		plt.title('Number of Unique Terms per Topic (after merging)')
		plt.xlabel('Topic ID')
		plt.ylabel('Number of Unique Terms')
		plt.xticks(range(len(merged_topics)))
		plt.savefig(os.path.join(dataset_dir, 'merged_topic_diversity.png'), bbox_inches='tight')
		plt.close()

		# Compute topic diversity (Jaccard similarity between topics)
		topic_sets = [set(topic) for topic in merged_topics]
		jaccard_matrix = np.zeros((len(merged_topics), len(merged_topics)))
		for i in range(len(merged_topics)):
			for j in range(i + 1, len(merged_topics)):
				intersection = len(topic_sets[i] & topic_sets[j])
				union = len(topic_sets[i] | topic_sets[j])
				jaccard_matrix[i, j] = intersection / union if union > 0 else 0
				jaccard_matrix[j, i] = jaccard_matrix[i, j]
		plt.figure(figsize=(15, 11))
		sns.heatmap(
			data=jaccard_matrix, 
			# annot=True, 
			cmap='YlGnBu', 
			vmin=0, 
			vmax=1, 
			square=True,
		)
		plt.title(f'Merged Topic Diversity (Jaccard Similarity Between Topics) meriging threshold {merge_threshold}')
		plt.xlabel('Topic ID')
		plt.ylabel('Topic ID')
		plt.savefig(os.path.join(dataset_dir, 'merged_topic_diversity_jaccard_similarity.png'), bbox_inches='tight')
		plt.close()

	# Visualization 12: Topic Distribution Comparison: Before vs. After Merging
	if enable_visualizations:
		plt.figure(figsize=(19, 10))
		bars_before = plt.bar(
			[i - 0.1 for i in range(n_clusters)],
			[cluster_text_counts[i] for i in range(n_clusters) if i in cluster_text_counts],
			width=0.2,
			color='#3785e6',
			label=f'Before Merging: {sum(cluster_text_counts[i] for i in range(n_clusters))} in {n_clusters} clusters'
		)
		bars_after = plt.bar(
			[i + 0.1 for i in range(len(merged_topics))],
			[len(topic) for topic in merged_topics],
			width=0.2,
			color='#e45151',
			label=f'After Merging (Terms): {sum(len(topic) for topic in merged_topics)} in {len(merged_topics)} clusters'
		)
		plt.title('Cluster Distribution: Before vs. After Merging')
		plt.xlabel('Cluster/Topic')
		plt.xticks(range(max(n_clusters, len(merged_topics))), labels=range(max(n_clusters, len(merged_topics))), fontsize=10, va='center', ha='center', rotation=90)
		plt.ylabel('Count')
		plt.legend()
		plt.savefig(os.path.join(dataset_dir, f'topic_distribution_original_vs_merged_thresh_{merge_threshold:.4f}.png'), bbox_inches='tight')
		plt.close()
	
	# Visualization 13: Noise Analysis Plot
	if enable_visualizations:
		noise_texts = [texts[i] for i, label in enumerate(labels) if label == -1]
		noise_lengths = [len(text.split()) for text in noise_texts]
		noise_stopword_ratios = [sum(1 for word in text.split() if word in CUSTOM_STOPWORDS) / max(1, len(text.split())) for text in noise_texts]
		
		plt.figure(figsize=(18, 10))
		plt.subplot(1, 2, 1)
		plt.hist(noise_lengths, bins=50, color='salmon', edgecolor='black')
		plt.title('Text Length Distribution of Noise Points')
		plt.xlabel('Text Length (Words)')
		plt.ylabel('Number of Texts')
		plt.subplot(1, 2, 2)
		plt.hist(noise_stopword_ratios, bins=50, color='lightgreen', edgecolor='black')
		plt.title('Stopword Ratio Distribution of Noise Points')
		plt.xlabel('Stopword Ratio')
		plt.ylabel('Number of Texts')
		plt.tight_layout()
		plt.savefig(os.path.join(dataset_dir, 'noise_analysis.png'), bbox_inches='tight')
		plt.close()

	# Visualization 14: Phrase Co-Occurrence Network for Each Topic [after merging]
	if enable_visualizations:
		print(f"Generating co-occurrence networks for {len(merged_topics)} topics [after merging]...")
		# Map merged topic indices to original cluster indices
		cluster_to_merged = {}
		current_merged_idx = 0
		used_indices = set()
		for i in range(len(topics_before_merging)):
			if i in used_indices:
				continue
			merged_indices = {i}
			used_indices.add(i)
			for j in range(len(topics_before_merging)):
				if j in used_indices or i == j:
					continue
				if similarity_matrix[i, j] > merge_threshold:
					merged_indices.add(j)
					used_indices.add(j)
			for orig_idx in merged_indices:
				cluster_to_merged[orig_idx] = current_merged_idx
			current_merged_idx += 1
		for merged_idx, topic_phrases in enumerate(merged_topics):
			if not topic_phrases:
					# print(f"Skipping Merged Topic {merged_idx}: No phrases available.")
					continue
			# Select top 10 phrases by frequency to reduce clutter
			counter = Counter()
			original_clusters = [orig_idx for orig_idx, m_idx in cluster_to_merged.items() if m_idx == merged_idx]
			for orig_idx in original_clusters:
					for phrase in cluster_phrases[orig_idx]:
							counter[phrase] += cluster_phrases[orig_idx][phrase]
			phrase_freq = {phrase: counter[phrase] for phrase in topic_phrases if phrase in counter}
			top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]
			top_phrases = [phrase for phrase, _ in top_phrases]
			if not top_phrases:
					# print(f"Skipping Merged Topic {merged_idx}: No phrases after filtering.")
					continue
			# Collect texts from all original clusters that were merged into this topic
			cluster_texts = [
					texts[i] for i, l in enumerate(labels)
					if l in original_clusters and is_english(texts[i], ft_model)
			]
			if not cluster_texts:
					print(f"Skipping Merged Topic {merged_idx}: No valid texts for co-occurrence.")
					continue
			phrase_set = set(top_phrases)
			cooc_matrix = defaultdict(int)
			for text in cluster_texts:
				phrases = kw_model.extract_keywords(
					text,
					keyphrase_ngram_range=(1, 3),
					stop_words="english",
					top_n=15 if len(text.split()) > 100 else 5,
					diversity=0.7,
				)
				valid_phrases = []
				seen_phrases = set()
				for phrase, _ in phrases:
					words = phrase.split()
					stopword_count = sum(1 for w in words if w in CUSTOM_STOPWORDS)
					if stopword_count / len(words) > 0.6 or len(words) < 2:
						continue
					normalized = " ".join(word for i, word in enumerate(words) if i == 0 or word != words[i-1])
					if len(normalized.split()) >= 2 and normalized not in seen_phrases:
						valid_phrases.append(normalized)
						seen_phrases.add(normalized)
				text_phrases = set(valid_phrases).intersection(phrase_set)
				for p1 in text_phrases:
					for p2 in text_phrases:
						if p1 < p2:
							cooc_matrix[(p1, p2)] += 1
			G = nx.Graph()
			for (p1, p2), count in cooc_matrix.items():
				if count >= 2: # An edge exists between two phrases if they appear together in the same text at least twice
					G.add_edge(p1, p2, weight=count)
			for phrase in top_phrases:
				if phrase not in G:
					G.add_node(phrase)
			plt.figure(figsize=(12, 8))
			pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjusted k for better spacing
			edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
			nx.draw_networkx_edges(
				G=G, 
				pos=pos,
				width=[w * 0.1 for w in edge_weights],  # Scale edge width
				alpha=0.6,
				edge_color='#474646'
			)
			nx.draw_networkx_nodes(
				G=G, 
				pos=pos,
				node_size=150,
				node_color='#df9100',
				alpha=0.9
			)
			nx.draw_networkx_labels(
				G=G, 
				pos=pos,
				font_size=5,
				alpha=0.95,
				verticalalignment='center',
				horizontalalignment='center'
			)
			plt.title(f'Phrase Co-Occurrence Network for Merged Topic {merged_idx} ({len(G.nodes())} nodes, {len(G.edges())} edges)')
			plt.axis('off')
			plt.savefig(os.path.join(dataset_dir, f'cooccurrence_network_merged_topic_{merged_idx}.png'), bbox_inches='tight', dpi=300)
			plt.close()

	flat_topics = set(word for topic in merged_topics for word in topic)
	print(f"Extracted {len(flat_topics)} unique topic terms after merging")
	return merged_topics, flat_topics

def clean_labels(labels):
		cleaned = set()
		# Predefined list of valid historical non-ASCII terms
		VALID_HISTORICAL_TERMS = {
				"blitzkrieg", "kübelwagen", "wehrmacht", "panzer", "luftwaffe",
				"stuka", "t-34", "afrika korps"
		}
		for label in labels:
				# Normalize the label
				label = label.lower().strip()
				# Remove non-alphanumeric characters except spaces and hyphens
				label = re.sub(r"[^a-z0-9\s\-äöüßñéèêëìíîïçåæœø]", "", label)
				# Skip short labels, stopwords, or invalid labels
				if label in CUSTOM_STOPWORDS or len(label) < 3:
						continue
				# Skip labels that are just numbers
				if label.isdigit():
						continue
				# Skip labels that start with numbers unless they're years (4 digits)
				if label[0].isdigit() and not (len(label) == 4 and label.isdigit()):
						continue
				# Allow non-ASCII labels if proper noun or in valid historical terms
				if not all(ord(char) < 128 for char in label):
						if not (label[0].isupper() or label in VALID_HISTORICAL_TERMS):
								continue
				cleaned.add(label)
		return sorted(cleaned)

def extract_named_entities(
		nlp: pipeline, 
		text: str, 
		ft_model: fasttext.FastText._FastText
	):
	# Skip if text is not primarily English
	if not is_english(text=text, ft_model=ft_model):
		return []
	try:
		ner_results = nlp(text)
		entities = []
		for entity in ner_results:
			if entity["entity_group"] in RELEVANT_ENTITY_TYPES and entity["word"].isalpha():
				# Clean and normalize entity text
				entity_text = re.sub(r'[^\w\s\-]', '', entity["word"].lower()).strip()
				if (
					entity_text 
					and len(entity_text) > 2 
					and entity_text not in CUSTOM_STOPWORDS 
					# and all(ord(char) < 128 for char in entity_text)
				):
					entities.append(entity_text)
		# Also extract multi-word phrases that might be significant
		tokens = [
			word.lower() for word in text.split() 
			if word.isalpha() and len(word) > 2 
			and word.lower() not in CUSTOM_STOPWORDS
			# and all(ord(char) < 128 for char in word)
		]

		# Return unique list of entities and tokens
		return list(set(entities + tokens))
	except Exception as e:
		print(f"NER error: {e}")
		print(f"Text: {text}")
		return []

def filter_metadata_terms(labels):
	metadata_fragments = [
		'kunststoff',
		'bildetekst',
		'arkiv',
		'quer',
		'riksarkivet',
		'museum donated',
		'association',
		'information received',
	]
	return [label for label in labels if not any(frag in label for frag in metadata_fragments)]

def filter_temporal_labels(labels, doc_year):
		if not doc_year:
				return labels
		year = float(doc_year.split('-')[0]) if isinstance(doc_year, str) else doc_year
		era_ranges = {
				"WWI era": (1914, 1918),
				"WWII era": (1939, 1945),
				"Cold War era": (1947, 1991)
		}
		return [label for label in labels if not any(era in label and not (start <= year <= end) for era, (start, end) in era_ranges.items())]

def handle_multilingual_labels(labels):
	processed_labels = []
	
	for label in labels:
		# Skip non-ASCII labels completely
		if not all(ord(char) < 128 for char in label):
			continue
				
		words = label.split()
		
		# Single word label
		if len(words) == 1:
			if is_likely_english_term(label) or label[0].isupper():
				processed_labels.append(label)
						
		# Multi-word label
		else:
			# Keep if all words are likely English or proper nouns
			if all(is_likely_english_term(word) or word[0].isupper() for word in words):
				processed_labels.append(label)

	return processed_labels

def assign_semantic_categories(labels):
	"""Categorize labels into semantic groups"""
	categorized_labels = []
	for label in labels:
		for category, terms in SEMANTIC_CATEGORIES.items():
			if any(term in label for term in terms):
				if label not in categorized_labels:
					categorized_labels.append(label)
				break
	return categorized_labels

def balance_label_count(
		image_labels_list,
		text_descriptions,
		sent_model,
		min_labels=1,
		max_labels=12,
		similarity_threshold=0.5
	):
	"""
	Balance the number of labels per image, prioritizing multi-word labels and ensuring semantic coherence.
	
	Args:
			image_labels_list: List of lists of labels
			text_descriptions: List of text descriptions for relevance scoring
			sent_model: SentenceTransformer model for embeddings
			min_labels: Minimum number of labels per image
			max_labels: Maximum number of labels per image
			similarity_threshold: Minimum cosine similarity for compound labels
	
	Returns:
			List of lists of balanced labels
	"""
	balanced_labels = []
	
	# Encode text descriptions once
	text_embeds = sent_model.encode(text_descriptions, show_progress_bar=False, convert_to_numpy=True)
	
	for idx, labels in tqdm(enumerate(image_labels_list), total=len(image_labels_list), desc="Label Balancing"):
			text_emb = text_embeds[idx]
			
			# Case 1: Too few labels - generate coherent compounds
			if len(labels) < min_labels:
					compound_labels = []
					if len(labels) >= 2:
							# Encode existing labels
							label_embs = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
							for i in range(len(labels)):
									for j in range(i+1, len(labels)):
											compound = f"{labels[i]} {labels[j]}"
											if 2 <= len(compound.split()) <= 3:  # 2-3 word compounds
													# Compute similarity of compound to text
													compound_emb = sent_model.encode(compound, show_progress_bar=False)
													similarity = np.dot(compound_emb, text_emb) / (
															np.linalg.norm(compound_emb) * np.linalg.norm(text_emb) + 1e-8
													)
													if similarity > similarity_threshold:
															compound_labels.append(compound)
					
					# Sort compounds by similarity
					if compound_labels:
							compound_embs = sent_model.encode(compound_labels, show_progress_bar=False, convert_to_numpy=True)
							similarities = np.dot(compound_embs, text_emb) / (
									np.linalg.norm(compound_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
							)
							compound_labels = [compound_labels[i] for i in np.argsort(similarities)[::-1]]
					
					# Add compounds to reach min_labels
					expanded_labels = labels + compound_labels[:min_labels - len(labels)]
					balanced_labels.append(expanded_labels[:min_labels] if len(expanded_labels) >= min_labels else expanded_labels)
			
			# Case 2: Too many labels - prioritize multi-word by relevance
			elif len(labels) > max_labels:
					# Encode labels and compute similarities
					label_embs = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
					similarities = np.dot(label_embs, text_emb) / (
							np.linalg.norm(label_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
					)
					
					# Separate multi-word and single-word labels
					multi_word = [(label, sim) for label, sim in zip(labels, similarities) if ' ' in label]
					single_word = [(label, sim) for label, sim in zip(labels, similarities) if ' ' not in label]
					
					# Sort by similarity
					multi_word.sort(key=lambda x: x[1], reverse=True)
					single_word.sort(key=lambda x: x[1], reverse=True)
					
					# Prioritize multi-word, fill with single-word if needed
					selected_labels = [label for label, _ in multi_word[:max_labels]]
					if len(selected_labels) < max_labels:
							selected_labels.extend([label for label, _ in single_word[:max_labels - len(selected_labels)]])
					
					balanced_labels.append(selected_labels)
			
			# Case 3: Good label count - keep as is
			else:
					balanced_labels.append(labels)
	
	return balanced_labels

def quick_filter_candidates(
		text: str, 
		labels: list[str], 
		sent_model: SentenceTransformer, 
		max_keep: int=30
	) -> list[str]:
	if not labels:
		print(f"<!> Empty labels list: {text}")
		return []
	text_emb = sent_model.encode(text, show_progress_bar=False)
	label_embs = sent_model.encode(labels, show_progress_bar=False)
	# Ensure label_embs is not empty (redundant but defensive)
	if label_embs.size == 0:
		print(f"<!> No embeddings generated for labels: {labels}")
		return []
	similarities = np.dot(label_embs, text_emb) / (np.linalg.norm(label_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8)
	sorted_labels = [label for _, label in sorted(zip(similarities, labels), reverse=True)]
	return sorted_labels[:max_keep]

def process_document_chunk(chunk_data):
	"""Process a chunk of documents in parallel"""
	start_idx, end_idx, texts, combined_labels = chunk_data
	
	# Initialize sent_model within the process
	local_model = SentenceTransformer("all-MiniLM-L6-v2")
	
	results = []
	for i in range(start_idx, end_idx):
		idx = i - start_idx  # Local index within chunk
		text = texts[idx]
		labels = combined_labels[idx]
		
		# Quick filter first
		filtered_labels = quick_filter_candidates(text, labels)
		
		if not filtered_labels:
				results.append([])
				continue
				
		# Encode text and labels
		text_emb = local_model.encode(text)
		label_embs = local_model.encode(filtered_labels)
		
		# Calculate similarities
		similarities = []
		for label_emb in label_embs:
				sim = np.dot(text_emb, label_emb) / (
						np.linalg.norm(text_emb) * np.linalg.norm(label_emb) + 1e-8
				)
				similarities.append(sim)
				
		# Filter by threshold
		threshold = 0.3
		relevant_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
		results.append([filtered_labels[i] for i in relevant_indices])
	
	return results

def get_all_unique_user_queries(df: pd.DataFrame) -> Set[str]:
		"""Collects all unique and cleaned user_query terms from the entire DataFrame."""
		unique_queries = set()
		if 'user_query' in df.columns:
				for query in df['user_query'].fillna('').astype(str):
						if query.strip():
								# Simple cleaning for user query before adding to global pool
								cleaned_query = re.sub(r'[^\w\s\-]', '', query.lower()).strip()
								if cleaned_query and cleaned_query not in CUSTOM_STOPWORDS and len(cleaned_query) > 2:
										unique_queries.add(cleaned_query)
		return unique_queries

def is_year_compatible(category, year):
		"""
		Check if a label's era is compatible with the document year.
		
		Args:
						category: String label
						year: Float or int document year
		
		Returns:
						Bool indicating compatibility
		"""
		if year is None:
						return True

		era_ranges = {
						"WWI era": (1914, 1918),
						"WWII era": (1939, 1945),
						"Cold War era": (1947, 1991)
		}
		for era, (start, end) in era_ranges.items():
						if era.lower() in category.lower() and year and not (start <= year <= end):
										return False
		return True

def post_process_labels(labels, text_description, sent_model, doc_year, vlm_scores, max_labels=10, similarity_threshold=0.8):
		"""
		Post-process visual labels to remove duplicates, validate temporally, and rank by relevance.
		
		Args:
				labels: List of labels
				text_description: String enriched document description
				sent_model: SentenceTransformer model
				doc_year: Float or int document year
				vlm_scores: List of VLM similarity scores
				max_labels: Maximum number of labels to retain
				similarity_threshold: Cosine similarity threshold for deduplication
		
		Returns:
				List of processed labels
		"""
		if not labels:
				return []
		
		# Validate vlm_scores length
		if len(vlm_scores) != len(labels):
				vlm_scores = vlm_scores + [0.0] * (len(labels) - len(vlm_scores)) if len(vlm_scores) < len(labels) else vlm_scores[:len(labels)]
		
		# Encode labels and text
		label_embs = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
		text_emb = sent_model.encode(text_description, show_progress_bar=False, convert_to_numpy=True)
		
		# Deduplicate based on semantic similarity
		deduplicated = []
		for i, (label, score) in enumerate(zip(labels, vlm_scores)):
				is_redundant = False
				for kept_label, kept_emb, _ in deduplicated:
						sim = np.dot(label_embs[i], kept_emb) / (np.linalg.norm(label_embs[i]) * np.linalg.norm(kept_emb) + 1e-8)
						if sim > similarity_threshold:
								is_redundant = True
								break
				if not is_redundant:
						deduplicated.append((label, label_embs[i], score))
		
		# Temporal validation
		validated = [
				(label, emb, score)
				for label, emb, score in deduplicated
				if is_year_compatible(label, doc_year)
		]
		
		# Rank by combined VLM and text similarity
		if validated:
				text_sims = np.dot(np.array([emb for _, emb, _ in validated]), text_emb) / (
						np.linalg.norm([emb for _, emb, _ in validated], axis=1) * np.linalg.norm(text_emb) + 1e-8
				)
				combined_scores = [0.6 * vlm_score + 0.4 * text_sim for vlm_score, text_sim in zip(
						[score for _, _, score in validated], text_sims
				)]
				ranked = [label for _, label in sorted(zip(combined_scores, [label for label, _, _ in validated]), reverse=True)]
				return ranked[:max_labels]
		
		return []

def process_category_batch(
		batch_paths, batch_descriptions, batch_indices, df, categories, prompt_embeds,
		category_type, sent_model, processor, model, device, verbose, base_thresholds, sub_batch_size
):
		"""
		Process a batch of images for a specific category type with adaptive thresholding and sparse metadata handling.
		
		Args:
				batch_paths: List of image file paths
				batch_descriptions: List of enriched document descriptions
				batch_indices: List of global indices for the batch
				df: DataFrame with metadata
				categories: List of category strings
				prompt_embeds: Pre-computed prompt embeddings
				category_type: String ('object', 'scene', 'era', 'activity')
				sent_model: SentenceTransformer model
				processor: ALIGN processor
				model: ALIGN model
				device: Torch device
				verbose: Bool for logging
				base_thresholds: Dict of base thresholds per category type
				sub_batch_size: Size of sub-batches for GPU memory management
		
		Returns:
				batch_results: List of lists of selected categories
				batch_scores: List of lists of VLM scores
		"""
		valid_images = []
		valid_indices = []
		valid_descriptions = []
		failed_images = []

		# Handle sparse metadata with fallback to title
		user_queries = df['user_query'].fillna('').iloc[batch_indices].tolist() if 'user_query' in df.columns else [''] * len(batch_indices)
		titles = df['title'].fillna('').iloc[batch_indices].tolist() if 'title' in df.columns else [''] * len(batch_indices)

		# Validate images and descriptions
		for i, path in enumerate(batch_paths):
			try:
				if os.path.exists(path):
					img = Image.open(path).convert('RGB')
					desc = batch_descriptions[i].strip()
					if not desc and user_queries[i].strip():
						desc = user_queries[i]
					if not desc and titles[i].strip():
						desc = titles[i]
					if len(desc.strip()) < 10:
						desc = ""  # Treat as sparse
					valid_images.append(img)
					valid_indices.append(i)
					valid_descriptions.append(desc)
				else:
					failed_images.append(path)
			except Exception as e:
				failed_images.append(path)
				if verbose:
					print(f"Error loading image {path}: {e}")

		if not valid_images:
			if verbose and failed_images:
				print(f"Failed to load {len(failed_images)} images in batch")
			return [[] for _ in range(len(batch_paths))], [[] for _ in range(len(batch_paths))]

		# Add dynamic categories from user_query
		extended_categories = categories.copy()
		new_queries = []
		for query in user_queries:
			if isinstance(query, str) and query.strip() and query not in extended_categories:
				# extended_categories.append(query)
				# new_queries.append(query)
				try:
					# Try to parse as a list
					parsed_query = ast.literal_eval(query) if query.startswith('[') and query.endswith(']') else [query]
					for q in parsed_query:
						if isinstance(q, str) and q.strip() and q not in extended_categories:
							extended_categories.append(q)
							new_queries.append(q)
				except (ValueError, SyntaxError):
					# Treat as single string if parsing fails
					if query not in extended_categories:
						extended_categories.append(query)
						new_queries.append(query)

		# Update prompt embeddings
		extended_prompt_embeds = prompt_embeds
		if new_queries:
			new_prompts = [f"a photo of {q}" for q in new_queries]
			new_embeds = sent_model.encode(new_prompts, device=device, convert_to_tensor=True, show_progress_bar=False)
			extended_prompt_embeds = torch.cat([prompt_embeds, new_embeds], dim=0)

		# Compute text similarities
		text_prompts = [f"a photo of {cat}" for cat in extended_categories]
		with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			text_embeds = sent_model.encode(valid_descriptions, device=device, convert_to_tensor=True, show_progress_bar=False)
			text_similarities = torch.mm(extended_prompt_embeds, text_embeds.T).cpu().numpy()
			prompt_norms = torch.norm(extended_prompt_embeds, dim=1, keepdim=True)
			text_norms = torch.norm(text_embeds, dim=1, keepdim=True)
			text_similarities = text_similarities / (torch.mm(prompt_norms, text_norms.T).cpu().numpy() + 1e-8)
			model.eval()
			batch_results = [[] for _ in range(len(batch_paths))]
			batch_scores = [[] for _ in range(len(batch_paths))]
			# Process sub-batches
			for sub_idx in range(0, len(valid_images), sub_batch_size):
				sub_end = min(sub_idx + sub_batch_size, len(valid_images))
				sub_images = valid_images[sub_idx:sub_end]
				sub_valid_indices = valid_indices[sub_idx:sub_end]
				inputs = processor(
					text=text_prompts,
					images=sub_images,
					padding="max_length",
					return_tensors="pt",
				).to(device)
				outputs = model(**inputs)
				image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
				vlm_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
				vlm_similarity = (100.0 * image_embeds @ vlm_embeds.T).softmax(dim=-1).cpu().numpy()
				threshold = base_thresholds[category_type]
				for i, img_idx in enumerate(range(sub_idx, sub_end)):
					batch_idx = sub_valid_indices[i]
					local_img_idx = img_idx - sub_idx
					# Entropy-based threshold adjustment
					img = sub_images[i]
					img_array = np.array(img.convert('L'))
					img_array = resize(img_array, (128, 128), anti_aliasing=True, preserve_range=True).astype(np.uint8)
					entropy = shannon_entropy(img_array)
					complexity_factor = max(0.7, 1.0 - 0.15 * (entropy / 8.0))
					adaptive_threshold = max(0.15, threshold * complexity_factor)
					is_sparse = not valid_descriptions[img_idx].strip()
					for cat_idx, cat in enumerate(extended_categories):
						# Moved text_score computation inside the loop
						text_score = text_similarities[cat_idx, img_idx]
						adjusted_threshold = adaptive_threshold
						if text_score > 0.5:
							adjusted_threshold *= 0.8
						elif text_score < 0.1:
							adjusted_threshold *= 1.2
						vlm_score = vlm_similarity[local_img_idx, cat_idx]
						combined_score = (
							0.5 * vlm_score + 0.5 * text_score if is_sparse
							else vlm_score * (0.8 + 0.2 * text_score)
						)
						if combined_score > adjusted_threshold:
							batch_results[batch_idx].append(cat)
							batch_scores[batch_idx].append(vlm_score)

		return batch_results, batch_scores

def combine_and_clean_labels(ner_labels, keywords, topic_labels, user_query, text, sent_model, min_threshold=0.4, max_threshold=0.7):
		"""
		Combine labels from NER, keywords, topics, and user_query, splitting concatenated phrases and ensuring coherence.
		
		Args:
				ner_labels: List of NER-extracted labels
				keywords: List of KeyBERT-extracted keywords
				topic_labels: List of topic-derived labels
				user_query: User-provided query (string or list)
				text: Original text for context
				sent_model: SentenceTransformer model
				min_threshold: Minimum similarity threshold for label retention
				max_threshold: Maximum similarity threshold for deduplication
		
		Returns:
				List of cleaned, unique labels
		"""
		# Initialize combined labels
		combined = set(ner_labels + keywords + topic_labels)
		
		# Parse user_query
		if isinstance(user_query, str) and user_query.strip():
				try:
						parsed_query = ast.literal_eval(user_query) if user_query.startswith('[') else [user_query]
						combined.update(str(q).strip().lower() for q in parsed_query if isinstance(q, str) and q.strip())
				except (ValueError, SyntaxError):
						combined.add(user_query.strip().lower())
		
		# Remove stopwords and invalid labels
		combined = [label for label in combined if label.lower() not in CUSTOM_STOPWORDS and len(label.split()) <= 2]
		
		if not combined:
				return []
		
		# Split concatenated phrases
		split_labels = []
		for label in combined:
				words = label.split()
				if len(words) > 2:
						# Split into 1-2 word phrases
						for i in range(0, len(words), 2):
								phrase = " ".join(words[i:i+2])
								if phrase and any(word in text.lower() for word in phrase.split()):
										split_labels.append(phrase)
						# Add single words if not covered
						for word in words:
								if word not in CUSTOM_STOPWORDS and word in text.lower():
										split_labels.append(word)
				else:
						split_labels.append(label)
		
		# Remove duplicates
		split_labels = list(set(split_labels))
		
		# Encode text and labels
		text_emb = sent_model.encode(text, show_progress_bar=False)
		label_embs = sent_model.encode(split_labels, show_progress_bar=False)
		
		# Filter by relevance to text
		similarities = np.dot(label_embs, text_emb) / (
				np.linalg.norm(label_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
		)
		relevant_labels = [split_labels[i] for i in range(len(split_labels)) if similarities[i] > min_threshold]
		
		if not relevant_labels:
				return []
		
		# Semantic deduplication
		label_embs = sent_model.encode(relevant_labels, show_progress_bar=False)
		deduplicated = []
		for i, (label, sim) in enumerate(zip(relevant_labels, similarities)):
				is_redundant = False
				for kept_label, _, kept_emb in deduplicated:
						sim_to_kept = np.dot(label_embs[i], kept_emb) / (
								np.linalg.norm(label_embs[i]) * np.linalg.norm(kept_emb) + 1e-8
						)
						if sim_to_kept > max_threshold or label in kept_label or kept_label in label:
								is_redundant = True
								break
				if not is_redundant and any(word in text.lower() for word in label.split()):
						deduplicated.append((label, sim, label_embs[i]))
		
		# Sort by similarity and limit to max 10 labels
		deduplicated.sort(key=lambda x: x[1], reverse=True)
		final_labels = [label for label, _, _ in deduplicated[:10]]
		
		return sorted(set(final_labels))

def batch_filter_by_relevance(
		sent_model: SentenceTransformer,
		texts: list,
		all_labels_list: list[list[str]],
		threshold: float,
		batch_size: int,
		verbose: bool = True
	):
	"""
	Filter labels by relevance to their corresponding texts, dynamically adjusting batch size based on GPU memory.
	
	Args:
			sent_model: SentenceTransformer model
			texts: List of texts
			all_labels_list: List of label lists
			threshold: Relevance threshold
			batch_size: Initial batch size for processing
			verbose: Boolean for logging
	
	Returns:
			List of filtered label lists
	"""
	results = []
	total = len(texts)
	
	# Check if using GPU
	device = torch.device(sent_model.device)
	is_cuda = device.type == 'cuda'
	if is_cuda and verbose:
			print(f"Using GPU: {device}, Initial batch_size: {batch_size}")
	# Function to estimate safe batch size based on GPU memory
	def get_safe_batch_size(initial_batch_size, texts, labels_list):
			if not is_cuda:
					return min(initial_batch_size, 32)  # Default for CPU
			
			total_memory = torch.cuda.get_device_properties(device).total_memory
			free_memory = total_memory - torch.cuda.memory_allocated(device)
			# Estimate memory per text (conservative estimate: 2 MiB per text for encoding)
			memory_per_text = 2 * 1024 * 1024  # 2 MiB
			# Account for labels (assume average 5 labels per text, 0.5 MiB per label)
			avg_labels = sum(len(labels) for labels in labels_list) / len(labels_list) if labels_list else 5
			memory_per_label = 0.5 * 1024 * 1024  # 0.5 MiB
			# Total memory per sample (text + labels)
			memory_per_sample = memory_per_text + avg_labels * memory_per_label
			# Reserve 20% of free memory for safety
			safe_memory = free_memory * 0.8
			# Calculate safe batch size
			safe_batch_size = max(1, int(safe_memory // memory_per_sample))
			# Cap at initial batch size or a reasonable maximum
			return min(initial_batch_size, safe_batch_size, 64)
	for batch_start in range(0, total, batch_size):
			batch_end = min(batch_start + batch_size, total)
			batch_texts = texts[batch_start:batch_end]
			batch_labels_list = all_labels_list[batch_start:batch_end]
			
			# Dynamically adjust batch size based on GPU memory
			adjusted_batch_size = get_safe_batch_size(batch_size, batch_texts, batch_labels_list)
			if verbose and is_cuda and adjusted_batch_size != batch_size:
					print(f"Adjusted batch_size to {adjusted_batch_size} due to GPU memory constraints "
								f"(free: {torch.cuda.memory_allocated(device)/1024**2:.2f} MiB)")
			# Encode batch texts with retries
			text_embeddings = None
			current_batch_size = adjusted_batch_size
			retries = 3
			while retries > 0 and text_embeddings is None:
					try:
							text_embeddings = sent_model.encode(
									batch_texts,
									show_progress_bar=False,
									batch_size=current_batch_size,
									convert_to_numpy=True
							)
					except torch.cuda.OutOfMemoryError as e:
							if verbose:
									print(f"CUDA OOM error with batch_size={current_batch_size}: {e}. Retrying with smaller batch...")
							current_batch_size = max(1, current_batch_size // 2)
							retries -= 1
							if retries == 0:
									print("Failed to encode batch after retries. Skipping batch.")
									results.extend([[] for _ in range(len(batch_texts))])
									continue
			
			if text_embeddings is None:
					continue
			batch_results = []
			for i, (text_emb, labels) in enumerate(zip(text_embeddings, batch_labels_list)):
					if not labels:
							batch_results.append([])
							continue
					# Encode labels with retry mechanism
					label_embeddings = None
					current_label_batch_size = adjusted_batch_size
					retries = 3
					while retries > 0 and label_embeddings is None:
							try:
									label_embeddings = sent_model.encode(
											labels,
											show_progress_bar=False,
											batch_size=current_label_batch_size,
											convert_to_numpy=True
									)
							except torch.cuda.OutOfMemoryError as e:
									if verbose:
											print(f"CUDA OOM error for labels with batch_size={current_label_batch_size}: {e}. Retrying with smaller batch...")
									current_label_batch_size = max(1, current_label_batch_size // 2)
									retries -= 1
									if retries == 0:
											print("Failed to encode labels after retries. Skipping labels.")
											batch_results.append([])
											break
					
					if label_embeddings is None:
							continue
					similarities = np.dot(label_embeddings, text_emb) / (
							np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(text_emb) + 1e-8
					)
					relevant_indices = np.where(similarities > threshold)[0]
					batch_results.append([labels[idx] for idx in relevant_indices])
			
			results.extend(batch_results)
	return results

def deduplicate_labels(labels):
		"""
		Deduplicate labels based on semantic similarity and substring containment.
		
		Args:
				labels: List of labels
		
		Returns:
				List of deduplicated labels
		"""
		if not labels:
				return []
		
		# Substring and semantic deduplication
		deduplicated = []
		for label in sorted(labels, key=len, reverse=True):
				if not any(
						label in kept_label or kept_label in label or label.lower() == kept_label.lower()
						for kept_label in deduplicated
				):
						deduplicated.append(label)
		
		return sorted(deduplicated)

def get_textual_based_annotation(
		csv_file: str, 
		num_workers: int,
		batch_size: int,
		relevance_threshold: float,
		metadata_fpth: str,
		device: str,
		st_model_name: str,
		ner_model_name: str,
		verbose: bool=True,
		use_parallel: bool=False,
	):
	if verbose:
		print(f"Automatic label extraction from text data".center(160, "-"))
		print(f"Loading metadata from {csv_file}...")
	text_based_annotation_start_time = time.time()
	dataset_dir = os.path.dirname(csv_file)
	
	if verbose:
		print(f"Loading sentence-transformer model: {st_model_name}...")
	sent_model = SentenceTransformer(model_name_or_path=st_model_name, device=device)
	
	ft_model = fasttext.load_model(FastText_Language_Identification)
	
	if verbose:
		print(f"Loading NER model: {ner_model_name}...")
	nlp = pipeline(
		task="ner", 
		model=AutoModelForTokenClassification.from_pretrained(ner_model_name),
		tokenizer=AutoTokenizer.from_pretrained(ner_model_name), 
		aggregation_strategy="simple",
		device=device,
		batch_size=batch_size,
	)
	dtypes = {
			'doc_id': str, 'id': str, 'label': str, 'title': str,
			'description': str, 'img_url': str, 'enriched_document_description': str,
			'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
			'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
			'user_query': str,
	}
	
	df = pd.read_csv(
			filepath_or_buffer=csv_file, 
			on_bad_lines='skip',
			dtype=dtypes, 
			low_memory=False,
	)
	if verbose:
		print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")
	
	df['content'] = df['enriched_document_description'].fillna('').astype(str)
	# Handle missing 'user_query' column
	if 'user_query' not in df.columns:
		if verbose:
			print("Warning: 'user_query' column missing in DataFrame. Using empty queries.")
		user_queries = [''] * len(df)
	else:
		user_queries = df['user_query'].fillna('').tolist()
	num_samples = df.shape[0]
	
	print(f"Filtering non-English entries for {num_samples} samples")
	t0 = time.time()
	english_mask = df['content'].apply(lambda x: is_english(text=x, ft_model=ft_model, verbose=False))
	english_indices = english_mask[english_mask].index.tolist()
	print(f"{sum(english_mask)} / {len(df)} texts are English [{sum(english_mask)/len(df)*100:.2f}%]")
	print(f"Elapsed_t: {time.time() - t0:.2f} sec")
	
	english_df = df[english_mask].reset_index(drop=True)
	english_texts = english_df['content'].tolist()
	english_queries = [user_queries[i] for i in english_indices]
	per_image_labels = [[] for _ in range(num_samples)]
	if len(english_texts) > 0:
			# Step 1: Topic Modeling
			print("Topic Modeling".center(160, "-"))
			t0 = time.time()
			topics, flat_topic_words = extract_semantic_topics(
					sent_model=sent_model,
					ft_model=ft_model,
					texts=english_texts,
					num_workers=num_workers,
					dataset_dir=dataset_dir,
			)
			print(f"{len(topics)} Topics (clusters) {type(topics)}:\n{[len(tp) for tp in topics]}")
			print(f"Elapsed_t: {time.time()-t0:.2f} sec".center(160, "-"))
			
			# Step 2: Named Entity Recognition
			print("Extracting NER per sample...")
			t0 = time.time()
			if len(english_texts) > 1000 and use_parallel:
					chunk_size = len(english_texts) // num_workers + 1
					chunks = [(english_texts[i:i+chunk_size]) for i in range(0, len(english_texts), chunk_size)]
					print(f"Using {num_workers} processes for NER extraction...")
					with multiprocessing.Pool(processes=num_workers) as pool:
							ner_results = pool.map(process_text_chunk, [(nlp, chunk) for chunk in chunks])
					per_image_ner_labels = []
					for chunk_result in ner_results:
							per_image_ner_labels.extend(chunk_result)
			else:
					per_image_ner_labels = []
					for text in tqdm(english_texts, desc="NER Progress"):
							entities = extract_named_entities(nlp=nlp, text=text, ft_model=ft_model)
							per_image_ner_labels.append(entities)
			print(f"NER done in {time.time() - t0:.1f} sec")
			
			# Step 3: Extract keywords
			print("Extracting keywords per image using KeyBERT...")
			t0 = time.time()
			per_image_keywords = []
			kw_model = KeyBERT(model=sent_model)
			for text in tqdm(english_texts, desc="Keyword Extraction"):
					if not is_english(text, ft_model):
							per_image_keywords.append([])
							continue
					try:
							phrases = kw_model.extract_keywords(
									text,
									keyphrase_ngram_range=(1, 2),
									stop_words=CUSTOM_STOPWORDS,
									top_n=10,
									diversity=0.7,
							)
							keywords = []
							seen_phrases = set()
							for phrase, _ in phrases:
									if phrase.lower() in CUSTOM_STOPWORDS or len(phrase.split()) > 2:
											continue
									normalized = " ".join(word for word in phrase.split() if word not in CUSTOM_STOPWORDS)
									if normalized and normalized not in seen_phrases and any(word in text.lower() for word in normalized.split()):
											keywords.append(normalized)
											seen_phrases.add(normalized)
							per_image_keywords.append(keywords)
					except Exception as e:
							print(f"KeyBERT error for text: {text[:100]}...: {e}")
							per_image_keywords.append([])
			print(f"Keyword extraction done in {time.time() - t0:.1f} sec")
			# Step 4: Add topic labels
			print("Assigning topic labels per image...")
			t0 = time.time()
			per_image_topic_labels = []
			for text in tqdm(english_texts, desc="Topic Assignment"):
					matching_topics = [word for word in flat_topic_words if word in text.lower() and word not in CUSTOM_STOPWORDS]
					per_image_topic_labels.append(matching_topics)
			print(f"Topic assignment done in {time.time() - t0:.1f} sec")
			
			# Step 5: Combine and clean labels
			print("Combining and cleaning labels...")
			t0 = time.time()
			per_image_combined_labels = []
			for text, query, ner, keywords, topics in tqdm(zip(english_texts, english_queries, per_image_ner_labels, per_image_keywords, per_image_topic_labels), total=len(english_texts), desc="Label Combination"):
					cleaned_labels = combine_and_clean_labels(ner, keywords, topics, query, text, sent_model, min_threshold=0.4, max_threshold=0.7)
					per_image_combined_labels.append(cleaned_labels)
			print(f"Label combination and cleaning done in {time.time() - t0:.3f} sec")
			# Step 6: Filter by relevance
			print(f"Filtering labels by relevance (thresh: {relevance_threshold})...")
			t0 = time.time()
			if use_parallel:
					print("Using parallel processing for relevance filtering...")
					per_image_relevant_labels = parallel_relevance_filtering(
							texts=english_texts,
							all_labels=per_image_combined_labels,
							n_processes=num_workers,
					)
			else:
					print(f"Using batch processing for textual-based annotation ({batch_size} batches) for relevance filtering (thresh: {relevance_threshold})...")
					per_image_relevant_labels = batch_filter_by_relevance(
						sent_model=sent_model,
						texts=english_texts,
						all_labels_list=per_image_combined_labels,
						threshold=relevance_threshold,
						batch_size=batch_size,
					)
			print(f"Relevance filtering done in {time.time() - t0:.1f} sec")
			# Step 7: Post-process
			print("Post-processing labels, deduplication, and semantic categorization...")
			t0 = time.time()
			english_labels = []
			for i, relevant_labels in enumerate(per_image_relevant_labels):
					filtered_labels = handle_multilingual_labels(relevant_labels)
					filtered_labels = deduplicate_labels(filtered_labels)
					categorized = assign_semantic_categories(filtered_labels)
					final_labels = sorted(set(filtered_labels + categorized))
					english_labels.append(final_labels)
			print(f"Post-processing done in {time.time() - t0:.1f} sec")
			print("Balancing label counts...")
			t0 = time.time()
			english_labels = balance_label_count(
					image_labels_list=english_labels, 
					text_descriptions=english_texts, 
					sent_model=sent_model, 
					min_labels=1, 
					max_labels=12,
			)
			print(f"Label balancing done in {time.time() - t0:.3f} sec")
			
			# Transfer results
			for i, orig_idx in enumerate(english_indices):
					if i < len(english_labels):
							per_image_labels[orig_idx] = english_labels[i]
	else:
			print("No English texts found. Returning empty labels for all entries.")
	
	df['textual_based_labels'] = per_image_labels
	df.to_csv(metadata_fpth, index=False)
	
	print(f">> Generated text labels for {sum(1 for labels in per_image_labels if labels)} out of {num_samples} entries")
	print(f"Text-based annotation Elapsed time: {time.time() - text_based_annotation_start_time:.2f} sec".center(160, " "))
	
	return per_image_labels

def get_visual_based_annotation(
		csv_file: str,
		st_model_name: str,
		vlm_model_name: str,
		batch_size: int,
		device: str,
		num_workers: int,
		verbose: bool,
		metadata_fpth: str,
		topk: int=5,
	):
	if verbose:
		print(f"Semi-Supervised label extraction from image data (using VLM) batch_size: {batch_size}".center(160, "-"))
	visual_based_annotation_start_time = time.time()
	CATEGORIES_FILE = "categories.json"
	object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
	candidate_labels = list(set(object_categories + scene_categories + activity_categories))
	texts = [f"This is a photo of {lbl}." for lbl in candidate_labels]
	gpu_name = torch.cuda.get_device_name(device)
	total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 # GB
	available_gpu_memory = torch.cuda.mem_get_info()[0] / 1024**3 # GB
	print(f"Total GPU memory: {total_gpu_memory:.2f} GB ({gpu_name})")
	print(f"Available GPU memory: {available_gpu_memory:.2f} GB ({gpu_name})")
	model = AutoModel.from_pretrained(
		pretrained_model_name_or_path=vlm_model_name, 
		torch_dtype=torch.float16 if available_gpu_memory < 10 else torch.float32,
		device_map=device,
	)
	processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=vlm_model_name)
	print(model.parameters().__next__().dtype)

	if verbose:
		print(f"Loading metadata from {csv_file}...")
	dtypes = {
		'doc_id': str, 'id': str, 'label': str, 'title': str,
		'description': str, 'img_url': str, 'enriched_document_description': str,
		'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
		'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	}
	df = pd.read_csv(
		filepath_or_buffer=csv_file,
		on_bad_lines='skip',
		dtype=dtypes,
		low_memory=False,
	)
	if verbose:
		print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")


	img_paths = df['img_path'].tolist()
	combined_labels = list()
	for i, pth in tqdm(enumerate(img_paths), total=len(img_paths), desc="Processing Images"):
		try:
			image = Image.open(pth).convert("RGB")
		except Exception as e:
			print(f"ERROR: failed to load image from {pth} => {e}")
			continue
		inputs = processor(
			text=texts,
			images=image,
			padding="max_length",
			max_num_patches=4096,
			max_length=64,
			return_tensors="pt",
		).to(device)
		with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			outputs = model(**inputs)
		torch.cuda.empty_cache()
		logits_per_image = outputs.logits_per_image
		probs = torch.sigmoid(logits_per_image)
		topk_probs, topk_indices = probs[0].topk(topk)
		topk_labels = [candidate_labels[idx] for idx in topk_indices]
		combined_labels.append(topk_labels)
	df['visual_based_labels'] = combined_labels
	df.to_csv(metadata_fpth, index=False)
	print(f"Visual-based annotation Elapsed time: {time.time() - visual_based_annotation_start_time:.2f} sec".center(160, " "))
	return combined_labels

def process_text_chunk(nlp, chunk):
	return [extract_named_entities(nlp=nlp, text=text) for text in chunk]

def parallel_relevance_filtering(texts, all_labels, n_processes=None):
	if n_processes is None:
		n_processes = max(1, multiprocessing.cpu_count() - 1)
	
	total_docs = len(texts)
	chunk_size = total_docs // n_processes + (1 if total_docs % n_processes else 0)
	chunks = []
	
	for i in range(0, total_docs, chunk_size):
			end_idx = min(i + chunk_size, total_docs)
			chunks.append((i, end_idx, texts[i:end_idx], all_labels[i:end_idx]))
	
	with multiprocessing.Pool(processes=n_processes) as pool:
			chunk_results = pool.map(process_document_chunk, chunks)
	
	all_results = []
	for chunk in chunk_results:
			all_results.extend(chunk)
	
	return all_results

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Multi-label annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--use_parallel", '-parallel', action="store_true")
	parser.add_argument("--num_workers", '-nw', type=int, default=10, help="Number of workers for parallel processing")
	parser.add_argument("--relevance_threshold", '-rth', type=float, default=0.25, help="Relevance threshold for textual-based annotation")
	parser.add_argument("--text_batch_size", '-tbs', type=int, default=64)
	parser.add_argument("--vision_batch_size", '-vbs', type=int, default=64, help="Batch size for vision processing")
	parser.add_argument("--sentence_model_name", '-smn', type=str, default="all-mpnet-base-v2", choices=["all-mpnet-base-v2", "all-MiniLM-L6-v2", "all-MiniLM-L12-v2"], help="Sentence-transformer model name")
	parser.add_argument("--vlm_model_name", '-vlm', type=str, default="google/siglip2-so400m-patch16-naflex", choices=["kakaobrain/align-base", "google/siglip2-so400m-patch16-naflex"], help="Vision-Language model name")
	parser.add_argument("--ner_model_name", '-ner', type=str, default="Babelscape/wikineural-multilingual-ner", choices=["dslim/bert-large-NER", "dslim/bert-base-NER", "Babelscape/wikineural-multilingual-ner"], help="NER model name")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)

	set_seeds(seed=42)
	DATASET_DIRECTORY = os.path.dirname(args.csv_file)
	text_output_path = os.path.join(DATASET_DIRECTORY, "metadata_textual_based_labels.csv")
	vision_output_path = os.path.join(DATASET_DIRECTORY, "metadata_visual_based_labels.csv")
	combined_output_path = os.path.join(DATASET_DIRECTORY, "metadata_multimodal_labels.csv")
	
	if os.path.exists(text_output_path):
		print(f"Found existing textual-based labels at {text_output_path}. Loading...")
		text_df = pd.read_csv(text_output_path)
		
		# Handle textual labels with proper null value handling
		textual_based_labels = []
		for label_str in text_df['textual_based_labels']:
			if pd.isna(label_str) or label_str == '[]' or not label_str:
				textual_based_labels.append([])
			else:
				try:
					labels = eval(label_str)
					textual_based_labels.append(labels if labels else [])
				except:
					textual_based_labels.append([])
		
		print(f"Loaded {len(textual_based_labels)} textual-based labels")
	else:
		textual_based_labels = get_textual_based_annotation(
			csv_file=args.csv_file,
			use_parallel=args.use_parallel,
			num_workers=args.num_workers,
			batch_size=args.text_batch_size,
			relevance_threshold=args.relevance_threshold,
			st_model_name=args.sentence_model_name,
			ner_model_name=args.ner_model_name,
			metadata_fpth=text_output_path,
			device=args.device,
			verbose=True,
		)

	if os.path.exists(vision_output_path):
		print(f"Found existing visual-based labels at {vision_output_path}. Loading...")
		vision_df = pd.read_csv(vision_output_path)
		
		# Handle visual labels with proper null value handling
		visual_based_labels = []
		for label_str in vision_df['visual_based_labels']:
			if pd.isna(label_str) or label_str == '[]' or not label_str:
				visual_based_labels.append([])
			else:
				try:
					labels = eval(label_str)
					visual_based_labels.append(labels if labels else [])
				except:
					visual_based_labels.append([])
		
		print(f"Loaded {len(visual_based_labels)} visual-based labels")
	else:
		visual_based_labels = get_visual_based_annotation(
			csv_file=args.csv_file,
			st_model_name=args.sentence_model_name,
			vlm_model_name=args.vlm_model_name,
			batch_size=args.vision_batch_size,
			device=args.device,
			num_workers=args.num_workers,
			metadata_fpth=vision_output_path,
			verbose=True,
		)
	
	print("Sample textual labels:", textual_based_labels[:15])
	print("Sample visual labels:", visual_based_labels[:15])
	assert len(textual_based_labels) == len(visual_based_labels), "Label lists must have same length"
	
	# Check if combined file already exists
	if os.path.exists(combined_output_path):
		print(f"Found existing combined labels at {combined_output_path}.")
		recompute = input("Do you want to recompute the combined labels? (y/n): ").lower() == 'y'
		if not recompute:
			print("Using existing combined labels.")
			combined_df = pd.read_csv(combined_output_path)
			# Handle combined labels with proper null value handling
			combined_labels = []
			for label_str in combined_df['multimodal_labels']:
				if pd.isna(label_str) or label_str == '[]' or not label_str:
					combined_labels.append([])
				else:
					try:
						labels = eval(label_str)
						combined_labels.append(labels if labels else [])
					except:
						combined_labels.append([])				
			print(f"Loaded {len(combined_labels)} combined labels")
			return combined_labels

	print("Merging text and vision annotations...")
	combined_labels = []
	empty_count = 0
	
	for text_labels, vision_labels in zip(textual_based_labels, visual_based_labels):
			# Check if both are empty - if so, store None for this entry
			if not text_labels and not vision_labels:
					combined_labels.append([])
					empty_count += 1
					continue
					
			# Ensure both are lists before combining
			if not isinstance(text_labels, list):
					text_labels = []
			if not isinstance(vision_labels, list):
					vision_labels = []
					
			# Combine all labels
			all_labels = list(set(text_labels + vision_labels))
			
			# Clean and filter
			all_labels = clean_labels(all_labels)
			all_labels = filter_metadata_terms(all_labels)
			all_labels = deduplicate_labels(all_labels)
			
			# Add semantic categories
			categorized = assign_semantic_categories(all_labels)
			final_labels = sorted(set(all_labels + categorized))
			
			combined_labels.append(final_labels)
	
	print(f"Created {len(combined_labels)} combined labels ({empty_count} empty entries)")
	
	# Save results
	df = pd.read_csv(args.csv_file)
	
	# Convert empty lists to None for better CSV representation
	df['textual_based_labels'] = [labels if labels else None for labels in textual_based_labels]
	df['visual_based_labels'] = [labels if labels else None for labels in visual_based_labels]
	df['multimodal_labels'] = [labels if labels else None for labels in combined_labels]
	
	# Save to CSV
	df.to_csv(combined_output_path, index=False)
	
	# Try to save as Excel
	try:
		excel_path = combined_output_path.replace('.csv', '.xlsx')
		print(f"Saving Excel file to {excel_path}...")
		df.to_excel(excel_path, index=False)
		print(f"Excel file saved successfully")
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	print("\nExample results:")
	sample_cols = ['title', 'description', 'label', 'user_query', 'img_url', 'enriched_document_description', 'textual_based_labels', 'visual_based_labels', 'multimodal_labels']
	available_cols = [col for col in sample_cols if col in df.columns]
	for i in range(min(25, len(df))):
		print(f"\nExample {i+1}:")
		for col in available_cols:
			value = df.iloc[i][col]
			if col in ['textual_based_labels', 'visual_based_labels', 'multimodal_labels']:
				print(f"{col}: {value if value else '[]'}")
			else:
				print(f"{col}: {value}")
	print(f"Combined labels saved to: {combined_output_path}")
	return combined_labels

if __name__ == "__main__":
	multiprocessing.set_start_method('spawn', force=True)
	main()