from utils import *

# how to run[Pouta]:
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata.csv -d "cuda:0" -nw 50 -tbs 1024 -vbs 512 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_EUROPEANA.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31/metadata.csv -d "cuda:1" -nw 50 -tbs 1024 -vbs 512 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_NA.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata.csv -d "cuda:2" -nw 50 -tbs 1024 -vbs 512 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_WWII.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/metadata.csv -d "cuda:3" -nw 50 -tbs 1024 -vbs 512 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_HISTORY_X4.out &

# Make language detection deterministic
DetectorFactory.seed = 42

# Custom stopwords and metadata patterns
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS.union(
	{
		"bildetekst", "photo", "image", "archive", "arkivreferanse", "caption", "following", "below", "above",
		"copyright", "description", "riksarkivet", "ntbs", "ra", "pa", "bildetekst",
		"showing", "shown", "shows", "depicts", "depicting", "pictured", "picture", "pinterest",
		"copy", "version", "view", "looking", "seen", "visible", "illustration",
		"photograph", "photography", "photo", "image", "img", "photographer",
		"sent", "received", "taken", "made", "created", "produced", "found",
		"across", "opposite", "near", "under", "over", "inside", "outside",
		"collection", "collections", "number", "abbreviation", "abbreviations",
		"folder", "box", "file", "document", "page", "index", "label", "code", "icon", "type", "unknown", "unknow",
		"folder icon", "box icon", "file icon", "document icon", "page icon", "index icon", "label icon", "code icon",
		"used", "southern", "built",
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

FastText_Language_Identification = "lid.176.ftz"
if "lid.176.ftz" not in os.listdir():
	url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
	urllib.request.urlretrieve(url, "lid.176.ftz")

# Semantic categories for organization
SEMANTIC_CATEGORIES = {
	'military': ['military', 'army', 'navy', 'air force', 'soldier', 'officer', 'troop', 'regiment', 'division', 'corps', 'battalion'],
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

def get_hdbscan_parameters(embeddings, use_static=False):
	print(f"get hdbscan parameters for embeddings {embeddings.shape}...")
	if use_static:
		return 100, 10
	num_samples, num_embeddings = embeddings.shape

	# Method 1: Silhouette-based
	silhouette_size = find_optimal_min_cluster_size(embeddings, num_samples)
	
	# Method 2: Knee point detection
	knee_size = find_knee_point(embeddings)
	
	# Method 3: Density-based
	density_size, density_samples = density_based_parameters(embeddings)
	
	# Combine results (median of suggestions)
	suggestions = [
		silhouette_size,
		knee_size,
		density_size,
		int(np.sqrt(num_samples) // 2),
		int(np.sqrt(num_samples) / 2), # ~102 for 42,146
		int(np.log(num_samples)**2),
		int(np.log(num_samples)**3),
		50,
		70,
		80,  
		100,
		120,
	]
	print(f"Suggestions: {suggestions}")
	# Remove outliers (values outside 25%-75% percentile)
	q25, q75 = np.percentile(suggestions, [25, 75])
	iqr = q75 - q25
	print(f"q25: {q25}, q75: {q75}, iqr: {iqr}")
	filtered = [x for x in suggestions if (x >= q25 - 2*iqr) and (x <= q75 + 2*iqr)]
	print(f"Filtered: {filtered} median: {np.median(filtered)} min: {min(filtered)} max: {max(filtered)} mean: {np.mean(filtered)}")
	
	min_cluster_size = max(50, int(np.median(filtered)))
	min_samples = max(5, min(10, density_samples)) # Cap min_samples lower
	
	return min_cluster_size, min_samples

def is_english(
		text: str, 
		ft_model: fasttext.FastText._FastText,
		verbose: bool=False,
		min_length: int=5,
	) -> bool:
	if verbose:
		print(f"text({len(text)}): {text}")
	if not text or len(text) < min_length:
		if verbose:
			print(f"text({len(text)}) is too short")
		return False
	text = text.replace("\n", " ").replace("\r", " ").strip()  # 🛠️ sanitize input
	if len(text) < 20:
		# Short texts: rely on ASCII + stopword heuristics
		ascii_chars = sum(c.isalpha() and ord(c) < 128 for c in text)
		total_chars = sum(c.isalpha() for c in text)
		if total_chars == 0 or ascii_chars / total_chars < 0.9:
			if verbose:
				print(f"text({len(text)}) is not English")
			return False
		common_words = {'the', 'and', 'of', 'to', 'in', 'is', 'was', 'for', 'with', 'on'}
		words = text.lower().split()
		return any(word in common_words for word in words)
	# Long texts: fasttext is preferred
	try:
		prediction = ft_model.predict(text)[0][0]
		if verbose:
			print(f"Fasttext prediction: {prediction}")
		return prediction == '__label__en'
	except ValueError as e:
		if verbose:
			print(f"FastText error: {e}")
		return False

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
	emb_fpth = os.path.join(dataset_dir, f'text_embeddings_{dataset_size}_samples.gz')
	t0 = time.time()
	try:
		embeddings = load_pickle(fpath=emb_fpth)
	except Exception as e:
		print(e)
		print(f"Generating Text embeddings for {len(texts)} texts [might take a while]...")
		embeddings = sent_model.encode(texts, show_progress_bar=False)
		save_pickle(pkl=embeddings, fname=emb_fpth)

	print(f"Raw Embeddings: {embeddings.shape} generated in {time.time() - t0:.2f} sec")

	# # Visualization 0: Raw Embeddings into 2D for better debugging
	# print(f"Reducing embeddings: {embeddings.shape} to 2D for visualization using UMAP")
	# umap_reducer = umap.UMAP(
	# 	n_neighbors=15,
	# 	min_dist=0.1,
	# 	densmap=True,
	# 	spread=1.0,
	# 	n_components=2, 
	# 	random_state=42, 
	# 	metric='cosine',
	# )
	# emb_umap = umap_reducer.fit_transform(embeddings)
	# if enable_visualizations:
	# 	# Visualize embeddings with UMAP
	# 	print(f"Reducing embeddings: {embeddings.shape} to 2D for visualization using UMAP")
	# 	plt.figure(figsize=(18, 10))
	# 	plt.scatter(emb_umap[:, 0], emb_umap[:, 1], s=25, c='#f1f8ff', edgecolors='#0078e9', alpha=0.8)
	# 	plt.title(f"UMAP Visualization of Embeddings ({dataset_size} Texts)")
	# 	plt.xlabel("UMAP Dimension 1")
	# 	plt.ylabel("UMAP Dimension 2")
	# 	plt.savefig(os.path.join(dataset_dir, f'umap_raw_embeddings_{embeddings.shape[0]}_x_{embeddings.shape[1]}.png'), bbox_inches='tight')
	# 	print(f"UMAP visualization saved to {os.path.join(dataset_dir, f'umap_raw_embeddings_{embeddings.shape[0]}_x_{embeddings.shape[1]}.png')}")

	t0 = time.time()
	if dataset_size < 500:
		print("Dataset is small, using KMeans for clustering...")
		kmeans = KMeans(n_clusters=min(10, max(2, int(np.sqrt(dataset_size)))), random_state=42)
		labels = kmeans.fit_predict(embeddings)
	else:
		print(f"Clustering embeddings {embeddings.shape} into topics with HDBSCAN...")
		min_cluster_size, min_samples = get_hdbscan_parameters(
			embeddings=embeddings,
			use_static=False,
		)
		cluster_selection_method = 'eom' if dataset_size < 500 else 'leaf'
		print(f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, cluster_selection_method: {cluster_selection_method}")
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

	# Visualization 1: Cluster Distribution Bar Plot (Before Merging)
	if enable_visualizations:
		topic_counts = Counter(labels)
		plt.figure(figsize=(16, 7))
		bars = plt.bar(range(len(topic_counts)), [topic_counts[i] for i in sorted(topic_counts.keys())], color='#3785e6', label='Before Merging')
		plt.title('Document Distribution Across Clusters (Before Merging)')
		plt.xlabel('Cluster ID')
		plt.ylabel('Number of Documents')
		plt.xticks(range(len(topic_counts)), labels=sorted(topic_counts.keys()))
		for bar in bars:
			yval = bar.get_height()
			plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')
		plt.legend()
		plt.savefig(os.path.join(dataset_dir, f'topic_distribution_before_merging_{n_clusters}_clusters.png'), bbox_inches='tight')
		plt.close()

	# # Visualization 2: Interactive UMAP Scatter Plot with Plotly
	# if enable_visualizations:
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
	for i, (text, label) in enumerate(zip(texts, labels)):
		if label == -1:  # Skip noise points
			continue
		if is_english(text=text, ft_model=ft_model):
			# Extract keyphrases with KeyBERT
			phrases = kw_model.extract_keywords(
				text,
				keyphrase_ngram_range=(1, 3),  # 1-3 word phrases
				stop_words="english",
				top_n=5,  # Top 5 phrases
				# diversity=0.7,  # Reduce redundancy
				# use_maxsum=True
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
					# Length filter: ensure phrase has 2+ words
					if len(words) < 2:
							phrase_filter_log['length_filtered'] += 1
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
					if len(normalized.split()) >= 2 and normalized not in seen_phrases:
							normalized_phrases.append(normalized)
							seen_phrases.add(normalized)
					else:
							phrase_filter_log['length_filtered'] += 1
			
			if not normalized_phrases:
					print(f"Text {i} has no valid phrases: {text[:500]}")
			else:
					cluster_phrases[label].update(normalized_phrases)
			cluster_text_counts[label] += 1
		else:
			print(f"Text {i} is not English: {text[:500]}")
	print(f"Phrase collection done in {time.time() - t0:.2f} sec")
	
	# Visualization 4: Phrase Retention Histogram
	if enable_visualizations:
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
		# print(f"Topic {label}: {cluster_text_counts[label]} texts, {len(counter)} phrases, Sample: {list(counter.items())[:5]}")
		if not counter:
			print(f"Warning: Topic {label} has no phrases.")
		# Score phrases with diversity bonus
		phrase_scores = []
		top_k_words = max(10, len(counter) // 15)
		print(f"Topic {label}: {len(counter)} phrases, Selecting Top-{top_k_words}")
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

	# Visualization 7: Phrase Co-Occurrence Network for Each Topic [before merging]
	if enable_visualizations:
		print(f"Generating co-occurrence networks for {len(topics_before_merging)} topics [before merging]...")
		for label, topic_phrases in enumerate(topics_before_merging):
			if not topic_phrases:
					print(f"Skipping Topic {label}: No phrases available.")
					continue
			counter = cluster_phrases[label]
			cluster_texts = [texts[i] for i, l in enumerate(labels) if l == label and is_english(texts[i], ft_model)]
			if not cluster_texts:
					print(f"Skipping Topic {label}: No valid texts for co-occurrence.")
					continue
			phrase_set = set(topic_phrases)
			cooc_matrix = defaultdict(int)
			for text in cluster_texts:
					# Extract keyphrases with KeyBERT
					phrases = kw_model.extract_keywords(
							text,
							keyphrase_ngram_range=(1, 3),  # 1-3 word phrases
							stop_words="english",
							top_n=5,  # Top 5 phrases
							# diversity=0.7,  # Reduce redundancy
							# use_maxsum=True # makes it slow
					)
					# Filter and normalize phrases
					valid_phrases = []
					seen_phrases = set()
					for phrase, _ in phrases:
							words = phrase.split()
							stopword_count = sum(1 for w in words if w in CUSTOM_STOPWORDS)
							if stopword_count / len(words) > 0.7 or len(words) < 2:
									continue
							normalized = " ".join(word for i, word in enumerate(words) if i == 0 or word != words[i-1])
							if len(normalized.split()) >= 2 and normalized not in seen_phrases:
									valid_phrases.append(normalized)
									seen_phrases.add(normalized)
					# Compute co-occurrences within topic phrases
					text_phrases = set(valid_phrases).intersection(phrase_set)
					for p1 in text_phrases:
							for p2 in text_phrases:
									if p1 < p2:
											cooc_matrix[(p1, p2)] += 1
			G = nx.Graph()
			for (p1, p2), count in cooc_matrix.items():
					if count > 0:
							G.add_edge(p1, p2, weight=count)
			for phrase in topic_phrases:
					if phrase not in G:
							G.add_node(phrase)
			plt.figure(figsize=(18, 12))
			pos = nx.spring_layout(G)
			edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
			nx.draw_networkx_edges(G, pos, width=[w * 1.1 for w in edge_weights], alpha=0.7, edge_color='#000a16')
			nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#54a1ff', alpha=0.8)
			nx.draw_networkx_labels(G, pos, font_size=10, font_family='monospace')
			plt.title(f'Phrase Co-Occurrence Network for Topic {label} [before merging] with {len(G.nodes())} nodes and {len(G.edges())} edges')
			plt.savefig(os.path.join(dataset_dir, f'cooccurrence_network_topic_{label}_before_merging.png'), bbox_inches='tight')
			plt.close()

	# Visualization 12: Top-K Phrases Bar Plot for Each Topic [before merging]
	if enable_visualizations:
		for label, topic_phrases in enumerate(topics_before_merging):
			if not topic_phrases:
				print(f"Skipping Topic {label}: No phrases available.")
				continue
			counter = cluster_phrases[label]
			print(f"Topic {label}: {len(counter)} phrases, Selecting Top-{len(topic_phrases)}")
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

	# Visualization 5: Topic Similarity Heatmap (before merging)
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
	
	# Visualization 6: Dendrogram of Topic Similarities (before merging)
	if enable_visualizations:
		sim_values = similarity_matrix[np.triu_indices(len(topics_before_merging), k=1)]
		if sim_values.size > 0:
			mean_sim = np.mean(sim_values)
			min_sim = np.min(sim_values)
			max_sim = np.max(sim_values)
			print(f"Similarity matrix stats: Mean={mean_sim:.3f}, Min={min_sim:.3f}, Max={max_sim:.3f}")
			merge_threshold = np.percentile(sim_values, 75) + 0.10  # Add 0.10 to preserve diversity (dynamic threshold)
			print(f"Dynamic merge threshold (75th percentile): {merge_threshold}")
			plt.figure(figsize=(17, 10))
			linkage_matrix = linkage(sim_values, method='average')
			dendrogram(linkage_matrix, labels=[f'Topic {i}' for i in range(len(topics_before_merging))])
			plt.title(f'Dendrogram of Topic Similarities [before merging]')
			plt.xlabel('Topics')
			plt.ylabel('Distance (1 - Cosine Similarity)')
			plt.axhline(y=1-merge_threshold, color='red', linestyle='--', label=f'Merge Threshold ({merge_threshold:.4f})')
			plt.legend()
			plt.xticks(rotation=90, fontsize=8)
			plt.savefig(os.path.join(dataset_dir, f'similarity_dendrogram_before_merging_thresh_{merge_threshold:.4f}.png'), bbox_inches='tight')
			plt.close()
		else:
			print("Similarity matrix is empty.")
		
	# # Visualization 10: UMAP with Top Phrases
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

	# Visualization 11: Cluster Size vs. Term Count Plot
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

	# Visualization 13: Interactive Topic Similarity Network
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
	print(f"\n>> Merging similar topics with dynamic merging threshold {merge_threshold}")
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
	print(f"Topics Reduced from {len(topics_before_merging)} to {len(merged_topics)} topics after merging")

	# Visualization 14: Topic Diversity Plot (after merging)
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
		plt.title('Merged Topic Diversity (Jaccard Similarity Between Topics)')
		plt.xlabel('Topic ID')
		plt.ylabel('Topic ID')
		plt.savefig(os.path.join(dataset_dir, 'merged_topic_diversity_jaccard_similarity.png'), bbox_inches='tight')
		plt.close()

	# Visualization 15: Topic Distribution Comparison: Before vs. After Merging
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
	
	# Visualization 16: Noise Analysis Plot
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
	
	flat_topics = set(word for topic in merged_topics for word in topic)
	print(f"Extracted {len(flat_topics)} unique topic terms after merging")
	return merged_topics, flat_topics

def clean_labels(labels):
	cleaned = set()
	for label in labels:
		# Normalize the label
		label = label.lower().strip()
		# Remove non-alphanumeric characters except spaces and hyphens
		label = re.sub(r"[^a-z0-9\s\-]", "", label)
		# Skip short labels and stopwords
		if label in CUSTOM_STOPWORDS or len(label) < 3:
				continue
		# Skip labels that are just numbers
		if label.isdigit():
				continue
		# Skip labels that start with numbers unless they're years (4 digits)
		if label[0].isdigit() and not (len(label) == 4 and label.isdigit()):
				continue
		# Skip non-English labels
		if not all(ord(char) < 128 for char in label):
				continue
		# Add the cleaned label
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
				if (entity_text and len(entity_text) > 2 and 
					entity_text not in CUSTOM_STOPWORDS and 
					all(ord(char) < 128 for char in entity_text)):
					entities.append(entity_text)
		# Also extract multi-word phrases that might be significant
		tokens = [
			word.lower() for word in text.split() 
			if word.isalpha() and len(word) > 2 
			and word.lower() not in CUSTOM_STOPWORDS
			and all(ord(char) < 128 for char in word)
		]

		# Return unique list of entities and tokens
		return list(set(entities + tokens))
	except Exception as e:
		print(f"NER error: {e}")
		print(f"Text: {text}")
		return []

def extract_keywords(
		text: str, 
		ft_model: fasttext.FastText._FastText, 
		min_count: int=3, 
		max_df: float=0.9, 
		min_df: float=0.01, 
		max_features: int=1000, 
		ngram_range: tuple=(1, 3), 
		top_k: int=50
	):
	if not is_english(text, ft_model) or len(text) < 5:
		return []
			
	vectorizer = TfidfVectorizer(
		# max_df=max_df,
		# min_df=min_df,
		# max_features=max_features,
		ngram_range=ngram_range,
		stop_words=CUSTOM_STOPWORDS,
	)
	try:
		X = vectorizer.fit_transform([text])
		feature_names = vectorizer.get_feature_names_out()
		# Get scores for the first document
		scores = zip(feature_names, X.toarray()[0])
		# Sort by score in descending order
		sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
		# Return top keywords
		keywords = [
			kw for kw, score in sorted_scores[:top_k] 
			if score > 0.01 and all(ord(char) < 128 for char in kw)
		]
		return keywords
	except:
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

def deduplicate_labels(labels):
	"""Remove semantically redundant labels"""
	if not labels:
		return []

	# Sort labels by length (typically longer labels are more specific)
	sorted_labels = sorted(labels, key=len, reverse=True)
	deduplicated = []
	for label in sorted_labels:
		# Check if this label is a substring of any already-kept label
		is_redundant = False
		for kept_label in deduplicated:
			# If label is completely contained in a kept label, it's redundant
			if label in kept_label:
				is_redundant = True
				break

			# Check for high token overlap in multi-word labels
			if ' ' in label and ' ' in kept_label:
				label_tokens = set(label.split())
				kept_tokens = set(kept_label.split())

				# If 75% or more tokens overlap, consider redundant
				overlap_ratio = len(label_tokens & kept_tokens) / len(label_tokens)
				if overlap_ratio > 0.75:
					is_redundant = True
					break
		
		if not is_redundant:
			deduplicated.append(label)
	
	return deduplicated

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
		min_labels=3, 
		max_labels=10,
	):
	balanced_labels = []
	for labels in image_labels_list:
		# Case 1: Too few labels - attempt to generate more
		if len(labels) < min_labels:
			# Create compound labels by combining existing ones
			compound_labels = []
			if len(labels) >= 2:
				for i in range(len(labels)):
					for j in range(i+1, len(labels)):
						compound = f"{labels[i]} {labels[j]}"
						if 3 <= len(compound.split()) <= 4:  # Reasonable phrase length
							compound_labels.append(compound)

			# Add compound labels to reach minimum
			expanded_labels = labels + compound_labels
			balanced_labels.append(expanded_labels[:min_labels] if len(expanded_labels) >= min_labels else expanded_labels)
						
		# Case 2: Too many labels - prioritize the most relevant ones
		elif len(labels) > max_labels:
			# Sort labels by potential relevance (length can be a simple heuristic)
			sorted_labels = sorted(labels, key=lambda x: len(x.split()), reverse=True)
			
			# Ensure we keep some single-word labels for searchability
			single_word = [label for label in labels if ' ' not in label][:max_labels//3]
			multi_word = [label for label in sorted_labels if ' ' in label][:max_labels - len(single_word)]
			
			balanced_labels.append(single_word + multi_word)
		
		# Case 3: Good label count - keep as is
		else:
			balanced_labels.append(labels)
	
	return balanced_labels

def quick_filter_candidates(text, labels, max_keep=30):
	"""Quick pre-filtering using simple word overlap"""
	if not labels:
		return []
			
	text_words = set(text.lower().split())
	scores = []
	
	for label in labels:
		label_words = set(label.lower().split())
		# Calculate simple overlap score
		overlap = len(text_words.intersection(label_words))
		scores.append((label, overlap))
	
	# Sort by score and keep top candidates
	sorted_labels = [l for l, s in sorted(scores, key=lambda x: x[1], reverse=True)]
	return sorted_labels[:max_keep]

def batch_filter_by_relevance(
		sent_model: SentenceTransformer,
		texts: list,
		all_labels_list: list[list[str]],
		threshold: float=0.3,
		batch_size: int=128,
		print_every=10,
	):
	"""Process document relevance filtering in efficient batches"""
	results = []
	total = len(texts)
	
	# Process in batches to avoid memory issues
	for batch_start in range(0, total, batch_size):
		batch_end = min(batch_start + batch_size, total)
		batch_texts = texts[batch_start:batch_end]
		batch_labels_list = all_labels_list[batch_start:batch_end]

		
		# First apply quick filtering to reduce candidates
		quick_filtered_batch = []
		for text, labels in zip(batch_texts, batch_labels_list):
			quick_filtered = quick_filter_candidates(text, labels)
			quick_filtered_batch.append(quick_filtered)
		
		# Get text embeddings for the batch (more efficient)
		text_embeddings = sent_model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True)
		
		batch_results = []
		for i, (text_emb, labels) in enumerate(zip(text_embeddings, quick_filtered_batch)):
			if not labels:
				batch_results.append([])
				continue
					
			# Encode all labels at once for this document
			label_embeddings = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
			
			# Calculate similarities all at once (vectorized operation)
			similarities = np.dot(label_embeddings, text_emb) / (np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(text_emb) + 1e-8)
			
			# Filter by threshold
			relevant_indices = np.where(similarities > threshold)[0]
			batch_results.append([labels[idx] for idx in relevant_indices])
			
			# # Progress indicator within batch
			# if (i + 1) % print_every == 0 or i + 1 == len(batch_texts):
			# 	print(f"  Processed {i+1}/{len(batch_texts)} documents in current batch")

		results.extend(batch_results)

		if batch_start % print_every == 0 or batch_start == 0:
			print(f"batch {batch_start//batch_size + 1}/{(total-1)//batch_size + 1} with {sum(len(labels) for labels in batch_results)} relevant labels found")
	
	return results

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

def parallel_relevance_filtering(texts, all_labels, n_processes=None):
		"""Run relevance filtering in parallel using multiple processes"""
		if n_processes is None:
			n_processes = max(1, multiprocessing.cpu_count() - 1)
		
		total_docs = len(texts)
		chunk_size = total_docs // n_processes + (1 if total_docs % n_processes else 0)
		chunks = []
		
		# Create chunks of data
		for i in range(0, total_docs, chunk_size):
			end_idx = min(i + chunk_size, total_docs)
			chunks.append((i, end_idx, texts[i:end_idx], all_labels[i:end_idx]))
		
		# Process chunks in parallel
		with multiprocessing.Pool(processes=n_processes) as pool:
			chunk_results = pool.map(process_document_chunk, chunks)
		
		# Combine results
		all_results = []
		for chunk in chunk_results:
			all_results.extend(chunk)
		
		return all_results

def process_text_chunk(nlp, chunk):
	return [extract_named_entities(nlp=nlp, text=text) for text in chunk]

def get_textual_based_annotation(
		csv_file: str, 
		num_workers: int,
		batch_size: int,
		relevance_threshold: float,
		metadata_fpth: str,
		device: str,
		verbose: bool=True,
		use_parallel: bool=False,
	):
	if verbose:
		print(f"Automatic label extraction from text data".center(160, "-"))
		print(f"Loading metadata from {csv_file}...")
	text_based_annotation_start_time = time.time()

	dataset_dir = os.path.dirname(csv_file)

	sent_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
	ft_model = fasttext.load_model("lid.176.ftz")
	tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
	model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
	nlp = pipeline(
		task="ner", 
		model=model, 
		tokenizer=tokenizer, 
		aggregation_strategy="simple",
		device=device,
		batch_size=batch_size,
	)

	# Load the full dataset
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

	print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")
	
	df['content'] = df['enriched_document_description'].fillna('').astype(str)
	num_samples = len(df)
	print(f"Total number of samples: {num_samples}")
	
	print("Filtering non-English entries...")
	t0 = time.time()
	english_mask = df['content'].apply(lambda x: is_english(text=x, ft_model=ft_model, verbose=False))
	english_indices = english_mask[english_mask].index.tolist()
	print(f"{sum(english_mask)} / {len(df)} texts are English")
	print(f"Language filter done in {time.time() - t0:.2f} sec")
	
	# Create a filtered dataframe with only English entries
	english_df = df[english_mask].reset_index(drop=True)
	english_texts = english_df['content'].tolist()

	# Initialize results for all entries with empty lists
	per_image_labels = [[] for _ in range(num_samples)]
	if len(english_texts) > 0:
		# Step 1: Topic Modeling with redundancy reduction
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
		
		# Step 2: Named Entity Recognition per image
		print("Extracting NER per sample...")
		t0 = time.time()
		if len(english_texts) > 1000 and use_parallel:
			chunk_size = len(english_texts) // num_workers + 1
			chunks = [
				(english_texts[i:i+chunk_size]) 
				for i in range(0, len(english_texts), chunk_size)
			]
			print(f"Using {num_workers} processes for NER extraction...")
			with multiprocessing.Pool(processes=num_workers) as pool:
				ner_results = pool.map(process_text_chunk, chunks)
					
			per_image_ner_labels = []
			for chunk_result in ner_results:
				per_image_ner_labels.extend(chunk_result)
		else:
			per_image_ner_labels = []
			for i, text in enumerate(tqdm(english_texts, desc="NER Progress")):
				entities = extract_named_entities(nlp=nlp, text=text, ft_model=ft_model)
				per_image_ner_labels.append(entities)
		print(f"NER done in {time.time() - t0:.1f} sec")
		
		# # Step 3: Extract keywords per image
		# print("Extracting keywords per image using TF-IDF...")
		# t0 = time.time()
		# # per_image_keywords = [extract_keywords(text, ft_model) for text in english_texts]
		# per_image_keywords = [
		# 	kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10) 
		# 	for text in english_texts
		# ]
		# print(f"Keyword extraction done in {time.time() - t0:.1f} sec")		
		print("Extracting keywords per image using KeyBERT...")
		t0 = time.time()
		per_image_keywords = []
		kw_model = KeyBERT(model=sent_model)
		for text in tqdm(english_texts, desc="Keyword Extraction"):
			if not is_english(text, ft_model):  # Skip non-English texts
				per_image_keywords.append([])
				continue
			try:
				# Extract keyphrases with KeyBERT
				phrases = kw_model.extract_keywords(
					text,
					keyphrase_ngram_range=(1, 3),  # 1-3 word phrases
					stop_words=CUSTOM_STOPWORDS,  # Use custom stopwords
					top_n=10,  # Top 10 phrases
					# diversity=0.7,  # Reduce redundancy
					# use_maxsum=True
				)
				# Filter and normalize phrases
				keywords = []
				seen_phrases = set()
				for phrase, _ in phrases:
					words = phrase.split()
					stopword_count = sum(1 for w in words if w in CUSTOM_STOPWORDS)
					if stopword_count / len(words) > 0.7 or len(words) < 1:  # Allow single words
						continue
					normalized = " ".join(word for i, word in enumerate(words) if i == 0 or word != words[i-1])
					if len(normalized.split()) >= 1 and normalized not in seen_phrases:
						keywords.append(normalized)
						seen_phrases.add(normalized)
				per_image_keywords.append(keywords)
			except Exception as e:
				print(f"KeyBERT error for text: {text[:100]}...: {e}")
				per_image_keywords.append([])
		print(f"Keyword extraction done in {time.time() - t0:.1f} sec")


		# Step 4: Add individual topic labels
		print("Assigning topic labels per image...")
		t0 = time.time()
		per_image_topic_labels = []
		for text in english_texts:
			matching_topics = [word for word in flat_topic_words if word in text]
			per_image_topic_labels.append(matching_topics)
		print(f"Topic assignment done in {time.time() - t0:.1f} sec")
		
		# Step 5: Combine all label sources and clean
		print("Combining and cleaning labels...")
		t0 = time.time()
		per_image_combined_labels = []
		for ner, keywords, topics in zip(per_image_ner_labels, per_image_keywords, per_image_topic_labels):
			# Combine all sources
			all_labels = list(set(ner + keywords + topics))
			# Clean the labels
			cleaned_labels = clean_labels(all_labels)
			# Filter metadata terms
			cleaned_labels = filter_metadata_terms(cleaned_labels)
			per_image_combined_labels.append(cleaned_labels)
		print(f"Label combination and cleaning done in {time.time() - t0:.3f} sec")
		# Step 6: Filter by relevance and handle languages (optimized)
		print("Filtering labels by relevance...")
		t0 = time.time()
		if use_parallel:
			print("Using parallel processing for relevance filtering...")
			per_image_relevant_labels = parallel_relevance_filtering(
				texts=english_texts,
				all_labels=per_image_combined_labels,
				n_processes=num_workers,
			)
		else:
			print(f"Using batch processing for relevance filtering (thresh: {relevance_threshold})...")
			per_image_relevant_labels = batch_filter_by_relevance(
				sent_model=sent_model,
				texts=english_texts,
				all_labels_list=per_image_combined_labels,
				threshold=relevance_threshold,
				batch_size=batch_size,
				print_every=500,
			)
		print(f"Relevance filtering done in {time.time() - t0:.1f} sec")
		# Post-process: language handling, deduplication, etc.
		print("Post-processing labels, deduplication, and semantic categorization...")
		t0 = time.time()
		english_labels = []
		for i, relevant_labels in enumerate(tqdm(per_image_relevant_labels, desc="Post-processing", unit="image")):
			filtered_labels = handle_multilingual_labels(relevant_labels)
			
			if "user_query" in english_df.columns:
				original_label = english_df.iloc[i]["user_query"]
				if isinstance(original_label, str) and original_label.strip():
					original_label_clean = re.sub(r"[^a-z0-9\s\-]", "", original_label.lower().strip())
					if all(ord(char) < 128 for char in original_label_clean):
						filtered_labels.append(original_label_clean)
			
			filtered_labels = deduplicate_labels(filtered_labels)
			# Add semantic categories
			categorized = assign_semantic_categories(filtered_labels)
			final_labels = sorted(set(filtered_labels + categorized))
			english_labels.append(final_labels)
		print(f"Post-processing done in {time.time() - t0:.1f} sec")
		# Balance label counts
		print("Balancing label counts...")
		t0 = time.time()
		english_labels = balance_label_count(english_labels, min_labels=3, max_labels=12)
		print(f"Label balancing done in {time.time() - t0:.3f} sec")
		
		# Transfer results to the full-sized results array using original indices
		for i, orig_idx in enumerate(english_indices):
			if i < len(english_labels):
				per_image_labels[orig_idx] = english_labels[i]
	else:
		print("No English texts found. Returning empty labels for all entries.")
	
	# Save the results in a separate column
	df['textual_based_labels'] = per_image_labels
	
	df.to_csv(metadata_fpth, index=False)
	
	print(f">> Generated text labels for {sum(1 for labels in per_image_labels if labels)} out of {num_samples} entries")
	print(f"Text-based annotation Elapsed time: {time.time() - text_based_annotation_start_time:.2f} sec".center(160, " "))
	
	return per_image_labels

def get_visual_based_annotation(
		csv_file: str,
		confidence_threshold: float,
		batch_size: int,
		device: str,
		verbose: bool,
		metadata_fpth: str,
	) -> List[List[str]]:
	print(f"Automatic label extraction from image data".center(160, "-"))
	start_time = time.time()
	
	# Load dataset
	if verbose:
		print(f"Loading metadata from {csv_file}...")
	df = pd.read_csv(csv_file, dtype={'img_path': str}, low_memory=False)
	image_paths = df['img_path'].tolist()
	if verbose:
		print(f"Found {len(image_paths)} samples to process...")
	
	if verbose:
		print("Loading VLM model for image labeling...")
	# model_name = "openai/clip-vit-large-patch14"
	# model = CLIPModel.from_pretrained(model_name).to(device)
	# processor = CLIPProcessor.from_pretrained(model_name)
	processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
	model = AlignModel.from_pretrained("kakaobrain/align-base")
	model.to(device) # Add this line to move the model to the GPU
	model.eval()

	object_categories = [
		# Tanks & Armored Vehicles (WWI-WWII)
		"tank", "light tank", "medium tank", "heavy tank", "super-heavy tank", 
		"tank destroyer", "self-propelled gun", "armored car", "half-track", 
		"armored personnel carrier", "armored train", "reconnaissance vehicle",
		"Mark IV tank", "Tiger tank", "Panther tank", "T-34 tank", "Sherman tank",
		"Churchill tank", "KV-1 tank", "Panzer IV", "Panzer III", "Stuart tank",
		"Sonderkraftfahrzeug", "Kettenkrad", "M4 Sherman", "T-34/85", "IS-2 tank",
		
		# Light & Utility Vehicles
		"jeep", "staff car", "command car", "ambulance", "motorcycle", 
		"military truck", "supply truck", "fuel truck", "artillery tractor", 
		"amphibious vehicle", "scout car", "Willys Jeep", "Kubelwagen", 
		"Dodge WC series", "Opel Blitz", "Zis truck", "weapons carrier",
			
		# Aircraft
		"military aircraft", "fighter aircraft", "bomber aircraft", "reconnaissance aircraft", 
		"dive bomber", "torpedo bomber", "transport aircraft", "seaplane", "flying boat",
		"biplane", "monoplane", "fighter-bomber", "ground attack aircraft", "night fighter",
		"Spitfire", "Messerschmitt Bf 109", "P-51 Mustang", "Focke-Wulf Fw 190", 
		"B-17 Flying Fortress", "Lancaster bomber", "Heinkel He 111", "Junkers Ju 87 Stuka",
		"Mitsubishi Zero", "Il-2 Sturmovik", "P-47 Thunderbolt", "Hurricane fighter", "helicopter",
		"aircraft engine", "aircraft propeller", "aircraft wing", "aircraft fuselage", "aircraft tail",
		"aircraft manufacturing company",
	
		# Naval Vessels
		"submarine", "U-boat", "destroyer", "cruiser", "battleship", "aircraft carrier", 
		"battlecruiser", "corvette", "frigate", "minesweeper", "torpedo boat", 
		"landing craft", "PT boat", "pocket battleship", "gunboat", "escort carrier",
		"liberty ship", "merchant vessel", "hospital ship", "troop transport",
			
		# Military Personnel
		"soldier", "infantryman", "officer", "NCO", "general", "field marshal",
		"pilot", "bomber crew", "aircraft crew", "tanker", "artilleryman", "sailor", "marine", 
		"paratrooper", "commando", "sniper", "medic", "military police", 
		"cavalry", "SS officer", "Wehrmacht soldier", "Red Army soldier", 
		"Desert Rat", "Afrika Korps soldier", "Luftwaffe personnel", "naval officer",
		
		# Weapons & Ordnance
		"rifle", "machine gun", "submachine gun", "pistol", "bayonet", "flamethrower", 
		"mortar", "artillery piece", "howitzer", "field gun", "anti-tank gun", "cannon", 
		"anti-aircraft gun", "rocket launcher", "grenade", "hand grenade", "rifle grenade",
		"landmine", "naval mine", "depth charge", "torpedo", "aerial bomb", "incendiary bomb",
		"Thompson submachine gun", "MG-42", "Karabiner 98k", "M1 Garand", "Sten gun",
		"Luger pistol", "PIAT", "Bazooka", "Panzerfaust", "88mm gun",
		
		# Military Infrastructure
		"bunker", "pillbox", "gun emplacement", "observation post", "barbed wire", 
		"trenches", "foxhole", "dugout", "fortification", "coastal defense", 
		"anti-tank obstacle", "dragon's teeth", "minefield", "floating bridge",
		"portable bridge", "military headquarters", "command post", "communications center",
		
		# Military Insignia & Symbols
		"military flag", "swastika flag", "rising sun flag", "Soviet flag", "Union Jack", 
		"American flag", "regimental colors", "military insignia", "rank insignia", 
		"unit patch", "medal", "military decoration", "Iron Cross", "Victoria Cross",
		"Medal of Honor", "military helmet", "steel helmet", "Brodie helmet",
		"Stahlhelm", "Adrian helmet", "gas mask",
		
		# Military Equipment
		"military uniform", "combat uniform", "field equipment", "backpack", "mess kit", 
		"entrenching tool", "canteen", "ammunition belt", "bandolier", "map case", 
		"binoculars", "field telephone", "radio equipment", "signal equipment",
		"parachute", "life vest", "fuel drum", "jerry can", "ration box",
		"military stretcher", "field kitchen", "anti-gas equipment"
	]

	scene_categories = [
		# European Theaters
		"Western Front", "Eastern Front", "Italian Front", "North African Front",
		"Normandy beaches", "coastline", "Soviet urban ruins",
		
		# Pacific & Asian Theaters
		"Pacific island", "jungle battlefield", "Pacific beach landing", "atoll",
		"tropical forest", "coral reef", "bamboo grove", "rice paddy",
		"Burmese jungle", "Chinese village", "Philippine beach", "volcanic island",
		"Japanese homeland", "Pacific airfield", "jungle airstrip", "coconut plantation",
		
		# Military Settings
		"prisoner of war camp", "concentration camp", "military hospital", "field hospital",
		"military cemetery", "aircraft factory", "tank factory", "shipyard",
		"military depot", "ammunition dump", "fuel depot", "supply dump",
		"military port", "embarkation point", "submarine pen", "naval dry dock",
		
		# Terrain Types
		"desert", "desert oasis", "desert dunes", "rocky desert", "forest",
		"dense forest", "winter forest", "urban area", "bombed city", "city ruins",
		"beach", "landing beach", "rocky beach", "mountain", "mountain pass",
		"field", "farm field", "snow-covered field", "ocean", "open ocean",
		"coastal waters", "river", "river crossing", "flooded river", "bridge",
		
		# Military Infrastructure
		"airfield", "temporary airstrip", "bomber base", "fighter base",
		"naval base", "submarine base", "army barracks", "training camp",
		"military headquarters", "command bunker", "coastal defense",
		"fortified line", "defensive position", "artillery position",
		
		# Military Activities
		"battlefield", "active battlefield", "battlefield aftermath",
		"military parade", "victory parade", "surrender ceremony",
		"military exercise", "amphibious landing exercise", "tank maneuvers",
		"war zone", "civilian evacuation", "occupation zone", "frontline",
		"military checkpoint", "border checkpoint", "military convoy route",
		
		# Home Front
		"war factory", "armaments factory", "aircraft assembly line",
		"vehicle assembly line", "shipyard", "munitions factory",
		"civilian air raid shelter", "bombed civilian area", "rationing center",
		"recruitment office", "propaganda poster display", "war bonds office",
		"civil defense drill", "air raid aftermath", "victory celebration"
	]

	era_categories = [
		# Pre-War & Early War
		"pre-World War I era", "World War I era", "interwar period",
		"pre-1939 military",
		"Phoney War", "Blitzkrieg era", "1939-1940 military equipment",
		
		# World War II Specific Periods
		"World War II era", "Battle of Britain era", "North African campaign",
		"Pearl Harbor era", "Midway period", "Stalingrad era", "Normandy invasion",
		"Operation Barbarossa", "Battle of the Bulge", "Italian campaign",
		"D-Day preparations", "Market Garden operation", "Fall of Berlin",
		"Island-hopping campaign", "Battle of the Atlantic", "V-E Day era",
		"Pacific War late stage", "atomic bomb era", "Japanese surrender period",
		
		# Post-War Periods
		"immediate post-war", "occupation period", "Cold War",
		"Korean War era", "1950s military", "Vietnam era", "Japanse World War era",
					
		# Military Technology Eras
		"early tank warfare", "biplane era", "early radar period", "monoplane transition",
		"early jet aircraft", "V-weapon period", "heavy bomber era", "aircraft carrier warfare",
		"submarine warfare golden age", "amphibious assault development",
		"mechanized warfare", "combined arms doctrine", "early nuclear era",
		
		# National Military Period Styles
		"Wehrmacht prime", "Soviet military buildup", "British Empire forces",
		"American war production peak", "Imperial Japanese forces",
		"Nazi Germany zenith", "Allied powers ascendancy", "Axis powers decline",
		"Red Army resurgence", "American military dominance"
	]

	activity_categories = [
		# Combat Activities
		"fighting", "tank battle", "infantry assault", "naval engagement",
		"aerial dogfight", "bombing run", "strafing run", "artillery barrage",
		"firing weapon", "machine gun firing", "mortar firing", "shelling",
		"anti-aircraft firing", "sniper activity", "flamethrower attack",
		"bayonet charge", "hand-to-hand combat", "urban combat", "house-to-house fighting",
		
		# Movement & Transportation
		"driving", "convoy movement", "tank column", "troop transport",
		"marching", "infantry advance", "tactical retreat", "military withdrawal",
		"flying", "air patrol", "reconnaissance flight", "bombing mission",
		"parachute drop", "airborne landing", "glider landing", "air resupply",
		"crossing terrain", "river crossing", "beach landing", "amphibious assault",
		"fording stream", "mountain climbing", "moving through jungle",
		"naval convoy", "fleet movement", "submarine patrol", "naval blockade",
		
		# Military Operations
		"digging trenches", "building fortifications", "laying mines", "clearing mines",
		"constructing bridge", "demolishing bridge", "breaching obstacles",
		"setting up artillery", "camouflaging position", "establishing perimeter",
		"setting up command post", "establishing field hospital", "creating airstrip",
		
		# Logistics & Support
		"loading equipment", "unloading equipment", "loading ammunition", "refueling",
		"resupplying troops", "distributing rations", "loading wounded", "evacuating casualties",
		"loading ships", "unloading landing craft", "airdrop receiving", "gathering supplies",
		"towing disabled vehicle", "vehicle recovery", "aircraft maintenance", "tank repair",
		"weapon cleaning", "equipment maintenance", "vehicle maintenance",
		
		# Military Life & Routines
		"training", "infantry drill", "weapons training", "tank crew training", "pilot training",
		"field exercise", "receiving briefing", "map reading", "radio communication",
		"standing guard", "sentry duty", "prisoner handling", "military inspection",
		"cooking field rations", 'military rations', "eating meal", "resting between battles", "writing letters",
		"medical treatment", "field surgery", "distributing supplies", "receiving orders",
		
		# Ceremonial & Administrative
		"military parade", "award ceremony", "flag raising", "surrender ceremony", 
		"prisoner processing", "military funeral", "military wedding", "religious service",
		"officer briefing", "signing documents", "military trial", "propaganda filming",
		"press conference", "VIP visit", "civilian interaction", "occupation duty",
		"war crime investigation", "reconnaissance reporting"
	]

	# Function to process image batches
	def process_batch(batch_paths, categories):
		valid_images = []
		valid_indices = []
		
		# Load images
		for i, path in enumerate(batch_paths):
			try:
				if os.path.exists(path):
					img = Image.open(path).convert('RGB')
					valid_images.append(img)
					valid_indices.append(i)
			except Exception as e:
				if verbose:
					print(f"Error loading image {path}: {e}")
		
		if not valid_images:
			return [[] for _ in range(len(batch_paths))]
				
		# Prepare text prompts
		text_prompts = [f"a photo of {cat}" for cat in categories]
		
		# Process with VLM
		with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			# Prepare inputs
			inputs = processor(
				text=text_prompts,
				images=valid_images, 
				return_tensors="pt", 
				padding=True
			).to(device)
			
			# Get embeddings
			outputs = model(**inputs)
			
			# Normalize embeddings
			image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
			text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
			
			# Calculate similarity scores
			similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)
			
			# Create results
			batch_results = [[] for _ in range(len(batch_paths))]
			
			# Extract predictions above threshold
			for img_idx, similarities in enumerate(similarity.cpu().numpy()):
				batch_idx = valid_indices[img_idx]
				for cat_idx, score in enumerate(similarities):
					if score > confidence_threshold:
						batch_results[batch_idx].append(categories[cat_idx])

		return batch_results
	
	# Process images in batches
	all_labels = []
	
	# 1. Visual Object Detection
	if verbose:
		print("Performing visual object detection...")
	for i in tqdm(range(0, len(image_paths), batch_size), desc="Object Detection"):
		batch_paths = image_paths[i:i+batch_size]
		batch_results = process_batch(batch_paths, object_categories)
		all_labels.extend(batch_results)
	
	# 2. Scene Classification
	if verbose:
		print("Performing scene classification...")
	scene_labels = []
	for i in tqdm(range(0, len(image_paths), batch_size), desc="Scene Classification"):
		batch_paths = image_paths[i:i+batch_size]
		batch_results = process_batch(batch_paths, scene_categories)
		scene_labels.extend(batch_results)
	
	# 3. Era Detection (Temporal Visual Cues)
	if verbose:
		print("Detecting temporal visual cues...")
	era_labels = []
	for i in tqdm(range(0, len(image_paths), batch_size), desc="Era Detection"):
		batch_paths = image_paths[i:i+batch_size]
		# Lower threshold for era detection (more challenging)
		batch_results = process_batch(batch_paths, era_categories)
		era_labels.extend(batch_results)
	
	# 4. Activity Recognition (Visual Relationship Detection)
	if verbose:
		print("Detecting visual relationships and activities...")
	activity_labels = []
	for i in tqdm(range(0, len(image_paths), batch_size), desc="Activity Recognition"):
		batch_paths = image_paths[i:i+batch_size]
		batch_results = process_batch(batch_paths, activity_categories)
		activity_labels.extend(batch_results)
	
	# Combine all visual annotations
	combined_labels = []
	for i in range(len(image_paths)):
		image_labels = []
		
		# Add object labels if available
		if i < len(all_labels):
			image_labels.extend(all_labels[i])
		
		# Add scene labels if available
		if i < len(scene_labels):
			image_labels.extend(scene_labels[i])
		
		# Add era labels if available
		if i < len(era_labels):
			image_labels.extend(era_labels[i])
		
		# Add activity labels if available
		if i < len(activity_labels):
			image_labels.extend(activity_labels[i])
		
		# Remove duplicates and sort
		combined_labels.append(sorted(set(image_labels)))
	
	if verbose:
		total_labels = sum(len(labels) for labels in combined_labels)
		print(f"Vision-based annotation completed in {time.time() - start_time:.2f} seconds")
		print(f"Generated {total_labels} labels for {len(image_paths)} images")
		print(f"Average labels per image: {total_labels/len(image_paths):.2f}")

	df['visual_based_labels'] = combined_labels
	df.to_csv(metadata_fpth, index=False)
	print(f"Visual-based annotation Elapsed time: {time.time() - start_time:.2f} sec".center(160, " "))

	return combined_labels

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Multi-label annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--use_parallel", '-parallel', action="store_true")
	parser.add_argument("--num_workers", '-nw', type=int, default=16)
	parser.add_argument("--text_batch_size", '-tbs', type=int, default=512)
	parser.add_argument("--vision_batch_size", '-vbs', type=int, default=16, help="Batch size for vision processing")
	parser.add_argument("--relevance_threshold", '-rth', type=float, default=0.25, help="Relevance threshold for text-based filtering")
	parser.add_argument("--vision_threshold", '-vth', type=float, default=0.20, help="Confidence threshold for VLM-based filtering")
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
			batch_size=args.vision_batch_size,
			verbose=True,
			confidence_threshold=args.vision_threshold,
			device=args.device,
			metadata_fpth=vision_output_path,
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
	sample_cols = ['title', 'description', 'label', 'img_url', 'textual_based_labels', 'visual_based_labels', 'multimodal_labels']
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