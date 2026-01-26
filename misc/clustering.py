# Step 1: Import necessary libraries
from utils import *
from sklearn.metrics import silhouette_score
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seeds(seed=42)

COLORMAP = "Dark2"
cmap = plt.colormaps.get_cmap(COLORMAP)
# model_id = "google/embeddinggemma-300M"
model_id = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_id).to(device)
print(f"Total number of parameters in {model_id}: {sum([p.numel() for _, p in model.named_parameters()]):,}")

labels_file_path = "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal_multimodal.pkl"
DATASET_DIR = os.path.dirname(labels_file_path)
OUTPUTS_DIR = os.path.join(DATASET_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
documents = load_pickle(fpath=labels_file_path)
print(f"Loaded {type(documents)} {len(documents)} docs")
documents = [list(set(doc)) for doc in documents]
print(f"Loaded {type(documents)} {len(documents)} docs after deduplication")
# # ["keyword1, keyword2, keyword3, ..."]
all_labels = []

for doc in documents:
	for label in doc:
		all_labels.append(label)
		# print(label)

# for doc in documents:
# 	all_labels.append("; ".join(doc))

all_labels = list(set(all_labels))


print(f"Loaded {type(all_labels)} {len(all_labels)} labels")
for i, label in enumerate(all_labels[:20]):
	print(f"{i}: {label}")

# Encode the documents to get sentence embeddings
X = model.encode(all_labels, show_progress_bar=True)
print(f"Shape of sentence embeddings: {type(X)} {X.shape}")

# Define a range of cluster numbers to evaluate
range_n_clusters = range(2, 111, 3)
silhouette_scores = []

for n_clusters in range_n_clusters:
	kmeans_model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
	cluster_labels = kmeans_model.fit_predict(X)
	silhouette_avg = silhouette_score(X=X, labels=cluster_labels, random_state=0, metric='euclidean')
	silhouette_scores.append(silhouette_avg)
	print(f"cluster: {n_clusters:<8} silhouette_score: {silhouette_avg:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Score for Various Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range_n_clusters)
plt.grid(True)

# Highlight the optimal number of clusters
optimal_n_clusters_idx = np.argmax(silhouette_scores)
optimal_n_clusters = range_n_clusters[optimal_n_clusters_idx]
plt.axvline(x=optimal_n_clusters, color='red', linestyle='--', label=f'Optimal N_clusters: {optimal_n_clusters}')
plt.legend()
plt.savefig(os.path.join(OUTPUTS_DIR, f"clusters_silhouette_score_{optimal_n_clusters}.png"), dpi=100)
print(f"The optimal number of clusters based on Silhouette Score is: {optimal_n_clusters}")

# Clustering using K-Means
# optimal_n_clusters = 90
kmeans_optimal = KMeans(n_clusters=optimal_n_clusters, random_state=0, n_init=10)
clusters_optimal = kmeans_optimal.fit_predict(X)

# Dimensionality Reduction (optional, for visualization)
pca = PCA(n_components=2, random_state=0)
X_reduced = pca.fit_transform(X)

print(f"X_pca: {type(X_reduced)} {X_reduced.shape}")

# Step 6: Visualization
plt.figure(figsize=(19, 15))
scatter = plt.scatter(
	X_reduced[:, 0], 
	X_reduced[:, 1], 
	c=clusters_optimal, 
	# cmap=COLORMAP,
	facecolors='none',
	s=12,
	alpha=0.95,
	marker='o',
	label=f'{len(all_labels)}',
)
plt.title(f"Text Clustering Visualization (Optimal N_clusters = {optimal_n_clusters})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Adding cluster centers to the plot
centers_optimal = kmeans_optimal.cluster_centers_
centers_reduced_optimal = pca.transform(centers_optimal)

for i, center_coords in enumerate(centers_reduced_optimal):
	plt.scatter(
		center_coords[0], 
		center_coords[1], 
		# c=[cmap(i / (kmeans_optimal.n_clusters - 1 if kmeans_optimal.n_clusters > 1 else 1))], 
		s=200, 
		alpha=0.94, 
		marker='X'
	)
	plt.scatter(center_coords[0], center_coords[1], facecolors='none', edgecolors=[cmap(i / (kmeans_optimal.n_clusters - 1 if kmeans_optimal.n_clusters > 1 else 1))], s=120, alpha=0.8, marker='o', linewidths=2)

# # Adding labels to the plot
# for i, txt in enumerate(all_labels):
# 	plt.annotate(txt[:20], (X_reduced[i, 0], X_reduced[i, 1]), fontsize=6, alpha=0.75, rotation=60)

# plt.colorbar(scatter, label='Cluster Label')
plt.legend(loc='best', frameon=False, fancybox=True, edgecolor='black', facecolor='white')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, f"clustering_optimal_{optimal_n_clusters}.png"), dpi=250)

# how many samples each cluster has:
unique, counts = np.unique(clusters_optimal, return_counts=True)
print(np.asarray((unique, counts)).T)


# create a pandas dataframe with text column and their corresponding cluster index and print them
df_clusters = pd.DataFrame({'text': all_labels, 'cluster': clusters_optimal})
print(df_clusters.head(10))
print(df_clusters.tail(10))

# save df to csv:
df_clusters.to_csv(os.path.join(OUTPUTS_DIR, f"clustering_optimal_{optimal_n_clusters}.csv"), index=False)
try:
	df_clusters.to_excel(os.path.join(OUTPUTS_DIR, f"clustering_optimal_{optimal_n_clusters}.xlsx"), index=False)
except Exception as e:
	print(f"Failed to write Excel file: {e}")

# those with largest number of samples:
print("-"*120)
print(df_clusters['cluster'].value_counts().head(10))
print("-"*120)
# print 20 samples of each cluster:
for cluster_id in range(optimal_n_clusters):
	print(f"Cluster {cluster_id}:")
	print(df_clusters[df_clusters['cluster'] == cluster_id]['text'].head(50).tolist())
	print()