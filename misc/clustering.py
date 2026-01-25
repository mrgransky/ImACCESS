# Step 1: Import necessary libraries
from utils import *
from sklearn.metrics import silhouette_score

# Load a pre-trained Sentence-BERT model. 'all-MiniLM-L6-v2' is a good general-purpose model.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Load your data
documents = load_pickle(fpath="/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal_multimodal.pkl")
print(f"Loaded {type(documents)} {len(documents)} docs")
documents = [list(set(doc)) for doc in documents]
print(f"Loaded {type(documents)} {len(documents)} docs after deduplication")
# ["keyword1, keyword2, keyword3, ..."]
all_labels = []
for doc in documents:
	for label in doc:
		all_labels.append(label)
		# print(label)

# for doc in documents:
# 	all_labels.append(", ".join(doc))

print(f"Loaded {type(all_labels)} {len(all_labels)} labels")
for i, label in enumerate(all_labels[:20]):
	print(f"{i}: {label}")

# Encode the documents to get sentence embeddings
X = model.encode(all_labels, show_progress_bar=True)
print(f"Shape of sentence embeddings: {type(X)} {X.shape}")


# Define a range of cluster numbers to evaluate
range_n_clusters = range(2, 100, 5)
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
plt.savefig(f"clusters_silhouette_score_{optimal_n_clusters}.png", dpi=100)
# plt.show()
print(f"The optimal number of clusters based on Silhouette Score is: {optimal_n_clusters}")

# Clustering using K-Means
kmeans_optimal = KMeans(n_clusters=optimal_n_clusters, random_state=0, n_init=10)
clusters_optimal = kmeans_optimal.fit_predict(X)


# Dimensionality Reduction (optional, for visualization)
pca = PCA(n_components=2, random_state=0)
X_reduced = pca.fit_transform(X)

print(f"Shape of sentence embeddings: {type(X_reduced)} {X_reduced.shape}")

# Step 6: Visualization
plt.figure(figsize=(27, 17))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_optimal, cmap='viridis')
plt.title(f"Text Clustering Visualization (Optimal N_clusters = {optimal_n_clusters})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Adding cluster centers to the plot
centers_optimal = kmeans_optimal.cluster_centers_
centers_reduced_optimal = pca.transform(centers_optimal)

cmap = plt.colormaps.get_cmap('viridis')

for i, center_coords in enumerate(centers_reduced_optimal):
	plt.scatter(center_coords[0], center_coords[1], c=[cmap(i / (kmeans_optimal.n_clusters - 1 if kmeans_optimal.n_clusters > 1 else 1))], s=200, alpha=0.94, marker='X')
	plt.scatter(center_coords[0], center_coords[1], facecolors='none', edgecolors=[cmap(i / (kmeans_optimal.n_clusters - 1 if kmeans_optimal.n_clusters > 1 else 1))], s=120, alpha=0.8, marker='o', linewidths=2)

# Adding labels to the plot
for i, txt in enumerate(all_labels):
	plt.annotate(txt[:20], (X_reduced[i, 0], X_reduced[i, 1]), fontsize=6, alpha=0.75, rotation=60)

# plt.colorbar(scatter, label='Cluster Label')
plt.tight_layout()
plt.savefig(f"clustering_optimal_{optimal_n_clusters}.png", dpi=200)

# how many samples each cluster has:
unique, counts = np.unique(clusters_optimal, return_counts=True)
print(np.asarray((unique, counts)).T)


# create a pandas dataframe with text column and their corresponding cluster index and print them
df_clusters = pd.DataFrame({'text': all_labels, 'cluster': clusters_optimal})
print(df_clusters.head(10))
print(df_clusters.tail(10))


# those with largest number of samples:
print(df_clusters['cluster'].value_counts().head(10))