from utils import *
import visualize as viz

try:
	import fastcluster
	use_fastcluster = True
	print("[FASTCLUSTER] Using fastcluster for O(n² log n) performance")
except ImportError:
	use_fastcluster = False
	print("[SCIPY] Using scipy (slower for large n)")

try:
	from sentence_transformers.sentence_transformer.modules import Normalize
except ImportError:
	try:
		from sentence_transformers.base.modules import Normalize
	except ImportError:
		from sentence_transformers.models import Normalize

if not getattr(Normalize, "_rogue_kwargs_patched", False):
	_orig_norm_init = Normalize.__init__
	def _patched_norm_init(self, *args, **kwargs):
		kwargs.pop("normalize_embeddings", None)
		return _orig_norm_init(self, *args, **kwargs)
	Normalize.__init__ = _patched_norm_init
	Normalize._rogue_kwargs_patched = True

CUSTOM_ENCODE_INSTRUCTION = (
	"Instruct: Given a label describing a historical photograph, "
	"retrieve labels that name the same concept\nQuery:"
)

def summarize_canonical_selection(
	path,
	benign=("building", "room", "flag", "station", "plant", "sign", "camera", "suit", "cap", "camp", "debris", "hospital"),
	homonyms=("tank", "float", "race", "gear", "party", "arm", "press", "ward"),
	detail=("building", "tank"),
	detail_rows=8,
	print_summary=True,
	return_text=False,
):
	path = pathlib.Path(path)
	with path.open(encoding="utf-8") as f:
		data = json.load(f)

	clusters = data["clusters"]
	meta = data.get("meta", {})
	lines = []

	def emit(text=""):
		lines.append(text)

	def section(title):
		emit("\n" + title)

	def pct(a, b):
		return f"{a / max(b, 1) * 100:.1f}%"

	def instances(cluster):
		return sum(
			(candidate.get("corpus_freq") or 0)
			for candidate in cluster.get("candidates", [])
			if not candidate.get("is_virtual", False)
		)

	n = len(clusters)
	labels = sum(c.get("size", 0) for c in clusters)
	total_instances = sum(instances(c) for c in clusters)
	emit(f"RUN SUMMARY  {path.name}")
	emit(
		f"clusters {n:,} | labels {labels:,} "
		f"({labels / max(n, 1):.2f} per cluster) | label instances {total_instances:,}"
	)

	section("1. cluster sizes")
	if clusters:
		sizes = Counter(min(c.get("size", 0), 11) for c in clusters)
		emit("   " + " | ".join(
			f"{'11+' if k == 11 else k}: {sizes[k]}" for k in sorted(sizes)
		) + f" | max {max(c.get('size', 0) for c in clusters)}")
	else:
		emit("   (no clusters)")

	section("2. how canonicals were chosen")
	for method, count in meta.get("selection_method_counts", {}).items():
		emit(f"   {method:<34} {count:6d} ({pct(count, n)})")
	virtual_gates = meta.get("virtual_gates")
	if virtual_gates:
		emit(
			"   virtual gates: "
			f"min_sim_ratio={virtual_gates.get('min_sim_ratio')} "
			f"min_distinct={virtual_gates.get('min_distinct_concepts')} "
			f"rejections={virtual_gates.get('rejections')}"
		)

	def is_degenerate_virtual(label):
		return bool(
			re.fullmatch(r"[\d.,/\- ]+", label)
			or re.fullmatch(r"(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})", label.lower())
			or (len(re.sub(r"[^A-Za-z]", "", label)) < 2
				and not any(ch.isdigit() for ch in label))
			or label.lower() in {
				"co", "corp", "company", "inc", "ltd", "limited", "corporation"
			}
		)

	degenerate = [
		c.get("canonical", "") for c in clusters
		if c.get("is_virtual") and is_degenerate_virtual(c.get("canonical", ""))
	]
	emit(
		"   degenerate virtual canonicals (numerals / single letters / "
		f"corporate suffixes): {len(degenerate)} {degenerate[:8]}"
	)

	section("3. vocabulary and tail")
	class_instances = Counter()
	for c in clusters:
		class_instances[c.get("canonical", "")] += instances(c)
	values = sorted(class_instances.values(), reverse=True)
	total = sum(values)
	if values:
		tail = " | ".join(
			f"<{k}: {sum(v < k for v in values) / len(values) * 100:.0f}%"
			for k in (5, 10, 20, 50)
		)
		top_share = sum(values[:len(values) // 10]) / max(total, 1) * 100
		largest_share = values[0] / max(total, 1) * 100
		emit(
			f"   classes {len(values):,} | {tail} | top 10% hold {top_share:.0f}% "
			f"| largest {values[0]:,} ({largest_share:.1f}%)"
		)
		emit("   largest classes: " + ", ".join(
			f"{name!r} {count:,}" for name, count in class_instances.most_common(8)
		))
	else:
		emit("   (no class data)")

	# 4. shared names and the shared-name resolver
	section("4. names shared by several clusters and the shared-name resolver")
	by_name = Counter(c.get("canonical", "") for c in clusters)
	shared = {name: count for name, count in by_name.items() if count > 1}
	emit(
		f"   {len(shared)} names cover {sum(shared.values())} clusters | top: "
		+ ", ".join(f"{name!r}x{count}" for name, count in by_name.most_common(8))
	)
	resolver_counts = meta.get("shared_resolution_counts")
	if resolver_counts:
		emit(f"   resolver outcomes: {resolver_counts}")

	# a cluster in a shared group is KEPT when its centroid is within `threshold` of the group's anchor,
	# otherwise DEMOTED to a real member; groups are keyed by the ORIGINAL shared name
	groups = defaultdict(list)
	for c in clusters:
		s = c.get("shared_canonical") or {}
		if s.get("group_size") is None or s.get("anchor_similarity") is None:
			continue
		groups[(s.get("demoted_from") or c.get("canonical", "")).lower()].append(c)

	def anchor(c):
		return c["shared_canonical"]["anchor_similarity"]

	def demoted_flag(c):
		return c["shared_canonical"].get("resolution") == "demoted"

	if groups:
		in_groups = [c for g in groups.values() for c in g]
		demoted = [c for c in in_groups if demoted_flag(c)]
		kept = [c for c in in_groups if not demoted_flag(c)]
		thresholds = sorted({
			(c.get("shared_canonical") or {}).get("threshold") for c in clusters
		} - {None})
		emit(
			f"   threshold used {thresholds} | groups {len(groups)} | clusters in groups {len(in_groups)} "
			f"| kept {len(kept)} | demoted {len(demoted)} ({pct(len(demoted), len(in_groups))})"
		)
		if demoted:
			emit("   anchor similarity of DEMOTED clusters: " + " ".join(
				f"p{p} {np.percentile([anchor(c) for c in demoted], p):.3f}" for p in (10, 50, 90)
			))
		if kept:
			emit("   anchor similarity of KEPT clusters   : " + " ".join(
				f"p{p} {np.percentile([anchor(c) for c in kept], p):.3f}" for p in (1, 5, 10, 50)
			))
		most_demoted = Counter({
			name: sum(demoted_flag(c) for c in g) for name, g in groups.items()
		}).most_common(10)
		emit("   names with the most demotions: " + ", ".join(
			f"{name!r} {count}" for name, count in most_demoted if count
		))

		def group_line(name):
			g = groups.get(name)
			if not g:
				return f"   {name!r:12} (no shared group)"
			sims = [anchor(c) for c in g]
			d = sum(demoted_flag(c) for c in g)
			return (
				f"   {name!r:12} clusters {len(g):3d} | kept {len(g) - d:3d} demoted {d:3d} "
				f"| anchor sim min {min(sims):.3f} median {np.median(sims):.3f}"
			)

		emit("   benign names (should stay together):")
		for name in benign:
			emit(group_line(name))
		emit("   homonyms (should split):")
		for name in homonyms:
			emit(group_line(name))
	else:
		emit("   (no shared-name diagnostics in this schema)")
		for name in homonyms:
			if by_name.get(name):
				emit(f"   homonym watch {name!r}: {by_name[name]} clusters;")

	section("5. harmonization (lowercase -> Capitalised renames can cross word senses)")
	harmonized = [c for c in clusters if c.get("harmonize", {}).get("changed")]
	if harmonized:
		flips = [
			(c["harmonize"].get("from", ""), c.get("canonical", ""))
			for c in harmonized
			if c["harmonize"].get("from", "")[:1].islower()
			and c.get("canonical", "")[:1].isupper()
		]
		emit(
			f"   clusters renamed {len(harmonized)} | lowercase->Capitalised "
			f"flips {len(flips)}: {sorted(set(flips))[:14]}"
		)
	else:
		emit("   (no harmonize field in this schema)")

	def punctuation_key(text):
		return re.sub(r"[^a-z0-9]", "", text.lower())

	normalized = defaultdict(set)
	for name in by_name:
		normalized[punctuation_key(name)].add(name)
	duplicates = [sorted(names) for names in normalized.values() if len(names) > 1]
	emit(
		"   names that differ only by case / spacing / punctuation: "
		f"{len(duplicates)} groups {duplicates[:8]}"
	)

	# synonym variants of one concept under different names ('fighter aircraft' / 'fighter airplane')
	synonym_map = {"airplane": "aircraft", "aeroplane": "aircraft", "plane": "aircraft", "automobile": "car"}
	synonym_targets = set(synonym_map.values())

	def synonym_tokens(name):
		out = []
		for token in re.findall(r"[a-z0-9]+", name.lower()):
			base = token[:-1] if token.endswith("s") and (token[:-1] in synonym_map or token[:-1] in synonym_targets) else token
			out.append(synonym_map.get(base, base))
		return " ".join(out)

	synonym_groups = defaultdict(set)
	for name in by_name:
		synonym_groups[synonym_tokens(name)].add(name)
	synonym_variants = [
		sorted(names) for names in synonym_groups.values()
		if len({punctuation_key(x) for x in names}) > 1
	]
	synonym_variants.sort(key=lambda names: -sum(class_instances[x] for x in names))
	emit(
		"   synonym variants (aircraft/airplane/plane, car/automobile) under different names: "
		f"{len(synonym_variants)} groups, {sum(class_instances[x] for names in synonym_variants for x in names):,} instances | top: "
		+ "; ".join(
			f"{names} {sum(class_instances[x] for x in names):,}" for names in synonym_variants[:5]
		)
	)

	section("6. literal labels displaced by a virtual's name")
	home = {
		label: c for c in clusters for label in c.get("members", [])
	}
	frequencies = {
		candidate.get("label", ""): candidate.get("corpus_freq") or 0
		for c in clusters for candidate in c.get("candidates", [])
		if not candidate.get("is_virtual", False)
	}
	displaced = {}
	for c in clusters:
		canonical = c.get("canonical", "")
		if (c.get("is_virtual") and canonical in home
				and home[canonical].get("canonical", "").lower() != canonical.lower()):
			displaced[canonical] = home[canonical].get("canonical", "")
	virtual_count = sum(bool(c.get("is_virtual")) for c in clusters)
	displaced_instances = sum(frequencies.get(label, 0) for label in displaced)
	top_displaced = sorted(
		((label, frequencies.get(label, 0), target)
		 for label, target in displaced.items()),
		key=lambda item: -item[1],
	)[:5]
	emit(
		f"   virtuals: {virtual_count} ({pct(virtual_count, n)}) | literal labels "
		"that map to a different class than their own text: "
		f"{len(displaced)} ({displaced_instances:,} instances) | top: {top_displaced}"
	)

	section("7. designation labels (B-17, Ki-46, Bf 109 ...): is the canonical the same family?")

	def family(text):
		match = re.search(
			r"\b((?:Bf|Fw|Ju|He|Me|Do|Ar)[- ]?\d{2,3}|[A-Z]{1,3}[- ]?\d{1,4})",
			text,
		)
		return re.sub(r"[- ]", "", match.group(1)).upper() if match else None

	categories = Counter()
	for c in clusters:
		canonical_family = family(c.get("canonical", ""))
		for label in c.get("members", []):
			member_family = family(label)
			if not member_family:
				continue
			weight = frequencies.get(label, 1)
			categories[
				"same family" if canonical_family == member_family
				else "generic / named" if canonical_family is None
				else "DIFFERENT family"
			] += weight
	category_total = sum(categories.values())
	category_summary = " | ".join(
		f"{name} {count / max(category_total, 1) * 100:.1f}%"
		for name, count in categories.most_common()
	)
	mixed_families = sum(
		len({family(label) for label in c.get("members", [])} - {None}) >= 2
		for c in clusters
	)
	emit(f"   {category_summary} | clusters mixing >=2 families: {mixed_families}")

	section("8. nearest-cluster structure")
	nearest = meta.get("nearest_cluster_similarity")
	if nearest:
		emit(
			f"   median {nearest.get('median_nearest_similarity')} "
			f"p90 {nearest.get('p90_nearest_similarity')} | "
			f"nearest at/above level: {nearest.get('nearest_at_or_above')}"
		)
		emit(
			"   ... with a DIFFERENT name: "
			f"{nearest.get('nearest_at_or_above_different_name')}"
		)
		mutual = sum(
			bool(c.get("nearest_cluster", {}).get("mutual")) for c in clusters
		)
		emit(f"   mutual nearest neighbours {pct(mutual, n)}")
	else:
		emit("   (no neighbour diagnostics in this schema)")

	if groups and detail:
		section("9. shared-name resolver: lowest anchor similarities (the decision boundary)")
		for name in detail:
			g = sorted(groups.get(name, []), key=anchor)
			emit(f"   {name!r}:")
			for c in g[:detail_rows]:
				emit(
					f"      {anchor(c):.3f} {c['shared_canonical'].get('resolution', ''):8} "
					f"-> {c.get('canonical', '')!r:30} {c.get('members', [])[:7]}"
				)

	summary = "\n".join(lines) + "\n"
	if print_summary:
		print(summary)
	return summary if return_text else None

def _encode_(
	model,
	texts: List[str],
	batch_size: int = 128,
	prompt: Optional[str] = None,
	show_progress_bar: bool = False,
	normalize: bool = True,
) -> np.ndarray:
	"""
	The single place where label text becomes a vector.
	
	Guarantees unit-norm float32 embeddings across standard models,
	BF16 models (e.g., Nemotron), and models with internal Normalize layers (e.g., Octen).
	"""
	def _has_internal_normalize(m) -> bool:
		if hasattr(m, "modules"):
			return any(isinstance(mod, Normalize) for mod in m.modules())
		return False

	kw = {
		"batch_size": batch_size,
		"show_progress_bar": show_progress_bar,
		"convert_to_numpy": True,
		"precision": "float32",
	}
	
	if prompt is not None:
		kw["prompt"] = prompt
	
	# Only request external normalization if the model pipeline doesn't already do it
	if normalize and not _has_internal_normalize(model):
		kw["normalize_embeddings"] = True

	embeddings = model.encode(list(texts), **kw)
	
	return np.asarray(embeddings, dtype=np.float32)

def _nearest_cluster_neighbors(centroids: np.ndarray, chunk: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
	"""
	For every row of `centroids` (n, d): the index and cosine similarity of the
	most similar OTHER row. Chunked, so memory stays at chunk * n floats even
	for ~10k clusters x 4096 dims.
	"""
	V = np.asarray(centroids, dtype=np.float32)
	V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
	
	n = V.shape[0]
	nn_idx = np.zeros(n, dtype=np.int64)
	nn_sim = np.full(n, -1.0, dtype=np.float32)
	
	if n < 2:
		return nn_idx, nn_sim
	
	for s in range(0, n, chunk):
		S = V[s:s + chunk] @ V.T
		rows = np.arange(S.shape[0])
		S[rows, rows + s] = -np.inf # exclude the cluster itself
		j = S.argmax(axis=1)
		nn_idx[s:s + chunk] = j
		nn_sim[s:s + chunk] = S[rows, j]
	
	return nn_idx, nn_sim

def _json_default(obj):
	if isinstance(obj, np.generic):
		return obj.item()
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def _estimate_dynamic_neighbor_thresholds(
	sims: np.ndarray,
	mutual_mask: np.ndarray,
	verbose: bool = False,
) -> Dict[str, Any]:
	"""
	Data‑driven heuristic that turns a distribution of pairwise similarity scores into a handful 
	of threshold values that are later used to decide whether two clusters 
	(or two “member centroids”) should be merged.

	Derive data-driven similarity thresholds from the intrinsic 1-NN cluster distribution
	(prior to canonical selection, directly on member centroids):
	  1. t_mad        : Robust upper dispersion fence = median(sims) + 1.0 * sigma_MAD,
	                    where sigma_MAD = 1.4826 * median(|sims - median(sims)|).
	  2. t_elbow      : Kneedle maximum-curvature inflection point on the sorted upper tail
	                    (sims >= median), separating background concept spacing from split duplicates.
	  3. t_mutual_p75 : 75th percentile of mutual 1-NN similarities (reciprocal closest pairs).
	  4. threshold    : Consensus median of the active estimators (used for review & Step 4b merge).

	
	Full Algorithmic Pipeline (Step‑by‑Step):

	INPUT: sims (array of similarity scores), mutual_mask (bool mask)

	IF len(sims) < 2:
			RETURN constant fallback dict

	# 1️⃣ Robust statistics
	med   = median(sims)
	mad   = median(|sims - med|)
	sigma = 1.4826 * mad
	t_mad = clip(med + sigma, lower=med, upper=max(sims))

	# 2️⃣ Upper‑tail elbow (Kneedle)
	upper = sort(sims where sims >= med)
	IF len(upper) >= 3 AND upper[-1] > upper[0]:
			x = linspace(0, 1, len(upper))
			y = (upper - upper[0]) / (upper[-1] - upper[0])
			idx_elbow = argmax(x - y)
			t_elbow = upper[idx_elbow]
	ELSE:
			t_elbow = percentile(sims, 90)

	# 3️⃣ Mutual 1‑NN 75th percentile (if enough data)
	n_mut = sum(mutual_mask)
	IF n_mut >= 5:
			t_mut_p75 = percentile(sims[mutual_mask], 75)
	ELSE:
			t_mut_p75 = None

	# 4️⃣ Consensus threshold
	estimators = [t_mad, t_elbow]
	IF t_mut_p75 is not None:
			estimators.append(t_mut_p75)

	threshold = median(estimators)   # robust aggregation
	ROUND all numbers to 4 decimals

	RETURN {
			"threshold": threshold,
			"t_mad": t_mad,
			"t_elbow": t_elbow,
			"t_mutual_p75": t_mut_p75,
			"sigma_mad": sigma,
	}
	"""

	n = len(sims)
	if n < 2:
		return {
			'threshold': 0.88,
			't_mad': None,
			't_elbow': None,
			't_mutual_p75': None,
			'sigma_mad': 0.0,
		}

	if verbose:
		print(f"[DYNAMIC] threshold estimation")
		print(f"  ├─ sims        {type(sims)} {sims.shape}")
		print(f"  ├─ mutual_mask {type(mutual_mask)} {mutual_mask.shape} True: {mutual_mask.sum()}")

	# Stage 1: Estimator 1 — Robust Statistical Dispersion Fence (t_mad)
	med = float(np.median(sims))
	mad = float(np.median(np.abs(sims - med))) # Median Absolute Deviation (MAD)
	sigma_mad = 1.4826 * mad
	t_mad = float(np.clip(med + 1.0 * sigma_mad, med, float(np.max(sims))))

	# Stage 2: Estimator 2 — Kneedle Maximum-Curvature Elbow (t_elbow)
	upper = np.sort(sims[sims >= med])
	if len(upper) >= 3 and (upper[-1] > upper[0]):
		x = np.linspace(0.0, 1.0, len(upper))
		y = (upper - upper[0]) / (upper[-1] - upper[0])
		t_elbow = float(upper[int(np.argmax(x - y))])
	else:
		t_elbow = float(np.percentile(sims, 90))

	# Stage 3: Estimator 3 — Topological Mutual 1-NN Quartile (t_mutual_p75)
	n_mut = int(mutual_mask.sum())
	t_mut_p75 = float(np.percentile(sims[mutual_mask], 75)) if n_mut >= 5 else None

	# Stage 4: Consensus Ensemble Aggregation (threshold)
	estimators = [t_mad, t_elbow] + ([t_mut_p75] if t_mut_p75 is not None else [])
	dynamic_threshold = round(float(np.median(estimators)), 4)

	result = {
		'threshold': dynamic_threshold,
		't_mad': round(t_mad, 4),
		't_elbow': round(t_elbow, 4),
		't_mutual_p75': round(t_mut_p75, 4) if t_mut_p75 is not None else None,
		'sigma_mad': round(sigma_mad, 4),
	}

	if verbose:
		print(f"  └─ {result}")

	return result

def get_cluster_medoid_nearest_neighbor(
	embedding_model_id: str,
	X: np.ndarray,
	labels: np.ndarray,
	label_texts: List[str],
	max_merged_size: Optional[int] = None,
	threshold: Optional[float] = None,
	review_path: Optional[str] = None,
	verbose: bool = False,
) -> Tuple[Dict[int, Dict], Dict]:
	"""
	Pre-merge cluster proximity diagnostics computed directly on member centroids
	after `get_clustered_labels()` and before `merge_cluster_medoid_nearest_neighbor()`.

	If `threshold` is None (default), it is derived dynamically from the
	clusters via three intrinsic signals (robust MAD fence, Kneedle upper-tail
	elbow, and the 75th percentile of mutual 1-NN similarities).

	Returns
	-------
	info    : {cid: {cluster_id, similarity, mutual, merge_eligible, medoid, members}}
				describing each cluster's nearest other cluster.
	summary : distribution quantiles, counts at similarity levels (overall, mutual,
				one-way, and merge-eligible mutual pairs), and resolved threshold.
	"""
	ids = [int(c) for c in np.unique(labels)]
	n_clusters = len(ids)

	cluster_members: Dict[int, List[str]] = {}
	cluster_medoids: Dict[int, str] = {}
	cluster_medoid_sims: Dict[int, float] = {}
	sizes = np.zeros(n_clusters, dtype=np.int64)
	centroid_list = []

	for pos, c in enumerate(ids):
		m_idx = np.where(labels == c)[0]
		members_c = [label_texts[i] for i in m_idx]
		cluster_members[c] = members_c
		sizes[pos] = len(m_idx)

		emb_c = X[m_idx]
		centroid = emb_c.mean(axis=0).astype(np.float32)
		centroid_list.append(centroid)

		unit_c = centroid / (np.linalg.norm(centroid) + 1e-12)
		sims_to_centroid = emb_c @ unit_c
		medoid_local = int(np.argmax(sims_to_centroid))
		cluster_medoids[c] = members_c[medoid_local]

		# Average similarity of the medoid to all OTHER members in the cluster
		if len(m_idx) > 1:
			medoid_emb = emb_c[medoid_local]
			medoid_sims_to_others = emb_c @ medoid_emb
			# Exclude self-similarity (which equals 1.0)
			avg_medoid_sim = float((medoid_sims_to_others.sum() - 1.0) / (len(m_idx) - 1))
		else:
			avg_medoid_sim = 1.0
		cluster_medoid_sims[c] = round(avg_medoid_sim, 4)

	centroids = np.vstack(centroid_list) if n_clusters > 0 else np.empty((0, X.shape[1]), dtype=np.float32)
	nn_idx, nn_sim = _nearest_cluster_neighbors(centroids=centroids, chunk=4096)

	info: Dict[int, Dict] = {}
	for pos, c in enumerate(ids):
		npos = int(nn_idx[pos])
		n = ids[npos]
		sim = float(nn_sim[pos])
		is_mutual = bool(int(nn_idx[npos]) == pos)
		comb_size = int(sizes[pos] + sizes[npos])
		merge_ok = bool(is_mutual and (max_merged_size is None or comb_size <= max_merged_size))
		info[c] = {
			'cluster_id': int(n),
			'similarity': sim,
			'mutual': is_mutual,
			'combined_size': comb_size,
			'merge_eligible': merge_ok,
			'medoid': cluster_medoids[n],
			'medoid_similarity': cluster_medoid_sims[n],
			'members': cluster_members[n][:4],
		}

	sims = nn_sim.astype(float)
	mutual_mask = np.array([bool(info[c]['mutual']) for c in ids], dtype=bool)
	oneway_mask = ~mutual_mask
	eligible_mask = np.array([bool(info[c]['merge_eligible']) for c in ids], dtype=bool)

	dyn = _estimate_dynamic_neighbor_thresholds(sims, mutual_mask, verbose=verbose)
	is_dynamic_thresh = threshold is None
	active_threshold = float(dyn['threshold'] if is_dynamic_thresh else threshold)

	pairs = {}
	for pos, c in enumerate(ids):
		n = info[c]['cluster_id']
		sim = float(nn_sim[pos])
		if sim >= active_threshold:
			pairs[(min(c, n), max(c, n))] = sim

	levels = [0.98, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80]
	summary = {
		'n_clusters': n_clusters,
		'threshold': round(active_threshold, 4),
		'threshold_mode': 'dynamic' if is_dynamic_thresh else 'manual',
		'max_merged_size': int(max_merged_size) if max_merged_size is not None else None,
		'dynamic_thresholds': dyn,
		'min_nearest_similarity': round(float(np.min(sims)), 4) if n_clusters else 0.0,
		'p25_nearest_similarity': round(float(np.percentile(sims, 25)), 4) if n_clusters else 0.0,
		'median_nearest_similarity': round(float(np.median(sims)), 4) if n_clusters else 0.0,
		'p75_nearest_similarity': round(float(np.percentile(sims, 75)), 4) if n_clusters else 0.0,
		'p90_nearest_similarity': round(float(np.percentile(sims, 90)), 4) if n_clusters else 0.0,
		'p95_nearest_similarity': round(float(np.percentile(sims, 95)), 4) if n_clusters else 0.0,
		'max_nearest_similarity': round(float(np.max(sims)), 4) if n_clusters else 0.0,
		'nearest_at_or_above': {f'{t:.2f}': int((sims >= t).sum()) for t in levels},
		'nearest_at_or_above_mutual': {f'{t:.2f}': int(((sims >= t) & mutual_mask).sum()) for t in levels},
		'nearest_at_or_above_oneway': {f'{t:.2f}': int(((sims >= t) & oneway_mask).sum()) for t in levels},
		'mergeable_pairs_at_or_above': {f'{t:.2f}': int(((sims >= t) & eligible_mask).sum()) // 2 for t in levels},
	}

	if review_path:
		def _side(c):
			return {
				'cluster_id': int(c),
				'size': len(cluster_members[c]),
				'medoid': cluster_medoids[c],
				'medoid_similarity': cluster_medoid_sims[c],
				'members': cluster_members[c]
			}

		review = []
		for (a, b), s in sorted(pairs.items(), key=lambda kv: -kv[1]):
			a_to_b = bool(info[a]['cluster_id'] == b)
			b_to_a = bool(info[b]['cluster_id'] == a)
			is_mutual = bool(a_to_b and b_to_a)
			comb_sz = int(len(cluster_members[a]) + len(cluster_members[b]))
			merge_ok = bool(is_mutual and (max_merged_size is None or comb_sz <= max_merged_size))

			record = {
				'similarity': float(s),
				'mutual': is_mutual,
				'a_to_b': a_to_b,
				'b_to_a': b_to_a,
			}

			if not a_to_b:
				pref_id = int(info[a]['cluster_id'])
				pref_sim = float(info[a]['similarity'])
				record['a_prefers'] = {
					'cluster_id': pref_id,
					'medoid': cluster_medoids[pref_id],
					'similarity': pref_sim,
					'margin_over_b': float(round(pref_sim - float(s), 7)),
				}

			if not b_to_a:
				pref_id = int(info[b]['cluster_id'])
				pref_sim = float(info[b]['similarity'])
				record['b_prefers'] = {
					'cluster_id': pref_id,
					'medoid': cluster_medoids[pref_id],
					'similarity': pref_sim,
					'margin_over_a': float(round(pref_sim - float(s), 6)),
				}

			record.update({
				'merge_eligible': merge_ok,
				'combined_size': comb_sz,
				'a': _side(a),
				'b': _side(b),
			})
			review.append(record)

		with open(review_path, 'w', encoding='utf-8') as f:
			pairs_results = {
				'embedding': embedding_model_id,
				'threshold': float(active_threshold),
				'threshold_mode': summary['threshold_mode'],
				'max_merged_size': summary['max_merged_size'],
				'dynamic_thresholds': dyn,
				'n_pairs': len(review),
				'n_merge_eligible_pairs': sum(1 for r in review if r['merge_eligible']),
				'pairs': review,
			}
			json.dump(pairs_results, f, indent=2, ensure_ascii=False, default=_json_default)

	if verbose:
		n_mutual_total = int(mutual_mask.sum())
		n_eligible_total = int(eligible_mask.sum()) // 2
		size_limit_str = f"<= max_merged_size={max_merged_size}" if max_merged_size is not None else "no size limit"
		size_header_str = f"Mergeable Pairs (<={max_merged_size})" if max_merged_size is not None else "Mergeable Pairs (all mutual)"
		print("\n" + "=" * 120)
		print("[CLUSTER NEIGHBOURS] Pre-Merge 1-Nearest-Neighbor Centroid Cosine Similarity Diagnostics")
		print(
			f"Total Clusters: {n_clusters:,} | "
			f"Mutual 1-NN: {n_mutual_total:,}/{n_clusters:,} ({100.0 * n_mutual_total / max(n_clusters, 1):.1f}%) "
			f"[{n_mutual_total // 2:,} mutual pairs, {n_eligible_total:,} pairs ({size_limit_str})]"
		)
		if n_clusters > 1:
			print(
				f"Distribution: min={summary['min_nearest_similarity']:.3f} | "
				f"p25={summary['p25_nearest_similarity']:.3f} | "
				f"median={summary['median_nearest_similarity']:.3f} | "
				f"p75={summary['p75_nearest_similarity']:.3f} | "
				f"p90={summary['p90_nearest_similarity']:.3f} | "
				f"p95={summary['p95_nearest_similarity']:.3f} | "
				f"max={summary['max_nearest_similarity']:.3f}"
			)
			t_mut_str = f"{dyn['t_mutual_p75']:.4f}" if dyn['t_mutual_p75'] is not None else "N/A (<5 mutual)"
			print(
				f"Intrinsic Estimators: "
				f"MAD fence={dyn['t_mad']:.4f} (med+1.0*MAD, σ={dyn['sigma_mad']:.4f}) | "
				f"Kneedle elbow={dyn['t_elbow']:.4f} | "
				f"Mutual 1-NN p75={t_mut_str}"
			)
			print(f"Active Threshold: {active_threshold:.4f} ({summary['threshold_mode']})")
		print("-" * 120)
		print(
			f"  {'>= Sim':<8} {'All Clusters':>16} {'Mutual 1-NN':>18} {'One-Way 1-NN':>18} "
			f"{size_header_str:>28}  {'Intrinsic Landmark / Role'}"
		)
		print(f"  {'-'*8:<8} {'-'*16:>16} {'-'*18:>18} {'-'*18:>18} {'-'*28:>28}  {'-'*32}")

		raw_rows = [(t, None) for t in levels]
		if n_clusters > 1:
			raw_rows.extend([
				(summary['p95_nearest_similarity'], 'p95 (top 5% tail)'),
				(dyn['t_elbow'], 'kneedle elbow'),
				(summary['p90_nearest_similarity'], 'p90 (top 10% tail)'),
				(active_threshold, f"ACTIVE threshold ({summary['threshold_mode']})"),
				(dyn['t_mad'], 'median + 1.0*MAD'),
				(summary['p75_nearest_similarity'], 'p75 (upper quartile)'),
				(summary['median_nearest_similarity'], 'p50 (median)'),
			])
			if dyn['t_mutual_p75'] is not None:
				raw_rows.append((dyn['t_mutual_p75'], 'mutual 1-NN p75'))

		merged_rows = []
		for val, tag in sorted(raw_rows, key=lambda r: -r[0]):
			if val is None:
				continue
			val_r = round(float(val), 4)
			if merged_rows and abs(merged_rows[-1][0] - val_r) < 1e-6:
				if tag and tag not in merged_rows[-1][1]:
					merged_rows[-1][1].append(tag)
			else:
				merged_rows.append((val_r, [tag] if tag else []))

		for val_r, tags in merged_rows:
			n_all = int((sims >= val_r).sum())
			n_mut = int(((sims >= val_r) & mutual_mask).sum())
			n_one = int(((sims >= val_r) & oneway_mask).sum())
			n_mrg_pairs = int(((sims >= val_r) & eligible_mask).sum()) // 2
			pct_all = 100.0 * n_all / max(n_clusters, 1)
			pct_mut = 100.0 * n_mut / max(n_clusters, 1)
			pct_one = 100.0 * n_one / max(n_clusters, 1)
			tag_str = f"<- {', '.join(tags)}" if tags else ""
			print(
				f"  {val_r:<8.4f} {f'{n_all:,} ({pct_all:5.1f}%)':>16} "
				f"{f'{n_mut:,} ({pct_mut:5.1f}%)':>18} {f'{n_one:,} ({pct_one:5.1f}%)':>18} "
				f"{f'{n_mrg_pairs:,} pairs':>28}  {tag_str}"
			)

		print("-" * 120)

		if pairs:
			n_cand_clusters = int((sims >= active_threshold).sum())
			n_mut_pairs = sum(1 for (a, b) in pairs if info[a]['cluster_id'] == b and info[b]['cluster_id'] == a)

			n_mrg_pairs = sum(
				1 for (a, b) in pairs
				if info[a]['cluster_id'] == b
				and info[b]['cluster_id'] == a
				and (max_merged_size is None or len(cluster_members[a]) + len(cluster_members[b]) <= max_merged_size)
			)

			n_size_blocked = n_mut_pairs - n_mrg_pairs
			n_asym_pairs = len(pairs) - n_mut_pairs
			size_note = f" (all {n_mut_pairs:,} mutual pairs mergeable; no size limit)" if max_merged_size is None else f" ({n_size_blocked:,} mutual pairs blocked by size > {max_merged_size})"
			
			print(f"  Candidate Pairs at Active Threshold (similarity >= {active_threshold:.4f} [{summary['threshold_mode']}]):")
			print(f"    • Unique pairs extracted {len(pairs):,} ({n_mut_pairs:,} mutual 1-NN, {n_asym_pairs:,} one-way 1-NN)")
			print(f"    • Step 4b merge-eligible {n_mrg_pairs:,} mutual pairs{size_note}")
			print(f"    • Individual clusters    {n_cand_clusters:,} ({100.0 * n_cand_clusters / max(n_clusters, 1):.1f}% of total clusters)")

			top_k = min(10, len(pairs))
			sorted_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:top_k]
			print(f"\n  Top {top_k} Closest Neighbor Pairs:")
			for rank, ((a, b), s) in enumerate(sorted_pairs, start=1):
				is_mut = info[a]['cluster_id'] == b and info[b]['cluster_id'] == a
				comb_sz = len(cluster_members[a]) + len(cluster_members[b])

				if is_mut and (max_merged_size is None or comb_sz <= max_merged_size):
					limit_desc = f"<={max_merged_size}" if max_merged_size is not None else "no limit"
					status_tag = f"mutual (A<->B), merge-eligible (combined={comb_sz}, {limit_desc})"
				elif is_mut:
					status_tag = f"mutual (A<->B), size-blocked (combined={comb_sz}>{max_merged_size})"
				elif info[a]['cluster_id'] == b:
					pref_b = int(info[b]['cluster_id'])
					status_tag = f"one-way (A->B, B prefers C{pref_b} '{cluster_medoids[pref_b]}' sim={info[b]['similarity']:.4f})"
				else:
					pref_a = int(info[a]['cluster_id'])
					status_tag = f"one-way (B->A, A prefers C{pref_a} '{cluster_medoids[pref_a]}' sim={info[a]['similarity']:.4f})"
				
				print(f"    {rank}. sim={s:.4f} {status_tag}")
				print(f"       Cluster {a:5d} ({len(cluster_members[a]):3d} labels, medoid: {repr(cluster_medoids[a]):<35} sim={cluster_medoid_sims[a]:.4f}) {cluster_members[a][:7]}")
				print(f"       Cluster {b:5d} ({len(cluster_members[b]):3d} labels, medoid: {repr(cluster_medoids[b]):<35} sim={cluster_medoid_sims[b]:.4f}) {cluster_members[b][:7]}")
				print()

		if review_path:
			print(f"\n[SAVED] {review_path}")
		print("=" * 120 + "\n")

	return summary

def merge_cluster_medoid_nearest_neighbor(
	X: np.ndarray,
	labels: np.ndarray,
	label_texts: List[str],
	threshold: float,
	max_merged_size: Optional[int] = None,
	max_rounds: int = 10,
	report_path: Optional[str]=None,
	verbose: bool = False,
) -> np.ndarray:

	"""
	Second, conservative pass over the clusters produced by the tree cut.

	Why: the tree is cut into a fixed number of clusters (about one per five
	labels), so a concept with 11 labels ('reconnaissance aircraft', 'recon
	plane', ...) can be split along its cheapest line and end up as two
	clusters with two different names. Two clusters are merged only when

		* they are mutual nearest neighbours (each is the other's closest cluster),
		* their centroid cosine similarity is >= threshold, and
		* (if max_merged_size is given) the merged cluster would hold at most max_merged_size labels.

	Merged centroids are recomputed (size-weighted) and the process repeats for
	up to max_rounds rounds. Because each round compares the RECOMPUTED centroid,
	chains (A~B, B~C, A far from C) cannot form: after A+B merge, the merged
	centroid must itself be close to C.

	Returns new labels (contiguous, same row order).
	"""

	ids = np.unique(labels)
	cluster_ids = [int(c) for c in ids]
	members = [np.where(labels == c)[0] for c in ids]
	initial_sizes = np.array([len(m) for m in members])
	sums = np.vstack([X[m].sum(axis=0) for m in members]).astype(np.float32)
	sizes = np.array([len(m) for m in members])
	n_before = len(members)
	merges = []
	round_progression = []
	termination_reason = "max_rounds_reached"

	def _get_medoid_info(m_indices: np.ndarray, sum_vec: np.ndarray) -> Tuple[str, float]:
		if len(m_indices) == 0:
			return "", 0.0
		emb = X[m_indices]
		norm = np.linalg.norm(sum_vec)
		unit = sum_vec / (norm + 1e-12)
		sims = emb @ unit
		local_idx = int(np.argmax(sims))
		medoid_text = label_texts[m_indices[local_idx]]
		if len(m_indices) > 1:
			med_emb = emb[local_idx]
			med_sims = emb @ med_emb
			avg_sim = float((med_sims.sum() - 1.0) / (len(m_indices) - 1))
		else:
			avg_sim = 1.0
		return medoid_text, round(avg_sim, 4)

	for rnd in range(1, max_rounds + 1):
		if len(members) < 2:
			termination_reason = "insufficient_clusters"
			break

		nn_idx, nn_sim = _nearest_cluster_neighbors(centroids=sums, chunk=4096)
		drop = set()
		round_merges = 0
		round_sims = []
		clusters_in_round = len(members)

		for i in range(len(members)):
			j = int(nn_idx[i])
			
			if not (i < j and int(nn_idx[j]) == i):
				continue # mutual pairs only: each cluster is in at most one pair
			
			if nn_sim[i] < threshold:
				continue
			if max_merged_size is not None and sizes[i] + sizes[j] > max_merged_size:
				continue

			sim_val = float(nn_sim[i])
			round_sims.append(sim_val)
			round_merges += 1

			# Compute medoids for cluster A, cluster B, and the merged cluster
			merged_indices = np.concatenate([members[i], members[j]])
			merged_sum = sums[i] + sums[j]
			med_a, med_a_sim = _get_medoid_info(members[i], sums[i])
			med_b, med_b_sim = _get_medoid_info(members[j], sums[j])
			med_merged, med_merged_sim = _get_medoid_info(merged_indices, merged_sum)

			members_a_texts = [label_texts[k] for k in members[i]]
			members_b_texts = [label_texts[k] for k in members[j]]

			merge = {
				'merge_id': len(merges) + 1,
				'round': rnd,
				'similarity': round(sim_val, 6),
				'cluster_a': {
					'id': cluster_ids[i],
					'size': int(sizes[i]),
					'medoid': med_a,
					'medoid_similarity': med_a_sim,
					'members': members_a_texts,
				},
				'cluster_b': {
					'id': cluster_ids[j],
					'size': int(sizes[j]),
					'medoid': med_b,
					'medoid_similarity': med_b_sim,
					'members': members_b_texts,
				},
				'merged': {
					'size': int(sizes[i] + sizes[j]),
					'medoid': med_merged,
					'medoid_similarity': med_merged_sim,
					'members': members_a_texts + members_b_texts,
				},
			}

			merges.append(merge)
			
			members[i] = merged_indices
			sums[i] = merged_sum
			sizes[i] += sizes[j]
			# cluster_ids[i] remains the target cluster id
			drop.add(j)
		
		if not drop:
			termination_reason = "converged_no_more_pairs"
			break

		round_progression.append({
			'round': rnd,
			'clusters_in': clusters_in_round,
			'merges': round_merges,
			'clusters_out': clusters_in_round - round_merges,
			'min_similarity': round(float(min(round_sims)), 6) if round_sims else None,
			'median_similarity': round(float(np.median(round_sims)), 6) if round_sims else None,
			'max_similarity': round(float(max(round_sims)), 6) if round_sims else None,
		})
		
		keep = [k for k in range(len(members)) if k not in drop]
		members = [members[k] for k in keep]
		cluster_ids = [cluster_ids[k] for k in keep]
		sums = sums[keep]
		sizes = sizes[keep]

	new_labels = np.empty_like(labels)
	for new_id, m in enumerate(members):
		new_labels[m] = new_id

	rounds_completed = len(round_progression)
	final_sizes = np.array([len(m) for m in members])
	all_sims = [m['similarity'] for m in merges]

	if all_sims:
		merge_sim_stats = {
			'count': len(all_sims),
			'min': round(float(np.min(all_sims)), 4),
			'p25': round(float(np.percentile(all_sims, 25)), 4),
			'median': round(float(np.median(all_sims)), 4),
			'mean': round(float(np.mean(all_sims)), 4),
			'p75': round(float(np.percentile(all_sims, 75)), 4),
			'p90': round(float(np.percentile(all_sims, 90)), 4),
			'max': round(float(np.max(all_sims)), 4),
		}
	else:
		merge_sim_stats = {}

	size_stats = {
		'before': {
			'min': int(np.min(initial_sizes)) if len(initial_sizes) > 0 else 0,
			'median': float(np.median(initial_sizes)) if len(initial_sizes) > 0 else 0.0,
			'mean': round(float(np.mean(initial_sizes)), 2) if len(initial_sizes) > 0 else 0.0,
			'p75': float(np.percentile(initial_sizes, 75)) if len(initial_sizes) > 0 else 0.0,
			'p90': float(np.percentile(initial_sizes, 90)) if len(initial_sizes) > 0 else 0.0,
			'max': int(np.max(initial_sizes)) if len(initial_sizes) > 0 else 0,
		},
		'after': {
			'min': int(np.min(final_sizes)) if len(final_sizes) > 0 else 0,
			'median': float(np.median(final_sizes)) if len(final_sizes) > 0 else 0.0,
			'mean': round(float(np.mean(final_sizes)), 2) if len(final_sizes) > 0 else 0.0,
			'p75': float(np.percentile(final_sizes, 75)) if len(final_sizes) > 0 else 0.0,
			'p90': float(np.percentile(final_sizes, 90)) if len(final_sizes) > 0 else 0.0,
			'max': int(np.max(final_sizes)) if len(final_sizes) > 0 else 0,
		}
	}

	if verbose:
		size_limit_str = f"max_size={max_merged_size}" if max_merged_size is not None else "no size limit"
		reduction_count = n_before - len(members)
		reduction_pct = (1.0 - len(members) / max(1, n_before)) * 100.0

		print("=" * 120)
		print(f"[MERGE CLUSTERS (medoid similarity)] Conservative Multi-Round 1-NN Mutual Pair Merging Diagnostics")
		print(f"Policy: threshold >= {threshold:.5f} | Size Limit: {size_limit_str} | Max Rounds: {max_rounds}")
		print(
			f"Reduction: {n_before:,} -> {len(members):,} clusters "
			f"(-{reduction_count:,} clusters, -{reduction_pct:.2f}%) "
			f"across {rounds_completed} round(s) (status: {termination_reason})"
		)
		print("-" * 120)

		if round_progression:
			print("Round-by-Round Progression:")
			print(
				f"  {'Round':<9} {'Clusters In':>13} {'Merges Done':>13} {'Clusters Out':>14} "
				f"{'Sim Min':>12} {'Sim Med':>12} {'Sim Max':>12}"
			)
			print(
				f"  {'-'*9} {'-'*13} {'-'*13} {'-'*14} {'-'*12} {'-'*12} {'-'*12}"
			)
			for rp in round_progression:
				sim_min_s = f"{rp['min_similarity']:.4f}" if rp['min_similarity'] is not None else "-"
				sim_med_s = f"{rp['median_similarity']:.4f}" if rp['median_similarity'] is not None else "-"
				sim_max_s = f"{rp['max_similarity']:.4f}" if rp['max_similarity'] is not None else "-"
				print(
					f"  Round {rp['round']:<3} {rp['clusters_in']:>13,} {rp['merges']:>13,} {rp['clusters_out']:>14,} "
					f"{sim_min_s:>12} {sim_med_s:>12} {sim_max_s:>12}"
				)
			print("-" * 120)

		print("Cluster Size Distributions:")
		print(
			f"  Before Merge: min={size_stats['before']['min']} | median={size_stats['before']['median']:.1f} | "
			f"mean={size_stats['before']['mean']:.2f} | p75={size_stats['before']['p75']:.1f} | "
			f"p90={size_stats['before']['p90']:.1f} | max={size_stats['before']['max']}"
		)
		print(
			f"  After Merge : min={size_stats['after']['min']} | median={size_stats['after']['median']:.1f} | "
			f"mean={size_stats['after']['mean']:.2f} | p75={size_stats['after']['p75']:.1f} | "
			f"p90={size_stats['after']['p90']:.1f} | max={size_stats['after']['max']}"
		)

		if merge_sim_stats:
			print(
				f"\nMerge Similarity Percentiles (N={merge_sim_stats['count']:,}):\n"
				f"  min={merge_sim_stats['min']:.4f} "
				f"p25={merge_sim_stats['p25']:.4f} "
				f"median={merge_sim_stats['median']:.4f} "
				f"mean={merge_sim_stats['mean']:.4f} "
				f"p75={merge_sim_stats['p75']:.4f} "
				f"p90={merge_sim_stats['p90']:.4f} "
				f"max={merge_sim_stats['max']:.4f}"
			)

		if merges:
			topk = min(10, len(merges))
			topk_confident_merges = sorted(merges, key=lambda m: m['similarity'], reverse=True)[:topk]
			topk_borderline_merges = sorted(merges, key=lambda m: m['similarity'], reverse=False)[:topk]
			
			print(f"\nTop-{topk} Closest Merges (Highest Similarity / Highest Confidence):")
			for rank, m in enumerate(topk_confident_merges, 1):
				comb_sz = m['merged']['size']
				ca, cb = m['cluster_a'], m['cluster_b']
				print(
					f"  {rank}. sim={m['similarity']:.4f} (Rnd {m['round']}) | Combined size: {comb_sz} ({ca['size']} + {cb['size']})\n"
					f"     Cluster {ca['id']:5d} medoid: {ca['medoid']:<35} {ca['members'][:3]}\n"
					f"     Cluster {cb['id']:5d} medoid: {cb['medoid']:<35} {cb['members'][:3]}\n"
					f"     -> Merged medoid: {m['merged']['medoid']}\n"
				)

			print(f"\nTop-{topk} Borderline Merges (Near Cutoff Threshold):")
			for rank, m in enumerate(topk_borderline_merges, 1):
				comb_sz = m['merged']['size']
				ca, cb = m['cluster_a'], m['cluster_b']
				print(
					f"  {rank}. sim={m['similarity']:.4f} (Rnd {m['round']}) | Combined size: {comb_sz} ({ca['size']} + {cb['size']})\n"
					f"     Cluster {ca['id']:5d} medoid: {ca['medoid']:<35} {ca['members'][:3]}\n"
					f"     Cluster {cb['id']:5d} medoid: {cb['medoid']:<35} {cb['members'][:3]}\n"
					f"     -> Merged medoid: {m['merged']['medoid']}\n"
				)

			print(f"  (lowest-similarity merges shown; borderline merges to review) cutoff threshold: {threshold:.6f}")

		if report_path:
			print(f"\n[SAVED] {report_path}")
		print("=" * 120 + "\n")

	if report_path:
		with open(report_path, 'w', encoding='utf-8') as f:
			new_result = {
				'meta': {
					'threshold': threshold,
					'max_merged_size': max_merged_size,
					'max_rounds': max_rounds,
					'rounds_completed': rounds_completed,
					'termination_reason': termination_reason,
					'clusters_before': n_before,
					'clusters_after': len(members),
					'total_merges': len(merges),
					'reduction_pct': round((1.0 - len(members) / max(1, n_before)) * 100, 2),
					'round_progression': round_progression,
					'cluster_size_stats': size_stats,
					'merge_similarity_stats': merge_sim_stats,
				},
				'threshold': threshold,
				'max_merged_size': max_merged_size,
				'clusters_before': n_before,
				'clusters_after': len(members),
				'rounds': rounds_completed,
				'merges': merges,
			}

			json.dump(new_result, f, indent=2, ensure_ascii=False, default=_json_default)
	
	return new_labels

def _build_canonical_embedding_lookup(
	cluster_canonicals: Dict[int, Dict],
	label_texts: List[str],
	X: np.ndarray,
	model,
	batch_size: int = 128,
	encode_prompt: Optional[str] = None,
	verbose: bool = False,
) -> Dict[str, np.ndarray]:
	"""
	Build a unit-normalized embedding lookup dictionary for all canonical labels
	(both internal 'canonical' and display 'canonical_harmonized').
	Real canonicals present in `label_texts` slice their vector directly from `X`;
	virtual or harmonized canonicals absent from `label_texts` are batch-encoded once.
	"""
	label_to_idx = {lbl: idx for idx, lbl in enumerate(label_texts)}
	needed: set = set()
	for meta in cluster_canonicals.values():
		c_int = meta.get("canonical")
		c_har = meta.get("canonical_harmonized", c_int)
		if c_int:
			needed.add(c_int)
		if c_har:
			needed.add(c_har)

	unique_canonicals = sorted(needed)
	canonical_emb_map: Dict[str, np.ndarray] = {}
	missing_texts: List[str] = []

	for c_str in unique_canonicals:
		idx = label_to_idx.get(c_str)
		if idx is not None:
			vec = np.asarray(X[idx], dtype=np.float32)
			canonical_emb_map[c_str] = vec / (np.linalg.norm(vec) + 1e-12)
		else:
			missing_texts.append(c_str)

	if missing_texts:
		encoded = _encode_(model, missing_texts, batch_size, encode_prompt)
		for c_str, vec in zip(missing_texts, encoded):
			v = np.asarray(vec, dtype=np.float32)
			canonical_emb_map[c_str] = v / (np.linalg.norm(v) + 1e-12)

	if verbose:
		n_in_x = len(unique_canonicals) - len(missing_texts)
		print(
			f"\n[CANONICAL EMBEDDING LOOKUP] {len(cluster_canonicals):,} clusters -> "
			f"{len(unique_canonicals):,} unique canonical strings "
			f"({n_in_x:,} indexed in X, {len(missing_texts):,} virtual/harmonized encoded on-the-fly)"
		)

	return canonical_emb_map

def _nearest_canonical_neighbors(
	canon_embs: np.ndarray,
	unit_centroids: np.ndarray,
	chunk: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	For every cluster row (n, d): find the index of the most similar OTHER cluster
	in canonical embedding space, breaking exact canonical ties (e.g. identical
	canonical strings with sim=1.0) via member-centroid cosine similarity.

	Returns
	-------
	nn_idx          : (n,) int64 index of nearest neighbor cluster
	nn_canon_sim    : (n,) float32 pure canonical-to-canonical cosine similarity
	nn_centroid_sim : (n,) float32 member-centroid cosine similarity for that pair
	"""
	C = np.asarray(canon_embs, dtype=np.float32)
	C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

	U = np.asarray(unit_centroids, dtype=np.float32)
	U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

	n = C.shape[0]
	nn_idx = np.zeros(n, dtype=np.int64)
	nn_canon_sim = np.full(n, -1.0, dtype=np.float32)
	nn_centroid_sim = np.full(n, -1.0, dtype=np.float32)

	if n < 2:
		return nn_idx, nn_canon_sim, nn_centroid_sim

	for s in range(0, n, chunk):
		S_canon = C[s:s + chunk] @ C.T
		S_cent = U[s:s + chunk] @ U.T
		# Tiny centroid tie-breaker resolves identical canonical strings (sim=1.0)
		# in favor of the cluster with the closest member centroid.
		S_score = S_canon + 1e-6 * S_cent
		rows = np.arange(S_score.shape[0])
		S_score[rows, rows + s] = -np.inf
		j = S_score.argmax(axis=1)
		nn_idx[s:s + chunk] = j
		nn_canon_sim[s:s + chunk] = np.clip(S_canon[rows, j], -1.0, 1.0)
		nn_centroid_sim[s:s + chunk] = np.clip(S_cent[rows, j], -1.0, 1.0)

	return nn_idx, nn_canon_sim, nn_centroid_sim

def get_cluster_canonical_nearest_neighbor(
	embedding_model_id: str,
	X: np.ndarray,
	labels: np.ndarray,
	label_texts: List[str],
	cluster_canonicals: Dict[int, Dict],
	canonical_emb_map: Dict[str, np.ndarray],
	min_centroid_similarity: float,
	max_merged_size: Optional[int] = None,
	threshold: Optional[float] = None,
	review_path: Optional[str] = None,
	verbose: bool = False,
) -> Dict[str, Any]:
	"""
	Post-canonical cluster proximity diagnostics computed directly on per-cluster
	canonical label embeddings (after `assign_canonical_labels()` and before
	`merge_cluster_canonical_nearest_neighbor()`).

	If `threshold` is None (default), it is derived dynamically from the
	non-identical canonical 1-NN similarity distribution via three intrinsic
	signals (MAD fence, Kneedle elbow, and Mutual 1-NN p75).
	"""
	ids = [int(c) for c in np.unique(labels)]
	n_clusters = len(ids)

	cluster_members: Dict[int, List[str]] = {}
	cluster_medoids: Dict[int, str] = {}
	cluster_medoid_sims: Dict[int, float] = {}
	cluster_canon_strs: Dict[int, str] = {}
	cluster_is_virtual: Dict[int, bool] = {}
	cluster_canon_cent_sims: Dict[int, float] = {}
	cluster_canon_mem_sims: Dict[int, float] = {}

	sizes = np.zeros(n_clusters, dtype=np.int64)
	unit_centroid_list = []
	canon_emb_list = []

	for pos, c in enumerate(ids):
		m_idx = np.where(labels == c)[0]
		members_c = [label_texts[i] for i in m_idx]
		cluster_members[c] = members_c
		sizes[pos] = len(m_idx)

		emb_c = X[m_idx]
		centroid = emb_c.mean(axis=0).astype(np.float32)
		unit_c = centroid / (np.linalg.norm(centroid) + 1e-12)
		unit_centroid_list.append(unit_c)

		sims_to_centroid = emb_c @ unit_c
		medoid_local = int(np.argmax(sims_to_centroid))
		cluster_medoids[c] = members_c[medoid_local]

		if len(m_idx) > 1:
			medoid_emb = emb_c[medoid_local]
			medoid_sims_to_others = emb_c @ medoid_emb
			avg_medoid_sim = float((medoid_sims_to_others.sum() - 1.0) / (len(m_idx) - 1))
		else:
			avg_medoid_sim = 1.0
		cluster_medoid_sims[c] = round(avg_medoid_sim, 4)

		meta = cluster_canonicals.get(c, {})
		canon_str = meta.get("canonical_harmonized") or meta.get("canonical") or cluster_medoids[c]
		is_virt = bool(meta.get("virtual", False))
		cluster_canon_strs[c] = canon_str
		cluster_is_virtual[c] = is_virt

		c_emb = canonical_emb_map[canon_str]
		canon_emb_list.append(c_emb)
		cluster_canon_cent_sims[c] = round(float(c_emb @ unit_c), 4)
		cluster_canon_mem_sims[c] = round(float((emb_c @ c_emb).mean()), 4)

	unit_centroids = np.vstack(unit_centroid_list) if n_clusters > 0 else np.empty((0, X.shape[1]), dtype=np.float32)
	canon_embs = np.vstack(canon_emb_list) if n_clusters > 0 else np.empty((0, X.shape[1]), dtype=np.float32)

	nn_idx, nn_canon_sim, nn_centroid_sim = _nearest_canonical_neighbors(
		canon_embs=canon_embs,
		unit_centroids=unit_centroids,
		chunk=4096,
	)

	info: Dict[int, Dict] = {}
	for pos, c in enumerate(ids):
		npos = int(nn_idx[pos])
		n = ids[npos]
		c_sim = float(nn_canon_sim[pos])
		cent_sim = float(nn_centroid_sim[pos])
		is_mutual = bool(int(nn_idx[npos]) == pos)
		comb_size = int(sizes[pos] + sizes[npos])
		size_ok = bool(max_merged_size is None or comb_size <= max_merged_size)
		centroid_ok = bool(cent_sim >= min_centroid_similarity)
		merge_ok = bool(is_mutual and size_ok and centroid_ok)
		info[c] = {
			'cluster_id': int(n),
			'canonical_similarity': c_sim,
			'centroid_similarity': cent_sim,
			'same_canonical_string': bool(cluster_canon_strs[c].lower() == cluster_canon_strs[n].lower()),
			'mutual': is_mutual,
			'combined_size': comb_size,
			'centroid_guard_passed': centroid_ok,
			'merge_eligible': merge_ok,
			'canonical': cluster_canon_strs[n],
			'is_virtual': cluster_is_virtual[n],
			'medoid': cluster_medoids[n],
			'medoid_similarity': cluster_medoid_sims[n],
			'members': cluster_members[n][:4],
		}

	sims = nn_canon_sim.astype(float)
	cent_sims = nn_centroid_sim.astype(float)
	mutual_mask = np.array([bool(info[c]['mutual']) for c in ids], dtype=bool)
	oneway_mask = ~mutual_mask
	eligible_mask = np.array([bool(info[c]['merge_eligible']) for c in ids], dtype=bool)

	dyn = _estimate_dynamic_neighbor_thresholds(sims, mutual_mask, verbose=verbose)

	is_dynamic_thresh = threshold is None
	active_threshold = float(dyn['threshold'] if is_dynamic_thresh else threshold)

	pairs: Dict[Tuple[int, int], Tuple[float, float]] = {}
	for pos, c in enumerate(ids):
		n = info[c]['cluster_id']
		c_sim = float(nn_canon_sim[pos])
		cent_sim = float(nn_centroid_sim[pos])
		if c_sim >= active_threshold:
			pairs[(min(c, n), max(c, n))] = (c_sim, cent_sim)

	levels = [0.98, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80]
	n_unique_canon_before = len({s.lower() for s in cluster_canon_strs.values()})
	summary = {
		'n_clusters': n_clusters,
		'n_unique_canonicals': n_unique_canon_before,
		'threshold': round(active_threshold, 4),
		'threshold_mode': 'dynamic' if is_dynamic_thresh else 'manual',
		'min_centroid_similarity': round(float(min_centroid_similarity), 4),
		'max_merged_size': int(max_merged_size) if max_merged_size is not None else None,
		'dynamic_thresholds': dyn,
		'min_nearest_similarity': round(float(np.min(sims)), 4) if n_clusters else 0.0,
		'p25_nearest_similarity': round(float(np.percentile(sims, 25)), 4) if n_clusters else 0.0,
		'median_nearest_similarity': round(float(np.median(sims)), 4) if n_clusters else 0.0,
		'p75_nearest_similarity': round(float(np.percentile(sims, 75)), 4) if n_clusters else 0.0,
		'p90_nearest_similarity': round(float(np.percentile(sims, 90)), 4) if n_clusters else 0.0,
		'p95_nearest_similarity': round(float(np.percentile(sims, 95)), 4) if n_clusters else 0.0,
		'max_nearest_similarity': round(float(np.max(sims)), 4) if n_clusters else 0.0,
		'nearest_at_or_above': {f'{t:.2f}': int((sims >= t).sum()) for t in levels},
		'nearest_at_or_above_mutual': {f'{t:.2f}': int(((sims >= t) & mutual_mask).sum()) for t in levels},
		'nearest_at_or_above_oneway': {f'{t:.2f}': int(((sims >= t) & oneway_mask).sum()) for t in levels},
		'mergeable_pairs_at_or_above': {f'{t:.2f}': int(((sims >= t) & eligible_mask).sum()) // 2 for t in levels},
	}

	if review_path:
		def _side(c: int):
			return {
				'cluster_id': int(c),
				'size': len(cluster_members[c]),
				'canonical': cluster_canon_strs[c],
				'is_virtual': cluster_is_virtual[c],
				'canonical_to_centroid_similarity': cluster_canon_cent_sims[c],
				'canonical_to_members_similarity': cluster_canon_mem_sims[c],
				'medoid': cluster_medoids[c],
				'medoid_similarity': cluster_medoid_sims[c],
				'members': cluster_members[c],
			}

		review = []
		for (a, b), (c_sim, cent_sim) in sorted(pairs.items(), key=lambda kv: (-kv[1][0], -kv[1][1])):
			a_to_b = bool(info[a]['cluster_id'] == b)
			b_to_a = bool(info[b]['cluster_id'] == a)
			is_mutual = bool(a_to_b and b_to_a)
			comb_sz = int(len(cluster_members[a]) + len(cluster_members[b]))
			size_ok = bool(max_merged_size is None or comb_sz <= max_merged_size)
			centroid_ok = bool(cent_sim >= min_centroid_similarity)
			merge_ok = bool(is_mutual and size_ok and centroid_ok)
			same_canon = bool(cluster_canon_strs[a].lower() == cluster_canon_strs[b].lower())

			record = {
				'canonical_similarity': round(float(c_sim), 6),
				'centroid_similarity': round(float(cent_sim), 6),
				'same_canonical_string': same_canon,
				'mutual': is_mutual,
				'a_to_b': a_to_b,
				'b_to_a': b_to_a,
			}

			if not a_to_b:
				pref_id = int(info[a]['cluster_id'])
				pref_sim = float(info[a]['canonical_similarity'])
				record['a_prefers'] = {
					'cluster_id': pref_id,
					'canonical': cluster_canon_strs[pref_id],
					'is_virtual': cluster_is_virtual[pref_id],
					'medoid': cluster_medoids[pref_id],
					'canonical_similarity': round(pref_sim, 6),
					'centroid_similarity': round(float(info[a]['centroid_similarity']), 6),
					'margin_over_b': float(round(pref_sim - float(c_sim), 6)),
				}

			if not b_to_a:
				pref_id = int(info[b]['cluster_id'])
				pref_sim = float(info[b]['canonical_similarity'])
				record['b_prefers'] = {
					'cluster_id': pref_id,
					'canonical': cluster_canon_strs[pref_id],
					'is_virtual': cluster_is_virtual[pref_id],
					'medoid': cluster_medoids[pref_id],
					'canonical_similarity': round(pref_sim, 6),
					'centroid_similarity': round(float(info[b]['centroid_similarity']), 6),
					'margin_over_a': float(round(pref_sim - float(c_sim), 6)),
				}

			record.update({
				'merge_eligible': merge_ok,
				'centroid_guard_passed': centroid_ok,
				'size_guard_passed': size_ok,
				'combined_size': comb_sz,
				'a': _side(a),
				'b': _side(b),
			})
			review.append(record)

		with open(review_path, 'w', encoding='utf-8') as f:
			pairs_results = {
				'embedding': embedding_model_id,
				'threshold': float(active_threshold),
				'threshold_mode': summary['threshold_mode'],
				'min_centroid_similarity': float(min_centroid_similarity),
				'max_merged_size': summary['max_merged_size'],
				'dynamic_thresholds': dyn,
				'n_clusters': n_clusters,
				'n_unique_canonicals': n_unique_canon_before,
				'n_pairs': len(review),
				'n_merge_eligible_pairs': sum(1 for r in review if r['merge_eligible']),
				'n_centroid_blocked_pairs': sum(1 for r in review if r['mutual'] and not r['centroid_guard_passed']),
				'pairs': review,
			}
			json.dump(pairs_results, f, indent=2, ensure_ascii=False, default=_json_default)

	if verbose:
		n_mutual_total = int(mutual_mask.sum())
		n_eligible_total = int(eligible_mask.sum()) // 2
		size_limit_str = f"<= max_merged_size={max_merged_size}" if max_merged_size is not None else "no size limit"
		size_header_str = f"Mergeable Pairs (<={max_merged_size})" if max_merged_size is not None else f"Mergeable (cent>={min_centroid_similarity:.2f})"
		print("\n" + "=" * 120)
		print("[CANONICAL NEIGHBOURS] Pre-Merge 1-Nearest-Neighbor Canonical Label Cosine Similarity Diagnostics")
		print(
			f"Total Clusters: {n_clusters:,} (Unique Canonicals: {n_unique_canon_before:,}) | "
			f"Mutual 1-NN: {n_mutual_total:,}/{n_clusters:,} ({100.0 * n_mutual_total / max(n_clusters, 1):.1f}%) "
			f"[{n_mutual_total // 2:,} mutual pairs, {n_eligible_total:,} eligible pairs ({size_limit_str}, centroid_sim>={min_centroid_similarity:.2f})]"
		)
		if n_clusters > 1:
			print(
				f"Distribution: min={summary['min_nearest_similarity']:.3f} | "
				f"p25={summary['p25_nearest_similarity']:.3f} | "
				f"median={summary['median_nearest_similarity']:.3f} | "
				f"p75={summary['p75_nearest_similarity']:.3f} | "
				f"p90={summary['p90_nearest_similarity']:.3f} | "
				f"p95={summary['p95_nearest_similarity']:.3f} | "
				f"max={summary['max_nearest_similarity']:.3f}"
			)
			t_mut_str = f"{dyn['t_mutual_p75']:.4f}" if dyn['t_mutual_p75'] is not None else "N/A (<5 mutual)"
			print(
				f"Intrinsic Estimators: "
				f"MAD fence={dyn['t_mad']:.4f} (med+1.0*MAD, σ={dyn['sigma_mad']:.4f}) | "
				f"Kneedle elbow={dyn['t_elbow']:.4f} | "
				f"Mutual 1-NN p75={t_mut_str}"
			)
			print(
				f"Active Threshold "
				f"canonical_threshold={active_threshold:.4f} ({summary['threshold_mode']}) "
				f"min_centroid_similarity={min_centroid_similarity}"
			)
		print("-" * 120)
		print(
			f"  {'>= Sim':<8} {'All Clusters':>16} {'Mutual 1-NN':>18} {'One-Way 1-NN':>18} "
			f"{size_header_str:>28}  {'Intrinsic Landmark / Role'}"
		)
		print(f"  {'-'*8:<8} {'-'*16:>16} {'-'*18:>18} {'-'*18:>18} {'-'*28:>28}  {'-'*32}")

		raw_rows = [(t, None) for t in levels]
		if n_clusters > 1:
			raw_rows.extend([
				(summary['p95_nearest_similarity'], 'p95 (top 5% tail)'),
				(dyn['t_elbow'], 'kneedle elbow'),
				(summary['p90_nearest_similarity'], 'p90 (top 10% tail)'),
				(active_threshold, f"ACTIVE threshold ({summary['threshold_mode']})"),
				(dyn['t_mad'], 'median + 1.0*MAD'),
				(summary['p75_nearest_similarity'], 'p75 (upper quartile)'),
				(summary['median_nearest_similarity'], 'p50 (median)'),
			])
			if dyn['t_mutual_p75'] is not None:
				raw_rows.append((dyn['t_mutual_p75'], 'mutual 1-NN p75'))

		merged_rows = []
		for val, tag in sorted(raw_rows, key=lambda r: -r[0]):
			if val is None:
				continue
			val_r = round(float(val), 4)
			if merged_rows and abs(merged_rows[-1][0] - val_r) < 1e-6:
				if tag and tag not in merged_rows[-1][1]:
					merged_rows[-1][1].append(tag)
			else:
				merged_rows.append((val_r, [tag] if tag else []))

		for val_r, tags in merged_rows:
			n_all = int((sims >= val_r).sum())
			n_mut = int(((sims >= val_r) & mutual_mask).sum())
			n_one = int(((sims >= val_r) & oneway_mask).sum())
			n_mrg_pairs = int(((sims >= val_r) & eligible_mask).sum()) // 2
			pct_all = 100.0 * n_all / max(n_clusters, 1)
			pct_mut = 100.0 * n_mut / max(n_clusters, 1)
			pct_one = 100.0 * n_one / max(n_clusters, 1)
			tag_str = f"<- {', '.join(tags)}" if tags else ""
			print(
				f"  {val_r:<8.4f} {f'{n_all:,} ({pct_all:5.1f}%)':>16} "
				f"{f'{n_mut:,} ({pct_mut:5.1f}%)':>18} {f'{n_one:,} ({pct_one:5.1f}%)':>18} "
				f"{f'{n_mrg_pairs:,} pairs':>28}  {tag_str}"
			)

		print("-" * 120)

		if pairs:
			n_cand_clusters = int((sims >= active_threshold).sum())
			n_mut_pairs = sum(1 for (a, b) in pairs if info[a]['cluster_id'] == b and info[b]['cluster_id'] == a)
			n_cent_blocked = sum(
				1 for (a, b), (_, cent_s) in pairs.items()
				if info[a]['cluster_id'] == b and info[b]['cluster_id'] == a and cent_s < min_centroid_similarity
			)
			n_mrg_pairs = sum(
				1 for (a, b), (_, cent_s) in pairs.items()
				if info[a]['cluster_id'] == b
				and info[b]['cluster_id'] == a
				and cent_s >= min_centroid_similarity
				and (max_merged_size is None or len(cluster_members[a]) + len(cluster_members[b]) <= max_merged_size)
			)
			n_size_blocked = (n_mut_pairs - n_cent_blocked) - n_mrg_pairs
			n_asym_pairs = len(pairs) - n_mut_pairs
			n_same_canon = sum(1 for (a, b) in pairs if cluster_canon_strs[a].lower() == cluster_canon_strs[b].lower())
			n_synonym_pairs = len(pairs) - n_same_canon

			print(f"Candidate Canonical Pairs at Active Threshold (canonical_similarity >= {active_threshold:.4f} [{summary['threshold_mode']}]):")
			print(f"  ├─ Unique pairs extracted {len(pairs):,} ({n_mut_pairs:,} mutual 1-NN, {n_asym_pairs:,} one-way 1-NN)")
			print(f"  ├─ Lexical breakdown      {n_synonym_pairs:,} distinct synonym pairs, {n_same_canon:,} identical-canonical pairs")
			print(
				f"  ├─ Step 6b merge-eligible {n_mrg_pairs:,} mutual pairs "
				f"({n_cent_blocked:,} blocked by centroid_sim < {min_centroid_similarity:.2f}, "
				f"{n_size_blocked:,} blocked by size)"
			)
			print(f"  └─ Individual clusters    {n_cand_clusters:,} ({100.0 * n_cand_clusters / max(n_clusters, 1):.1f}% of total clusters)")

			# Show top synonym pairs (distinct canonical strings)
			synonym_pairs = [
				((a, b), (c_sim, cent_sim))
				for (a, b), (c_sim, cent_sim) in pairs.items()
				if cluster_canon_strs[a].lower() != cluster_canon_strs[b].lower()
			]

			top_k = min(10, len(synonym_pairs))

			if top_k > 0:
				sorted_syn = sorted(synonym_pairs, key=lambda kv: (-kv[1][0], -kv[1][1]))[:top_k]
				print(f"\n  Top-{top_k} Closest Synonym Canonical Neighbor Pairs (Distinct Canonicals):")

				for rank, ((a, b), (c_sim, cent_sim)) in enumerate(sorted_syn, start=1):
					is_mut = info[a]['cluster_id'] == b and info[b]['cluster_id'] == a
					comb_sz = len(cluster_members[a]) + len(cluster_members[b])
					cent_ok = cent_sim >= min_centroid_similarity
					size_ok = max_merged_size is None or comb_sz <= max_merged_size
					
					if is_mut and cent_ok and size_ok:
						status_tag = f"mutual (A<->B), merge-eligible (combined={comb_sz})"
					elif is_mut and not cent_ok:
						status_tag = f"mutual (A<->B), CENTROID-BLOCKED (centroid_sim={cent_sim:.4f} < {min_centroid_similarity:.2f})"
					elif is_mut:
						status_tag = f"mutual (A<->B), size-blocked (combined={comb_sz}>{max_merged_size})"
					elif info[a]['cluster_id'] == b:
						pref_b = int(info[b]['cluster_id'])
						status_tag = f"one-way (A->B, B prefers C{pref_b} {cluster_canon_strs[pref_b]!r} sim={info[b]['canonical_similarity']:.4f})"
					else:
						pref_a = int(info[a]['cluster_id'])
						status_tag = f"one-way (B->A, A prefers C{pref_a} {cluster_canon_strs[pref_a]!r} sim={info[a]['canonical_similarity']:.4f})"

					v_tag_a = " [VIRTUAL]" if cluster_is_virtual[a] else ""
					v_tag_b = " [VIRTUAL]" if cluster_is_virtual[b] else ""
					
					print(f"  {rank:2d}. canon_sim={c_sim:.4f} centroid_sim={cent_sim:.4f} {status_tag}")
					print(
						f"      Cluster {a:6d} ({len(cluster_members[a]):3d} labels, "
						f"canonical={repr(cluster_canon_strs[a]) + v_tag_a:<34} "
						f"medoid={repr(cluster_medoids[a]):<35}) {cluster_members[a][:5]}"
					)

					print(
						f"      Cluster {b:6d} ({len(cluster_members[b]):3d} labels, "
						f"canonical={repr(cluster_canon_strs[b]) + v_tag_b:<34} "
						f"medoid={repr(cluster_medoids[b]):<35}) {cluster_members[b][:5]}\n"
					)

		if review_path:
			print(f"\n[SAVED] {review_path}")
		print("=" * 120 + "\n")

	return summary

def merge_cluster_canonical_nearest_neighbor(
	X: np.ndarray,
	labels: np.ndarray,
	label_texts: List[str],
	cluster_canonicals: Dict[int, Dict],
	canonical_emb_map: Dict[str, np.ndarray],
	threshold: Optional[float] = None,
	min_centroid_similarity: float = 0.70,
	max_merged_size: Optional[int] = None,
	max_rounds: int = 10,
	report_path: Optional[str] = None,
	verbose: bool = False,
) -> Tuple[np.ndarray, Dict[int, Dict]]:
	"""
	Conservative multi-round pass over clusters using their assigned canonical labels.

	Two clusters A and B are merged in a round only when:
		1. Their canonical labels (cA, cB) are mutual 1-nearest neighbors in canonical embedding space,
		2. Their canonical cosine similarity is >= threshold (derived dynamically if threshold=None),
		3. Their real-member centroid cosine similarity is >= min_centroid_similarity (prevents merging
		   unrelated clusters that happen to share polysemous or surface-similar canonicals), and
		4. (if max_merged_size is set) their combined size is <= max_merged_size.

	Winning Canonical Selection (Centroid Closeness):
		When A and B merge into AB, the combined real-member centroid u_AB is computed from X.
		All canonical candidates accumulated from the merging clusters (cA, cB, ...) are scored
		by cosine similarity against u_AB. The canonical label closest to u_AB wins (with ties
		broken by larger constituent cluster size).

	Returns
	-------
	new_labels             : contiguous re-indexed cluster array (same length as `labels`)
	new_cluster_canonicals : updated {new_cid: meta_dict} aligned with `new_labels`
	"""
	ids = [int(c) for c in np.unique(labels)]
	cluster_ids = list(ids)
	members = [np.where(labels == c)[0] for c in ids]
	initial_sizes = np.array([len(m) for m in members])
	sums = np.vstack([X[m].sum(axis=0) for m in members]).astype(np.float32)
	unit_centroids = sums / (np.linalg.norm(sums, axis=1, keepdims=True) + 1e-12)
	sizes = np.array([len(m) for m in members])
	n_before = len(members)

	def _get_medoid_info(m_indices: np.ndarray, sum_vec: np.ndarray) -> Tuple[str, float]:
		if len(m_indices) == 0:
			return "", 0.0
		emb = X[m_indices]
		norm = np.linalg.norm(sum_vec)
		unit = sum_vec / (norm + 1e-12)
		sims = emb @ unit
		local_idx = int(np.argmax(sims))
		medoid_text = label_texts[m_indices[local_idx]]
		if len(m_indices) > 1:
			med_emb = emb[local_idx]
			med_sims = emb @ med_emb
			avg_sim = float((med_sims.sum() - 1.0) / (len(m_indices) - 1))
		else:
			avg_sim = 1.0
		return medoid_text, round(avg_sim, 4)

	# Initialize canonical state per cluster
	canon_strs: List[str] = []
	canon_embs_list: List[np.ndarray] = []
	canon_metas: List[Dict] = []
	candidate_canonicals: List[List[str]] = []

	for pos, c in enumerate(ids):
		meta = dict(cluster_canonicals.get(c, {}))
		c_str = meta.get("canonical_harmonized") or meta.get("canonical")
		if not c_str:
			c_str, _ = _get_medoid_info(members[pos], sums[pos])
		canon_strs.append(c_str)
		canon_embs_list.append(canonical_emb_map[c_str])
		canon_metas.append(meta)
		candidate_canonicals.append([c_str])

	canon_embs = np.vstack(canon_embs_list) if n_before > 0 else np.empty((0, X.shape[1]), dtype=np.float32)
	n_unique_canon_before = len({s.lower() for s in canon_strs})

	if threshold is None:
		nn_idx_init, nn_canon_sim_init, _ = _nearest_canonical_neighbors(
			canon_embs=canon_embs,
			unit_centroids=unit_centroids,
			chunk=4096,
		)
		mutual_mask_init = np.array([
			bool(int(nn_idx_init[int(nn_idx_init[pos])]) == pos)
			for pos in range(n_before)
		], dtype=bool)
		dyn_init = _estimate_dynamic_neighbor_thresholds(
			sims=nn_canon_sim_init.astype(float),
			mutual_mask=mutual_mask_init,
			verbose=verbose,
		)
		threshold = float(dyn_init['threshold'])
		threshold_mode = "dynamic"
	else:
		threshold = float(threshold)
		threshold_mode = "manual"

	merges = []
	centroid_blocked_pairs = []
	seen_blocked_keys = set()
	round_progression = []
	termination_reason = "max_rounds_reached"

	for rnd in range(1, max_rounds + 1):
		if len(members) < 2:
			termination_reason = "insufficient_clusters"
			break

		nn_idx, nn_canon_sim, nn_centroid_sim = _nearest_canonical_neighbors(
			canon_embs=canon_embs,
			unit_centroids=unit_centroids,
			chunk=4096,
		)
		drop = set()
		round_merges = 0
		round_centroid_blocked = 0
		round_canon_sims = []
		round_cent_sims = []
		clusters_in_round = len(members)

		for i in range(len(members)):
			j = int(nn_idx[i])

			if not (i < j and int(nn_idx[j]) == i):
				continue  # mutual 1-NN pairs only

			c_sim_val = float(nn_canon_sim[i])
			cent_sim_val = float(nn_centroid_sim[i])

			if c_sim_val < threshold:
				continue
			if max_merged_size is not None and sizes[i] + sizes[j] > max_merged_size:
				continue

			# Guard against merging unrelated clusters with polysemous/surface-similar canonicals
			if cent_sim_val < min_centroid_similarity:
				round_centroid_blocked += 1
				bkey = (min(cluster_ids[i], cluster_ids[j]), max(cluster_ids[i], cluster_ids[j]))
				if bkey not in seen_blocked_keys:
					seen_blocked_keys.add(bkey)
					med_a, med_a_sim = _get_medoid_info(members[i], sums[i])
					med_b, med_b_sim = _get_medoid_info(members[j], sums[j])
					centroid_blocked_pairs.append({
						'round': rnd,
						'canonical_similarity': round(c_sim_val, 6),
						'centroid_similarity': round(cent_sim_val, 6),
						'min_centroid_similarity': round(float(min_centroid_similarity), 4),
						'cluster_a': {
							'id': cluster_ids[i],
							'size': int(sizes[i]),
							'canonical': canon_strs[i],
							'medoid': med_a,
							'medoid_similarity': med_a_sim,
							'members': [label_texts[k] for k in members[i]],
						},
						'cluster_b': {
							'id': cluster_ids[j],
							'size': int(sizes[j]),
							'canonical': canon_strs[j],
							'medoid': med_b,
							'medoid_similarity': med_b_sim,
							'members': [label_texts[k] for k in members[j]],
						},
					})
				continue

			round_canon_sims.append(c_sim_val)
			round_cent_sims.append(cent_sim_val)
			round_merges += 1

			merged_indices = np.concatenate([members[i], members[j]])
			merged_sum = sums[i] + sums[j]
			unit_merged = merged_sum / (np.linalg.norm(merged_sum) + 1e-12)

			med_a, med_a_sim = _get_medoid_info(members[i], sums[i])
			med_b, med_b_sim = _get_medoid_info(members[j], sums[j])
			med_merged, med_merged_sim = _get_medoid_info(merged_indices, merged_sum)

			members_a_texts = [label_texts[k] for k in members[i]]
			members_b_texts = [label_texts[k] for k in members[j]]
			merged_members_texts = members_a_texts + members_b_texts
			merged_members_set = set(merged_members_texts)

			# Option 2: Winner Selection by closeness to the merged real-member centroid
			canon_a = canon_strs[i]
			canon_b = canon_strs[j]
			emb_a = canon_embs[i]
			emb_b = canon_embs[j]
			sim_a_to_merged = float(emb_a @ unit_merged)
			sim_b_to_merged = float(emb_b @ unit_merged)

			# Evaluate all candidate canonicals accumulated across this merge history
			cand_pool = list(dict.fromkeys(candidate_canonicals[i] + candidate_canonicals[j]))
			cand_scores = {}
			for cand in cand_pool:
				cand_scores[cand] = float(canonical_emb_map[cand] @ unit_merged)

			# Tie-break identical similarity (e.g. when canon_a == canon_b) by larger cluster size
			if abs(sim_a_to_merged - sim_b_to_merged) < 1e-6:
				if sizes[i] >= sizes[j]:
					winning_canon = canon_a
					winner_from = "cluster_a_tiebreak_size" if canon_a != canon_b else "identical_canonical"
				else:
					winning_canon = canon_b
					winner_from = "cluster_b_tiebreak_size"
			elif sim_a_to_merged > sim_b_to_merged:
				winning_canon = canon_a
				winner_from = "cluster_a"
			else:
				winning_canon = canon_b
				winner_from = "cluster_b"

			# Check if an earlier constituent canonical in cand_pool strictly beats both
			best_pool_cand = max(cand_pool, key=lambda c: cand_scores[c])
			if cand_scores[best_pool_cand] > max(sim_a_to_merged, sim_b_to_merged) + 1e-6:
				winning_canon = best_pool_cand
				winner_from = "constituent_pool"

			winning_sim_to_merged = cand_scores[winning_canon]
			runner_up_sim = min(sim_a_to_merged, sim_b_to_merged)
			winner_margin = round(float(abs(sim_a_to_merged - sim_b_to_merged)), 6)

			is_virt_a = canon_a not in set(members_a_texts)
			is_virt_b = canon_b not in set(members_b_texts)
			is_virt_merged = winning_canon not in merged_members_set

			merge = {
				'merge_id': len(merges) + 1,
				'round': rnd,
				'canonical_similarity': round(c_sim_val, 6),
				'centroid_similarity': round(cent_sim_val, 6),
				'same_canonical_string': bool(canon_a.lower() == canon_b.lower()),
				'cluster_a': {
					'id': cluster_ids[i],
					'size': int(sizes[i]),
					'canonical': canon_a,
					'is_virtual': is_virt_a,
					'canonical_to_merged_centroid_sim': round(sim_a_to_merged, 4),
					'medoid': med_a,
					'medoid_similarity': med_a_sim,
					'members': members_a_texts,
				},
				'cluster_b': {
					'id': cluster_ids[j],
					'size': int(sizes[j]),
					'canonical': canon_b,
					'is_virtual': is_virt_b,
					'canonical_to_merged_centroid_sim': round(sim_b_to_merged, 4),
					'medoid': med_b,
					'medoid_similarity': med_b_sim,
					'members': members_b_texts,
				},
				'merged': {
					'size': int(sizes[i] + sizes[j]),
					'canonical': winning_canon,
					'is_virtual': is_virt_merged,
					'winner_from': winner_from,
					'winner_margin': winner_margin,
					'canonical_to_merged_centroid_sim': round(winning_sim_to_merged, 4),
					'medoid': med_merged,
					'medoid_similarity': med_merged_sim,
					'medoid_agrees_with_canonical': bool(med_merged.lower() == winning_canon.lower()),
					'candidate_pool_scores': {k: round(v, 4) for k, v in cand_scores.items()},
					'members': merged_members_texts,
				},
			}

			merges.append(merge)

			members[i] = merged_indices
			sums[i] = merged_sum
			unit_centroids[i] = unit_merged
			sizes[i] += sizes[j]
			canon_strs[i] = winning_canon
			canon_embs[i] = canonical_emb_map[winning_canon]
			candidate_canonicals[i] = cand_pool
			drop.add(j)

		if not drop:
			termination_reason = "converged_no_more_pairs"
			break

		round_progression.append({
			'round': rnd,
			'clusters_in': clusters_in_round,
			'merges': round_merges,
			'centroid_blocked': round_centroid_blocked,
			'clusters_out': clusters_in_round - round_merges,
			'min_canonical_similarity': round(float(min(round_canon_sims)), 6) if round_canon_sims else None,
			'median_canonical_similarity': round(float(np.median(round_canon_sims)), 6) if round_canon_sims else None,
			'max_canonical_similarity': round(float(max(round_canon_sims)), 6) if round_canon_sims else None,
			'min_centroid_similarity': round(float(min(round_cent_sims)), 6) if round_cent_sims else None,
			'median_centroid_similarity': round(float(np.median(round_cent_sims)), 6) if round_cent_sims else None,
		})

		keep = [k for k in range(len(members)) if k not in drop]
		members = [members[k] for k in keep]
		cluster_ids = [cluster_ids[k] for k in keep]
		sums = sums[keep]
		unit_centroids = unit_centroids[keep]
		sizes = sizes[keep]
		canon_strs = [canon_strs[k] for k in keep]
		canon_embs = canon_embs[keep]
		canon_metas = [canon_metas[k] for k in keep]
		candidate_canonicals = [candidate_canonicals[k] for k in keep]

	new_labels = np.empty_like(labels)
	new_cluster_canonicals: Dict[int, Dict] = {}

	for new_id, m in enumerate(members):
		new_labels[m] = new_id
		merged_member_set = {label_texts[k] for k in m}
		w_canon = canon_strs[new_id]
		is_virt_now = w_canon not in merged_member_set
		meta = dict(canon_metas[new_id])
		meta.update({
			'canonical': w_canon,
			'canonical_harmonized': w_canon,
			'changed_by_harmonize': False,
			'virtual': is_virt_now,
			'size': int(len(m)),
			'score': round(float(canon_embs[new_id] @ unit_centroids[new_id]), 4),
		})
		new_cluster_canonicals[new_id] = meta

	rounds_completed = len(round_progression)
	final_sizes = np.array([len(m) for m in members])
	n_unique_canon_after = len({s.lower() for s in canon_strs})
	all_canon_sims = [m['canonical_similarity'] for m in merges]
	all_cent_sims = [m['centroid_similarity'] for m in merges]

	if all_canon_sims:
		merge_sim_stats = {
			'count': len(all_canon_sims),
			'synonym_merges': sum(1 for m in merges if not m['same_canonical_string']),
			'identical_canonical_merges': sum(1 for m in merges if m['same_canonical_string']),
			'canonical_similarity': {
				'min': round(float(np.min(all_canon_sims)), 4),
				'p25': round(float(np.percentile(all_canon_sims, 25)), 4),
				'median': round(float(np.median(all_canon_sims)), 4),
				'mean': round(float(np.mean(all_canon_sims)), 4),
				'p75': round(float(np.percentile(all_canon_sims, 75)), 4),
				'p90': round(float(np.percentile(all_canon_sims, 90)), 4),
				'max': round(float(np.max(all_canon_sims)), 4),
			},
			'centroid_similarity': {
				'min': round(float(np.min(all_cent_sims)), 4),
				'p25': round(float(np.percentile(all_cent_sims, 25)), 4),
				'median': round(float(np.median(all_cent_sims)), 4),
				'mean': round(float(np.mean(all_cent_sims)), 4),
				'p75': round(float(np.percentile(all_cent_sims, 75)), 4),
				'p90': round(float(np.percentile(all_cent_sims, 90)), 4),
				'max': round(float(np.max(all_cent_sims)), 4),
			},
		}
	else:
		merge_sim_stats = {}

	size_stats = {
		'before': {
			'min': int(np.min(initial_sizes)) if len(initial_sizes) > 0 else 0,
			'median': float(np.median(initial_sizes)) if len(initial_sizes) > 0 else 0.0,
			'mean': round(float(np.mean(initial_sizes)), 2) if len(initial_sizes) > 0 else 0.0,
			'p75': float(np.percentile(initial_sizes, 75)) if len(initial_sizes) > 0 else 0.0,
			'p90': float(np.percentile(initial_sizes, 90)) if len(initial_sizes) > 0 else 0.0,
			'max': int(np.max(initial_sizes)) if len(initial_sizes) > 0 else 0,
		},
		'after': {
			'min': int(np.min(final_sizes)) if len(final_sizes) > 0 else 0,
			'median': float(np.median(final_sizes)) if len(final_sizes) > 0 else 0.0,
			'mean': round(float(np.mean(final_sizes)), 2) if len(final_sizes) > 0 else 0.0,
			'p75': float(np.percentile(final_sizes, 75)) if len(final_sizes) > 0 else 0.0,
			'p90': float(np.percentile(final_sizes, 90)) if len(final_sizes) > 0 else 0.0,
			'max': int(np.max(final_sizes)) if len(final_sizes) > 0 else 0,
		},
	}

	if verbose:
		size_limit_str = f"max_size={max_merged_size}" if max_merged_size is not None else "no size limit"
		reduction_count = n_before - len(members)
		reduction_pct = (1.0 - len(members) / max(1, n_before)) * 100.0
		canon_red = n_unique_canon_before - n_unique_canon_after

		print("=" * 120)
		print("[MERGE CLUSTERS (with canonical neighbors)]\nMulti-Round 1-NN Canonical Label Merging Diagnostics")
		print(
			f"Policy: canonical_threshold >= {threshold:.5f} ({threshold_mode}) | "
			f"min_centroid_similarity >= {min_centroid_similarity:.2f} | "
			f"Size Limit: {size_limit_str} | Max Rounds: {max_rounds}"
		)
		print(
			f"Cluster Reduction  : {n_before:,} -> {len(members):,} clusters "
			f"(-{reduction_count:,} clusters, -{reduction_pct:.2f}%) "
			f"across {rounds_completed} round(s) (status: {termination_reason})"
		)
		print(
			f"Canonical Reduction: {n_unique_canon_before:,} -> {n_unique_canon_after:,} unique canonical labels "
			f"(-{canon_red:,} near-duplicate canonicals consolidated)"
		)
		print("-" * 120)

		if round_progression:
			print("Round-by-Round Progression:")
			print(
				f"  {'Round':<9} {'Clusters In':>13} {'Merges Done':>13} {'Cent Blocked':>13} {'Clusters Out':>14} "
				f"{'Canon Min':>11} {'Canon Med':>11} {'Canon Max':>11} {'Cent Med':>11}"
			)
			print(
				f"  {'-'*9} {'-'*13} {'-'*13} {'-'*13} {'-'*14} {'-'*11} {'-'*11} {'-'*11} {'-'*11}"
			)
			for rp in round_progression:
				c_min_s = f"{rp['min_canonical_similarity']:.4f}" if rp['min_canonical_similarity'] is not None else "-"
				c_med_s = f"{rp['median_canonical_similarity']:.4f}" if rp['median_canonical_similarity'] is not None else "-"
				c_max_s = f"{rp['max_canonical_similarity']:.4f}" if rp['max_canonical_similarity'] is not None else "-"
				ct_med_s = f"{rp['median_centroid_similarity']:.4f}" if rp['median_centroid_similarity'] is not None else "-"
				print(
					f"  Round {rp['round']:<3} {rp['clusters_in']:>13,} {rp['merges']:>13,} {rp['centroid_blocked']:>13,} "
					f"{rp['clusters_out']:>14,} {c_min_s:>11} {c_med_s:>11} {c_max_s:>11} {ct_med_s:>11}"
				)
			print("-" * 120)

		print("Cluster Size Distributions:")
		print(
			f"  Before Merge: min={size_stats['before']['min']} | median={size_stats['before']['median']:.1f} | "
			f"mean={size_stats['before']['mean']:.2f} | p75={size_stats['before']['p75']:.1f} | "
			f"p90={size_stats['before']['p90']:.1f} | max={size_stats['before']['max']}"
		)
		print(
			f"  After Merge : min={size_stats['after']['min']} | median={size_stats['after']['median']:.1f} | "
			f"mean={size_stats['after']['mean']:.2f} | p75={size_stats['after']['p75']:.1f} | "
			f"p90={size_stats['after']['p90']:.1f} | max={size_stats['after']['max']}"
		)

		if merge_sim_stats:
			cs = merge_sim_stats['canonical_similarity']
			cts = merge_sim_stats['centroid_similarity']
			n_virt_winners = sum(1 for m in merges if m['merged']['is_virtual'])
			n_med_agree = sum(1 for m in merges if m['merged']['medoid_agrees_with_canonical'])
			print(
				f"\nMerge Similarity Percentiles (N={merge_sim_stats['count']:,}: "
				f"{merge_sim_stats['synonym_merges']:,} synonym merges, "
				f"{merge_sim_stats['identical_canonical_merges']:,} identical-canonical merges):\n"
				f"  Canonical Sim: min={cs['min']:.4f} | p25={cs['p25']:.4f} | median={cs['median']:.4f} | "
				f"mean={cs['mean']:.4f} | p75={cs['p75']:.4f} | p90={cs['p90']:.4f} | max={cs['max']:.4f}\n"
				f"  Centroid Sim : min={cts['min']:.4f} | p25={cts['p25']:.4f} | median={cts['median']:.4f} | "
				f"mean={cts['mean']:.4f} | p75={cts['p75']:.4f} | p90={cts['p90']:.4f} | max={cts['max']:.4f}\n"
				f"  Winner Stats : {n_virt_winners:,}/{len(merges):,} virtual canonical winners | "
				f"{n_med_agree:,}/{len(merges):,} ({100.0 * n_med_agree / max(1, len(merges)):.1f}%) agree with raw member medoid"
			)

		if merges:
			print("-" * 120)
			synonym_merges = [m for m in merges if not m['same_canonical_string']]
			display_pool = synonym_merges if synonym_merges else merges
			sorted_by_sim = sorted(display_pool, key=lambda m: (m['canonical_similarity'], m['centroid_similarity']), reverse=True)

			topk = min(10, len(synonym_merges))
			print(f"Top {topk} Closest Canonical Synonym Merges (Highest Confidence):")
			for rank, m in enumerate(sorted_by_sim[:topk], 1):
				comb_sz = m['merged']['size']
				ca, cb, cm = m['cluster_a'], m['cluster_b'], m['merged']
				va = " [VIRTUAL]" if ca['is_virtual'] else ""
				vb = " [VIRTUAL]" if cb['is_virtual'] else ""
				vm = " [VIRTUAL]" if cm['is_virtual'] else ""
				print(
					f"  {rank}. canon_sim={m['canonical_similarity']:.4f} | centroid_sim={m['centroid_similarity']:.4f} "
					f"(Rnd {m['round']}) | Combined size: {comb_sz} ({ca['size']} + {cb['size']})\n"
					f"     Cluster {ca['id']:6d} canonical: {repr(ca['canonical']) + va:<40} sim_to_merged={ca['canonical_to_merged_centroid_sim']:.4f} medoid: {repr(ca['medoid']):<40} {ca['members'][:3]}\n"
					f"     Cluster {cb['id']:6d} canonical: {repr(cb['canonical']) + vb:<40} sim_to_merged={cb['canonical_to_merged_centroid_sim']:.4f} medoid: {repr(cb['medoid']):<40} {cb['members'][:3]}\n"
					f"     -> Winning canonical: {repr(cm['canonical']) + vm} (from {cm['winner_from']}, margin=+{cm['winner_margin']:.4f}, merged_medoid={repr(cm['medoid'])})\n"
				)

			print(f"\nTop {topk} Borderline Canonical Merges (Near Cutoff Threshold):")
			borderline = sorted(display_pool, key=lambda m: (m['canonical_similarity'], m['centroid_similarity']))[:topk]
			for rank, m in enumerate(borderline, 1):
				comb_sz = m['merged']['size']
				ca, cb, cm = m['cluster_a'], m['cluster_b'], m['merged']
				va = " [VIRTUAL]" if ca['is_virtual'] else ""
				vb = " [VIRTUAL]" if cb['is_virtual'] else ""
				vm = " [VIRTUAL]" if cm['is_virtual'] else ""
				print(
					f"  {rank}. canon_sim={m['canonical_similarity']:.4f} | centroid_sim={m['centroid_similarity']:.4f} "
					f"(Rnd {m['round']}) | Combined size: {comb_sz} ({ca['size']} + {cb['size']})\n"
					f"     Cluster {ca['id']:6d} canonical: {repr(ca['canonical']) + va:<40} sim_to_merged={ca['canonical_to_merged_centroid_sim']:.4f} medoid={repr(ca['medoid']):<40} {ca['members'][:3]}\n"
					f"     Cluster {cb['id']:6d} canonical: {repr(cb['canonical']) + vb:<40} sim_to_merged={cb['canonical_to_merged_centroid_sim']:.4f} medoid={repr(cb['medoid']):<40} {cb['members'][:3]}\n"
					f"     -> Winning canonical: {repr(cm['canonical']) + vm} (from {cm['winner_from']}, margin=+{cm['winner_margin']:.4f}, merged_medoid={repr(cm['medoid'])})\n"
				)
			print("  (lowest-similarity canonical merges shown; borderline ones to review)")

		if centroid_blocked_pairs:
			print("-" * 120)
			print(
				f"[BLOCKED] Centroid Coherence Guard Blocked {len(centroid_blocked_pairs)} Mutual Canonical Pair(s) "
				f"(canonical_sim >= {threshold:.4f} but centroid_sim < {min_centroid_similarity}):"
			)
			for rank, bp in enumerate(sorted(centroid_blocked_pairs, key=lambda x: -x['canonical_similarity']), 1):
				ca, cb = bp['cluster_a'], bp['cluster_b']
				print(
					f"  {rank}. canon_sim={bp['canonical_similarity']:.4f} centroid_sim={bp['centroid_similarity']:.4f} < {min_centroid_similarity}\n"
					f"     Cluster {ca['id']:6d} canonical: {repr(ca['canonical']):<25} medoid: {repr(ca['medoid']):<40} {ca['members'][:3]}\n"
					f"     Cluster {cb['id']:6d} canonical: {repr(cb['canonical']):<25} medoid: {repr(cb['medoid']):<40} {cb['members'][:3]}\n"
				)

		if report_path:
			print(f"\n[SAVED] {report_path}")
		print("=" * 120 + "\n")

	if report_path:
		with open(report_path, 'w', encoding='utf-8') as f:
			new_result = {
				'meta': {
					'threshold': threshold,
					'threshold_mode': threshold_mode,
					'min_centroid_similarity': min_centroid_similarity,
					'max_merged_size': max_merged_size,
					'max_rounds': max_rounds,
					'rounds_completed': rounds_completed,
					'termination_reason': termination_reason,
					'clusters_before': n_before,
					'clusters_after': len(members),
					'unique_canonicals_before': n_unique_canon_before,
					'unique_canonicals_after': n_unique_canon_after,
					'total_merges': len(merges),
					'total_centroid_blocked_pairs': len(centroid_blocked_pairs),
					'reduction_pct': round((1.0 - len(members) / max(1, n_before)) * 100, 2),
					'round_progression': round_progression,
					'cluster_size_stats': size_stats,
					'merge_similarity_stats': merge_sim_stats,
				},
				'threshold': threshold,
				'min_centroid_similarity': min_centroid_similarity,
				'max_merged_size': max_merged_size,
				'clusters_before': n_before,
				'clusters_after': len(members),
				'rounds': rounds_completed,
				'merges': merges,
				'centroid_blocked_pairs': centroid_blocked_pairs,
			}
			json.dump(new_result, f, indent=2, ensure_ascii=False, default=_json_default)

	return new_labels, new_cluster_canonicals

def _validate_embeddings(X: np.ndarray, unique_labels: List[str]) -> None:
	if np.isnan(X).any():
		nan_rows = np.where(np.isnan(X).any(axis=1))[0]
		print(f"\n❌ ERROR: {np.isnan(X).sum()} NaN values in embeddings!")
		for idx in nan_rows[:10]:
			print(f"  - {unique_labels[idx]}")
		raise ValueError("Cannot proceed with NaN embeddings")
	if np.isinf(X).any():
		raise ValueError(f"Infinite values detected ({np.isinf(X).sum()}) - numerical overflow!")
	if X.shape[0] == 0:
		raise ValueError("No embeddings generated")
	if np.allclose(X, 0):
		raise ValueError("All embeddings are zero vectors")

def _compute_linkage(
	X: np.ndarray, 
	linkage_method: str, 
	distance_metric: str, 
	verbose: bool = False
) -> np.ndarray:

	if linkage_method == "ward":
		return fastcluster.linkage(X, method='ward', metric='euclidean') if use_fastcluster \
			else linkage(X, method='ward', metric='euclidean')

	if distance_metric == "cosine":
		distance_matrix = np.clip(1 - (X @ X.T), 0, 2)
		np.fill_diagonal(distance_matrix, 0)
		condensed_dist = squareform(distance_matrix, checks=False)
		
		if verbose:
			print(f"[LINKAGE] Using {linkage_method} linkage with {distance_metric} distance")
		
		return fastcluster.linkage(condensed_dist, method=linkage_method) if use_fastcluster \
			else scipy.cluster.hierarchy.linkage(condensed_dist, method=linkage_method)

	if distance_metric == "euclidean":
		if verbose:
			print(f"[LINKAGE] Using {linkage_method} linkage with Euclidean distance")
		return fastcluster.linkage(X, method=linkage_method, metric='euclidean') if use_fastcluster \
			else scipy.cluster.hierarchy.linkage(X, method=linkage_method, metric='euclidean')
	
	raise ValueError(f"Unsupported distance metric: {distance_metric}")

def _caching(
	clusters_fname: str,
	model: SentenceTransformer,
	linkage_method: str,
	distance_metric: str,
	unique_labels: List[str],
	encode_prompt: Optional[str] = None,
	verbose: bool = False,
) -> Tuple[str, str]:

	model_id = model.model_card_data.base_model 
	dtype = next(model.parameters()).dtype

	# Canonical representation of the label set
	label_blob = "\x1f".join(unique_labels)
	label_hash = hashlib.sha1(
			label_blob.encode("utf-8")
	).hexdigest()[:16]
	
	# Full cache key
	# The prompt changes every embedding, so it is part of the key. It is added ONLY when set:
	# with encode_prompt=None the key is byte-for-byte what it was, so existing caches stay valid.
	key_parts = [
		model_id, 
		str(dtype), 
		linkage_method, 
		distance_metric
	]

	if encode_prompt:
		key_parts.append("PROMPT:" + encode_prompt)

	key_parts.append(label_blob)
	cache_blob = "\x1f".join(key_parts)
	
	key = hashlib.sha1(
		cache_blob.encode("utf-8")
	).hexdigest()[:16]
	
	stem = os.path.join(
		os.path.dirname(clusters_fname) or ".",
		f"cache_{key}"
	)
	
	x_path = stem + "_embeddings_X.npy"
	z_path = stem + "_linkage_Z.npy"
	
	if verbose:
		print(f"[CACHE PATHS]")
		print(f"  ├─ model          : {model_id}")
		print(f"  ├─ dtype          : {dtype}")
		print(f"  ├─ linkage        : {linkage_method}")
		print(f"  ├─ distance       : {distance_metric}")
		print(f"  ├─ encode prompt  : {encode_prompt!r}")
		print(f"  ├─ labels         : {len(unique_labels)}")
		print(f"  ├─ label hash     : {label_hash}")
		print(f"  ├─ cache key      : {key}")
		print(f"  ├─ x_path exists  : {os.path.exists(x_path)}")
		print(f"  ├─ z_path exists  : {os.path.exists(z_path)}")
		print(f"  ├─ embeddings     : {x_path}")
		print(f"  └─ linkage        : {z_path}")
	
	return x_path, z_path

def _save_npy_atomic(path: str, arr: np.ndarray) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	tmp = path + ".tmp.npy"
	try:
		np.save(tmp, arr)
		os.replace(tmp, path)
	except Exception:
		if os.path.exists(tmp):
			try:
				os.remove(tmp)
			except OSError:
				pass
		raise

def _save_cache_manifest(
	x_path: str,
	model_id: str,
	dtype: Any,
	distance_metric: str,
	unique_labels: List[str],
	embedding_shape: tuple,
	embedding_dtype: Any,
	linkage_method: str,
	encode_prompt: Optional[str]=None,
	verbose: bool=False,
) -> str:
	manifest_path = x_path.replace("_embeddings_X.npy", "_manifest.json")
	
	label_hash = hashlib.sha1(
		"\x1f".join(unique_labels).encode("utf-8")
	).hexdigest()
	
	manifest = {
		"model_id": model_id,
		"dtype": str(dtype),
		"linkage_method": linkage_method,
		"distance_metric": distance_metric,
		"n_labels": len(unique_labels),
		"label_hash": label_hash,
		"embedding_shape": list(embedding_shape),
		"embedding_dtype": str(embedding_dtype),
		"encode_prompt": encode_prompt,
		"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
	}
	
	if verbose:
		print(json.dumps(manifest, indent=2))

	tmp_path = manifest_path + ".tmp"
	
	try:
		with open(tmp_path, "w", encoding="utf-8") as f:
			json.dump(manifest, f, indent=2)
		os.replace(tmp_path, manifest_path)
	except Exception:
		if os.path.exists(tmp_path):
			try:
				os.remove(tmp_path)
			except OSError:
				pass
		raise
	
	return manifest_path

def get_clustering_artifacts(
	clusters_fname: str,
	unique_labels: List[str],
	model: SentenceTransformer,
	batch_size: int,
	linkage_method: str,
	distance_metric: str,
	use_cache: bool = True,
	encode_prompt: Optional[str] = None,
	verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Load cached embeddings/linkage when valid; otherwise compute and
	independently cache each artifact.

	Embeddings and linkage are cached independently so that if linkage
	computation fails after embeddings are saved, the next run can reuse X.
	"""
	model_id = model.model_card_data.base_model 
	dtype = next(model.parameters()).dtype

	x_path, z_path = _caching(
		clusters_fname=clusters_fname,
		model=model,
		linkage_method=linkage_method,
		distance_metric=distance_metric,
		unique_labels=unique_labels,
		encode_prompt=encode_prompt,
		verbose=verbose,
	)

	X = None
	Z = None

	n_labels = len(unique_labels)

	# STEP 1: LOAD / COMPUTE EMBEDDINGS
	if use_cache and os.path.exists(x_path):

		if verbose:
			print(f"\n[CACHE HIT] Embeddings {x_path}")

		try:
			X = np.load(x_path, mmap_mode=None)

			expected_rows = n_labels

			if X.ndim != 2 or X.shape[0] != expected_rows:
				if verbose:
					print(
						f"[CACHE MISS] embedding shape mismatch: "
						f"{X.shape} vs expected "
						f"({expected_rows}, embedding_dim)"
					)
				X = None

			else:
				_validate_embeddings(X, unique_labels)

				if verbose:
					print(
						f"[CACHE HIT] embeddings: "
						f"{X.shape} {X.dtype} "
						f"({X.nbytes / 1e6:.2f} MB)"
					)

		except Exception as e:
			print(
				f"[CACHE] Failed to load embeddings "
				f"({type(e).__name__}: {e})"
			)
			X = None

	if X is None:

		if verbose:
			print(f"\n[EMBEDDING] {n_labels} unique labels...")

		t0 = time.time()

		X = _encode_(model, unique_labels, batch_size, encode_prompt)

		_validate_embeddings(X, unique_labels)

		if verbose:
			print(f"[EMBEDDING] {type(X)} {X.shape} {X.dtype} ({X.nbytes / 1e6:.2f} MB)")
			print(f"[EMBEDDING] Encoding time: {time.time() - t0:.1f} sec")

		if use_cache:
			_save_npy_atomic(x_path, X)
			manifest_path = _save_cache_manifest(
				x_path=x_path,
				model_id=model_id,
				dtype=dtype,
				linkage_method=linkage_method,
				distance_metric=distance_metric,
				unique_labels=unique_labels,
				embedding_shape=X.shape,
				embedding_dtype=X.dtype,
				encode_prompt=encode_prompt,
				verbose=verbose,
			)

			if verbose:
				print(f"[CACHE SAVE] embeddings: {x_path}  ({X.nbytes / 1e6:.2f} MB)")
				print(f"[CACHE SAVE] manifest  : {manifest_path}")

	# STEP 2: LOAD / COMPUTE LINKAGE
	if use_cache and os.path.exists(z_path):

		if verbose:
			print(f"\n[CACHE HIT] Linkage {z_path}")

		try:
			Z = np.load(z_path)

			expected_shape = (n_labels - 1, 4)

			if Z.shape != expected_shape:

				if verbose:
					print(
						f"[CACHE MISS] linkage shape mismatch: "
						f"{Z.shape} vs expected {expected_shape}"
					)

				Z = None

			elif verbose:
				print(f"[CACHE HIT] linkage: {Z.shape} {Z.dtype} ({Z.nbytes / 1e6:.2f} MB)")

		except Exception as e:
			print(f"[CACHE] Failed to load linkage ({type(e).__name__}: {e})")
			Z = None

	if Z is None:
		if verbose:
			print(f"[LINKAGE] {linkage_method} {X.shape} embeddings [takes a while...]")

		t0 = time.time()

		Z = _compute_linkage(X, linkage_method, distance_metric, verbose=verbose,)

		if verbose:
			print(
				f"[LINKAGE] Z[{linkage_method}] "
				f"{type(Z)} {Z.shape} {Z.dtype} "
				f"{Z.strides} {Z.itemsize} {Z.nbytes} "
				f"| {time.time() - t0:.1f} sec"
			)

		if use_cache:
			_save_npy_atomic(z_path, Z)

			if verbose:
				print(f"[CACHE SAVE] linkage {z_path} ({Z.nbytes / 1e6:.2f} MB)")

	# viz.plot_dimensional_reduction(
	# 	embedding=X,
	# 	clusters_fname=clusters_fname,
	# )

	return X, Z

def _build_case_registry(original_label_counts: Dict[str, int]) -> Dict[str, str]:
	"""
	Build a lowercase -> preferred-surface-form registry from the corpus-wide
	label frequency dict.

	Since original_label_counts is built from documents that already passed
	through _normalize_label_case() upstream (in cluster() Step 1), there
	should be at most one real surface form per lowercase key already. This
	function guards against the rare case where two forms slip through
	(e.g. if this function is called with pre-fold data) by keeping the
	higher-frequency form.

	Parameters
	----------
	original_label_counts : Dict[str, int]
		Corpus-wide frequency of every real (case-folded) label.

	Returns
	-------
	Dict[str, str]
		{lowercase_key: preferred_surface_form}
	"""
	registry: Dict[str, str] = {}
	for label, freq in original_label_counts.items():
		key = label.lower()
		if key not in registry:
			registry[key] = label
		else:
			# Guard: keep whichever surface form has higher corpus frequency
			existing_freq = original_label_counts.get(registry[key], 0)
			if freq > existing_freq:
				registry[key] = label
	return registry

def _normalize_label_case(
	documents: List[List[str]],
	verbose: bool = False,
) -> List[List[str]]:
	"""
	Collapse case-only duplicate labels ('trench' / 'Trench' / 'TRENCH')
	into a single winning surface form, BEFORE `unique_labels` is computed
	in cluster().
	Why this is needed
	------------------
	cluster() Step 1 deduplicates case-SENSITIVELY:
			documents.append(list(set(lbl for lbl in doc if lbl is not None)))
			unique_labels = sorted(set(label for doc in documents for label in doc))
	Neither `set()` nor `sorted(set(...))` folds case, so 'trench' and
	'Trench' survive as two distinct unique labels.  Each then gets its own
	row in the embedding matrix X, its own leaf in the linkage matrix, and —
	worse — each contributes a separate (smaller) entry to label_freq_dict
	(STEP 5 of cluster()), weakening the frequency signal that
	assign_canonical_labels() relies on to pick cluster representatives.
	Why not blanket .lower()
	-----------------------
	Lowercasing everything would destroy meaningful capitalisation that
	canonical selection depends on ('Ausf', 'GR Mk', 'Constitution', proper
	nouns like 'Pepperell').  Instead, surface forms are grouped by their
	lowercase key and the most frequent REAL surface form in each group
	becomes the representative; every occurrence is rewritten to it.
	Determinism guarantees (reproducibility)
	-----------------------------------------
	1. Pass 1 counts frequencies — an order-free sum over documents.
	2. Pass 2 iterates surfaces in SORTED order, so group construction,
		 dict insertion order, and the verbose printout are functions of
		 input CONTENT only — immune to the hash-randomised per-document
		 label order produced by `list(set(...))` upstream (PYTHONHASHSEED).
	3. Pass 3 picks winners with a TOTAL-order key (frequency, surface).
		 Within a group all surfaces are distinct strings, so no two keys
		 can tie: the argmax is unique and cannot depend on iteration order.
		 Frequency ties are broken by the LEXICOGRAPHICALLY GREATEST surface
		 (under ASCII, lowercase > uppercase, so 'trench' beats 'Trench').
	4. Pass 4 output is a pure function of the input's content and
		 per-document order (first-occurrence order is preserved).
	What this function canNOT do
	-----------------------------
	It cannot create or destroy surface forms: the distinct raw string
	count (printed as "raw surface forms") is fixed by the caller's input.
	A run-to-run difference in that number therefore PROVES the input
	changed upstream — use the verbose block as a cheap drift detector.
	Parameters
	----------
	documents : List[List[str]]
			Per-sample label lists, expected to be None-filtered and per-doc
			deduplicated (case-sensitively) by cluster() Step 1.  A defensive
			Pass 0 additionally drops non-string / empty / whitespace-only
			labels in case that contract is violated (e.g. by a CSV round-trip).
	verbose : bool
			Print a reproducible summary of the fold (sorted examples, capped).
	Returns
	-------
	List[List[str]]
			Same number of documents; every label replaced by its group's
			winning surface form; each document re-deduplicated, ordered by
			first occurrence of each winner (documents may shrink: invalid
			labels dropped, case-variants collapsed onto one entry).
	"""

	# ── Pass 0 (defensive): enforce the input contract ───────────────────
	# cluster() Step 1 filters None, but a CSV round-trip can smuggle in
	# NaN floats or whitespace-only strings.  Dropping them here — with a
	# count, so nothing disappears silently — keeps the frequency counts
	# and the embeddings honest.  Note: surfaces are NOT stripped;
	# ' trench' stays ' trench' — only unusable labels are removed.
	if verbose:
		print(f"\n[CASE NORMALISATION]")
		print(f"  ├─ documents: {type(documents)} {len(documents)} {type(documents[0])} {len(documents[0])} {documents[0]}")
	clean_documents: List[List[str]] = []
	n_invalid = 0
	for doc in documents:
		kept = [lbl for lbl in doc if isinstance(lbl, str) and lbl.strip()]
		n_invalid += len(doc) - len(kept)
		clean_documents.append(kept)
	if n_invalid and verbose:
		print(f"[CASE NORMALISATION] dropped {n_invalid} non-string/empty label(s)")
	documents = clean_documents

	# ── Pass 1: corpus-wide surface-form frequency (order-free) ──────────
	# Counted once per document occurrence, matching label_freq_dict in
	# cluster() STEP 5 (input is per-doc deduped, so this equals "number of
	# documents containing the surface").  Counted BEFORE folding:
	# 'trench' and 'Trench' accrue separate frequencies, which Pass 3
	# then compares to elect the winner.
	surface_freq: Dict[str, int] = {}
	for doc in documents:
			for lbl in doc:
					surface_freq[lbl] = surface_freq.get(lbl, 0) + 1

	# ── Pass 2: group surface forms by lowercase key (SORTED iteration) ──
	# Iterating sorted(surface_freq) instead of the dict's insertion order
	# makes group construction AND the verbose example printout fully
	# deterministic.  This matters: the per-document label order upstream
	# comes from `list(set(...))`, which is hash-randomised per process —
	# without this sort, the log churns between runs (example lines swap
	# places) and log diffs become useless for spotting real input drift.
	#
	# str.lower() (not casefold()): sufficient for the ASCII English labels
	# in this corpus; casefold's extra aggressiveness ('ß'->'ss') is not
	# needed and would alter grouping for non-ASCII text.
	case_groups: Dict[str, List[str]] = {}
	for surface in sorted(surface_freq):
			case_groups.setdefault(surface.lower(), []).append(surface)

	# ── Pass 3: elect one winning surface form per group ─────────────────
	# Highest raw frequency wins; ties broken by the lexicographically
	# GREATEST surface (max() over the tuple (frequency, surface); ASCII
	# lowercase code points > uppercase, so 'trench' beats 'Trench').
	# Surfaces within a group are unique strings => the key is a total
	# order => the argmax is unique => the winner is iteration-order-
	# independent.  This is the core reproducibility property of the fold.
	#
	# sorted(case_groups) is redundant determinism-wise (insertion order is
	# already fixed by Pass 2) but keeps this pass self-evidently
	# order-independent even if Pass 2 is edited later.
	winner_map: Dict[str, str] = {}
	collapsed_groups = 0
	for key in sorted(case_groups):
		surfaces = case_groups[key]
		if len(surfaces) == 1:
			# No case variants: identity mapping, so Pass 4 can look up
			# every label unconditionally.
			winner_map[surfaces[0]] = surfaces[0]
			continue
		collapsed_groups += 1
		winner = max(surfaces, key=lambda s: (surface_freq[s], s))
		for s in surfaces:
			winner_map[s] = winner

	# ── Pass 4: rewrite every document, RE-DEDUPLICATING on the way ──────
	# Two jobs, one comprehension:
	#
	#  (a) REPLACE each label with its group winner.
	#  (b) RE-DEDUP each document.  The upstream per-doc dedup in
	#      cluster() was case-SENSITIVE, so a document holding both
	#      'Trench' and 'trench' (e.g. human annotation + LLM keywords
	#      disagreeing on casing) would collapse to
	#      ['trench', 'trench'] after mapping.  Left in, that double-counts
	#      in label_freq_dict (STEP 5) and skews the frequency signal used
	#      by assign_canonical_labels().
	#
	# dict.fromkeys() removes duplicates while preserving FIRST-OCCURRENCE
	# order — deterministic given the input document's order, and O(n).
	normalized_documents: List[List[str]] = [
		list(dict.fromkeys(winner_map[lbl] for lbl in doc))
		for doc in documents
	]
	if verbose:
		n_before = len(surface_freq)             # distinct RAW surface forms
		n_after = len(set(winner_map.values()))  # distinct post-fold labels
		print(f"  ├─ {n_before:,} raw surface forms")
		print(f"  └─ Case-duplicate groups collapsed: {collapsed_groups:,}")

		if n_after < n_before:
			print(f"  Unique labels after fold: {n_after:,} (was {n_before:,})")
		if collapsed_groups > 0:
			examples = sorted(
				(surfaces, winner_map[surfaces[0]])
				for surfaces in case_groups.values()
				if len(surfaces) > 1
			)
			for surfaces, winner in examples:
				print(f"\t{surfaces} -> {repr(winner)}")

	return normalized_documents

def _check_json_csv_consistency(df: pd.DataFrame, json_path: str, verbose: bool = False) -> list:
	with open(json_path, encoding="utf-8") as f:
		js_canon = {c["cluster_id"]: c["canonical"] for c in json.load(f)["clusters"]}
	csv_canon = df[~df["is_injected"]].groupby("cluster")["canonical"].first().to_dict()
	mismatch = [(cid, js_canon[cid], csv_canon.get(cid)) for cid in js_canon if csv_canon.get(cid) != js_canon[cid]]

	if verbose:
		print(
			f"[CONSISTENCY] JSON vs CSV mismatches: {len(mismatch)} | distinct canonicals "
			f"JSON {len(set(js_canon.values()))}, CSV {len(set(csv_canon.values()))}"
		)
	
	return mismatch

def dissolve_low_cohesion_clusters(
	df,
	embeddings,
	threshold=0.5,
	verbose=True
):
		"""
		Dissolve clusters with intra-similarity < threshold.
		Ensures cluster IDs remain contiguous (0, 1, 2, ..., n-1).
		"""
		
		if verbose:
				print(f"\n[DISSOLUTION] Analyzing clusters...")
				print(f"  Threshold: {threshold}")
		
		clusters_to_dissolve = list()
		
		# Find low cohesion clusters
		for cluster_id in df['cluster'].unique():
				cluster_mask = df['cluster'] == cluster_id
				cluster_size = cluster_mask.sum()
				
				if cluster_size < 2:
						continue
				
				clustered_labels = df[cluster_mask]['label'].tolist()
				cluster_indices = df[cluster_mask].index.tolist()
				cluster_embeddings = embeddings[cluster_indices]
				
				sim_matrix = sklearn.metrics.pairwise.cosine_similarity(cluster_embeddings)
				n = len(cluster_embeddings)
				intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
				
				if intra_sim < threshold:
						clusters_to_dissolve.append({
								'cluster_id': cluster_id,
								'size': cluster_size,
								'intra_sim': intra_sim,
								'labels': clustered_labels
						})
		
		if verbose:
				print(f"\n[DISSOLUTION] Found {len(clusters_to_dissolve)} low cohesion clusters")
				print(f"  Total labels affected: {sum(c['size'] for c in clusters_to_dissolve)}")
		
		if len(clusters_to_dissolve) == 0:
				print("\n✅ No low cohesion clusters found. Nothing to dissolve.")
				return df
		
		# Get next available cluster ID
		max_cluster_id = df['cluster'].max()
		next_cluster_id = max_cluster_id + 1
		
		if verbose:
				print(f"\n[DISSOLVING] Reassigning labels to new clusters...")
		
		# Dissolve each low cohesion cluster
		for cluster_info in clusters_to_dissolve:
				cluster_id = cluster_info['cluster_id']
				cluster_mask = df['cluster'] == cluster_id
				
				for idx in df[cluster_mask].index:
						label_name = df.loc[idx, 'label']
						df.loc[idx, 'cluster'] = next_cluster_id
						df.loc[idx, 'canonical'] = label_name
						next_cluster_id += 1
		
		if verbose:
				print(f"\n[RE-INDEXING] Making cluster IDs contiguous...")
		
		unique_clusters = sorted(df['cluster'].unique())
		cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
		
		df['cluster'] = df['cluster'].map(cluster_mapping)
		
		# statistics
		old_n_clusters = max_cluster_id + 1
		new_n_clusters = df['cluster'].nunique()
		old_consolidation = len(df) / old_n_clusters
		new_consolidation = len(df) / new_n_clusters
		
		if verbose:
				print(f"\n[RESULTS]")
				print(f"  Old clusters: {old_n_clusters:,}")
				print(f"  New clusters: {new_n_clusters:,}")
				print(f"  Change: +{new_n_clusters - old_n_clusters:,}")
				print(f"  Old consolidation: {old_consolidation:.2f}x")
				print(f"  New consolidation: {new_consolidation:.2f}x")
				print(f"  Cluster ID range: 0 to {df['cluster'].max()} (contiguous: {df['cluster'].max() == new_n_clusters - 1})")
				print(f"\n✅ Dissolution complete!")
		
		return df

def fix_poor_canonical_clusters(
	df,
	embeddings,
	threshold=0.60,
	verbose=True
):
	print(f"\n[FIX POOR CANONICAL] Threshold: {threshold}")
	
	# Identify poor canonical clusters
	poor_canonical_clusters = list()
	
	for cluster_id in df['cluster'].unique():
		cluster_mask = df['cluster'] == cluster_id
		clustered_labels = df[cluster_mask]['label'].tolist()
		cluster_size = len(clustered_labels)
		
		if cluster_size < 2:
			continue
		
		cluster_indices = df[cluster_mask].index.tolist()
		cluster_embeddings = embeddings[cluster_indices]
		
		# Get current canonical
		current_canonical = df[cluster_mask]['canonical'].iloc[0]
		canonical_idx = clustered_labels.index(current_canonical)
		canonical_emb = cluster_embeddings[canonical_idx].reshape(1, -1)
		
		# Compute representativeness (avg similarity to all members)
		canonical_rep = sklearn.metrics.pairwise.cosine_similarity(canonical_emb, cluster_embeddings).mean()
		
		if canonical_rep < threshold:
			poor_canonical_clusters.append(
				{
					'cluster_id': cluster_id,
					'current_canonical': current_canonical,
					'representativeness': canonical_rep,
					'size': cluster_size,
					'labels': clustered_labels
				}
			)
	
	if verbose:
		print(f"  Found {len(poor_canonical_clusters)} clusters with poor canonical")
	
	if len(poor_canonical_clusters) == 0:
		print("  ✓ All canonical labels are representative!")
		return df
	
	# Re-select canonical for each poor cluster
	fixed_count = 0
	
	for cluster_info in poor_canonical_clusters:
		cluster_id = cluster_info['cluster_id']
		clustered_labels = cluster_info['labels']
		old_canonical = cluster_info['current_canonical']
		
		cluster_mask = df['cluster'] == cluster_id
		cluster_indices = df[cluster_mask].index.tolist()
		cluster_embeddings = embeddings[cluster_indices]
		
		# Method 1: Centroid-nearest (most representative)
		centroid = cluster_embeddings.mean(axis=0, keepdims=True)
		similarities = sklearn.metrics.pairwise.cosine_similarity(centroid, cluster_embeddings)[0]
		best_idx = similarities.argmax()
		new_canonical = clustered_labels[best_idx]
		new_rep = similarities[best_idx]
		
		# Update canonical in dataframe
		df.loc[cluster_mask, 'canonical'] = new_canonical
		
		fixed_count += 1
		
		if verbose:
			print(f"\nCluster {cluster_id} ({len(clustered_labels)} labels):")
			print(f"\tOld: '{old_canonical}' (rep={cluster_info['representativeness']:.4f})")
			print(f"\tNew: '{new_canonical}' (rep={new_rep:.4f})")
			print(f"\tImprovement: {(new_rep - cluster_info['representativeness']):.4f}")
			print(f"\tLabels: {clustered_labels}")
	
	if verbose:
		print(f"\n✓ Fixed {fixed_count} clusters")
	
	return df

def generate_recommendations(
	global_summary:         dict,
	cluster_df:             pd.DataFrame,
	problematic_clusters:   list,
	consolidation_impact:   dict,
) -> list:
	recs = []

	mean_sim = global_summary['mean_intra_sim']
	if mean_sim < 0.75:
		recs.append(
			f"Mean intra-cluster similarity is low ({mean_sim:.3f}, target >= 0.80). "
			f"Consider raising target_intra_similarity in get_clustered_labels()."
		)
	elif mean_sim >= 0.82:
		recs.append(
			f"Cohesion is strong (mean intra_sim = {mean_sim:.3f}). "
			f"No changes to clustering parameters needed."
		)

	mean_rep = global_summary['mean_canon_rep']
	if mean_rep < 0.82:
		recs.append(
			f"Avg canonical representativeness is {mean_rep:.3f} (target >= 0.85). "
			f"Review the canonical selection weights in assign_canonical_labels()."
		)

	high_sev = [p for p in problematic_clusters if p['severity'] == 'HIGH']
	if high_sev:
		total_labels = sum(p['count'] for p in high_sev)
		recs.append(
			f"{len(high_sev)} HIGH-severity issue type(s) covering ~{total_labels} clusters. "
			f"Check low_cohesion_clusters.json for manual review."
		)

	singleton_pct = consolidation_impact['singleton_percentage']
	if singleton_pct > 15:
		recs.append(
			f"High singleton rate ({singleton_pct:.1f}%). "
			f"Consider lowering min_consolidation in get_clustered_labels()."
		)

	ratio = consolidation_impact['reduction_ratio']
	if ratio < 2.0:
		recs.append(
			f"Consolidation ratio is only {ratio:.1f}x. "
			f"Increase max_consolidation or lower target_intra_similarity."
		)
	elif ratio > 8.0:
		recs.append(
			f"High consolidation ratio ({ratio:.1f}x) — "
			f"clusters may be over-merged. Check very_large_clusters.csv."
		)

	if not recs:
		recs.append("No issues detected. Clustering quality is acceptable.")

	return recs

def get_optimal_super_clusters(
	linkage_matrix,
	embeddings,
	clustered_labels,
	unique_labels,
	linkage_method,
	clusters_fname,
	n_thresholds=50,
	min_clusters=3,
	max_clusters=10,
	verbose=False,
):
	print(f"\n[SUPER-CLUSTERS] Analyzing hierarchy...")

	distances = linkage_matrix[:, 2]
	candidate_distances = np.linspace(distances.min(), distances.max(), n_thresholds)
	best_score = -np.inf
	best_distance = None
	best_n_clusters = None
	print(f"[SUPER-CLUSTERS] Testing {len(candidate_distances)} distance thresholds...")
	for dist in candidate_distances:
			labels = scipy.cluster.hierarchy.fcluster(linkage_matrix, t=dist, criterion='distance')
			n_clusters = len(np.unique(labels))
			if n_clusters < min_clusters or n_clusters > max_clusters:
					continue
			score = silhouette_score(embeddings, labels, metric='cosine')
			if score > best_score:
					best_score = score
					best_distance = dist
					best_n_clusters = n_clusters
	if best_distance is None:
			# Fallback: choose dist whose n_clusters is closest to min_clusters
			print(f"[SUPER-CLUSTERS] No distance produced n_clusters in [{min_clusters}, {max_clusters}]. Falling back...")
			best_gap = np.inf
			for dist in candidate_distances:
					labels = fcluster(linkage_matrix, t=dist, criterion='distance')
					n_clusters = len(np.unique(labels))
					gap = abs(n_clusters - min_clusters)
					if gap < best_gap:
							best_gap = gap
							best_distance = dist
							best_n_clusters = n_clusters
			print(f"[SUPER-CLUSTERS] Fallback: distance={best_distance:.4f}, n_clusters={best_n_clusters}")
	else:
			print(f"[SUPER-CLUSTERS] Best silhouette: {best_score:.4f} at {best_n_clusters} clusters")

	super_cluster_distance, n_super_clusters = best_distance, best_n_clusters

	print(f"[SUPER-CLUSTERS] Optimal distance: {super_cluster_distance:.4f} ({n_super_clusters} super-clusters)")

	super_clustered_labels = scipy.cluster.hierarchy.fcluster(linkage_matrix, t=super_cluster_distance, criterion='distance') - 1

	print(f"\n[VERIFICATION] super-cluster alignment...")
	print(f"  ├─ Distance threshold: {super_cluster_distance:.4f}")
	print(f"  ├─ Expected clusters: {n_super_clusters}")

	# Recompute to verify
	labels_check = scipy.cluster.hierarchy.fcluster(linkage_matrix, t=super_cluster_distance, criterion='distance')
	n_clusters_check = len(np.unique(labels_check))
	print(f"  ├─ Actual clusters from fcluster: {n_clusters_check}")

	if n_clusters_check == n_super_clusters:
		print(f"  └─ Confirmed Alignment: {n_super_clusters} clusters at t={super_cluster_distance:.4f}")
	else:
		print(f"  └─ MISMATCH ALERT: Expected {n_super_clusters}, got {n_clusters_check}")

	# Map fine-grained clusters to super-clusters
	# Get the number of fine-grained clusters dynamically
	n_fine_clusters = len(np.unique(clustered_labels))
	
	# Create mapping: fine_cluster_id -> super_cluster_id
	cluster_to_supercluster = {}
	supercluster_stats = {}
	
	for fine_cluster_id in range(n_fine_clusters):
			# Get all label indices in this fine cluster
			fine_cluster_mask = clustered_labels == fine_cluster_id
			fine_cluster_indices = np.where(fine_cluster_mask)[0]
			
			# Find which super-cluster these labels belong to (majority vote)
			super_ids = super_clustered_labels[fine_cluster_indices]
			super_cluster_id = int(np.bincount(super_ids).argmax())
			
			cluster_to_supercluster[fine_cluster_id] = super_cluster_id
			
			# Track super-cluster stats
			if super_cluster_id not in supercluster_stats:
					supercluster_stats[super_cluster_id] = {
							'fine_clusters': [],
							'total_labels': 0
					}
			supercluster_stats[super_cluster_id]['fine_clusters'].append(fine_cluster_id)
			supercluster_stats[super_cluster_id]['total_labels'] += fine_cluster_mask.sum()
	
	print(f"\n[SUPER-CLUSTER] HIERARCHY")
	print(f"Total fine-grained clusters: {n_fine_clusters}")
	print(f"Total super-clusters: {n_super_clusters}")
	
	for super_id in sorted(supercluster_stats.keys()):
		stats = supercluster_stats[super_id]
		print(f"\n[Super-Cluster {super_id}]")
		print(f"  ├─ Fine clusters: {len(stats['fine_clusters'])} clusters")
		print(f"  ├─ Total labels: {stats['total_labels']} ({stats['total_labels']/len(unique_labels)*100:.1f}%)")
		print(f"  └─ Cluster IDs: {stats['fine_clusters'][:25]}{'...' if len(stats['fine_clusters']) > 25 else ''}")
	
	# 2D cluster visualizations
	plt.figure(figsize=(10, 7))
	scipy.cluster.hierarchy.dendrogram(
		linkage_matrix, 
		truncate_mode='lastp', 
		p=40, 
		show_leaf_counts=True, 
		color_threshold=super_cluster_distance
	)
	# plt.axhline(
	# 	y=super_cluster_distance, 
	# 	color='#000000', 
	# 	linestyle='--', 
	# 	label=f'Cut at {super_cluster_distance:.4f} ({n_super_clusters} super-clusters)',
	# 	linewidth=2.5,
	# 	zorder=10,
	# )

	plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} Linkage)\n{n_super_clusters} Super-Clusters at distance={super_cluster_distance:.4f}')
	plt.xlabel('Cluster')
	plt.ylabel('Distance')
	plt.legend(loc='upper right', fontsize=12)
	plt.grid(False)
	out_dendogram = clusters_fname.replace(".csv", "_dendrogram.png")
	plt.savefig(out_dendogram, dpi=200, bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(24, 15))
	scipy.cluster.hierarchy.dendrogram(
		linkage_matrix,
		color_threshold=super_cluster_distance,
		leaf_font_size=8
	)
	plt.axhline(
		y=super_cluster_distance, 
		color='#000000', 
		linestyle='--', 
		linewidth=2.5,
		label=f'Cut at {super_cluster_distance:.4f}',
		zorder=10
	)
	plt.title(f'Full Dendrogram (All {n_fine_clusters} Fine Clusters)')
	plt.xlabel('Fine Cluster ID')
	plt.ylabel('Distance')
	plt.legend()
	
	out_full_dendogram = clusters_fname.replace(".csv", "_dendrogram_full.png")
	plt.savefig(out_full_dendogram, dpi=200, bbox_inches='tight')
	plt.close()

	# PCA
	pca_projection = PCA(n_components=2, random_state=0).fit_transform(embeddings)
	
	# t-SNE (subsample if too large)
	if len(unique_labels) > 10000:
		tsne_indices = np.random.choice(len(unique_labels), 10000, replace=False)
		tsne_projection = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(embeddings[tsne_indices])
		tsne_labels = clustered_labels[tsne_indices]
	else:
		tsne_projection = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(embeddings)
		tsne_labels = clustered_labels
	
	# Color palette
	n_colors = min(len(clustered_labels), 256)
	palette = sns.color_palette('tab20', n_colors) if n_colors <= 20 else sns.color_palette('husl', n_colors)
	colors = [palette[i % len(palette)] for i in clustered_labels]
	# PCA plot
	plt.figure(figsize=(27, 17))
	plt.scatter(*pca_projection.T, s=40, c=colors, alpha=0.6, marker='o')
	plt.title(f"PCA - Agglomerative Clustering ({len(clustered_labels)} clusters, {len(unique_labels)} labels)")
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	out_pca = clusters_fname.replace(".csv", "_pca_agglomerative.png")
	plt.savefig(out_pca, dpi=150, bbox_inches='tight')
	plt.close()
	
	# t-SNE plot
	tsne_colors = [palette[i % len(palette)] for i in tsne_labels]
	plt.figure(figsize=(27, 17))
	plt.scatter(*tsne_projection.T, s=40, c=tsne_colors, alpha=0.6, marker='o')
	plt.title(f"t-SNE - Agglomerative Clustering ({len(clustered_labels)} clusters)")
	plt.xlabel("t-SNE 1")
	plt.ylabel("t-SNE 2")
	out_tsne = clusters_fname.replace(".csv", "_tsne_agglomerative.png")
	plt.savefig(out_tsne, dpi=150, bbox_inches='tight')
	plt.close()

def get_clustered_labels(
	X: np.ndarray,
	linkage_matrix: np.ndarray,
	label_texts: List[str],
	min_cluster_size: int,
	merge_singletons: bool,
	target_intra_similarity: float,
	min_consolidation: float,
	max_consolidation: float,
	min_singleton_ratio: float,
	max_singleton_ratio: float,
	target_singleton_ratio: float,
	quality_vs_consolidation_weight: float,
	min_singleton_merge_sim: Optional[float],
	nc: Optional[int]=None,
	verbose: bool=False,
):
	if nc:
		clustered_labels = scipy.cluster.hierarchy.fcluster(
			Z=linkage_matrix,
			t=nc,
			criterion='maxclust'
		) - 1 # make it 0‑based # <class 'numpy.ndarray'> (num_samples,)
		unique_labels = np.unique(clustered_labels)
		if verbose:
			num_clusters = len(unique_labels)
			print(f"\nManually defined number of clusters (nc) k={nc}")
			print(f"  ├─ clustered_labels: {type(clustered_labels)} {clustered_labels.shape} {clustered_labels}")
			print(f"  └─ {len(unique_labels)} labels mapped to {num_clusters} clusters")
		
		return clustered_labels

	num_samples = X.shape[0]
	valid_k_min = int(num_samples // max_consolidation)
	valid_k_max = int(num_samples // min_consolidation)
	if verbose:
		print("\n[ADAPTIVE OPTIMAL CLUSTER SELECTION]")
		print(f"  ├─ num_samples                     {num_samples}")
		print(f"  ├─ Embeddings (X)                  {type(X)} {X.shape} {X.dtype}")
		print(f"  ├─ Linkage (Z)                     {type(linkage_matrix)} {linkage_matrix.shape}")
		print(f"  ├─ Min cluster size                {min_cluster_size}")
		print(f"  ├─ Merge singletons                {merge_singletons}")
		print(f"  ├─ Consolidation                   {min_consolidation}x - {max_consolidation}x")
		print(f"  ├─ Required clusters               {valid_k_min} ≤ k ≤ {valid_k_max}")
		print(f"  ├─ Target singleton ratio          {target_singleton_ratio}")
		print(f"  ├─ Quality weight                  {quality_vs_consolidation_weight*100}%")
		print(f"  └─ Target intra-cluster similarity {target_intra_similarity}")
	
	# Adaptive coarse range based on dataset size
	if num_samples > int(3e4):
		coarse_step = 1000
	elif num_samples > int(1e4):
		coarse_step = 500
	elif num_samples > int(5e3):
		coarse_step = 100
	else:
		coarse_step = max(1, (valid_k_max - valid_k_min) // 5)  # ~5 steps in valid zone
	
	# Extend range to cover from 10% of valid_k_min up to valid_k_max
	coarse_start = max(10, valid_k_min // 2)
	coarse_end = valid_k_max + coarse_step # Use overshoot to ensure valid_k_max is covered and peeked
	coarse_range = range(coarse_start, coarse_end, coarse_step)  # explicit +1, drop overshoot

	if verbose:
		print(f"\n[STAGE 1] COARSE SEARCH - Finding quality plateau: {valid_k_min} ≤ k ≤ {valid_k_max}")
		print(f"\t{len(coarse_range)} Testing Config {list(coarse_range)} (step={coarse_step})")
		print(f"\n{'k':<8} {'IntraSim':<12} {'Consol':<10} {'SingleR':<10} {'Status':<50} {'Reason'}")
		print("-" * 200)
	
	coarse_results = list()
	best_intra_sim = 0
	plateau_k = None
	
	for n_clusters in coarse_range:
		labels = scipy.cluster.hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
		
		if len(np.unique(labels)) < 2:
			continue
		
		# Mean intra-cluster Cosine similarity
		unique_labels = np.unique(labels)
		intra_sims = list()
		
		for cid in unique_labels:
			cluster_mask = labels == cid
			cluster_X = X[cluster_mask]
			
			if len(cluster_X) > 1:
				sim_matrix = sklearn.metrics.pairwise.cosine_similarity(cluster_X)
				n = len(cluster_X)
				intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
				intra_sims.append(intra_sim)
		
		mean_intra_sim = np.mean(intra_sims) if intra_sims else 0
		
		# Cluster statistics
		cluster_sizes = np.bincount(labels)
		n_singletons = np.sum(cluster_sizes == 1)

		singleton_ratio = n_singletons / n_clusters
		consolidation = num_samples / n_clusters
		
		coarse_results.append(
			{
				'k': n_clusters,
				'intra_sim': mean_intra_sim,
				'consolidation': consolidation,
				'singleton_ratio': singleton_ratio,
				'n_singletons': n_singletons
			}
		)
		
		# Check if in target range
		in_consol_range = min_consolidation <= consolidation <= max_consolidation
		in_singleton_range = min_singleton_ratio <= singleton_ratio <= max_singleton_ratio

		status = ""
		reason = ""
		if mean_intra_sim >= target_intra_similarity and in_consol_range:
			status = "✓ TARGET REACHED (quality + consolidation)"
			reason = (
				f"IntraSim {mean_intra_sim:.4f} ≥ {target_intra_similarity} "
				f"AND consol {consolidation:.2f}x in [{min_consolidation}, {max_consolidation}]"
			)
			if plateau_k is None:
				plateau_k = n_clusters
		elif in_consol_range and in_singleton_range:
			status = "✓ OPTIMAL RANGE (consolidation + singletons)"
			reason = (
				f"Consol {consolidation:.2f}x in [{min_consolidation}, {max_consolidation}] "
				f"AND singleton: {n_singletons} {singleton_ratio*100:.3f}%"
			)
			if plateau_k is None:
				plateau_k = n_clusters
		elif mean_intra_sim > best_intra_sim:
			best_intra_sim = mean_intra_sim
			status = "↑ Staying on track"
			reason = (
				f"IntraSim {mean_intra_sim:.4f} > prev best {best_intra_sim:.4f}"
				+ (f" | consol {consolidation:5.2f}x outside [{min_consolidation},{max_consolidation}]" if not in_consol_range else "")
				+ (f" | singleton {singleton_ratio:.4f} outside [0.005,0.03]" if not in_singleton_range else "")
			)
		else:
			status = "→ Plateau region"
			reason = (
				f"IntraSim {mean_intra_sim:.4f} ≤ prev best {best_intra_sim:.4f} (no improvement)"
			)
		
		if verbose:
			print(f"{n_clusters:<8} {mean_intra_sim:<12.4f} {consolidation:<10.2f} {singleton_ratio:<10.4f} {status:<50} {reason}")

		# Early stopping: If in optimal range for 2 consecutive steps
		if len(coarse_results) >= 2:
			recent = coarse_results[-2:] # # Looks at the last TWO evaluations
			both_optimal = all(
				r['intra_sim'] >= target_intra_similarity * 0.95 and
				min_consolidation <= r['consolidation'] <= max_consolidation
				for r in recent
			)
			if both_optimal and plateau_k is not None:
				break

	if not coarse_results:
		raise ValueError("No valid cluster configurations found in coarse search")
	
	max_observed_intra = max(r['intra_sim'] for r in coarse_results)
	gap = (target_intra_similarity - max_observed_intra) / target_intra_similarity
	if gap > 0.10:  # Target is >10% above what's achievable
		adjusted_target = max_observed_intra * 1.02  # 2% headroom above observed max
		if verbose:
			print(f"\n[STAGE 1] ⚠ Target intra-sim {target_intra_similarity:.2f} unreachable.")
			print(f"           Max observed: {max_observed_intra:.4f} (gap={gap*100:.1f}%)")
			print(f"           Auto-adjusting target → {adjusted_target:.4f}")
		target_intra_similarity = adjusted_target

	# Determine search region for Stage 2
	if plateau_k is None:
		# Use k that best balances quality and consolidation
		def score_coarse(r):
			# Penalize if outside consolidation range
			consol_penalty = 1.0
			
			if r['consolidation'] < min_consolidation:
				consol_penalty = 0.5
			elif r['consolidation'] > max_consolidation:
				consol_penalty = 0.7
			
			# Reward if near target singleton ratio
			singleton_penalty = 1.0 - abs(r['singleton_ratio'] - target_singleton_ratio)
			singleton_penalty = max(0.5, singleton_penalty)
			
			return r['intra_sim'] * consol_penalty * singleton_penalty
		
		best_coarse = max(coarse_results, key=score_coarse)
		plateau_k = best_coarse['k']
		
	if verbose:
		print(f"\n[STAGE 2] FINE SEARCH - Optimizing around plateau: k={plateau_k}")
	
	# Fine search range: ±20% around plateau with smaller steps
	fine_min = max(int(plateau_k * 0.8), valid_k_min)  # Never go below valid zone
	_step_proxy = max(1, (valid_k_max - valid_k_min) // 20)  # temporary, only for fine_max
	# Allow exactly 2 fine steps beyond the strict boundary
	fine_max = min(
		int(plateau_k * 1.2), # Strict 20% above plateau
		int(num_samples // min_consolidation) + 2 * _step_proxy  # +2 steps of headroom
	)
	
	if verbose:
		print(f"Pareto Principle [80:20]: {fine_min} ≤ k ≤ {fine_max}")

	# Guard: if plateau_k was outside valid range, just search the valid range
	if fine_min >= fine_max:
		if verbose:
			print(f"[WARNING] Plateau k={plateau_k} outside valid range: {valid_k_min} ≤ k ≤ {valid_k_max}. Searching valid range only.")
		fine_min = valid_k_min
		fine_max = valid_k_max

	fine_step = max(1, (fine_max - fine_min) // 20)  # ← always ~20 evaluations
	fine_range = range(fine_min, fine_max + 1, fine_step)
	
	if verbose:
		print(f"\t{len(fine_range)} Testing Config {list(fine_range)} (step={fine_step})")
		print(f"\n{'k':<8} {'IntraSim':<12} {'Consol':<10} {'SingleR':<10} {'Score':<10} {'Status':<20} {'Reason'}")
		print("-" * 150)
	
	fine_results = list()
	for n_clusters in fine_range:
		labels = scipy.cluster.hierarchy.fcluster(
			Z=linkage_matrix, 
			t=n_clusters, 
			criterion='maxclust'
		) - 1
		
		unique_labels = np.unique(labels)

		if len(unique_labels) < 2:
			continue
		
		# Compute intra-cluster similarity
		intra_sims = list()
		for cid in unique_labels:
			cluster_mask = labels == cid
			cluster_X = X[cluster_mask]
			
			if len(cluster_X) > 1:
				sim_matrix = sklearn.metrics.pairwise.cosine_similarity(cluster_X)
				n = len(cluster_X)
				intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
				intra_sims.append(intra_sim)
		
		mean_intra_sim = np.mean(intra_sims) if intra_sims else 0
		
		# Compute statistics
		cluster_sizes = np.bincount(labels)
		n_singletons = np.sum(cluster_sizes == 1)
		singleton_ratio = n_singletons / n_clusters
		consolidation = num_samples / n_clusters
		
		# COMPOSITE SCORING FUNCTION (Key Innovation!)
		# Quality component (normalized to 0-1)
		quality_score = mean_intra_sim / target_intra_similarity
		quality_score = min(1.0, quality_score)  # Cap at 1.0
		
		# Consolidation component (penalize if outside range)
		if consolidation < min_consolidation:
			consol_score = consolidation / min_consolidation  # Penalize low consolidation
		elif consolidation > max_consolidation:
			consol_score = max_consolidation / consolidation  # Penalize high consolidation
		else:
			consol_score = 1.0  # Perfect
		
		# Singleton component (penalize if far from target)
		singleton_error = abs(singleton_ratio - target_singleton_ratio)
		singleton_score = 1.0 - min(1.0, singleton_error / target_singleton_ratio)
		
		# Final composite score
		score = (
			quality_vs_consolidation_weight * quality_score +
			(1 - quality_vs_consolidation_weight) * 0.7 * consol_score +
			(1 - quality_vs_consolidation_weight) * 0.3 * singleton_score
		)
		
		fine_results.append(
			{
				'k': n_clusters,
				'intra_sim': mean_intra_sim,
				'consolidation': consolidation,
				'singleton_ratio': singleton_ratio,
				'n_singletons': n_singletons,
				'quality_score': quality_score,
				'consol_score': consol_score,
				'singleton_score': singleton_score,
				'composite_score': score
			}
		)
		
		# Status + Reason
		status = ""
		reason = ""
		if quality_score >= 0.95 and consol_score >= 0.9:
			status = "EXCELLENT"
			reason = (
				f"QualityScore {quality_score:.3f} ≥ 0.95 "
				f"AND ConsolScore {consol_score:.3f} ≥ 0.90 "
				f"| SingletonScore {singleton_score:.3f}"
			)
		elif quality_score >= 0.90 and consol_score >= 0.8:
			status = "✓ GOOD"
			reason = (
				f"QualityScore {quality_score:.3f} ≥ 0.90 "
				f"AND ConsolScore {consol_score:.3f} ≥ 0.80 "
				f"| SingletonScore {singleton_score:.3f}"
			)
		elif score >= 0.70:
			status = "→ Acceptable"
			reason = (
				f"CompositeScore {score:.3f} ≥ 0.70 "
				f"| QualityScore {quality_score:.3f} ConsolScore {consol_score:.3f} SingletonScore {singleton_score:.3f}"
			)
		else:
			status = "✗ Below target"
			reason = (
				f"CompositeScore {score:.3f} < 0.70 "
				f"| QualScore {quality_score:.3f} ConsolScore {consol_score:.3f} SingletonScore {singleton_score:.3f}"
			)

		if verbose:
			print(f"{n_clusters:<8} {mean_intra_sim:<12.4f} {consolidation:<10.2f} {singleton_ratio:<10.4f} {score:<10.4f} {status:<20}{reason}")
	
	if not fine_results:
		raise ValueError("No valid cluster configurations found in fine search")
	
	# SELECT BEST CONFIGURATION
	# Priority: Highest composite score
	best = max(fine_results, key=lambda x: x['composite_score'])
	
	if verbose:
		print(f"\n[STAGE 2]")
		print(f"  ├─ k={best['k']} (highest composite score)")
		print(f"  ├─ Composite score: {best['composite_score']:.4f}")
		print(f"  ├─ Quality score: {best['quality_score']:.4f}")
		print(f"  ├─ Consolidation score: {best['consol_score']:.4f}")
		print(f"  ├─ Singleton score: {best['singleton_score']:.4f}")
		print(f"  └─ Intra-similarity: {best['intra_sim']:.4f}")
		print()
	
	# STAGE 3: Merge singletons
	labels = scipy.cluster.hierarchy.fcluster(
		linkage_matrix, 
		best['k'], 
		criterion='maxclust'
	) - 1 # <class 'numpy.ndarray'> (num_samples,)

	# Count number of occurrences of each value in array of non-negative ints.
	cluster_sizes = np.bincount(labels)
	singleton_clusters_indices = np.where(cluster_sizes == 1)[0]
	if len(singleton_clusters_indices) > 0:
		if verbose:
			print(f"[WARNING] {len(singleton_clusters_indices)} singleton cluster(s), e.g., containing only one label")
			print("[SINGLETONS BEFORE MERGING]")
			for singleton_id in singleton_clusters_indices:
				singleton_idx = np.where(labels == singleton_id)[0][0]
				print(f"  ├─ {singleton_id:5d} {label_texts[singleton_idx]}")

	kept_singletons = [] # singletons left alone because their best cluster was too dissimilar
	if len(singleton_clusters_indices) > 0 and merge_singletons:
		if verbose:
			print(f"\n[STAGE 3] MERGING {len(singleton_clusters_indices)} SINGLETON CLUSTERS")

		unique_labels = np.unique(labels)
		centroids = np.array([X[labels == cid].mean(axis=0) for cid in unique_labels])
		new_labels = labels.copy()
		merged_count = 0

		for singleton_id in singleton_clusters_indices:
			singleton_idx = np.where(labels == singleton_id)[0][0]
			singleton_vec = X[singleton_idx].reshape(1, -1)
			sims = sklearn.metrics.pairwise.cosine_similarity(singleton_vec, centroids)[0]
			sorted_ids = np.argsort(sims)[::-1]
 
			# Find nearest non-singleton cluster
			for nearest_id in sorted_ids:
				if cluster_sizes[nearest_id] < min_cluster_size:
					continue

				# Similarity floor: if even the best cluster is a weak match, the label
				# stays alone (it then simply names itself) instead of being forced into
				# a cluster it does not belong to (forced merges at sim 0.60-0.65 were seen).
				if min_singleton_merge_sim is not None and sims[nearest_id] < min_singleton_merge_sim:
					kept_singletons.append((label_texts[singleton_idx], float(sims[nearest_id]), int(nearest_id)))
					status = "KEPT"
					if verbose:
						print(
							f"  ├─ [{status:7s}] singleton cluster {singleton_id:5d} {repr(label_texts[singleton_idx]):<65}"
							f"best sim={sims[nearest_id]:.4f} < {min_singleton_merge_sim}"
						)
					break

				new_labels[singleton_idx] = nearest_id
				merged_count += 1
				status = "MERGED"

				if verbose:
					print(
						f"  ├─ [{status:7s}] singleton cluster {singleton_id:5d} {repr(label_texts[singleton_idx]):<65} "
						f"=> cluster {nearest_id:5d} sim: {sims[nearest_id]:.4f}"
					)

				break

		unique_new = np.unique(new_labels)
		label_map = {
			old: new 
			for new, old in enumerate(unique_new)
		}
		labels = np.array([label_map[l] for l in new_labels])
		
		if verbose:
			print(f"  ├─ Total merged: {merged_count} singleton(s)")
			print(f"  └─ Final clusters: {len(unique_new)} (initial: {len(unique_labels)})")

	# FINAL STATISTICS
	final_cluster_sizes = np.bincount(labels)
	final_n_clusters = len(np.unique(labels))

	final_singletons = np.sum(final_cluster_sizes == 1)
	final_max_size = final_cluster_sizes.max()
	
	# Recompute final intra-similarity
	final_intra_sims = list()
	for cid in np.unique(labels):
		cluster_X = X[labels == cid]
		if len(cluster_X) > 1:
			sim_matrix = sklearn.metrics.pairwise.cosine_similarity(cluster_X)
			n = len(cluster_X)
			intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
			final_intra_sims.append(intra_sim)
	
	final_mean_intra_sim = np.mean(final_intra_sims) if final_intra_sims else 0
	final_std_intra_sim = np.std(final_intra_sims) if final_intra_sims else 0
	
	stats = {
		'n_clusters': final_n_clusters,
		'max_cluster_size': int(final_max_size),
		'n_singletons': int(final_singletons),
		'max_size_ratio': round(float(final_max_size / num_samples), 4),	
		'mean_cluster_size': round(float(num_samples / final_n_clusters), 4),
		'consolidation_ratio': round(float(num_samples / final_n_clusters), 4),
		'mean_intra_similarity': round(float(final_mean_intra_sim), 4),
		'std_intra_similarity': round(float(final_std_intra_sim), 4),
		'singleton_ratio': round(float(final_singletons / final_n_clusters), 4) if final_n_clusters > 0 else 0,
		'n_singletons_kept_by_floor': len(kept_singletons),
	}
	
	if verbose:
		print(f"\n[STATISTICS]\n{json.dumps(stats, indent=2, ensure_ascii=False)}")

	return labels

def remove_problematic_clustered_labels(
	df,
	embeddings,
	low_cohesion_threshold: float,
	poor_canonical_threshold: float,
	min_labels_per_cluster: int = 2,
	verbose=False
):
	"""
	Remove ALL labels from problematic clusters.

	Removes labels from:
		1. Low-cohesion clusters (intra_sim < threshold)
		2. Poor canonical clusters (canonical_rep < threshold)

	Amputation rather than surgery:
	This is an aggressive but clean approach that eliminates
	problematic labels entirely rather than trying to fix them.

	Invariant (enforced by the caller — Step 8 of cluster())
	---------------------------------------------------------
	Every cluster's canonical is guaranteed to be a real row in df before
	this function is called. 
	Virtual hypernyms are injected as genuine rows into df+X in Step 8, 
	so the lookup `clustered_labels.index(canonical)` is
	always safe and the old "canonical not in clustered_labels" guard is gone.

	Parameters
	----------
	df : pd.DataFrame
		Clustering results with ['label', 'cluster', 'canonical'].
		df.index must be 0-based and contiguous (reset_index applied upstream).
	embeddings : np.ndarray
		Label embeddings in the same row-order as df.
	low_cohesion_threshold : float
		Intra-similarity threshold for low-cohesion detection.
	poor_canonical_threshold : float
		Canonical representativeness threshold.
	verbose : bool
		Print detailed statistics.

	Returns
	-------
	df_clean : pd.DataFrame
		Cleaned clustering with problematic labels removed.
	embeddings_clean : np.ndarray
		Embeddings aligned with df_clean.
	removed_labels : list
		List of removed labels for reference.
	"""
	if verbose:
		print("="*100)
		print(f"[DETECTION] Problematic clusters and labels")
		print(f"  ├─ clusters                  {len(df['cluster'].unique())}")
		print(f"  ├─ X                         {type(embeddings)} {embeddings.shape}")
		print(f"  ├─ [threshold] Low cohesion: {low_cohesion_threshold} Poor canonical: {poor_canonical_threshold}")
		print(f"  └─ min_labels_per_cluster    {min_labels_per_cluster}")

	problematic_cluster_ids = set()
	removed_labels = list()

	# PART 1: Identify Low-Cohesion Clusters
	low_cohesion_clusters = list()
	for cluster_id in df['cluster'].unique():
		cluster_mask = df['cluster'] == cluster_id
		clustered_labels = df[cluster_mask]['label'].tolist()
		cluster_size = len(clustered_labels)
		current_canonical = df[cluster_mask]['canonical'].iloc[0]
		canonical_idx = clustered_labels.index(current_canonical)

		if cluster_size < min_labels_per_cluster:
			continue

		cluster_indices = df[cluster_mask].index.tolist()
		cluster_embeddings = embeddings[cluster_indices]

		n = len(cluster_embeddings)
		sim_matrix = sklearn.metrics.pairwise.cosine_similarity(cluster_embeddings)

		intra_sim  = (sim_matrix.sum() - n) / (n * (n - 1))

		if intra_sim < low_cohesion_threshold:
			low_cohesion_clusters.append(
				{
					'cluster_id': cluster_id,
					'intra_sim':  intra_sim,
					'size':       cluster_size,
					'labels':     clustered_labels,
					'canonical':  current_canonical,
				}
			)
			problematic_cluster_ids.add(cluster_id)
			removed_labels.extend(clustered_labels)

	if low_cohesion_clusters and verbose:
		print(f"\n[LOW COHESION] {len(low_cohesion_clusters)} clusters ( < {low_cohesion_threshold})")
		for i, cluster in enumerate(low_cohesion_clusters):
			print(
				f"{i+1:3d}/{len(low_cohesion_clusters)} "
				f"Cluster {cluster['cluster_id']:5d} "
				f"intra_sim: {cluster['intra_sim']:.5f} "
				f"canonical: {cluster['canonical']:<35}"
				f"{cluster['labels']}"
			)

	# PART 2: Identify Poor Canonical Clusters:
	poor_canonical_clusters = list()
	for cluster_id in df['cluster'].unique():
		if cluster_id in problematic_cluster_ids:
			continue  # Already marked for removal

		cluster_mask = df['cluster'] == cluster_id
		clustered_labels = df[cluster_mask]['label'].tolist()
		cluster_size = len(clustered_labels)

		if cluster_size < min_labels_per_cluster:
			continue

		cluster_indices = df[cluster_mask].index.tolist()
		cluster_embeddings = embeddings[cluster_indices]

		# Canonical is guaranteed to be in clustered_labels (see docstring invariant).
		current_canonical = df[cluster_mask]['canonical'].iloc[0]
		canonical_idx = clustered_labels.index(current_canonical)
		canonical_emb = cluster_embeddings[canonical_idx].reshape(1, -1)

		canonical_representativeness = sklearn.metrics.pairwise.cosine_similarity(canonical_emb, cluster_embeddings).mean()

		if canonical_representativeness < poor_canonical_threshold:
			poor_canonical_clusters.append(
				{
					'cluster_id': cluster_id,
					'canonical': current_canonical,
					'canonical_representativeness': canonical_representativeness,
					'size': cluster_size,
					'labels': clustered_labels,
				}
			)
			problematic_cluster_ids.add(cluster_id)
			removed_labels.extend(clustered_labels)

	if poor_canonical_clusters and verbose:
		print(
			f"\n[POOR CANONICAL] {len(poor_canonical_clusters)} clusters (th: {poor_canonical_threshold}) -> "
			f"Labels to remove: {sum(c['size'] for c in poor_canonical_clusters)}")
		for i, cluster in enumerate(poor_canonical_clusters):
			print(f"{i+1:3d}/{len(poor_canonical_clusters)} Cluster {cluster['cluster_id']:5d} rep: {cluster['canonical_representativeness']:.3f} canonical: {cluster['canonical']:<27} {cluster['labels']}")
	
	if verbose:
		print(f"\n[REMOVAL SUMMARY]")
		print(f"  ├─ problematic clusters {len(problematic_cluster_ids):6d} = {len(poor_canonical_clusters)} (poor canonical) + {len(low_cohesion_clusters)} (low cohesion)")
		print(f"  ├─ problematic labels   {len(removed_labels):6d} = {sum(c['size'] for c in poor_canonical_clusters)} (poor canonical) + {sum(c['size'] for c in low_cohesion_clusters)} (low cohesion)")
		print(f"  ├─ labels to remove     {len(removed_labels):6d}/{len(df):<10}({len(removed_labels)/len(df)*100:.3f}%)")
		print(f"  └─ clusters to remove   {len(problematic_cluster_ids):6d}/{len(df['cluster'].unique()):<10}({len(problematic_cluster_ids)/len(df['cluster'].unique())*100:.3f}%)")

	# PART 3: Remove Problematic Labels
	df_clean = df[~df['cluster'].isin(problematic_cluster_ids)].copy()
	kept_indices = df_clean.index.tolist()
	embeddings_clean = embeddings[kept_indices]

	# Re-index cluster IDs to be contiguous
	unique_clusters  = sorted(df_clean['cluster'].unique())
	cluster_mapping  = {
		old_id: new_id 
		for new_id, old_id in enumerate(unique_clusters)
	}
	df_clean['cluster'] = df_clean['cluster'].map(cluster_mapping)
	df_clean = df_clean.reset_index(drop=True)

	if verbose:
		original_consolidation = len(df) / df['cluster'].nunique()
		new_consolidation      = len(df_clean) / df_clean['cluster'].nunique()

		print(f"\n[RESULTS]")
		print(f"df            {df.shape} -> {df_clean.shape} (Removed labels: {len(df) - len(df_clean):,})")
		print(f"embeddings    {embeddings.shape} -> {embeddings_clean.shape}")
		print(f"clusters      {df['cluster'].nunique():,} -> {df_clean['cluster'].nunique():,} (Removed: {df['cluster'].nunique() - df_clean['cluster'].nunique():,})")		
		print(f"consolidation {original_consolidation:.4f}x -> {new_consolidation:.4f}x (diff: {(new_consolidation - original_consolidation):.5f}x)")
		print("="*100)

	return df_clean, embeddings_clean, removed_labels

_HARMONIZE_PROTECTED_PLURALS = {
	# plurale tantum
	"pants", "shorts", "glasses", "scissors", "pliers", "tongs", "trousers",
	"binoculars", "goggles", "barracks", "headquarters", "clothes", "belongings",
	"remains", "surroundings", "outskirts", "archives", "overalls",
	# protected plurals / proper names
	"united nations", "united states", "howards", "reins", "airlines", "marines",
	"stables", "lines", "life savers", "pyrotechnics", "general motors",
	"marine corps", "corps",
	# plural is a different word from the singular
	"arms", "papers", "grounds", "works", "steelworks", "quarters", "customs",
	"goods", "colors", "colours", "forces", "spectacles", "manners", "means",
	"premises", "provisions", "letters", "minutes", "terms",

	"breeches", "pantaloons", "tights", "drawers", "leggings", "bloomers",
	"braces", "suspenders", "bifocals",
}

def _spacing_key(s: str) -> str:
	"""Lowercase, letters and digits only ('ice breaker' == 'icebreaker'; 'C-46' != 'C-47')."""
	return re.sub(r'[^a-z0-9]', '', s.lower())

def _is_acronym(s: str) -> bool:
	letters = re.sub(r'[^A-Za-z]', '', s)
	return len(letters) >= 2 and letters.isupper() and len(s.split()) == 1

def _cap_score(s: str) -> int:
	return sum(1 for c in s if c.isupper())

def _singular_candidates(low: str) -> List[str]:
	out = []
	if low.endswith("ies") and len(low) > 4:
		out.append(low[:-3] + "y")                      # batteries -> battery
	if low.endswith(("ches", "shes", "xes", "zes", "sses", "oes")) and len(low) > 4:
		out.append(low[:-2])                            # trenches -> trench
	if low.endswith("s") and not low.endswith("ss") and len(low) > 3:
		out.append(low[:-1])                            # cars -> car
	return out

def _canonical_group_keys(
	cluster_canonicals: Dict[int, Dict],
	protected_plurals: Optional[set] = None,
) -> Tuple[Dict[int, str], Dict[str, str]]:
	"""
	Single definition of "these canonicals are the same name", shared by
	_resolve_shared_canonicals and _harmonize_final_canonicals.

	Two canonicals have the same key when they differ only by case, spacing/punctuation,
	dotted-acronym form, or an attested singular/plural ('tank' / 'tanks' / 'Tank').
	A plural maps to its singular only if the singular is itself a canonical of some
	cluster and the plural is not protected ('arms', 'quarters', ...).

	Returns (keys, plural_of): keys[cid] -> group key; plural_of[surface] -> singular spacing key.
	"""
	protected = _HARMONIZE_PROTECTED_PLURALS if protected_plurals is None else protected_plurals

	surfaces = {cid: m['canonical'] for cid, m in cluster_canonicals.items() if m.get('canonical')}
	distinct = set(surfaces.values())
	attested_keys = {_spacing_key(s) for s in distinct if not _is_acronym(s)}

	plural_of: Dict[str, str] = {}
	for s in distinct:
		low = s.lower()
		if _is_acronym(s) or low in protected or low.endswith("ss"):
			continue
		for cand in _singular_candidates(low):
			k = _spacing_key(cand)
			if k in attested_keys and k != _spacing_key(low):
				plural_of[s] = k
				break

	keys = {
		cid: ("ACRONYM::" + _spacing_key(s)) if _is_acronym(s) else plural_of.get(s, _spacing_key(s))
		for cid, s in surfaces.items()
	}
	return keys, plural_of

def _pick_group_surface(
	variants,
	size_by_surface: Dict[str, int],
	plural_of: Dict[str, str],
	original_label_counts: Dict[str, int],
) -> str:
	"""Winner among surface variants: singular first, real corpus label, frequency, size, capitalisation, lexicographic."""
	pool = [s for s in variants if s not in plural_of] or list(variants)
	return max(
		pool,
		key=lambda s: (
			s in original_label_counts,
			original_label_counts.get(s, 0),
			size_by_surface[s],
			_cap_score(s),
			s,
		),
	)

def _label_tokens(label: str) -> set:
	"""Lowercased tokens with trailing punctuation stripped (same as selection)."""
	return {t for t in (re.sub(r'[^\w]+$', '', w.lower()) for w in label.split()) if t}

def _tokens_match(a: str, b: str) -> bool:
	"""Equal, or regular plural of each other ('tank'/'tanks', 'battery'/'batteries', 'trench'/'trenches').
	Deliberately no fuzzy matching and no -ing stripping ('stand' != 'standing')."""
	if a == b:
		return True
	if len(a) < 3 or len(b) < 3:
		return False
	for x, y in ((a, b), (b, a)):
		if y == x + "s" or y == x + "es":
			return True
		if x.endswith("y") and y == x[:-1] + "ies":
			return True
	return False

def _name_matches_label(name_toks: set, label: str) -> bool:
	"""True if every token of the canonical name is matched (plural-aware) by some token of `label`."""
	if not name_toks:
		return False
	label_toks = _label_tokens(label)
	return all(any(_tokens_match(n, l) for l in label_toks) for n in name_toks)

_CORE_PREPOSITIONS = frozenset(
	"of in on at with over under near from by for during into onto across along above below "
	"behind beside between inside outside through toward towards upon within without after "
	"before around against".split()
)

def _stem_token(t: str) -> str:
	"""Crude plural stemmer used only to compare phrases ('tanks' -> 'tank', 'batteries' -> 'battery')."""
	if t.endswith("ies") and len(t) > 4:
		return t[:-3] + "y"
	if t.endswith(("ches", "shes", "xes", "zes", "sses")) and len(t) > 4:
		return t[:-2]
	if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
		return t[:-1]
	return t

def _stem_seq(label: str) -> List[str]:
	return [_stem_token(t) for t in re.findall(r"[a-z0-9]+", label.lower())]

def _head_phrase_stems(label: str) -> List[str]:
	"""Stemmed tokens before the first preposition ('storage tank room' keeps all; 'tank of water' -> ['tank'])."""
	out = []
	for t in re.findall(r"[a-z0-9]+", label.lower()):
		if out and t in _CORE_PREPOSITIONS:
			break
		out.append(_stem_token(t))
	return out

def _build_label_index(original_label_counts: Dict[str, int]) -> Dict[str, str]:
	"""stem-key -> most frequent corpus surface ('storagetank' -> 'storage tank')."""
	best: Dict[str, Tuple[int, str]] = {}
	for label, cnt in original_label_counts.items():
		key = "".join(_stem_seq(label))
		if key and (key not in best or (cnt, label) > best[key]):
			best[key] = (cnt, label)
	return {k: v[1] for k, v in best.items()}

def _attested_core(
	members: List[str],
	surface: str,
	label_index: Dict[str, str],
	taken_keys: set,
	min_support_frac: float = 0.5,
) -> Optional[Tuple[str, bool]]:
	"""
	For a cluster demoted from the shared name `surface`, look for a QUALIFIED core:
	the name plus 1-2 qualifier words, taken from the END of the members' head phrases
	('storage tank', 'dressing station'), which is

	  * strictly longer than the shared name (a bare name is what we are leaving),
	  * supported by >= max(2, ceil(min_support_frac * n)) members (joint, contiguous),
	  * ATTESTED: it is itself a corpus label (plural-insensitive), and
	  * not already the canonical of another cluster (that would create a new shared name).

	Returns (surface_label, is_virtual) or None. is_virtual is True when the label is
	not one of this cluster's members, in which case it must be injected like a virtual.
	"""
	name = _stem_seq(surface)
	n = len(members)
	if not name or n < 2:
		return None
	need = max(2, -(-int(min_support_frac * 100) * n // 100))  # ceil without importing math
	heads = [_head_phrase_stems(m) for m in members]
	L = len(name)

	cands = set()
	for h in heads:
		if len(h) > L and h[-L:] == name:
			for q in (1, 2):
				if len(h) >= L + q:
					cands.add(tuple(h[-(L + q):]))
	scored = []
	for c in cands:
		support = sum(1 for h in heads if len(h) >= len(c) and tuple(h[-len(c):]) == c)
		if support >= need:
			scored.append((support, len(c), c))
	for support, _, c in sorted(scored, key=lambda t: (-t[0], -t[1], t[2])):
		key = "".join(c)
		if key in taken_keys:
			continue
		label = label_index.get(key)
		if label is None:
			continue
		member = next((m for m in members if "".join(_stem_seq(m)) == key), None)
		return (member, False) if member is not None else (label, True)
	return None

def _resolve_shared_canonicals(
	cluster_canonicals: Dict[int, Dict],
	cluster_centroids: Dict[int, np.ndarray],
	cluster_members: Dict[int, List[str]],
	original_label_counts: Dict[str, int],
	threshold: float,
	low_margin_ratio: float=1.5, # evidence decisions closer than this are flagged as coin flips
	verbose: bool = False,
) -> Dict[int, Dict]:
	"""
	Post-pass over clusters whose canonicals share a NAME.

	"Share a name" is decided by _canonical_group_keys (the same definition harmonize
	uses): case, spacing/punctuation, dotted acronyms and attested singular/plural all
	collapse, so 'tank' / 'tanks' / 'Tank' are ONE group. Previously the resolver grouped
	on the exact lowercase string and clusters later renamed by harmonize bypassed it.

	Steps per group
	---------------
	1. Group surface (name evidence only). Renaming is owned by _harmonize_final_canonicals.
	2. Primary neighbourhood by NAME EVIDENCE: evidence[c] = sum of corpus counts of the
	   members of c that contain the name (plural-aware token match).
	3. Anchored membership: keep clusters within `threshold` of the primary neighbourhood's
	   mean centroid. No chaining.
	4. Demoted clusters get, in order: an ATTESTED qualified core (see _attested_core: the name plus
	   1-2 qualifiers, a corpus label, majority support), else real_fallback, else real_runner_up.
	   A fallback in the same name group does not count. If none remains the cluster is
	   'kept_no_alternative'. shared_fallback_kind records which one was used.

	Fields written on every cluster in a multi-cluster group:
		shared_resolution, shared_group_size, shared_anchor_similarity,
		shared_name_evidence, shared_threshold, shared_key, shared_surface,
		shared_margin, shared_low_margin
		shared_demoted_from (demoted only)
	`shared_margin` is evidence(seed neighbourhood) / evidence(best other neighbourhood),
	None when the group is kept whole. `shared_low_margin` is True when it is < low_margin_ratio.
	"""
	if verbose:
		print("-"*130)
		print(f"[RESOLVE SHARED CANONICALS]")
		print(f"  ├─ unique labels {len(original_label_counts)}")
		print(f"  ├─ Clusters      {len(cluster_canonicals)} with {len(cluster_centroids)} centroids")
		print(f"  └─ threshold     {threshold}")

	# 0. Grouping on the harmonize key (before the loop)
	keys, plural_of = _canonical_group_keys(cluster_canonicals)
	label_index = _build_label_index(original_label_counts)  # stem-key -> corpus surface (for attested cores)
	canonical_keys = {_spacing_key(m["canonical"]) for m in cluster_canonicals.values() if m.get("canonical")}
	groups: Dict[str, List[int]] = defaultdict(list)
	for cid in cluster_canonicals:
		if cid in keys:
			groups[keys[cid]].append(int(cid))

	stats = Counter()
	for key in sorted(groups):
		cids = sorted(groups[key])
		if len(cids) == 1:
			continue

		# 1. Group surface and all its spelling variants
		variants = sorted({cluster_canonicals[c]["canonical"] for c in cids})
		size_by_surface = Counter()
		for c in cids:
			size_by_surface[cluster_canonicals[c]["canonical"]] += int(cluster_canonicals[c]["size"])
		surface = _pick_group_surface(variants, size_by_surface, plural_of, original_label_counts)
		if len(variants) > 1:
			stats["groups_with_variants"] += 1
		if len({v.lower() for v in variants}) > 1:
			stats["groups_beyond_exact_match"] += 1

		stats["shared_groups"] += 1

		# 2. Name evidence and primary neighbourhood
		name_tok_sets = [_label_tokens(v) for v in variants]
		evidence = {
			c: sum(
				original_label_counts.get(m, 1)
				for m in cluster_members[c]
				if any(_name_matches_label(nt, m) for nt in name_tok_sets)
			)
			for c in cids
		}
		volume = {
			c: sum(original_label_counts.get(m, 1) for m in cluster_members[c])
			for c in cids
		}

		V = np.vstack([cluster_centroids[c] for c in cids]).astype(float)
		V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
		S = V @ V.T

		scored = []
		for i, c in enumerate(cids):
			nb = [j for j in range(len(cids)) if S[i, j] >= threshold]
			score = (
				sum(evidence[cids[j]] for j in nb),
				sum(volume[cids[j]] for j in nb),
				len(nb),
				-c,
			)
			scored.append((score, i, nb))
		best = max(scored, key=lambda t: t[0])
		_, seed, nb = best

		# 3. Anchored membership threshold cut
		anchor = V[nb].mean(axis=0)
		anchor /= np.linalg.norm(anchor) + 1e-12
		anchor_sim = V @ anchor
		keep = {j for j in range(len(cids)) if anchor_sim[j] >= threshold} | {seed}
		stats["groups_kept_whole" if len(keep) == len(cids) else "groups_split"] += 1

		# How decisive was the evidence for the winning neighbourhood?
		margin = None
		if len(keep) < len(cids):
			nb_set = set(nb)
			others = [t[0][0] for t in scored if t[1] not in nb_set]
			margin = best[0][0] / max(max(others), 1) if others else None
		low_margin = margin is not None and margin < low_margin_ratio

		changes = []
		for j, c in enumerate(cids):
			meta = cluster_canonicals[c]
			meta.update(
				shared_group_size=len(cids),
				shared_anchor_similarity=float(anchor_sim[j]),
				shared_name_evidence=int(evidence[c]),
				shared_threshold=float(threshold),
				shared_key=key,
				shared_surface=surface,
				shared_margin=None if margin is None else float(margin),
				shared_low_margin=bool(low_margin),
			)
			if j in keep:
				meta["shared_resolution"] = "kept"
				continue

			same_name = {key} | {_spacing_key(v) for v in variants}
			fallback, kind, as_virtual = None, None, False

			# 4a. An ATTESTED qualified core with majority support beats a synthesized/bare name
			core = _attested_core(
				cluster_members[c], surface, label_index, taken_keys=canonical_keys - {_spacing_key(meta["canonical"])},
			)
			if core is not None:
				fallback, as_virtual = core
				kind = "attested_core"
			else:
				# 4b. Otherwise the best real member, then the real runner-up
				for field in ("real_fallback", "real_runner_up"):
					f = meta.get(field)
					if f and _spacing_key(f) not in same_name and plural_of.get(f, _spacing_key(f)) != key:
						fallback, kind = f, field
						break

			if fallback is None:
				meta["shared_resolution"] = "kept_no_alternative"
				stats["kept_no_alternative"] += 1
				continue

			meta["shared_resolution"] = "demoted"
			meta["shared_demoted_from"] = meta["canonical"]
			meta["shared_fallback_kind"] = kind
			meta["canonical"] = fallback
			meta["virtual"] = as_virtual  # an attested core that is not a member is injected like a virtual
			meta["score"] = meta.get("real_fallback_score", meta["score"])
			stats["demoted_" + kind] += 1
			stats["demoted"] += 1
			
			if low_margin:
				stats["demoted_low_margin"] += 1

			changes.append((c, fallback, anchor_sim[j]))

		if verbose and changes:
			demoted_ids = [c for c, _, _ in changes]
			flag = f"[LOW MARGIN {margin:.2f}x < {low_margin_ratio}x]" if low_margin else ""

			print(f"\n{len(cids)} clusters: {cids} SHARED CANONICAL LABEL: {repr(surface):<30}{flag}")

			for j, c in enumerate(cids):
				tag = "KEPT" if j in keep else "DEMOTED" if c in demoted_ids else "KEPT_NO_ALT"
				print(
					f"cluster {c:6d} "
					f"anchor_sim: {anchor_sim[j]:.4f} "
					f"evidence: {evidence[c]:<5}{tag:<12}{cluster_members[c]}"
				)

	if verbose:
		print(f"\nRESOLUTION: {stats}")
		print("-"*130)

	return cluster_canonicals

def report_shared_group_similarities(
	cluster_canonicals: Dict[int, Dict],
	cluster_centroids: Dict[int, np.ndarray],
	cluster_members: Dict[int, List[str]],
	names: List[str],
	n_members: int = 3,
) -> Dict[str, Dict]:
	print("-"*120)
	out = {}
	for name in names:
		cids = sorted(
			c for c, m in cluster_canonicals.items()
			if m['canonical'].lower() == name.lower()
		)

		if len(cids) < 2:
			print(f"SKIPPED {repr(name):<15} < {len(cids)} clusters!")
			continue
		
		V = np.vstack([cluster_centroids[c] for c in cids]).astype(float)
		V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
		S = V @ V.T
		off = S[~np.eye(len(cids), dtype=bool)]
		mean_to_others = (S.sum(axis=1) - 1.0) / (len(cids) - 1)

		print(
			f"\n[{name!r}] {len(cids)} clusters | pairwise sim min {off.min():.3f} "
			f"median {np.median(off):.3f} max {off.max():.3f}"
		)

		for i, c in sorted(enumerate(cids), key=lambda x: -mean_to_others[x[0]]):
			print(
				f"    cluster {c:6d}  mean sim to others {mean_to_others[i]:.3f} "
				f"{cluster_members[c][:n_members]}"
			)
		
		out[name] = {'cluster_ids': cids, 'pairwise': S.tolist()}

	print("-"*120)

	return out

def _harmonize_final_canonicals(
	cluster_canonicals: Dict[int, Dict],
	original_label_counts: Dict[str, int],
	protected_plurals: Optional[set] = None,
	verbose: bool = False,
) -> Dict[int, Dict]:
	"""
	Global post-pass that unifies surface variants of the chosen canonicals:
	1) case ('Nurse' / 'nurse'), 
	2) spacing and punctuation ('ice breaker' / 'icebreaker', 'dug-out' / 'dugout'),
	3) attested singular/plural ('car' / 'cars').

	It does NOT overwrite meta['canonical']. 
	Internal steps (virtual-row injection, remove_problematic_clustered_labels) 
	require the canonical to be a member of its cluster, 
	and a harmonized name often is not ('wing tip' -> 'wingtip').

	Instead each cluster gets a display name:

		meta['canonical_harmonized']   final name (equals 'canonical' if unchanged)
		meta['changed_by_harmonize']   bool
		meta['canonical_pre_harmonize']  old name, only when changed

	cluster() applies the display names as a string rename after
	remove_problematic_clustered_labels(). 
	All clusters sharing a surface are in
	the same group and get the same winner, so the rename is well defined.

	Rules
	-----
	* Names are compared on a normalised key: lowercase, with every character that is
		not a letter or digit removed. Digits are kept, so different designations never
		merge ('C-46' vs 'C-47'), while formatting variants do ('B-17G' / 'B17G').
	* Plural -> singular only when the singular is itself an attested canonical
		(compared on the same key, so 'icebreakers' finds 'ice breaker' and
		'gun-mounts' finds 'gun mount').
	* Protected plurals (different words: arms, papers, grounds, ...) never map.
	* All-caps acronyms never merge with ordinary words 
		('SPAR' vs 'spars', 'CARE' vs 'care'). 
		They are compared on their letters, so a dotted acronym and its plain form do merge 
		('A.E.F.' / 'AEF', 'L.C.V.P.' / 'LCVP').
	* Winner: singular form first, then a real corpus label, then corpus
		frequency, then total cluster size, then capitalisation, then lexicographic.
	"""
	if verbose:
		print("\n[CANONICAL LABEL HARMONIZATION]")
		print(f"  ├─ unique labels      {len(original_label_counts)}")
		print(f"  ├─ Clusters           {len(cluster_canonicals)}")

	keys, plural_of = _canonical_group_keys(cluster_canonicals, protected_plurals)
	surfaces = {cid: m['canonical'] for cid, m in cluster_canonicals.items() if m.get('canonical')}

	groups: Dict[str, list] = defaultdict(list)
	for cid in surfaces:
		groups[keys[cid]].append(cid)

	stats, examples = Counter(), []
	for cids in groups.values():
		variants = {
			cluster_canonicals[c]['canonical'] 
			for c in cids
		}
		if len(variants) == 1:
			for c in cids:
				m = cluster_canonicals[c]
				m['canonical_harmonized'] = m['canonical']
				m['changed_by_harmonize'] = False
			continue

		size = Counter()
		for c in cids:
			size[cluster_canonicals[c]['canonical']] += int(cluster_canonicals[c].get('size', 1))
		
		winner = _pick_group_surface(variants, size, plural_of, original_label_counts)

		stats['groups_changed'] += 1

		for c in cids:
			m = cluster_canonicals[c]
			m['canonical_harmonized'] = winner
			m['changed_by_harmonize'] = m['canonical'] != winner
			if m['changed_by_harmonize']:
				m['canonical_pre_harmonize'] = m['canonical']
				stats['clusters_renamed'] += 1

		examples.append((sorted(variants), winner))

	if verbose:
		print(f"  ├─ changed group(s)   {stats['groups_changed']}")
		print(f"  └─ renamed cluster(s) {stats['clusters_renamed']}")
		for variants, winner in examples:
			print(f"{repr(winner):<20}{variants}")

	return cluster_canonicals

def _finalize_selection_records(
	selection_records: List[Dict],
	cluster_canonicals: Dict[int, Dict],
	pre_postpass: Dict[int, str],
) -> None:
	"""In place: stamp final (post-resolver, post-harmonize) state onto each record."""
	for rec in selection_records:
		cid = rec['cluster_id']
		meta = cluster_canonicals[cid]
		final = meta.get('canonical_harmonized', meta['canonical'])
		rec['canonical_selected'] = final
		rec['changed_by_postpass'] = final != pre_postpass[cid]
		rec['is_virtual'] = meta['virtual']  # final state, not pre-post-pass
		rec['harmonize'] = {
			'changed': bool(meta.get('changed_by_harmonize', False)),
			'from':    meta.get('canonical_pre_harmonize'),
		}
		rec['shared'] = {
			'resolution':            meta.get('shared_resolution'),
			'group_size':            meta.get('shared_group_size'),
			'anchor_similarity':     meta.get('shared_anchor_similarity'),
			'name_evidence':         meta.get('shared_name_evidence'),
			'threshold':             meta.get('shared_threshold'),
			'demoted_from':          meta.get('shared_demoted_from'),
			'group_key':             meta.get('shared_key'),
			'fallback_kind':         meta.get('shared_fallback_kind'),
			'margin':                meta.get('shared_margin'),
			'low_margin':            meta.get('shared_low_margin'),
			'spelling_unified_from': meta.get('spelling_unified_from'),
		}

def _clean(v, ndigits: int = 4):
	"""numpy -> python; NaN -> None; floats rounded for readability."""
	if isinstance(v, (np.bool_, bool)):
		return bool(v)
	if isinstance(v, (np.integer, int)):
		return int(v)
	if isinstance(v, (np.floating, float)):
		return None if np.isnan(v) else round(float(v), ndigits)
	return v

def _candidate_json(r: Dict) -> Dict:
	return {
		'rank': _clean(r['rank']),
		'label': r['candidate'],
		'is_virtual': _clean(r['is_virtual']),
		'roles': [
			role for role, flag in (
				('selected', r['is_selected']),
				('pure_similarity_winner', r['is_pure_sim_winner']),
				('no_frequency_winner', r['is_no_freq_winner']),
			) if flag
		],
		'head_token': r['head_token'],
		'corpus_freq': _clean(r['raw_freq']),
		'scores': {
			'similarity':  _clean(r['sim']),
			'frequency':   _clean(r['freq_score']),
			'head':        _clean(r['head_score']),
			'containment': _clean(r['cont_score']),
			'brevity':     _clean(r['brevity_score']),
		},
		'composite_without_frequency': _clean(r['composite_no_freq']),
		'composite': _clean(r['composite']),
		'eligible': _clean(r.get('eligible', True)),
	}

def _cluster_json(rec: Dict, cands: List[Dict]) -> Dict:
	return {
		'cluster_id': _clean(rec['cluster_id']),
		'size': _clean(rec['cluster_size']),
		'members': rec['members'].split(" | "),
		'canonical': rec['canonical_selected'],
		'harmonize': rec['harmonize'],
		'canonical_pre_postpass': rec['canonical_pre_postpass'],
		'changed_by_postpass': _clean(rec['changed_by_postpass']),
		'is_virtual': _clean(rec['is_virtual']),
		'selection_method': rec['selection_method'],
		'freq_guard': rec['freq_guard'],
		'shared_canonical': {
			k: (v if isinstance(v, str) else _clean(v))
			for k, v in rec['shared'].items()
		},
		'margin_to_runner_up': _clean(rec['margin_to_runner_up']),
		'numeric': {
			'ineligible': _clean(rec.get('numeric_ineligible', 0)),
			'unresolved': _clean(rec.get('numeric_unresolved', False)),
		},
		'winners': {
			'selected': rec['canonical_pre_postpass'],
			'composite_before_guard': rec['composite_winner_pre_guard'] or None,
			'without_frequency': rec['no_freq_winner'] or None,
			'pure_similarity': rec['pure_sim_label'],
			'runner_up': rec['runner_up'] or None,
		},
		'virtual_hypernym': {
			'candidate': rec['virtual_candidate'] or None,
			'entered_pool': _clean(rec['virtual_entered_pool']),
			'route': rec['virtual_route'] or None,
			'support': _clean(rec['virtual_support']),
			'threshold': _clean(rec['virtual_threshold']),
			'note': rec['virtual_note'],
		},
		'candidates': [_candidate_json(r) for r in cands],
	}

def _build_clusters_json(
	selection_records: List[Dict],
	candidate_records: List[Dict],
) -> List[Dict]:
	by_cluster: Dict[int, List[Dict]] = defaultdict(list)
	for r in candidate_records:
		by_cluster[r['cluster_id']].append(r)
	return [
		_cluster_json(rec, sorted(by_cluster[rec['cluster_id']], key=lambda r: r['rank']))
		for rec in selection_records
	]

def _build_debug_payload(
	clusters_json: List[Dict],
	selection_records: List[Dict],
	n_candidates: int,
	weights: Dict[str, float],
	min_support: float,
	virtual_gates: Dict[str, Any],
	freq_gain_guard: float,
) -> Dict:
	method_counts = Counter(rec['selection_method'] for rec in selection_records)
	resolution_counts = Counter(
		rec['shared']['resolution']
		for rec in selection_records
		if rec['shared'].get('resolution')
	)
	return {
		'meta': {
			'n_clusters': len(clusters_json),
			'n_candidates': n_candidates,
			'weights': weights,
			'min_support': min_support,
			'virtual_gates': virtual_gates,
			'freq_gain_guard': freq_gain_guard,
			'selection_method_counts': dict(method_counts.most_common()),
			'virtual_winners': sum(1 for c in clusters_json if c['is_virtual']),
			'changed_by_postpass': sum(1 for c in clusters_json if c['changed_by_postpass']),
			'shared_resolution_counts': dict(resolution_counts.most_common()),
		},
		'clusters': clusters_json,
	}

def _write_debug_json(
	path: str,
	payload: Dict,
	verbose: bool = False,
) -> None:
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
	if verbose:
		print(f"[CANONICAL SELECTION]")
		print(f"  ├─ {len(payload['clusters'])} clusters")
		print(f"  ├─ {payload['meta']['n_candidates']} candidates")
		print(f"  ├─ {path}")

def _print_selection_overview(
	selection_records: List[Dict],
	virtual_rejections: Counter,
) -> None:
	sel = pd.DataFrame(selection_records)
	print("\nHow canonicals were chosen:")
	for m, cnt in sel['selection_method'].value_counts().items():
		print(f"  {m:<34} {cnt:6d} ({cnt / len(sel) * 100:5.1f}%)")

	routes = sel.loc[sel['virtual_entered_pool'], 'virtual_route'].value_counts()
	if len(routes):
		print("  Virtual candidates entering the pool, by route:")
		for r, cnt in routes.items():
			won = int(((sel['virtual_route'] == r) & sel['is_virtual']).sum())
			print(f"    {r:<14} {cnt:6d} entered, {won:6d} won")

	dup = sel['canonical_selected'].value_counts()
	dup = dup[dup > 1]
	if len(dup):
		print(
			f"  Canonicals shared by >1 cluster: {len(dup)} "
			f"(top: {', '.join(f'{k!r}x{v}' for k, v in dup.head(8).items())})"
		)

	print(f"  Postpass (case-collision) changed: {int(sel['changed_by_postpass'].sum())} cluster(s)")
	if virtual_rejections:
		print(f"  Virtual candidates rejected by the quality gates: {dict(virtual_rejections)}")

def _categorize_trades(
	total_sim_loss: List[float],
	total_freq_gain: List[float],
	questionable_examples: List[Dict],
) -> Dict[str, List[Dict]]:
	"""Mutually exclusive partition of every (sim_loss, freq_gain) trade."""
	buckets: Dict[str, List[Dict]] = {
		'free_upgrades': [],       # s <= 2%, f >= 1.0
		'high_yield': [],          # s <= 5%, f >= 5.0
		'structural': [],          # f < 1.0, s <= 5%
		'fair': [],                # s <= 8%, f >= 2.0
		'high_loss_good_gain': [], # s > 8%, f >= 5.0
		'questionable': [],        # (s > 8% and f < 2.0) or s > 12%
		'other_mild': [],
	}
	for idx, (s, f) in enumerate(zip(total_sim_loss, total_freq_gain)):
		ex = questionable_examples[idx] if idx < len(questionable_examples) else None
		item = {'sim_loss': s, 'freq_gain': f, 'example': ex}
		if s <= 0.02 and f >= 1.0:
			buckets['free_upgrades'].append(item)
		elif s <= 0.05 and f >= 5.0:
			buckets['high_yield'].append(item)
		elif f < 1.0 and s <= 0.05:
			buckets['structural'].append(item)
		elif s <= 0.08 and f >= 2.0:
			buckets['fair'].append(item)
		elif s > 0.08 and f >= 5.0:
			buckets['high_loss_good_gain'].append(item)
		elif (s > 0.08 and f < 2.0) or s > 0.12:
			buckets['questionable'].append(item)
		else:
			buckets['other_mild'].append(item)
	return buckets

def _print_tradeoff_analysis(
	total_sim_loss: List[float],
	total_freq_gain: List[float],
	questionable_examples: List[Dict],
) -> None:
	print(
		f"\nSIMILARITY LOSS "
		f"(min, max): ({np.min(total_sim_loss)}, {np.max(total_sim_loss)}) "
		f"μ±σ: {np.mean(total_sim_loss)} ± {np.std(total_sim_loss)} (Median: {np.median(total_sim_loss)})"
	)
	print(
		f"FREQUENCY GAIN  "
		f"(min, max): ({np.min(total_freq_gain)}, {np.max(total_freq_gain)}) "
		f"μ±σ: {np.mean(total_freq_gain)} ± {np.std(total_freq_gain)} (Median: {np.median(total_freq_gain)}x)"
	)

	b = _categorize_trades(total_sim_loss, total_freq_gain, questionable_examples)
	n_trades = len(total_sim_loss)

	log2_gains = np.log2(np.maximum(total_freq_gain, 0.01))
	sim_losses = np.maximum(total_sim_loss, 0.005)
	trade_roi = log2_gains / (sim_losses * 100)  # doublings per 1% sim loss

	rows = [
		("Free Upgrades      (s <= 2%,  f >= 1x)      ", 'free_upgrades',       "[Semantic drift < noise floor]"),
		("High-Yield Trades  (s <= 5%,  f >= 5x)      ", 'high_yield',          "[High frequency consolidation]"),
		("Fair / Modest      (s <= 8%,  f >= 2x)      ", 'fair',                "[Balanced compromise]"),
		("Structural/Brevity (f <  1x,  s <= 5%)      ", 'structural',          "[Driven by head-token / brevity]"),
		("Mild Adjustments   (s <= 8%,  f in 1-2x)    ", 'other_mild',          "[Benign low-loss variations]"),
		("High Loss & Gain   (s >  8%,  f >= 5x)      ", 'high_loss_good_gain', "[Aggressive trade, inspectable]"),
		("Truly Questionable (s >  8% & f < 2x / >12%)", 'questionable',        "[Poor ROI or high semantic loss]"),
	]
	print("\nCANONICAL SELECTION TRADEOFF ANALYSIS (100% Partitioned):")
	for i, (title, key, note) in enumerate(rows):
		branch = "└─" if i == len(rows) - 1 else "├─"
		cnt = len(b[key])
		print(f"  {branch} {title} {cnt:<5d} ({cnt / n_trades * 100:5.1f}%) {note}")
	print(f"\n  Average Trade ROI: {np.median(trade_roi):.2f} frequency doublings per 1% similarity loss (Median)")

	questionable = b['questionable']
	if questionable:
		print(f"\n[WARNING] {len(questionable)} genuinely questionable trades detected:")
		print(f"{'Sim Loss(%)':<15} {'Freq Gain':<15} {'ROI (doublings/1% loss)'}")
		print("-" * 60)
		for t in sorted(questionable, key=lambda x: x['sim_loss'], reverse=True)[:15]:
			roi = np.log2(max(t['freq_gain'], 0.01)) / (max(t['sim_loss'], 0.005) * 100)
			print(f"{t['sim_loss']*100:<15.2f} {t['freq_gain']:<15.2f}x {roi:<10.2f}")
	else:
		print("\n  ✅ Zero destructive trades detected. All canonical selections maintain strong semantic integrity.")

	_print_overall_verdict(
		avg_sim_loss_pct=np.mean(total_sim_loss) * 100,
		avg_freq_gain=np.mean(total_freq_gain),
	)

def _print_overall_verdict(avg_sim_loss_pct: float, avg_freq_gain: float) -> None:
	print(f"\nOVERALL VERDICT:")
	if avg_sim_loss_pct < 3 and avg_freq_gain > 50:
		print(f"[EXCELLENT] Small quality cost ({avg_sim_loss_pct:.1f}%) for huge frequency benefit ({avg_freq_gain:.0f}x)")
	elif avg_sim_loss_pct < 5 and avg_freq_gain > 10:
		print(f"[GOOD] Acceptable quality cost ({avg_sim_loss_pct:.1f}%) for strong frequency benefit ({avg_freq_gain:.0f}x)")
	elif avg_sim_loss_pct < 8 and avg_freq_gain > 5:
		print(f"[ACCEPTABLE] Moderate quality cost ({avg_sim_loss_pct:.1f}%) for moderate frequency benefit ({avg_freq_gain:.0f}x)")
	else:
		print(
			f"[POOR] High quality cost ({avg_sim_loss_pct:.1f}%) for limited frequency benefit ({avg_freq_gain:.0f}x) "
			f"Consider reducing frequency weight"
		)

def _print_frequency_impact(
	n_clusters: int,
	virtual_used_count: int,
	freq_changed_count: int,
	debug_json_path: Optional[str],
) -> None:
	print("-" * 100)
	print("[CLUSTERING] FREQUENCY WEIGHTING IMPACT")
	print(f"  Total clusters analyzed: {n_clusters}")
	print(f"  Virtual hypernym used as canonical: {virtual_used_count} ({virtual_used_count / n_clusters * 100:.1f}%)")
	print(f"  Clusters where score changed the canonical: {freq_changed_count} ({freq_changed_count / n_clusters * 100:.1f}%)")
	if debug_json_path:  # summarize_canonical_selection reads the JSON
		summarize_canonical_selection(path=debug_json_path)
	print("-" * 100)

def _synthesize_virtual(
	cluster_texts: List[str],
	synth_fn: Callable,
	restore_fn: Callable,
	norm_token_set_fn: Callable,
	norm_tokens_fn: Callable,
	case_registry: Dict[str, str],
) -> Tuple[Optional[str], Optional[str], Dict]:
	"""Returns (virtual_hypernym, vh_raw, vh_info). virtual_hypernym is None if absent or suppressed."""
	vh_info = {'route': '', 'support': 0, 'threshold': 0, 'note': 'cluster too small'}
	if len(cluster_texts) < 2:
		return None, None, vh_info

	vh_raw, vh_info = synth_fn(cluster_texts)
	if vh_raw is None:
		return None, None, vh_info

	norm_vh = set(norm_tokens_fn(vh_raw))
	twin = next((l for l in cluster_texts if norm_token_set_fn(l) == norm_vh), None)
	if twin is not None:
		vh_info['note'] += f" | suppressed: identical to real member {twin!r}"
		return None, vh_raw, vh_info

	virtual_hypernym = restore_fn(vh_raw, cluster_texts)
	# Align with an existing corpus spelling ('industrial' -> 'Industrial')
	registry_hit = case_registry.get(virtual_hypernym.lower())
	if registry_hit is not None and registry_hit != virtual_hypernym:
		vh_info['note'] += f" | respelled via corpus as {registry_hit!r}"
		virtual_hypernym = registry_hit
	return virtual_hypernym, vh_raw, vh_info

def _print_virtual_trace(
	vh_raw: Optional[str],
	virtual_hypernym: Optional[str],
	vh_info: Dict,
	cluster_size: int,
) -> None:
	if vh_raw is None:
		print(f"Virtual: none -- {vh_info['note']}")
	elif virtual_hypernym is None:
		print(f"Virtual: {vh_raw!r} [{vh_info['route']}] {vh_info['note']}")
	else:
		print(
			f"Virtual: {virtual_hypernym!r} [{vh_info['route']}, "
			f"support {vh_info['support']}/{cluster_size}, "
			f"need {vh_info['threshold']}] {vh_info['note']}"
		)

def _centroid_similarities(
	centroid: np.ndarray,
	cluster_embeddings: np.ndarray,
	virtual_hypernym: Optional[str],
	model,
	encode_prompt: Optional[str],
) -> np.ndarray:
	"""Cosine of centroid to every real member (+ the virtual, last, if present)."""
	if virtual_hypernym is not None:
		vh_emb = _encode_(model, [virtual_hypernym], 1, encode_prompt)[0]
		all_embeddings = np.vstack([cluster_embeddings, vh_emb[np.newaxis, :]])
	else:
		all_embeddings = cluster_embeddings
	
	centroid_sim = sklearn.metrics.pairwise.cosine_similarity(centroid.reshape(1, -1), all_embeddings)[0]

	return centroid_sim

def _gate_virtual_by_similarity(
	virtual_hypernym: Optional[str],
	similarities: np.ndarray,
	cluster_size: int,
	min_sim_ratio: Optional[float],
	abs_floor: float, # min acceptable virtual-label similarity
	vh_info: Dict,
	virtual_rejections: Counter,
) -> Tuple[Optional[str], np.ndarray, str]:
	"""
	A virtual is a SUMMARY of the cluster. 
	If its embedding is far from the centroid compared with
	the best real member, drop it and let a real member win.
	Returns (virtual_hypernym or None, similarities, rejection_tag). 
	The tag is '' if nothing was rejected.
	"""
	if virtual_hypernym is None or min_sim_ratio is None:
		return virtual_hypernym, similarities, ""

	# highest centroid similarity among the original labels
	best_real = float(similarities[:cluster_size].max())

	# similarity between the virtual label and the cluster centroid
	sim = float(similarities[cluster_size])

	ratio = sim / max(best_real, 1e-12)

	if ratio >= min_sim_ratio and sim >= abs_floor:
		return virtual_hypernym, similarities, ""

	reason = (
		f"sim ratio {ratio:.2f} < {min_sim_ratio}"
		if ratio < min_sim_ratio
		else f"abs sim {sim:.3f} < {abs_floor}"
	)

	vh_info['note'] += (
		f" | rejected: {reason} "
    f"(virtual {sim:.3f} vs best member {best_real:.3f})"
  )

	virtual_rejections['similarity_ratio'] += 1

	tag = (
		f"[REJECTED VIRTUAL] "
		f"{repr(virtual_hypernym):<25}"
		f"{reason} "
		f"(virtual={sim:.5f}, best_real={best_real:.5f})"
		# f"sim ratio {ratio:.5f} < {min_sim_ratio}"
		# f" or abs sim: {sim:.5f} < {abs_floor}"
	)

	return None, similarities[:cluster_size], tag

def _build_candidate_pool(
	cluster_texts: List[str],
	virtual_hypernym: Optional[str],
) -> Tuple[List[str], List[bool]]:
	n = len(cluster_texts)
	if virtual_hypernym:
		return cluster_texts + [virtual_hypernym], [False] * n + [True]
	return list(cluster_texts), [False] * n

# Unit aliases -> canonical unit. Only numbers followed by one of these (or an ordinal) take part
# in the signature, so designations ('Ki-46', 'Bf 109', 'Hill 272') are never touched.
_NUM_UNITS = {
	"mm": "mm", "cm": "cm", "km": "km", "inch": "inch", "inches": "inch",
	"ft": "ft", "foot": "ft", "feet": "ft", "yd": "yd", "yard": "yd", "yards": "yd", "mile": "mile", "miles": "mile",
	"lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb", "pdr": "pdr", "pounder": "pdr", "kg": "kg",
	"ton": "ton", "tons": "ton", "tonne": "ton", "tonnes": "ton", "gallon": "gallon", "gallons": "gallon",
	"barrel": "barrel", "barrels": "barrel", "hp": "hp", "kw": "kw", "mw": "mw", "kv": "kv", "volt": "volt", "volts": "volt",
	"mph": "mph", "knot": "knot", "knots": "knot", "kt": "knot", "cc": "cc",
	"gun": "gun", "guns": "gun", "cal": "cal", "caliber": "cal", "calibre": "cal",
	"day": "day", "days": "day", "hour": "hour", "hours": "hour", "minute": "minute", "minutes": "minute",
	"year": "year", "years": "year", "man": "man", "men": "man", "car": "car", "cars": "car",
	"story": "story", "storey": "story", "floor": "floor", "seat": "seat", "seater": "seat", "wheel": "wheel", "wheeler": "wheel",
}
_NUM_WORDS = {
	"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
	"eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
	"eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
	"seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_NUM_WORD_RE = re.compile(
	r'(?<![A-Za-z])(?:(' + "|".join(_TENS) + r')[\s-]+(one|two|three|four|five|six|seven|eight|nine)|(' + "|".join(_NUM_WORDS) + r'))(?=[\s-]*[A-Za-z])',
	re.I,
)

def _words_to_digits(label: str) -> str:
	"""'six inch gun' -> '6 inch gun', 'twenty-one gun salute' -> '21 gun salute' (only the number words;
	whether a unit follows is decided later), so spelled-out numbers cannot slip past the signature."""
	def sub(m):
		if m.group(1):
			return str(_TENS[m.group(1).lower()] + _NUM_WORDS[m.group(2).lower()])
		return str(_NUM_WORDS[m.group(3).lower()])
	return _NUM_WORD_RE.sub(sub, label)

_ORDINAL_RE = re.compile(r'(?<![A-Za-z0-9])(\d+)(?:st|nd|rd|th)(?![A-Za-z0-9])', re.I)
_NUM_UNIT_RE = re.compile(r'(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})+|\d*\.?\d+)\s*-?\s*([A-Za-z]+)')

def _numeric_signature(label: str) -> Dict[str, frozenset]:
	"""unit -> set of values, for number+unit pairs and ordinals: '10-ton trailers' -> {'ton': {10.0}};
	'18th century dress' -> {'ord': {18.0}}; 'French 75' / 'Ki-46' / 'Hill 272' -> {} (no unit)."""
	sig: Dict[str, set] = defaultdict(set)
	label = _words_to_digits(label)
	for m in _ORDINAL_RE.finditer(label):
		sig["ord"].add(float(m.group(1)))
	rest = _ORDINAL_RE.sub(" ", label)
	for m in _NUM_UNIT_RE.finditer(rest):
		unit = _NUM_UNITS.get(m.group(2).lower())
		if unit:
			sig[unit].add(float(m.group(1).replace(",", "")))
	return {u: frozenset(v) for u, v in sig.items()}

def _numeric_conflict(cand: Dict[str, frozenset], member: Dict[str, frozenset]) -> bool:
	"""True when a candidate's numeric claim is not backed by a member that states numbers.

	* cardinal units (ton, mm, gun, ...): every (unit, values) the candidate states must be stated
	  identically by the member, so '17 Pounder Guns' cannot name a cluster that holds '170mm gun'
	  and '10-ton trailers' cannot name one that holds '1-tons aanhangwagen'.
	* ordinals ('18th'): they conflict only when both labels state a different one.
	A label that is silent ('French 75', 'gun salute') never conflicts.
	"""
	for u, vals in cand.items():
		if u == "ord":
			if "ord" in member and member["ord"] != vals:
				return True
		elif member.get(u) != vals:
			return True
	return False

def _numeric_eligibility(candidates: List[str], cluster_texts: List[str]) -> np.ndarray:
	"""eligible[i] is False when candidate i makes a numeric claim that a number-bearing member does not back."""
	member_sigs = [s for s in (_numeric_signature(m) for m in cluster_texts) if s]
	if not member_sigs:
		return np.ones(len(candidates), dtype=bool)
	out = np.ones(len(candidates), dtype=bool)
	for i, c in enumerate(candidates):
		sc = _numeric_signature(c)
		if sc and any(_numeric_conflict(sc, s) for s in member_sigs):
			out[i] = False
	return out

INELIGIBLE_SCORE = -1.0  # below any real composite (weights sum to 1, sim >= -1)

def _score_candidates(
	candidates: List[str],
	cluster_texts: List[str],
	similarities: np.ndarray,
	raw_freqs: np.ndarray,
	head_token_fn: Callable,
	containment_fn: Callable,
	weights: Dict[str, float],
) -> Dict[str, np.ndarray]:
	"""Signals 2-5 and both composites (with / without the frequency term)."""
	cluster_size = len(cluster_texts)

	freq_scores = np.log1p(raw_freqs) / np.log1p(raw_freqs.max() + 1e-12)

	head_counts = Counter(head_token_fn(l) for l in cluster_texts)
	head_scores = np.array(
		[head_counts.get(head_token_fn(c), 0) / cluster_size for c in candidates]
	)

	cont_scores = containment_fn(candidates, cluster_texts)

	token_lengths = np.array([len(c.split()) for c in candidates], dtype=float)
	brevity_scores = 1.0 - (token_lengths - 1.0) / max(token_lengths.max(), 1)

	composite_no_freq = (
		weights['sim'] * similarities
		+ weights['head'] * head_scores
		+ weights['cont'] * cont_scores
		+ weights['brev'] * brevity_scores
	)
	combined_scores = composite_no_freq + weights['freq'] * freq_scores

	# Numeric-signature eligibility: a candidate that states a number contradicting a member
	# ('10-ton trailers' vs '1-tons aanhangwagen') cannot name the cluster. If EVERY candidate
	# conflicts the mask is skipped and the cluster is flagged instead of left without a name.
	eligible = _numeric_eligibility(candidates, cluster_texts)
	unresolved = not eligible.any()
	if unresolved:
		eligible = np.ones(len(candidates), dtype=bool)
	elif not eligible.all():
		composite_no_freq = np.where(eligible, composite_no_freq, INELIGIBLE_SCORE)
		combined_scores = np.where(eligible, combined_scores, INELIGIBLE_SCORE)

	return {
		'freq_scores': freq_scores,
		'head_scores': head_scores,
		'cont_scores': cont_scores,
		'brevity_scores': brevity_scores,
		'composite_no_freq': composite_no_freq,
		'combined_scores': combined_scores,
		'eligible': eligible,
		'numeric_unresolved': unresolved,
	}

def _unscored(n: int) -> Dict[str, np.ndarray]:
	"""All-NaN scores for singletons / no frequency data."""
	nan_vec = np.full(n, np.nan)
	out = {
		k: nan_vec for k in (
			'freq_scores', 'head_scores', 'cont_scores', 'brevity_scores',
			'composite_no_freq', 'combined_scores',
		)
	}
	out['eligible'] = np.ones(n, dtype=bool)
	out['numeric_unresolved'] = False
	return out

def _pick_best_candidate(
	scored: bool,
	scores: Dict[str, np.ndarray],
	similarities: np.ndarray,
	raw_freqs: np.ndarray,
	virtual_flags: List[bool],
	cluster_size: int,
	pure_sim_idx: int,
	freq_gain_guard: float,
	sim_gap_guard: float,
) -> Dict[str, Any]:
	"""
	Composite argmax + frequency guard (only when frequency flipped the decision).
	Returns best_idx, best_real_idx, composite_idx, nofreq_idx, freq_guard.
	"""
	if not scored:
		# Singleton or no frequency data: pure centroid similarity
		return {
			'best_idx': pure_sim_idx,
			'best_real_idx': pure_sim_idx,
			'composite_idx': None,
			'nofreq_idx': None,
			'freq_guard': 'not_applicable',
		}

	combined = scores['combined_scores']
	composite_idx = int(combined.argmax())
	nofreq_idx = int(scores['composite_no_freq'].argmax())
	best_idx = composite_idx

	if composite_idx == nofreq_idx:
		freq_guard = 'not_triggered'
	elif (
		virtual_flags[nofreq_idx]
		and not virtual_flags[composite_idx]
		and (similarities[composite_idx] - similarities[nofreq_idx]) > sim_gap_guard
	):
		sim_gap = similarities[composite_idx] - similarities[nofreq_idx]
		freq_guard = f'passed (real-vs-virtual sim gap {sim_gap:.3f} > {sim_gap_guard:.2f})'
		best_idx = composite_idx
	else:
		gain = raw_freqs[composite_idx] / max(raw_freqs[nofreq_idx], 1)
		if gain >= freq_gain_guard:
			freq_guard = f'passed ({gain:.1f}x)'
		else:
			freq_guard = f'reverted ({gain:.1f}x < {freq_gain_guard:.0f}x)'
			best_idx = nofreq_idx

	return {
		'best_idx': best_idx,
		'best_real_idx': int(combined[:cluster_size].argmax()),
		'composite_idx': composite_idx,
		'nofreq_idx': nofreq_idx,
		'freq_guard': freq_guard,
	}

def _label_selection_method(
	composite_idx: Optional[int],
	is_virtual_pick: bool,
	freq_guard: str,
	best_idx: int,
	pure_sim_idx: int,
) -> str:
	"""Why this label won."""
	if composite_idx is None:
		return 'pure_similarity_fallback'
	if is_virtual_pick:
		return 'virtual_hypernym'
	if freq_guard.startswith('reverted'):
		return 'freq_guard_revert'
	if freq_guard.startswith('passed'):
		return 'composite_frequency'
	if best_idx == pure_sim_idx:
		return 'composite_agrees_with_similarity'
	return 'composite_structural'

def _tradeoff_for_pick(
	cid: int,
	cluster_texts: List[str],
	candidates: List[str],
	similarities: np.ndarray,
	original_label_counts: Dict[str, int],
	best_idx: int,
	pure_sim_idx: int,
) -> Dict:
	real_freqs = np.array([original_label_counts.get(t, 1) for t in cluster_texts])
	sim_loss = float((similarities[pure_sim_idx] - similarities[best_idx]) / (similarities[pure_sim_idx] + 1e-12))
	freq_gain = float(real_freqs[best_idx] / max(real_freqs[pure_sim_idx], 1))
	return {
		'cluster_id':     cid,
		'pure_choice':    cluster_texts[pure_sim_idx],
		'freq_choice':    candidates[best_idx],
		'pure_freq':      real_freqs[pure_sim_idx],
		'freq_freq':      real_freqs[best_idx],
		'pure_sim':       similarities[pure_sim_idx],
		'freq_sim':       similarities[best_idx],
		'sim_loss':       sim_loss,
		'freq_gain':      freq_gain,
		'cluster_size':   len(cluster_texts),
		'clustered_labels': cluster_texts,
		'questionable':   sim_loss > 0.10 or freq_gain < 3.0,
	}

def _build_candidate_rows(
	cid: int,
	candidates: List[str],
	virtual_flags: List[bool],
	raw_freqs: np.ndarray,
	similarities: np.ndarray,
	scores: Dict[str, np.ndarray],
	best_idx: int,
	pure_sim_idx: int,
	nofreq_idx: Optional[int],
	head_token_fn: Callable,
) -> Tuple[List[Dict], List[int]]:
	"""One row per candidate (verbose table + JSON), with 'rank' filled in. Also returns the ranking."""
	rows = [
		{
			'cluster_id':         cid,
			'candidate':          c,
			'is_virtual':         virtual_flags[i],
			'head_token':         head_token_fn(c),
			'raw_freq':           int(raw_freqs[i]),
			'sim':                float(similarities[i]),
			'freq_score':         float(scores['freq_scores'][i]),
			'head_score':         float(scores['head_scores'][i]),
			'cont_score':         float(scores['cont_scores'][i]),
			'brevity_score':      float(scores['brevity_scores'][i]),
			'composite_no_freq':  float(scores['composite_no_freq'][i]),
			'composite':          float(scores['combined_scores'][i]),
			'eligible':           bool(scores['eligible'][i]),
			'is_selected':        i == best_idx,
			'is_pure_sim_winner': i == pure_sim_idx,
			'is_no_freq_winner':  nofreq_idx is not None and i == nofreq_idx,
		}
		for i, c in enumerate(candidates)
	]
	ranking = sorted(range(len(rows)), key=lambda i: -np.nan_to_num(rows[i]['composite'], nan=-1))
	for rank, i in enumerate(ranking, 1):
		rows[i]['rank'] = rank
	return rows, ranking

def _build_canonical_entry(
	candidates: List[str],
	cluster_texts: List[str],
	virtual_flags: List[bool],
	similarities: np.ndarray,
	combined_scores: np.ndarray,
	best_idx: int,
	best_real_idx: int,
	composite_idx: Optional[int],
	method: str,
	eligible: Optional[np.ndarray] = None,
) -> Dict:
	cluster_size = len(cluster_texts)
	order = [int(i) for i in np.argsort(-combined_scores[:cluster_size])]
	if eligible is not None:
		order = [i for i in order if eligible[i]]
	return {
		'canonical': candidates[best_idx],
		'score': float(similarities[best_idx]),
		'size': cluster_size,
		'virtual': virtual_flags[best_idx],
		'real_fallback': cluster_texts[best_real_idx],
		'real_fallback_score': float(similarities[best_real_idx]),
		'real_runner_up': (
			cluster_texts[order[1]]
			if composite_idx is not None and len(order) > 1
			else None
		),
		'method': method,
	}

def _build_selection_record(
	cid: int,
	cluster_texts: List[str],
	original_label_counts: Dict[str, int],
	candidates: List[str],
	virtual_flags: List[bool],
	virtual_hypernym: Optional[str],
	vh_raw: Optional[str],
	vh_info: Dict,
	similarities: np.ndarray,
	combined_scores: np.ndarray,
	raw_freqs: np.ndarray,
	best_idx: int,
	pure_sim_idx: int,
	composite_idx: Optional[int],
	nofreq_idx: Optional[int],
	runner_up: Optional[int],
	method: str,
	freq_guard: str,
) -> Dict:
	canonical = candidates[best_idx]
	margin = (
		combined_scores[best_idx] - combined_scores[runner_up]
		if runner_up is not None and composite_idx is not None
		else np.nan
	)
	return {
		'cluster_id':                 cid,
		'cluster_size':               len(cluster_texts),
		'members':                    " | ".join(cluster_texts),
		'member_freqs':               " | ".join(str(original_label_counts.get(t, 0)) for t in cluster_texts),
		'canonical_selected':         canonical,  # overwritten after post-passes
		'canonical_pre_postpass':     canonical,
		'changed_by_postpass':        False,
		'selection_method':           method,
		'is_virtual':                 virtual_flags[best_idx],
		'freq_guard':                 freq_guard,
		'virtual_candidate':          virtual_hypernym if virtual_hypernym else (vh_raw or ''),
		'virtual_entered_pool':       virtual_hypernym is not None,
		'virtual_route':              vh_info['route'],
		'virtual_support':            vh_info['support'],
		'virtual_threshold':          vh_info['threshold'],
		'virtual_note':               vh_info['note'],
		'pure_sim_label':             cluster_texts[pure_sim_idx],
		'pure_sim_score':             float(similarities[pure_sim_idx]),
		'no_freq_winner':             candidates[nofreq_idx] if nofreq_idx is not None else '',
		'composite_winner_pre_guard': candidates[composite_idx] if composite_idx is not None else '',
		'canonical_sim':              float(similarities[best_idx]),
		'canonical_composite':        float(combined_scores[best_idx]),
		'canonical_freq':             int(raw_freqs[best_idx]),
		'runner_up':                  candidates[runner_up] if runner_up is not None else '',
		'margin_to_runner_up':        float(margin),
	}

def assign_canonical_labels(
	df: pd.DataFrame,
	X: np.ndarray,
	model,
	original_label_counts: Dict[str, int],
	debug_json_path: Optional[str] = None,
	shared_calibration_names: Optional[List[str]] = None,
	encode_prompt: Optional[str] = None,
	shared_threshold: float = 0.70,
	virtual_min_sim_ratio: Optional[float] = 0.60, # None to disable
	min_distinct_concepts: int = 2,
	verbose: bool = False,
) -> Dict[int, Dict]:
	"""
	The most intricate block which has:
	1) per-cluster loop 
	2) global post-passes.

	Assign a canonical label to every cluster using a five-signal composite
	score, with optional virtual hypernym synthesis.

	Composite score (per candidate)
	-------------------------------
		0.30 * cosine similarity to the cluster centroid
		+ 0.15 * corpus frequency (log-normalised; virtual candidates get 0)
		+ 0.20 * head-noun dominance across the cluster
		+ 0.25 * lexical containment (fraction of members containing ALL of
				 the candidate's tokens)
		+ 0.10 * brevity (shorter -> more general)

	Virtual hypernym synthesis (two routes, tried in order)
	--------------------------------------------------------
	1. head_suffix : the members share a head noun phrase at the END of
		 their head phrase, e.g.
			 ['black aircraft', 'white aircraft']           -> 'aircraft'
			 ['aerial view of harbor', 'aerial view of lake'] -> 'aerial view'
		 A label's head phrase is everything before its first preposition,
		 after dropping a trailing version marker, so 'aerial view of harbor'
		 has head phrase 'aerial view' and 'Ventura Mk II' has head 'ventura'.
	2. title_prefix : the members share a leading designator, e.g.
			 ['USS Arizona', 'USS Iowa']       -> 'USS'
			 ['Cruiser Mk I', 'Cruiser Mk II'] -> 'Cruiser'  (a dangling Mk / Ausf is dropped)
		 Only vessel designators, 'Operation' and 'X Mk' / 'X Ausf' families
		 are allowed: a shared lowercase prefix is a MODIFIER, not a category.

	Both routes require JOINT support (the whole core, contiguous, in at
	least max(2, ceil(0.5 * n)) members), prefer the core covering the
	most members (ties -> longer core), and never end in a function word.

	Virtual hypernym quality gates
	------------------------------
	A synthesised head is only a good canonical if it still describes the images, so a candidate is
	dropped (a real member then names the cluster) when:
		1. the members are only spelling / typo / hyphen variants of ONE concept
		 ('torsion test machine' x3 -> not 'machine'); designation variants stay distinct
		 (min_distinct_concepts);
		2. the core is a bare numeral, Roman numeral or single letter ('18', 'II', 'E'), a two-letter
		 fragment ('Co', 'FA') or a corporate suffix ('company', 'Limited'); the next valid suffix is
		 then considered ('Rubber Company');
		3. its cosine to the cluster centroid is below virtual_min_sim_ratio of the best real member's
		 ('Act' for named laws, 'class' for mechanic classes). A ratio, not an absolute cosine, so
		 the gate holds when the embedding space changes (prompted vs un-prompted).
	Every rejection is counted in meta.virtual_gates.rejections of the JSON and keeps a note on the
	cluster's virtual_hypernym.

	Acronym spelling
	----------------
	Dotted acronyms compare equal to their plain form ('Y.M.C.A.' == 'YMCA'). A virtual restored from
	dotted members keeps the final period ('Y.M.C.A.', 'S.S.') and is aligned with the corpus spelling.

	Frequency guard
	---------------
	Applied only when frequency actually flipped the decision, i.e. the
	composite winner differs from the winner of the same score WITHOUT the
	frequency term. The flip stands only if the frequency gain is >= 3x;
	otherwise the no-frequency winner is kept. Structural wins (e.g.
	'Lighthouse' beating 'coastal lighthouse' on containment/brevity) are
	no longer vetoed.

	Debug output (when debug_json_path is given)
	--------------------------------------------
	One JSON file: a 'meta' block (weights, thresholds, method counts, virtual gates) and a
	'clusters' list. Each cluster holds its members, the final canonical, how
	it was chosen (method, frequency-guard outcome, margin to the runner-up),
	the virtual-hypernym trace (route, support, note), its nearest other cluster, and every
	candidate sorted by rank with all five score components.

	Parameters
	----------
	df : pd.DataFrame
		Columns ['label', 'cluster'] with contiguous cluster IDs.
	X : np.ndarray, shape (n_unique_labels, d)
		L2-normalised embeddings; row order matches df['label'].
	model : SentenceTransformer
		Used to encode virtual hypernym candidates (and for its model id in the neighbour report).
	original_label_counts : Dict[str, int]
		Corpus frequency of every label.
	debug_json_path : str, optional
		Where to write the per-cluster selection JSON (see above).
	shared_threshold : float
		Centroid-similarity threshold of the shared-name resolver.
	shared_calibration_names : list of str, optional
		Names whose shared-canonical groups are printed (verbose only) to calibrate shared_threshold.
	neighbor_review_min_sim : float
		Minimum similarity for the nearest-cluster review file.
	encode_prompt : str, optional
		Prompt used when encoding virtual hypernyms. Must equal the prompt used for X.
	virtual_min_sim_ratio : float or None
		Minimum (virtual cosine / best real member cosine) for a virtual to stay in the pool.
		None disables the similarity gate.
	min_distinct_concepts : int
		Minimum number of distinct concepts (after collapsing spelling variants) a cluster needs
		before a virtual may be synthesised.
	verbose : bool
		Print per-cluster decisions.

	Returns
	-------
	cluster_canonicals : Dict[int, Dict]
		{
			cid: {
				'canonical',
				'score',
				'size',
				'virtual',
				'real_fallback',
				'real_fallback_score',
				'real_runner_up',
				'method'
			}
		}
	"""

	if verbose:
		print(f"[CANONICAL SELECTION]")
		print(f"  ├─ df: {df.shape} {list(df.columns)}")
		print(f"  ├─ X:  {X.shape}")
		print(f"  ├─ virtual_min_sim_ratio: {virtual_min_sim_ratio}")
		print(f"  └─ min_distinct_concepts: {min_distinct_concepts}")
		print(df.head(10))
		print()

	W_SIM, W_FREQ, W_HEAD, W_CONT, W_BREV = 0.30, 0.15, 0.20, 0.25, 0.10
	weights = {
		'sim': W_SIM, 
		'freq': W_FREQ, 
		'head': W_HEAD, 
		'cont': W_CONT, # containment
		'brev': W_BREV  # brevity
	}

	MIN_SUPPORT     = 0.5
	FREQ_GAIN_GUARD = 3.0

	# Virtual-hypernym quality gates (see the docstring). Each rejection is counted for the JSON.
	VIRTUAL_MIN_SIM_RATIO = virtual_min_sim_ratio  # relative drop threshold (e.g. 0.60)
	VIRTUAL_ABSOLUTE_SIM_FLOOR = 0.68              # absolute floor aligned with shared_threshold buffer
	VIRTUAL_SIM_GAP_GUARD = 0.25                   # max allowable sim loss when virtual beats real on structure
	MIN_DISTINCT_CONCEPTS = min_distinct_concepts
	virtual_rejections    = Counter()

	# Prepositions introduce a post-modifier: in 'aerial view of harbor' the
	# head phrase is 'aerial view'. Function words can never start or end a
	# synthesised hypernym ('aerial view of' is rejected).
	PREPOSITIONS = {
		'of', 'in', 'on', 'at', 'with', 'over', 'under', 'near', 'from', 'by',
		'for', 'during', 'into', 'onto', 'across', 'along', 'above', 'below',
		'behind', 'beside', 'between', 'inside', 'outside', 'through',
		'toward', 'towards', 'upon', 'within', 'without', 'after', 'before',
		'around',
	}
	FUNCTION_WORDS = PREPOSITIONS | {'the', 'a', 'an', 'and', 'or', 'to', 'as'}

	# Token normalisation
	# Lowercase + strip trailing punctuation, so 'Ausf.' == 'Ausf' == 'ausf'.
	_TRAILING_PUNCT = re.compile(r'[^\w]+$')

	# 'Y.M.C.A.' and 'YMCA' (and 'S.S.' / 'SS', 'U.S.' / 'US') are ONE token in every comparison:
	# suffix support, containment, head scores and variant detection.
	_DOTTED_ACRONYM = re.compile(r'^(?:[a-z]\.)+[a-z]?$')

	def _norm_token(tok: str) -> str:
		t = _TRAILING_PUNCT.sub('', tok.lower())
		return t.replace('.', '') if _DOTTED_ACRONYM.match(t) else t

	def _norm_tokens(label: str) -> List[str]:
		return [_norm_token(t) for t in label.split() if _norm_token(t)]

	def _norm_token_set(label: str) -> set:
		return set(_norm_tokens(label))

	def _token_pairs(label: str) -> List[Tuple[str, str]]:
		"""(raw_token, normalised_token) pairs, 
		aligned, empties dropped.
		"""
		return [(t, _norm_token(t)) for t in label.split() if _norm_token(t)]

	# ── Version tails and hygiene ─────────────────────────────────────────────
	# 'Ventura II', 'Stalag 7A', 'Senate Resolution 77', 'Camp B': the trailing numeral / number /
	# letter is a version marker, not part of the category. Roman numerals are matched strictly
	# (I..XXXIX), so real words made of those letters ('mill', 'civil', 'mix') are never stripped.
	_VERSION_RE = re.compile(r'^(?:(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})|\d+[a-z]{0,2}|[a-z])$')
	_VERSION_MARKERS = {'mk', 'mark', 'type', 'typ', 'model', 'no', 'ausf', 'series'}
	_LEGAL_SUFFIXES = {'co', 'corp', 'corporation', 'company', 'inc', 'ltd', 'limited', 'plc', 'gmbh', 'ag', 'llc', 'bros', 'sons'}

	def _is_version_token(tok: str) -> bool:
		return bool(_VERSION_RE.match(tok))

	def _strip_version_tail(toks: List[str]) -> List[str]:
		toks = list(toks)
		while len(toks) > 1 and _is_version_token(toks[-1]):
			# 'He 111', 'Ju 87', 'Bf 109': a number right after a 1-3 letter code IS the designation, keep it
			if toks[-1][0].isdigit() and re.fullmatch(r'[a-z]{1,3}', toks[-2]) and toks[-2] not in _VERSION_MARKERS:
				break
			toks.pop()
			while len(toks) > 1 and toks[-1] in _VERSION_MARKERS:
				toks.pop()
		return toks

	def _core_rejection(core: List[str]) -> Optional[str]:
		"""Why a synthesised head can never be a canonical: bare numeral / Roman numeral / single letter,
		a 2-letter fragment, or a corporate suffix. Alphanumeric designations ('He 111', 'C-82') stay valid."""
		if all(_is_version_token(t) for t in core):
			return 'numeral_or_letter'
		if all(t in _LEGAL_SUFFIXES for t in core):
			return 'corporate_suffix'
		if len(re.sub(r'[^a-z]', '', ''.join(core))) < 3 and not any(ch.isdigit() for t in core for ch in t):
			return 'too_short'
		return None

	def _crude_stem(tok: str) -> str:
		for suf in ('ing', 'ed', 'es', 's'):
			if len(tok) > len(suf) + 2 and tok.endswith(suf):
				return tok[:-len(suf)]
		return tok

	def _distinct_concepts(lbls: List[str]) -> int:
		"""Number of distinct concepts once spelling variants are collapsed: hyphen / spacing / possessive /
		plural differences and typos ('Venezualan' / 'Venezuelan') merge. Numbers, Roman numerals and single
		letters must match exactly, so 'A-4 Skyhawk' / 'A-4B Skyhawk' and 'Mk I' / 'Mk II' stay distinct."""
		import difflib
		keys = []
		for l in lbls:
			lbl_lower = (l == l.lower())
			l = re.sub(r'(?:[A-Za-z]\.){2,}[A-Za-z]?', lambda mo: mo.group(0).replace('.', ''), l)   # 'U.S.S.' -> 'USS'

			# Normalize number-unit spacing so '88mm', '88 mm', '105mm' parse consistent numbers
			l = re.sub(r'(\d+)\s*(mm|cm|in|inch|inches|hp|ton|tons|pdr|pounder)\b', r'\1 \2', l, flags=re.IGNORECASE)

			toks = re.findall(r"[a-z0-9]+", l.lower())

			num = tuple(
				t 
				for t in toks 
				if any(ch.isdigit() for ch in t) 
				or (len(t) > 1 and _is_version_token(t))
				or (len(t) == 1 and t != 's')
			)
			
			alpha = ''.join(_crude_stem(t) for t in toks if t not in num and len(t) > 1)
			keys.append((num, alpha, lbl_lower))
		parent = list(range(len(keys)))

		def find(i):
			while parent[i] != i:
				parent[i] = parent[parent[i]]
				i = parent[i]
			return i

		for i in range(len(keys)):
			for j in range(i + 1, len(keys)):
				# Fuzzy (typo / morphology) matching is for common nouns and long keys: 'knit socks' /
				# 'knitted socks' and 'Venezualan lancers' / 'Venezuelan Lancers' are one concept, but short
				# capitalised names such as 'USS Texan' / 'USS Texana' / 'USS Texas' are three different ships.
				fuzzy_ok = min(len(keys[i][1]), len(keys[j][1])) >= 12 or (keys[i][2] and keys[j][2])
				if keys[i][0] == keys[j][0] and (
					keys[i][1] == keys[j][1]
					or (fuzzy_ok and difflib.SequenceMatcher(None, keys[i][1], keys[j][1]).ratio() >= 0.90)
				):
					parent[find(j)] = find(i)
		
		return len({find(i) for i in range(len(keys))})

	def _head_phrase(norm_toks: List[str]) -> List[str]:
		"""Tokens before the first preposition (never the very first token), after dropping a trailing
		version marker. Used for BOTH synthesis and scoring, so a virtual 'Ventura' and the members
		'Ventura II' / 'Ventura Mk V' agree on their head."""
		toks = _strip_version_tail(norm_toks)
		for i, t in enumerate(toks):
			if i > 0 and t in PREPOSITIONS:
				return toks[:i]
		return toks

	def _head_token(label: str) -> str:
		"""'aerial view of harbor' -> 'view'; 
		'coastal lighthouse' -> 'lighthouse'.
		"""

		hp = _head_phrase(_norm_tokens(label))

		return hp[-1] if hp else ''

	def _trim_function_words(toks) -> List[str]:
		toks = list(toks)
		while toks and toks[0] in FUNCTION_WORDS:
			toks.pop(0)
		while toks and toks[-1] in FUNCTION_WORDS:
			toks.pop()
		return toks

	def _head_suffix_core(lbls: List[str], threshold: int):
		"""Route 1: shared contiguous suffix of the members' head phrases.

		The winner is the suffix with the highest support (ties -> the longer one). Cores that can never
		be a canonical (numerals, single letters, corporate suffixes) are skipped, so the next valid
		suffix is considered: 'rubber company' instead of a bare 'company'.
		"""
		n = len(lbls)
		hps = [_head_phrase(_norm_tokens(l)) for l in lbls]
		max_len = max((len(h) for h in hps), default=0)
		best, best_below, blocked = None, None, []        # (support, length, core)
		for k in range(1, max_len + 1):
			counts = Counter(tuple(h[-k:]) for h in hps if len(h) >= k)
			for suffix, support in counts.items():
				core = _trim_function_words(suffix)
				if not core:
					continue
				why = _core_rejection(core)
				if why:
					if support >= threshold:
						blocked.append((' '.join(core), why))
					continue
				cand = (support, len(core), tuple(core))
				if support >= threshold:
					if best is None or cand > best:
						best = cand
				elif best_below is None or cand > best_below:
					best_below = cand
		if best:
			return list(best[2]), best[0], ''
		if blocked:
			virtual_rejections['hygiene:' + blocked[0][1]] += 1
			return None, 0, f"head_suffix: rejected {blocked[0][0]!r} ({blocked[0][1]})"
		if best_below:
			note = (f"head_suffix: best '{' '.join(best_below[2])}' only "
					f"{best_below[0]}/{n} (need {threshold})")
		else:
			note = "head_suffix: no shared head"
		return None, 0, note

	def _title_prefix_core(lbls: List[str], threshold: int):
		"""
		Route 2: conservative title / designation prefix synthesis.

		Allowed prefixes (nothing else):
			1. Named-vessel / transport designators:
					USS, HMS, HMCS, HMAS, HMNZS, USAT, USNS, USCGC, RMS, MV, and dotted
					S.S. (undotted 'SS' is excluded: in this corpus it is mostly the
					Schutzstaffel, an organisation).
			2. 'Operation', as an explicit archival designation.
			3. A base name followed by Mk / Ausf: 'Sunderland Mk', 'Grille Ausf'.
				A bare 'Mk' or 'Ausf' is never allowed.

		Deliberately NOT allowed: organisations (RAF, NATO, NASA, AAF, USMC),
		countries (U.S., British, Italian), and generic acronyms (VIP, TWA, NBC).
		As a canonical, those name the owner or origin, not the thing pictured.

		Dotted and undotted spellings are grouped together ('U.S.S. Luzon' and
		'USS Leyte' support the same prefix). The prefix is maximal: on equal
		support, the longer prefix wins.
		"""
		n = len(lbls)
		pairs = [_token_pairs(l) for l in lbls]
		max_len = max((len(p) for p in pairs), default=0)

		DESIGNATOR_PREFIXES = {
			'uss', 
			'hms', 
			'hmcs', 
			'hmas', 
			'hmnzs', 
			'usat', 
			'usns', 
			'uscgc', 
			'rms', 
			'mv',
		}

		EXPLICIT_PREFIXES   = {'operation'}

		DESIGNATION_ENDINGS = {'mk', 'ausf'}
		
		def _key_token(raw: str, norm: str) -> str:
			"""Dot-insensitive key; undotted 'SS' gets a key that is never allowed."""
			k = norm.replace('.', '')
			if k == 'ss' and '.' not in raw:
				return 'ss#undotted'
			return k

		def _is_allowed(key: tuple) -> bool:
			if len(key) == 1:
				return key[0] in DESIGNATOR_PREFIXES or key[0] in EXPLICIT_PREFIXES or key[0] == 'ss'
			if key[-1] in DESIGNATION_ENDINGS:
				base = key[:-1]
				return all(t not in FUNCTION_WORDS and t not in DESIGNATION_ENDINGS for t in base)
			
			return False

		best, best_below = None, None            # (support, length, core_tuple)
		for k in range(1, max_len):
			groups: Dict[tuple, List[tuple]] = defaultdict(list)
			for p in pairs:
				if len(p) > k:
					key = tuple(_key_token(r, t) for r, t in p[:k])
					groups[key].append(tuple(t for _, t in p[:k]))
			for key, variants in groups.items():
				if not _is_allowed(key):
					continue
				support = len(variants)
				counts = Counter(variants)
				core = max(counts, key=lambda v: (counts[v], v))   # most common spelling
				cand = (support, k, core)
				if support >= threshold:
					if best is None or cand > best:
						best = cand
				elif best_below is None or cand > best_below:
					best_below = cand

		if best:
			return list(best[2]), best[0], ''
		if best_below:
			note = (f"title_prefix: best allowed prefix '{' '.join(best_below[2])}' "
							f"only {best_below[0]}/{n} (need {threshold})")
		else:
			note = "title_prefix: no allowed designator prefix"
		return None, 0, note

	def _virtual_hypernym(lbls: List[str]):
		n = len(lbls)
		threshold = max(2, int(np.ceil(MIN_SUPPORT * n)))
		info = {'route': '', 'support': 0, 'threshold': threshold, 'note': ''}
		k = _distinct_concepts(lbls)
		if k < MIN_DISTINCT_CONCEPTS:
			virtual_rejections['one_concept'] += 1
			info['note'] = f"rejected: the {n} members are spelling variants of {k} concept (need >= {MIN_DISTINCT_CONCEPTS})"
			return None, info
		core, support, note_a = _head_suffix_core(lbls, threshold)
		if core:
			info.update(route='head_suffix', support=support,
						note=f"shared head phrase in {support}/{n}")
			return " ".join(core), info
		core, support, note_b = _title_prefix_core(lbls, threshold)
		if core:
			while len(core) > 1 and core[-1] in ('mk', 'ausf'):      # 'Cruiser Mk' -> 'Cruiser'
				core = core[:-1]
			info.update(route='title_prefix', support=support,
						note=f"shared title in {support}/{n}; {note_a}")
			return " ".join(core), info
		info['note'] = f"{note_a}; {note_b}"
		return None, info

	def _restore_surface(vh: str, cluster_texts: List[str]) -> str:
		"""Borrow each token's most frequent surface form from the members."""
		restored = []
		for ntok in _norm_tokens(vh):
			surface_forms: Dict[str, int] = {}
			for lbl in cluster_texts:
				for raw_tok in lbl.split():
					if _norm_token(raw_tok) == ntok:
						freq = original_label_counts.get(lbl, 1)
						surface_forms[raw_tok] = surface_forms.get(raw_tok, 0) + freq
			if surface_forms:
				# deterministic: highest weight, then lexicographic
				restored.append(max(surface_forms, key=lambda s: (surface_forms[s], s)))
			else:
				restored.append(ntok.title())
		out = " ".join(restored)
		# The final period belongs to a dotted abbreviation ('Y.M.C.A.', 'S.S.', 'U.S.'): stripping it would
		# create 'Y.M.C.A', a spelling that exists nowhere in the corpus and splits the class from 'Y.M.C.A.'.
		if re.search(r'(?:[A-Za-z]\.){2,}$', out):
			return out
		# strip trailing punctuation left over from a raw token like 'Ausf.'
		return _TRAILING_PUNCT.sub('', out) or out

	def _containment_scores(candidates: List[str], cluster_lbls: List[str]) -> np.ndarray:
		"""Fraction of members whose token set contains ALL candidate tokens."""
		cluster_norm_sets = [_norm_token_set(lbl) for lbl in cluster_lbls]
		return np.array([
			sum(1 for ns in cluster_norm_sets if _norm_token_set(c) <= ns)
			/ max(len(cluster_lbls), 1)
			for c in candidates
		])
	
	case_registry = _build_case_registry(original_label_counts)

	cluster_canonicals = {}
	virtual_used_count = 0
	freq_changed_count = 0
	selection_records  = [] # one per cluster -> debug CSV
	candidate_records  = [] # one per candidate -> nested into the JSON
	cluster_centroids: Dict[int, np.ndarray] = {}
	cluster_members:  Dict[int, List[str]]  = {}
	n_clusters_total: int = df.cluster.nunique()
	tradeoff_records: List[Dict] = [] # one per real pick (differs from pure similarity)

	for cid in sorted(df.cluster.unique()):
		cluster_mask       = df.cluster == cid
		cluster_texts      = df[cluster_mask]['label'].tolist()
		cluster_indices    = df[cluster_mask].index.tolist()
		cluster_embeddings = X[cluster_indices] # (n, d) L2-normalised
		cluster_size       = len(cluster_texts)

		if verbose:
			nl = '\n' if cluster_size > 10 else ' '
			print(f"\n[Cluster {cid:5d}/{n_clusters_total}] {cluster_size} labels{nl}{cluster_texts}")

		centroid = cluster_embeddings.mean(axis=0)  # real members only
		cluster_centroids[cid] = centroid
		cluster_members[cid]   = cluster_texts

		# ── Virtual hypernym: synthesise, then gate on similarity ────
		virtual_hypernym, vh_raw, vh_info = _synthesize_virtual(
			cluster_texts=cluster_texts,
			synth_fn=_virtual_hypernym,
			restore_fn=_restore_surface,
			norm_token_set_fn=_norm_token_set,
			norm_tokens_fn=_norm_tokens,
			case_registry=case_registry,
		)
		if verbose:
			_print_virtual_trace(vh_raw, virtual_hypernym, vh_info, cluster_size)

		similarities = _centroid_similarities(
			centroid, 
			cluster_embeddings, 
			virtual_hypernym, 
			model, 
			encode_prompt
		)
		pure_sim_idx = int(similarities[:cluster_size].argmax())  # real labels only

		virtual_hypernym, similarities, rejection_virtual_tag = _gate_virtual_by_similarity(
			virtual_hypernym,
			similarities,
			cluster_size,
			min_sim_ratio=VIRTUAL_MIN_SIM_RATIO,
			abs_floor=VIRTUAL_ABSOLUTE_SIM_FLOOR,
			vh_info=vh_info,
			virtual_rejections=virtual_rejections,
		)
		candidates, virtual_flags = _build_candidate_pool(cluster_texts, virtual_hypernym)

		raw_freqs = np.array(
			[0 if virtual_flags[i] else original_label_counts.get(c, 0) for i, c in enumerate(candidates)],
			dtype=float,
		)

		# ── Scoring + frequency guard ────
		scored = bool(original_label_counts) and cluster_size > 1
		scores = (
			_score_candidates(
				candidates, cluster_texts, similarities, raw_freqs,
				head_token_fn=_head_token,
				containment_fn=_containment_scores,
				weights=weights,
			)
			if scored else _unscored(len(candidates))
		)

		pick = _pick_best_candidate(
			scored, scores, similarities, raw_freqs, virtual_flags,
			cluster_size, pure_sim_idx,
			freq_gain_guard=FREQ_GAIN_GUARD,
			sim_gap_guard=VIRTUAL_SIM_GAP_GUARD,
		)

		best_idx      = pick['best_idx']
		composite_idx = pick['composite_idx']
		nofreq_idx    = pick['nofreq_idx']
		freq_guard    = pick['freq_guard']
		is_virtual_pick = virtual_flags[best_idx]

		method = _label_selection_method(
			composite_idx, 
			is_virtual_pick, 
			freq_guard, 
			best_idx, 
			pure_sim_idx
		)

		# statistics block
		if is_virtual_pick:
			virtual_used_count += 1
		elif best_idx != pure_sim_idx and composite_idx is not None:
			freq_changed_count += 1
			tradeoff_record = _tradeoff_for_pick(
				cid, 
				cluster_texts, 
				candidates, 
				similarities,
				original_label_counts, 
				best_idx, 
				pure_sim_idx,
			)
			tradeoff_records.append(tradeoff_record)

		# ── Records ────
		rows, ranking = _build_candidate_rows(
			cid, candidates, virtual_flags, raw_freqs, similarities, scores,
			best_idx, pure_sim_idx, nofreq_idx, head_token_fn=_head_token,
		)
		candidate_records.extend(rows)
		runner_up = next((i for i in ranking if i != best_idx and scores['eligible'][i]), None)

		cluster_canonicals[cid] = _build_canonical_entry(
			candidates, cluster_texts, virtual_flags, similarities,
			scores['combined_scores'], best_idx, pick['best_real_idx'],
			composite_idx, method,
			eligible=scores['eligible'],
		)
		selection_records.append(
			_build_selection_record(
				cid, cluster_texts, original_label_counts,
				candidates, virtual_flags, virtual_hypernym, vh_raw, vh_info,
				similarities, scores['combined_scores'], raw_freqs,
				best_idx, pure_sim_idx, composite_idx, nofreq_idx, runner_up,
				method, freq_guard,
			)
		)

		selection_records[-1]['numeric_ineligible'] = int((~scores['eligible']).sum() + (len(candidates) if scores['numeric_unresolved'] else 0))
		selection_records[-1]['numeric_unresolved'] = bool(scores['numeric_unresolved'])

		if verbose:
			tag = " [VIRTUAL]" if is_virtual_pick else ""
			if scores['numeric_unresolved']:
				tag += " [NUMERIC CONFLICT: no eligible candidate, mask skipped]"
			elif not scores['eligible'].all():
				dropped = [c for c, ok in zip(candidates, scores['eligible']) if not ok]
				tag += f" [NUMERIC: excluded {dropped}]"
			print(
				f"\t=> Selected Canonical: {repr(candidates[best_idx]):<45}"
				f"(sim={similarities[best_idx]:.4f}){tag} {rejection_virtual_tag}"
			)

	pre_postpass = {
		cid: meta['canonical']
		for cid, meta in cluster_canonicals.items()
	}

	if shared_calibration_names and verbose:
		report_shared_group_similarities(
			cluster_canonicals,
			cluster_centroids,
			cluster_members,
			names=shared_calibration_names,
		)

	cluster_canonicals = _resolve_shared_canonicals(
		cluster_canonicals=cluster_canonicals,
		cluster_centroids=cluster_centroids,
		cluster_members=cluster_members,
		original_label_counts=original_label_counts,
		threshold=shared_threshold,
		verbose=verbose,
	)

	cluster_canonicals = _harmonize_final_canonicals(
		cluster_canonicals=cluster_canonicals,
		original_label_counts=original_label_counts,
		verbose=verbose,
	)

	# stamp final state onto the per-cluster records
	_finalize_selection_records(
		selection_records=selection_records,
		cluster_canonicals=cluster_canonicals,
		pre_postpass=pre_postpass,
	)

	if debug_json_path:
		clusters_json = _build_clusters_json(selection_records, candidate_records)

		payload = _build_debug_payload(
			clusters_json=clusters_json,
			selection_records=selection_records,
			n_candidates=len(candidate_records),
			weights=weights,
			min_support=MIN_SUPPORT,
			virtual_gates={
				'min_sim_ratio': VIRTUAL_MIN_SIM_RATIO,
				'min_distinct_concepts': MIN_DISTINCT_CONCEPTS,
				'rejections': dict(virtual_rejections),
			},
			freq_gain_guard=FREQ_GAIN_GUARD,
		)
		_write_debug_json(debug_json_path, payload, verbose=verbose)

	if verbose:
		if selection_records:
			_print_selection_overview(selection_records, virtual_rejections)
		if tradeoff_records:
			_print_tradeoff_analysis(
				total_sim_loss=[r['sim_loss'] for r in tradeoff_records],
				total_freq_gain=[r['freq_gain'] for r in tradeoff_records],
				questionable_examples=[r for r in tradeoff_records if r['questionable']],
			)
		else:
			print("\nScore-based selection made no changes (all clusters picked highest similarity)")
			print("=" * 100)

		_print_frequency_impact(
			n_clusters=df.cluster.nunique(),
			virtual_used_count=virtual_used_count,
			freq_changed_count=freq_changed_count,
			debug_json_path=debug_json_path,
		)

	return cluster_canonicals

def map_inject_and_clean_clusters(
	df: pd.DataFrame,
	X: np.ndarray,
	cluster_canonicals: Dict[int, Dict],
	canonical_emb_map: Dict[str, np.ndarray],
	model: Any,
	batch_size: int,
	low_cohesion_threshold: float,
	poor_canonical_threshold: float,
	encode_prompt: Optional[str] = None,
	clusters_fname: Optional[str] = None,
	verbose: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
	"""
	STEP 7: Map canonical labels, inject virtual canonical rows for validation,
	filter problematic clusters, harmonize display surfaces, and export CSV.

	Parameters
	----------
	df : pd.DataFrame
		DataFrame containing 'label' and 'cluster' assignments.
	X : np.ndarray
		Embedding matrix aligned with rows of df.
	cluster_canonicals : Dict[int, Dict]
		Mapping of cluster IDs to canonical metadata.
	canonical_emb_map : Dict[str, np.ndarray]
		Lookup dictionary for precomputed canonical embeddings.
	model : Any
		SentenceTransformer model for encoding virtual canonicals if not in canonical_emb_map.
	batch_size : int
		Batch size for fallback encoding.
	encode_prompt : Optional[str]
		Prompt for fallback encoding.
	low_cohesion_threshold : float
		Cosine intra-similarity threshold to drop low-cohesion clusters.
	poor_canonical_threshold : float
		Representativeness threshold to drop poorly represented canonical clusters.
	clusters_fname : Optional[str]
		Base CSV path used to export the final consolidated DataFrame.
	verbose : bool
		Whether to print injection, cleaning, and harmonization summaries.

	Returns
	-------
	df : pd.DataFrame
		Final cleaned and harmonized DataFrame.
	X_clean : np.ndarray
		Embedding matrix aligned with cleaned df.
	removed_labels : List[str]
		List of labels removed by low-cohesion / poor-canonical filters.
	"""
	df["canonical"] = df["cluster"].map(lambda c: cluster_canonicals[c]["canonical"])
	df["is_injected"] = False

	# Every virtual canonical is injected as a row of its OWN cluster, even when
	# the same string is a real label in another cluster. Injected rows exist only
	# so internal checks (remove_problematic_clustered_labels, the quality report)
	# find the canonical among the cluster's rows. They are excluded from the
	# label -> canonical mapping in get_canonical_labels (is_injected == False),
	# so a duplicate string can never overwrite a real label's mapping.
	real_labels = set(df["label"])
	virtual_rows, virtual_texts = [], []
	for cid, meta in cluster_canonicals.items():
		if not meta.get("virtual"):
			continue
		vh = meta["canonical"]
		virtual_rows.append({"label": vh, "cluster": cid, "canonical": vh, "is_injected": True})
		virtual_texts.append(vh)

	if virtual_rows:
		if all(vh in canonical_emb_map for vh in virtual_texts):
			virtual_embs = np.vstack([canonical_emb_map[vh] for vh in virtual_texts]).astype(np.float32)
		else:
			virtual_embs = _encode_(model, virtual_texts, batch_size, encode_prompt)
		df = pd.concat([df, pd.DataFrame(virtual_rows)], ignore_index=True)
		X = np.vstack([X, virtual_embs])

	if verbose:
		shared = sum(1 for t in virtual_texts if t in real_labels)
		print(
			f"\n[INJECT] {len(virtual_rows)} virtual canonical row(s) injected "
			f"({shared} share a string with a real label in another cluster; "
			f"excluded from the label mapping via is_injected)"
		)
		print(df.info(verbose=verbose, memory_usage=True))

	df, X_clean, removed_labels = remove_problematic_clustered_labels(
		df=df,
		embeddings=X,
		low_cohesion_threshold=low_cohesion_threshold,
		poor_canonical_threshold=poor_canonical_threshold,
		verbose=verbose,
	)

	df["canonical_internal"] = df["canonical"]
	rename = {
		m["canonical"]: m["canonical_harmonized"]
		for m in cluster_canonicals.values()
		if m.get("changed_by_harmonize") and m.get("canonical_harmonized")
	}
	if rename:
		df["canonical"] = df["canonical"].replace(rename)
		if verbose:
			print(f"\n[HARMONIZE DISPLAY] {len(rename)} canonical surface rename(s)")
			for old, new in list(rename.items()):
				print(f"  {repr(old):<30} -> {new!r}")

	if clusters_fname is not None:
		out_csv = clusters_fname.replace(".csv", "_semantic_consolidation_agglomerative.csv")
		df.to_csv(out_csv, index=False)

	return df, X_clean, removed_labels

def cluster(
	labels: List[List[str]],
	model_id: str,
	clusters_fname: str,
	batch_size: int,
	device: Union[torch.device, str],
	nc: int,
	linkage_method: str="ward",
	distance_metric: str="euclidean",
	target_intra_similarity: float = 0.75, # 0.69,
	min_consolidation: float = 4.0, #3.5, #4.0, # 3.8, #5.0, 
	max_consolidation: float = 8.0, #6.0, #6.5, # 5.0, #7.5, 
	min_singleton_ratio: float = 1.5e-2, # 1.5%
	max_singleton_ratio: float = 3.0e-2, # 3.0%
	min_singleton_merge_sim: Optional[float] = 0.60,
	merge_near_duplicate_clusters_medoid: bool = False,
	merge_near_duplicate_clusters_canonical: bool = False,
	merge_close_clusters_threshold: Optional[float] = None,
	merge_canonical_clusters_threshold: Optional[float] = None,
	merge_canonical_min_centroid_sim: float = 0.65,
	merge_max_size: Optional[int] = None,
	min_cluster_size: int = 2,
	low_cohesion_threshold=0.60,
	poor_canonical_threshold=0.60,
	merge_singletons: bool = True,
	use_cache: bool = True,
	encode_prompt: Optional[str] = None,
	verbose: bool = False,
) -> pd.DataFrame:
	
	st_t = time.time()
	if encode_prompt is not None:
		# A prompted encoding is a different experiment: never overwrite the un-prompted outputs
		# (CSV, canonical-selection JSON, neighbour files). Delete these 3 lines when you adopt it for good.
		_stem, _ext = os.path.splitext(clusters_fname)
		clusters_fname = f"{_stem}_instruction_based_prompt{_ext}"
		if verbose:
			print(f"[ENCODE PROMPT] {encode_prompt!r}\n[ENCODE PROMPT] outputs are written with the suffix _instr -> {clusters_fname}")

	if verbose:
		print(f"\n[AGGLOMERATIVE CLUSTERING] {len(labels)} samples")
		print(f"  ├─ {model_id} | {device} | batch_size: {batch_size}")
		print(f"  ├─ linkage: {linkage_method}")
		print(f"  ├─ samples: {labels[:3]}")
		requires_type_exchange = isinstance(labels[0], str)
		print(f"  ├───> {type(labels[0])} requires_type_exchange? {requires_type_exchange}")
		print(f"  ├─ merge_near_duplicate_clusters_medoid: {merge_near_duplicate_clusters_medoid} (threshold: {merge_close_clusters_threshold if merge_close_clusters_threshold is not None else 'dynamic'})")
		print(f"  ├─ merge_near_duplicate_clusters_canonical: {merge_near_duplicate_clusters_canonical} (threshold: {merge_canonical_clusters_threshold if merge_canonical_clusters_threshold is not None else 'dynamic'}, min_centroid_sim: {merge_canonical_min_centroid_sim})")
		print(f"  ├─ encode_prompt: {encode_prompt!r}")
		print(f"  └─ nc: {nc} {f'Manually defined' if nc else '=> Adaptive Search'}")

	# STEP 1: DEDUP + FLATTEN
	documents = list()
	for i, doc in enumerate(labels):
		if doc is None:
			continue
		if isinstance(doc, str):
			try:
				doc = ast.literal_eval(doc)
			except (ValueError, SyntaxError):
				print(f"doc[{i}]: Failed to parse '{doc}' (skipping)")
				continue
		if not isinstance(doc, list):
			print(f"doc[{i}]: Invalid type {type(doc)} (skipping)")
			continue
		documents.append(list(set(lbl for lbl in doc if lbl is not None)))

	# Collapse case-only duplicates ('trench'/'Trench') 
	# before computing unique_labels,
	# so they are not embedded and clustered as if distinct.
	documents = _normalize_label_case(documents, verbose=verbose)
	unique_labels = sorted(
		set(
			label 
			for doc in documents 
			for label in doc
		)
	)

	if verbose:
		print("\n[DATASET SUMMARY]")
		print(f"  ├─ Total documents: {len(documents)} {documents[:3]}")
		print(f"  └─ Unique labels: {len(unique_labels)} {unique_labels[:15]}")

	attention, dtype = get_model_kwargs(verbose=verbose)
	model = SentenceTransformer(
		model_name_or_path=model_id,
		model_kwargs={"attn_implementation": attention, "dtype": dtype}, # no device_map
		trust_remote_code=True,
		device=device, # single device
		cache_folder=cache_directory,#cache_directory[os.getenv('USER')],
		token=os.getenv("HUGGINGFACE_TOKEN"),
		processor_kwargs={"padding_side": "left"}, # renamed from tokenizer_kwargs
	)

	if verbose:
		print(
			f"[ENCODING] {len(unique_labels)} unique labels | {model.model_card_data.base_model} | "
			f"dtype: {next(model.parameters()).dtype} "
			f"({sum(p.numel() for p in model.parameters()):,} parameters)"
		)

	# STEP 3: LOAD / COMPUTE EMBEDDINGS + LINKAGE
	X, Z = get_clustering_artifacts(
		clusters_fname=clusters_fname,
		unique_labels=unique_labels,
		model=model,
		batch_size=batch_size,
		linkage_method=linkage_method,
		distance_metric=distance_metric,
		use_cache=use_cache,
		encode_prompt=encode_prompt,
		verbose=verbose,
	)

	# STEP 4: clustering the labels (manual or automatic)
	clustered_labels = get_clustered_labels(
		X=X,
		linkage_matrix=Z,
		label_texts=unique_labels,
		target_intra_similarity=target_intra_similarity,
		min_consolidation=min_consolidation,
		max_consolidation=max_consolidation,
		min_singleton_ratio=min_singleton_ratio,
		max_singleton_ratio=max_singleton_ratio,
		target_singleton_ratio=0.015,
		quality_vs_consolidation_weight=0.5,
		merge_singletons=merge_singletons,
		min_cluster_size=min_cluster_size,
		min_singleton_merge_sim=min_singleton_merge_sim,
		nc=nc,
		verbose=verbose,
	)

	# STEP 4a: PRE-MERGE CLUSTER NEIGHBOUR DIAGNOSTICS & DYNAMIC THRESHOLDING
	neighbor_summary = get_cluster_medoid_nearest_neighbor(
		embedding_model_id=model.model_card_data.base_model,
		X=X,
		labels=clustered_labels,
		label_texts=unique_labels,
		max_merged_size=merge_max_size,
		threshold=merge_close_clusters_threshold,
		review_path=os.path.splitext(clusters_fname)[0] + "_medoid_neighbor_pairs.json",
		verbose=verbose,
	)

	# STEP 4b: OPTIONAL MERGE OF NEAR-DUPLICATE CLUSTERS (before canonical selection)
	if merge_near_duplicate_clusters_medoid:
		active_merge_threshold = (
			float(merge_close_clusters_threshold)
			if merge_close_clusters_threshold is not None
			else float(neighbor_summary['threshold'])
		)
		clustered_labels = merge_cluster_medoid_nearest_neighbor(
			X=X,
			labels=clustered_labels,
			label_texts=unique_labels,
			threshold=active_merge_threshold,
			max_merged_size=merge_max_size,
			report_path=os.path.splitext(clusters_fname)[0] + "_medoid_merges.json",
			verbose=verbose,
		)
 
	df = pd.DataFrame({'label': unique_labels, 'cluster': clustered_labels})

	# STEP 5: LABEL FREQUENCY DICT
	if verbose:
		print(f"\n[CLUSTERING] {len(np.unique(clustered_labels))} clusters for {clustered_labels.shape} {type(clustered_labels)} labels")

	label_freq_dict: dict = {}
	for doc in documents:
		for label in doc:
			label_freq_dict[label] = label_freq_dict.get(label, 0) + 1

	if verbose:
		print(f"  ├─ Frequency dict has {len(label_freq_dict)} labels")
		print(f"  ├─ Total label instances: {sum(label_freq_dict.values())}")
		print(f"  └─ Most frequent: {max(label_freq_dict.items(), key=lambda x: x[1])}")
		print('-' * 80)

	# STEP 6: CANONICAL SELECTION (with virtual hypernym synthesis)
	cluster_canonicals = assign_canonical_labels(
		df=df,
		X=X,
		model=model,
		original_label_counts=label_freq_dict,
		debug_json_path=os.path.splitext(clusters_fname)[0] + "_canonical_selection.json",
		encode_prompt=encode_prompt,
		# # detailed report:
		# shared_calibration_names=[
		# 	'camera', 'cap', 'camp', 'debris', 'building', 'hospital', # should stay together
		# 	'press', 'ward', 'tank', 'float', 'race', 'gear', 'party', # should split
		# 	'hangar', 'bridge', 'sign', 'station',
		# ],
		verbose=verbose,
	)

	# STEP 6a: POST-CANONICAL NEIGHBOUR DIAGNOSTICS & DYNAMIC THRESHOLDING
	canonical_emb_map = _build_canonical_embedding_lookup(
		cluster_canonicals=cluster_canonicals,
		label_texts=unique_labels,
		X=X,
		model=model,
		batch_size=batch_size,
		encode_prompt=encode_prompt,
		verbose=verbose,
	)

	canonical_neighbor_summary = get_cluster_canonical_nearest_neighbor(
		embedding_model_id=model.model_card_data.base_model,
		X=X,
		labels=clustered_labels,
		label_texts=unique_labels,
		cluster_canonicals=cluster_canonicals,
		canonical_emb_map=canonical_emb_map,
		max_merged_size=merge_max_size,
		threshold=merge_canonical_clusters_threshold,
		min_centroid_similarity=merge_canonical_min_centroid_sim,
		review_path=os.path.splitext(clusters_fname)[0] + "_canonical_neighbor_pairs.json",
		verbose=verbose,
	)

	# STEP 6b: OPTIONAL MERGE OF NEAR-DUPLICATE CANONICAL CLUSTERS (before virtual row injection)
	canonical_merges_done = 0
	if merge_near_duplicate_clusters_canonical:
		n_clusters_pre_canon_merge = len(np.unique(clustered_labels))
		clustered_labels, cluster_canonicals = merge_cluster_canonical_nearest_neighbor(
			X=X,
			labels=clustered_labels,
			label_texts=unique_labels,
			cluster_canonicals=cluster_canonicals,
			canonical_emb_map=canonical_emb_map,
			threshold=merge_canonical_clusters_threshold,
			min_centroid_similarity=merge_canonical_min_centroid_sim,
			max_merged_size=merge_max_size,
			report_path=os.path.splitext(clusters_fname)[0] + "_canonical_cluster_merges.json",
			verbose=verbose,
		)
		df["cluster"] = clustered_labels
		canonical_merges_done = n_clusters_pre_canon_merge - len(np.unique(clustered_labels))

	# STEP 7: MAP CANONICALS + INJECT VIRTUAL ROWS + CLEAN
	df, X_clean, removed_labels = map_inject_and_clean_clusters(
		df=df,
		X=X,
		cluster_canonicals=cluster_canonicals,
		canonical_emb_map=canonical_emb_map,
		model=model,
		batch_size=batch_size,
		encode_prompt=encode_prompt,
		low_cohesion_threshold=low_cohesion_threshold,
		poor_canonical_threshold=poor_canonical_threshold,
		clusters_fname=clusters_fname,
		verbose=verbose,
	)

	# cluster ids were re-indexed, so JSON and CSV ids no longer line up
	if removed_labels or canonical_merges_done > 0:
		reason = (
			"remove_problematic_clustered_labels removed clusters"
			if removed_labels
			else f"merge_cluster_canonical_nearest_neighbor merged {canonical_merges_done} cluster(s)"
		)
		if verbose:
			print(f"[CONSISTENCY] skipped pre-merge JSON vs post-merge CSV id check\n{reason}")
	else:
		_check_json_csv_consistency(
			df, 
			os.path.splitext(clusters_fname)[0] + "_canonical_selection.json",
			verbose=verbose,
		)

	if verbose:
		unique_labels_array = df["label"].values
		clustered_labels = df["cluster"].values
		
		# Quality metrics must use member-safe names, not display renames.
		canonical_map = df.groupby("cluster")["canonical_internal"].first().to_dict()

		print("-" * 100)
		print("[CLUSTER QUALITY]")
		print(f"  ├─ Updated clustered_labels {len(np.unique(clustered_labels))} unique clusters")
		print(f"  ├─ Updated canonical_map    {len(canonical_map)} mappings")
		print(f"  ├─ unique_labels_array      {type(unique_labels_array)} {unique_labels_array.shape}")
		print(f"  ├─ clustered_labels         {type(clustered_labels)} {clustered_labels.shape}")
		print(f"  ├─ label_freq_dict          {len(label_freq_dict)} labels with frequencies")
		print(f"  ├─ df reports               {df['cluster'].nunique()} clusters")
		print(f"  └─ {len(df)} labels -> {df['cluster'].nunique()} clusters")
	
		if df["cluster"].nunique() != len(np.unique(clustered_labels)):
			print("[WARNING] Mismatch detected! Analysis may be stale!")

		print(df.head(20))
		print(f"[TOTAL CLUSTERING ELAPSED TIME] {time.time()-st_t:.2f} sec")
		print("-" * 100)
	
	# sys.exit()

	return df

def get_canonical_labels(
	labels: List[List[str]],
	label_source: str,
	output_dir: str,
	model_id: str,
	batch_size: int,
	device: Union[str, torch.device],
	nc: Optional[int]=None,
	merge_near_duplicate_clusters_medoid: bool = True,
	merge_near_duplicate_clusters_canonical: bool = True,
	merge_close_clusters_threshold: Optional[float] = None,
	merge_canonical_clusters_threshold: Optional[float] = None,
	merge_canonical_min_centroid_sim: float = 0.70,
	encode_prompt: Optional[str] = None,
	verbose: bool = False,
) -> Tuple[List[List[str]], dict]:
	
	clusters_fname = os.path.join(output_dir, f"clustering_{label_source}.csv")
	
	if verbose:
		print("-" * 50)
		print("[CANONICALIZATION] Sequential Mapping")
		print(f"  ├─ {label_source}")
		print(f"  ├─ {model_id}")
		print(f"  ├─ Batch size  : {batch_size}")
		print(
			f"  ├─ labels      : {type(labels)} {len(labels)} "
			f"{type(labels[0])} {len(labels[0])} {labels[0]}"
		)
		print(f"  ├─ Output dir  : {output_dir}")
		print(f"  ├─ Clusters file: {clusters_fname}")
		print(
			f"  └─ ||Clusters||: {nc} "
			f"{'Manually defined' if nc else '=> Adaptive Search'}"
		)

	clustered_df = cluster(
		labels=labels,
		model_id=model_id,
		batch_size=batch_size,
		device=device,
		nc=nc,
		merge_near_duplicate_clusters_medoid=merge_near_duplicate_clusters_medoid,
		merge_near_duplicate_clusters_canonical=merge_near_duplicate_clusters_canonical,
		merge_close_clusters_threshold=merge_close_clusters_threshold,
		merge_canonical_clusters_threshold=merge_canonical_clusters_threshold,
		merge_canonical_min_centroid_sim=merge_canonical_min_centroid_sim,
		clusters_fname=clusters_fname,
		encode_prompt=encode_prompt,
		verbose=verbose,
	)

	# map from real rows only (exclude injected virtual hypernyms)
	real_rows = (
		clustered_df[~clustered_df["is_injected"]]
		if "is_injected" in clustered_df.columns
		else clustered_df
	)
	dup = real_rows["label"].duplicated()
	assert not dup.any(), f"duplicate real label rows: {real_rows.loc[dup, 'label'].head().tolist()}"
	canonical_map = real_rows.set_index("label")["canonical"].to_dict()

	lower_to_canonical = {k.lower(): v for k, v in canonical_map.items()}

	canonical_labels = list()
	missing_labels = set()

	for sample_labels in labels:
		if sample_labels is None:
			canonical_labels.append(None)
			continue
		
		if not isinstance(sample_labels, list):
			if isinstance(sample_labels, str):
				try:
					sample_labels = ast.literal_eval(sample_labels)
				except (ValueError, SyntaxError):
					canonical_labels.append(None)
					continue
			else:
				canonical_labels.append(None)
				continue
		
		mapped = list()
		for label in sample_labels:
			if label in canonical_map:
				mapped.append(canonical_map[label])
			elif label.lower() in lower_to_canonical:
				mapped.append(lower_to_canonical[label.lower()])
			else:
				missing_labels.add(label)
		
		canonical_labels.append(list(dict.fromkeys(mapped)))

	if verbose and missing_labels:
		print(f"[{label_source.upper()}]")
		print(f"  ├─ {len(missing_labels)} labels removed (not in canonical map)")
		print(f"  └─ {list(missing_labels)[:10]}")
		print("-"*100)

	# sys.exit()

	return canonical_labels, canonical_map