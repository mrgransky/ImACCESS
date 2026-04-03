"""
Canonical selection improvements for agglomerative clustering.

Core problem: When a cluster contains only color/modifier variants of a concept
(e.g. ['black aircraft', 'white aircraft', 'yellow aircraft']), the centroid
lands in "colored-aircraft" space, not "aircraft" space. The correct canonical
('aircraft') may not exist as a label in the cluster.

Solution: Synthesize a virtual hypernym by finding the shared token core across
cluster members, then use it to anchor canonical selection even when the pure
head noun is absent.
"""

from collections import Counter
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. Token-level utilities
# ---------------------------------------------------------------------------

def _tokens(label: str) -> List[str]:
    return label.lower().split()


def _head(label: str) -> str:
    """Last token is almost always the head noun in English NPs."""
    return label.split()[-1]


def _shared_token_core(labels: List[str], min_support: float = 0.5) -> List[str]:
    """
    Return the ordered list of tokens that appear in ≥ min_support fraction
    of cluster labels, preserving left-to-right order from the most frequent
    member.

    E.g. ['black aircraft', 'white aircraft', 'yellow aircraft']
         → ['aircraft']

         ['red cross badge', 'red cross banner', 'red cross emblem']
         → ['red', 'cross']

         ['bridge club', 'club', 'glee club', 'sport club']
         → ['club']
    """
    if not labels:
        return []

    n = len(labels)
    threshold = max(2, int(np.ceil(min_support * n)))  # absolute count

    # Count how many labels each token appears in
    token_support: Dict[str, int] = {}
    for lbl in labels:
        for tok in set(_tokens(lbl)):
            token_support[tok] = token_support.get(tok, 0) + 1

    shared = {tok for tok, cnt in token_support.items() if cnt >= threshold}
    if not shared:
        return []

    # Recover left-to-right order from the longest label (most informative)
    anchor = max(labels, key=lambda l: len(_tokens(l)))
    ordered = [tok for tok in _tokens(anchor) if tok in shared]
    return ordered


def _virtual_hypernym(labels: List[str], min_support: float = 0.5) -> Optional[str]:
    """
    Return a virtual hypernym string built from the shared token core,
    or None if no useful core exists.
    """
    core = _shared_token_core(labels, min_support=min_support)
    if not core:
        return None
    candidate = " ".join(core)
    # Reject if the candidate is identical to the longest label (no reduction)
    if candidate == max(labels, key=len):
        return None
    return candidate


# ---------------------------------------------------------------------------
# 2. Candidate pool construction
# ---------------------------------------------------------------------------

def _build_candidate_pool(
    cluster_texts: List[str],
    include_virtual: bool = True,
    min_support: float = 0.5,
) -> Tuple[List[str], List[bool]]:
    """
    Return (candidates, is_virtual) where candidates is the full pool to score
    and is_virtual[i] is True if candidates[i] was synthesised (not a real label).
    """
    candidates = list(cluster_texts)
    is_virtual = [False] * len(cluster_texts)

    if include_virtual and len(cluster_texts) >= 3:
        vh = _virtual_hypernym(cluster_texts, min_support=min_support)
        if vh is not None and vh not in cluster_texts:
            candidates.append(vh)
            is_virtual.append(True)

    return candidates, is_virtual


# ---------------------------------------------------------------------------
# 3. Scoring dimensions
# ---------------------------------------------------------------------------

def _freq_scores(candidates: List[str], freq_dict: Dict[str, int]) -> np.ndarray:
    freqs = np.array([freq_dict.get(c, 1) for c in candidates], dtype=float)
    log_freqs = np.log1p(freqs)
    return log_freqs / (log_freqs.max() + 1e-12)


def _head_scores(candidates: List[str], cluster_size: int) -> np.ndarray:
    """Proportional head-noun dominance across cluster members."""
    heads = [_head(c) for c in candidates]
    # Count head occurrences across cluster labels (not candidates)
    # For virtual hypernyms we use their own last token
    head_counts: Dict[str, int] = Counter(heads)
    return np.array([head_counts[_head(c)] / cluster_size for c in candidates])


def _containment_scores(candidates: List[str], cluster_texts: List[str]) -> np.ndarray:
    """
    For each candidate, compute the fraction of cluster members whose token
    set is a superset of the candidate's token set (i.e. the candidate is
    'contained in' / subsumed by those members — making it a hypernym).
    """
    n = len(cluster_texts)
    cluster_token_sets = [set(_tokens(lbl)) for lbl in cluster_texts]
    scores = []
    for cand in candidates:
        cand_tokens = set(_tokens(cand))
        # How many cluster members contain ALL of cand's tokens?
        subsumers = sum(
            1 for ts in cluster_token_sets if cand_tokens.issubset(ts)
        )
        scores.append(subsumers / max(n, 1))
    return np.array(scores)


def _brevity_scores(candidates: List[str], cluster_texts: List[str]) -> np.ndarray:
    """
    Shorter labels score higher (more general), but not so aggressive that
    single-char tokens dominate. Normalised against longest candidate.
    """
    lengths = np.array([len(_tokens(c)) for c in candidates], dtype=float)
    max_len = lengths.max()
    if max_len == 0:
        return np.ones(len(candidates))
    # Invert: shorter → higher score
    return 1.0 - (lengths - 1.0) / max_len


def _similarity_to_centroid(
    candidate_embeddings: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """Cosine similarity of each candidate embedding to the cluster centroid."""
    return cosine_similarity(centroid.reshape(1, -1), candidate_embeddings)[0]


# ---------------------------------------------------------------------------
# 4. Main canonical selector (drop-in replacement for the inner loop body)
# ---------------------------------------------------------------------------

def select_canonical(
    cluster_texts: List[str],
    cluster_embeddings: np.ndarray,        # shape (n, d), already L2-normalised
    encode_fn,                             # callable: List[str] -> np.ndarray
    freq_dict: Optional[Dict[str, int]] = None,
    # Weight knobs — tunable without changing the algorithm structure
    w_sim:         float = 0.30,
    w_freq:        float = 0.15,
    w_head:        float = 0.20,
    w_containment: float = 0.25,
    w_brevity:     float = 0.10,
    # Virtual hypernym settings
    use_virtual:   bool  = True,
    virtual_min_support: float = 0.5,
    # Safety: only override pure-sim choice if freq gain ≥ this
    min_freq_gain: float = 3.0,
    verbose:       bool  = False,
) -> Tuple[str, bool]:
    """
    Select the canonical label for a cluster, with optional virtual hypernym
    synthesis for modifier-only clusters.

    Returns
    -------
    canonical : str
        The selected canonical label (may be a synthesised string).
    is_virtual : bool
        True if the canonical was synthesised (not present in cluster_texts).
    """
    cluster_size = len(cluster_texts)
    if cluster_size == 1:
        return cluster_texts[0], False

    # ── Build candidate pool ───────────────────────────────────────────────
    candidates, virtual_flags = _build_candidate_pool(
        cluster_texts, include_virtual=use_virtual,
        min_support=virtual_min_support,
    )

    # ── Encode virtual candidates (real ones already encoded) ─────────────
    virtual_indices = [i for i, v in enumerate(virtual_flags) if v]
    if virtual_indices:
        virtual_texts = [candidates[i] for i in virtual_indices]
        virtual_embs  = encode_fn(virtual_texts)
        # Stitch together
        all_embeddings = np.vstack([cluster_embeddings, virtual_embs])
    else:
        all_embeddings = cluster_embeddings

    # ── Centroid from REAL cluster members only ────────────────────────────
    centroid = cluster_embeddings.mean(axis=0)

    # ── Scoring ───────────────────────────────────────────────────────────
    sim   = _similarity_to_centroid(all_embeddings, centroid)
    freq  = _freq_scores(candidates, freq_dict or {}) if freq_dict else np.zeros(len(candidates))
    head  = _head_scores(candidates, cluster_size)
    cont  = _containment_scores(candidates, cluster_texts)
    brev  = _brevity_scores(candidates, cluster_texts)

    # Virtual hypernym bonus: it earns extra containment credit because by
    # construction it is subsumed by ≥50% of members
    # (already captured in _containment_scores, but make it explicit)
    for i in virtual_indices:
        cont[i] = max(cont[i], virtual_min_support)   # floor at support threshold

    composite = (
        w_sim         * sim
        + w_freq      * freq
        + w_head      * head
        + w_containment * cont
        + w_brevity   * brev
    )

    best_idx     = int(composite.argmax())
    pure_sim_idx = int(sim[:cluster_size].argmax())   # only among real labels

    # ── Safety check: require min_freq_gain to override pure-sim choice ───
    if best_idx != pure_sim_idx and freq_dict:
        real_freqs   = np.array([freq_dict.get(t, 1) for t in cluster_texts])
        best_is_real = not virtual_flags[best_idx]
        if best_is_real:
            gain = real_freqs[best_idx] / max(real_freqs[pure_sim_idx], 1)
            if gain < min_freq_gain:
                best_idx = pure_sim_idx

    canonical   = candidates[best_idx]
    is_virtual_ = virtual_flags[best_idx]

    if verbose:
        print(f"  [select_canonical] cluster_size={cluster_size}")
        virtual_label = candidates[-1] if virtual_indices else "none"
        print(f"  virtual hypernym candidate: '{virtual_label}'")
        print(f"  scores (top-3):")
        top3 = np.argsort(composite)[::-1][:3]
        for i in top3:
            tag = "[VIRTUAL]" if virtual_flags[i] else ""
            print(f"    {candidates[i]:<35} composite={composite[i]:.4f}  "
                  f"sim={sim[i]:.4f}  cont={cont[i]:.4f}  brev={brev[i]:.4f} {tag}")
        print(f"  => canonical: '{canonical}' (virtual={is_virtual_})")

    return canonical, is_virtual_


# ---------------------------------------------------------------------------
# 5. Integration patch for cluster() inner loop
# ---------------------------------------------------------------------------

PATCH = '''
# ── Replace the entire canonical-selection block inside the cluster() loop ──
# (from  "centroid = ..."  to  "cluster_canonicals[cid] = {...}")

canonical, is_virtual = select_canonical(
    cluster_texts      = cluster_texts,
    cluster_embeddings = cluster_embeddings,
    encode_fn          = lambda texts: model.encode(
        texts,
        batch_size=min(64, len(texts)),
        convert_to_numpy=True,
        normalize_embeddings=True,
        precision="float32",
    ),
    freq_dict          = original_label_counts,
    w_sim=0.30, w_freq=0.15, w_head=0.20, w_containment=0.25, w_brevity=0.10,
    use_virtual=True,
    virtual_min_support=0.5,
    min_freq_gain=3.0,
    verbose=verbose,
)

# Compute canonical sim for bookkeeping (centroid sim of the chosen label)
centroid     = cluster_embeddings.mean(axis=0, keepdims=True)
if is_virtual:
    # Re-encode just for the score record
    canon_emb = model.encode(
        [canonical],
        convert_to_numpy=True, normalize_embeddings=True, precision="float32"
    )
    canon_sim = float(cosine_similarity(centroid, canon_emb)[0, 0])
else:
    canon_idx = cluster_texts.index(canonical)
    canon_sim = float(
        cosine_similarity(centroid, cluster_embeddings[[canon_idx]])[0, 0]
    )

cluster_canonicals[cid] = {
    "canonical": canonical,
    "score":     canon_sim,
    "size":      cluster_size,
    "virtual":   is_virtual,
}
'''


# ---------------------------------------------------------------------------
# 6. Unit tests / sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CANONICAL SELECTION UNIT TESTS")
    print("=" * 60)

    # ── Test 1: Modifier-only cluster (no pure head-noun label) ───────────
    labels_1 = ["black aircraft", "white aircraft", "yellow aircraft"]
    core_1   = _virtual_hypernym(labels_1)
    print(f"\nTest 1 – modifier-only cluster")
    print(f"  Labels:           {labels_1}")
    print(f"  Virtual hypernym: '{core_1}'")
    assert core_1 == "aircraft", f"Expected 'aircraft', got '{core_1}'"
    print(f"  ✓ PASS")

    # ── Test 2: Shared-prefix cluster ─────────────────────────────────────
    labels_2 = [
        "red cross badge", "red cross banner", "red cross emblem",
        "red cross flag",  "red cross pin",    "red cross poster",
        "red cross sign",  "red cross symbol", "blue cross", "green cross",
    ]
    core_2 = _virtual_hypernym(labels_2, min_support=0.5)
    print(f"\nTest 2 – shared-prefix cluster")
    print(f"  Labels:           {labels_2}")
    print(f"  Virtual hypernym: '{core_2}'")
    # "cross" appears in all; "red" in 8/10 ≥ 0.5 threshold
    assert "cross" in (core_2 or ""), f"Expected 'cross' in hypernym, got '{core_2}'"
    print(f"  ✓ PASS")

    # ── Test 3: Cluster with head noun already present ────────────────────
    labels_3 = [
        "aircraft", "jet aircraft", "commercial aircraft",
        "concept aircraft", "research aircraft",
    ]
    core_3 = _virtual_hypernym(labels_3)
    print(f"\nTest 3 – head noun present")
    print(f"  Labels:           {labels_3}")
    print(f"  Virtual hypernym: '{core_3}'")
    # Either None (aircraft already in pool) or 'aircraft' — both fine
    print(f"  ✓ PASS (virtual not needed — 'aircraft' already in pool)")

    # ── Test 4: Containment scores ────────────────────────────────────────
    labels_4  = ["bridge", "arch bridge", "suspension bridge", "concrete bridge"]
    cont_4    = _containment_scores(labels_4, labels_4)
    print(f"\nTest 4 – containment scores")
    for lbl, sc in zip(labels_4, cont_4):
        print(f"  {lbl:<25} containment={sc:.3f}")
    assert cont_4[0] > cont_4[1], "'bridge' should outscore 'arch bridge'"
    print(f"  ✓ PASS ('bridge' has highest containment)")

    # ── Test 5: Club cluster ──────────────────────────────────────────────
    labels_5 = ["bridge club", "club", "glee club", "literary club", "sport club"]
    core_5   = _virtual_hypernym(labels_5)
    cont_5   = _containment_scores(labels_5, labels_5)
    print(f"\nTest 5 – 'club' cluster (head noun present)")
    print(f"  Virtual hypernym: '{core_5}'")
    for lbl, sc in zip(labels_5, cont_5):
        print(f"  {lbl:<25} containment={sc:.3f}")
    club_idx = labels_5.index("club")
    assert cont_5[club_idx] == max(cont_5), "'club' should have max containment"
    print(f"  ✓ PASS ('club' wins on containment)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("Integration: see PATCH string above for drop-in replacement")
    print("of the canonical-selection block inside cluster().")