# Supervision Transparency Framework

### Core Contributions:
* **A**: Supervision Auditability Pipeline: produces per-label, per-sample supervision provenance in multi-label retrieval — tracing each label to its modality source, conflict regime, and CGD quality score.
* **B**: Modality Conflict Detection: characterizes cross-modal conflict as a structural property of image-text data and systematically predicts supervision quality degradation.
* **C**: Regime-Conditioned Label Quality: Conflict regime predicts supervision quality.

**(B → A → C):** Modality conflict is a structural property of multimodal data (**B**). We build an auditable pipeline to detect and route it per label (**A**). We use these audits to dynamically condition downstream loss, safely integrating soft conflicts and repelling hallucinations (**C**).

To ensure computational tractability across $N$ samples ($N \gg 10^5$), we formalize our framework as a *two-phase* architecture.
* Phase 1 performs stateless modality extraction and conflict **quantification** ★ *(measurement only — no regime decision)*.
* Phase 2 injects globally calibrated priors to perform stateful, per-label CGD auditing and consolidation.

This decoupling guarantees that our density metrics reflect true corpus-level statistics rather than localized batch artifacts. ★ *It also ensures that regime boundaries are induced from the full corpus distribution rather than hand-tuned thresholds.*

---

## PHASE 1: Stateless Map (GPU-Heavy / Per-Sample)

### Stage 1: Joint VLM Extraction with CoT Attribution
*   **Goal:** Extract distinct semantic concepts while forcing the VLM to explicitly attribute their modality source.
*   **Mechanism:** A highly structured System Prompt fed to a Joint VLM (Image + Caption). It enforces strict domain rules (Anti-Scope-Creep, Proper Noun Ban) and forces the VLM to categorize outputs.
*   **Output (JSON):** Three distinct lists of raw open-vocabulary concepts: `C_text` (Coverage), `C_vis` (Grounding), and `C_fused` (Density resolution). *Crucial constraint: outputs `[]` for `C_fused` if modalities are completely disjoint.*

### Stage 2: Modality Conflict Quantification ★ *(Measurement Only — Routing Deferred)*
* **Goal:** Mathematically quantify cross-modal dissonance and emit a **continuous conflict feature vector** per sample. ★ *No regime label is assigned here.*
* **Mechanism:**
  1. **Symmetric Audit (Cosine):** Uses `all-MiniLM` to find semantic overlap between `C_text` and `C_vis`.
     * Identifies unverified Orphans ($O_{text}, O_{vis}$); computes `set_similarity` and `orphan_ratio`.
  2. **Asymmetric Audit (NLI):** Uses `DeBERTa-NLI` cross-encoder to compute directional entailment.
     * **Asymmetry Gap ($\Delta_{density}$)** proves which modality is denser (Hyponym) vs. broader (Hypernym).
     * ★ *NLI is bypassed when `orphan_ratio` ≥ $\tau_{orphan}$; $\Delta_i$ is set to `None` in this case.*
  3. ★ **Heuristic Regime Tag $\hat{\mathcal{R}}_i$:** A deterministic label (*Agreement / Soft / Hard*) is stored as an **immutable audit field only** — it is never used as a routing gate downstream.
* **Output:** ★ Continuous feature vector $\mathbf{f}_i = [\text{set\_sim}_i,\; \text{orphan\_ratio}_i,\; |\Delta_i|]$ (2D when $\Delta_i = \text{None}$) serialised inside the `Evidence_Receipt` JSON alongside all raw metrics.

### Stage 2: Modality Conflict Quantification ★ *(Measurement Only — Routing Deferred)*
* **Goal:** Mathematically quantify cross-modal dissonance and emit a **continuous conflict feature vector** per sample. ★ *No regime label is assigned here.*
* **Mechanism:**
  1. **Symmetric Audit (Cosine):** Uses `all-MiniLM` to find semantic overlap between `C\_text` and `C\_vis`.
     * Identifies unverified Orphans ($O\_{text}, O\_{vis}$); computes `set\_similarity` and `orphan\_ratio`.
  2. **Asymmetric Audit (NLI):** Uses `DeBERTa-NLI` cross-encoder to compute directional entailment.
     * **Asymmetry Gap ($\Delta\_{density}$)** proves which modality is denser (Hyponym) vs. broader (Hypernym).
     * ★ *NLI is bypassed when `orphan\_ratio` ≥ $\tau\_{orphan}$; $\Delta\_i$ is set to `None` in this case.*
  3. ★ **Heuristic Regime Tag $\hat{\mathcal{R}}\_i$:** A deterministic label (*Agreement / Soft / Hard*) is stored as an **immutable audit field only** — it is never used as a routing gate downstream.
* **Output:** ★ Continuous feature vector $\mathbf{f}\_i = [\text{set\_sim}\_i,\; \text{orphan\_ratio}\_i,\;  \Delta\_i|]$ (2D when $\Delta\_i = \text{None}$) serialised inside the `Evidence_Receipt` JSON alongside all raw metrics.

---

## BRIDGE: Global Aggregation (CPU / Dataset-Level) ★ *+ GMM Regime Induction*

* **Goal:** Discover the Target Canonical Vocabulary ($V$), compute global dataset statistics, ★ *and induce data-driven conflict regime boundaries from the full corpus.*
* **Mechanism:**
  * **Step 1A — Vocabulary Induction:** Collect all raw concepts from Stage 1. Existing `clustering.py` engine (Agglomerative Linkage, Virtual Hypernym Synthesis, 5-Signal Canonical Assignment).
  * ★ **Step 1B — GMM Regime Induction:** Collect $\mathbf{f}_i$ vectors for all $N$ samples. Fit `GaussianMixture(K*)` via BIC search over $K \in \{2,\ldots,6\}$ with `StandardScaler` normalisation. Assign cluster semantics by centroid geometry: *Agreement* (high sim, low orphan) · *Soft Conflict* · *Hard Conflict* (low sim, high orphan). ★ *2D fallback used when the fraction of 3D-eligible samples falls below coverage threshold $\theta_{3D}$, ensuring Hard Conflict samples are not systematically underrepresented.*
* **Output:**
  * `canonical_map.json`, `emb_cache.pt`, global corpus frequencies *(unchanged)*
  * ★ `_conflict_gmm.pkl` — `{gmm_model, scaler, class_mapping, feature_dim, bic_scores, centroids_unscaled}`

---

## PHASE 2: Stateful Map (Fast Vector Math / Per-Sample)

### Stage 3: The Micro-CGD Audit
*(unchanged — C, G, D scores computed per concept as before)*

### Stage 4: Regime-Aware Consolidation & Weight Derivation
* **Goal:** Map raw concepts to canonical vocabulary $V$, gate them using the Conflict Regime, and derive gradient scaling weights ($\omega_{pos}, \omega_{neg}$) for training.
* **Mechanism:**
  * ★ **GMM Routing (opens Stage 4):** Load `_conflict_gmm.pkl` → standardise $\mathbf{f}_i$ → compute posterior $p(\mathcal{R}_k \mid \tilde{\mathbf{f}}_i)$ → assign $\mathcal{R}_i = \arg\max_k\; p(\mathcal{R}_k \mid \tilde{\mathbf{f}}_i)$. Samples with confidence $< 0.60$ are flagged. Cases where $\mathcal{R}_i \neq \hat{\mathcal{R}}_i$ are recorded as `regime_override = True` for post-hoc analysis.
  * **Agreement:** Union mapping. $\omega_{pos} = 1.0,\; \omega_{neg} = 0.0$.
  * **Soft Conflict:** Map to $V$, drop concepts failing $D(c) \geq \tau_D$. ★ $\omega_{pos} = \max(\omega_{floor},\; 1 - \lambda \cdot |\Delta_i|)$ *(GMM-weighted asymmetry gap rather than hard threshold)*.
  * **Hard Conflict:** Block all text concepts. Map visual concepts to $V$ as positive targets ($\omega_{pos} = \omega_{hard}$). Map orphaned text concepts to $V$ as Hard Negatives ($\omega_{neg} = 1 - \overline{G(\mathcal{T}^-_i)}$).
* **Output:** `auditable_supervision_matrix.parquet`
  * ★ *Schema: `sample_id | positive_targets | hn_targets | w_pos | w_neg | regime | gmm{confidence, probabilities, regime_override}`*

---

## DOWNSTREAM: Representation Learning (Fine-Tuning using PEFT)

### Stage 5: Regime-Conditioned Dual-Encoder Training
*   **Goal:** Train a VLR model (e.g., CLIP ViT-L/14) that safely learns from the long-tail, respects soft conflicts, and actively unlearns archival hallucinations.
*   **Mechanism:** Modulate a standard `BCEWithLogitsLoss` using two orthogonal axes:
    1.  **Class-Level Balance:** Your existing `pos_weight` mask to handle long-tail rare classes.
    2.  **Sample-Level Regime Weights:** 
        *   Multiply the positive target loss by **$\omega_{pos}$** (throttling learning from rescued/noisy labels).
        *   Multiply a repulsion loss by **$\omega_{neg}$** to explicitly push the image embedding *away* from hallucinated `hn_targets`.
*   **Output:** A fully trained, noise-resilient multimodal retrieval model achieving SOTA on the HISTORY-X4 benchmark.