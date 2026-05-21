# Supervision Transparency Framework 

### Core Contributions:
* **A**: Supervision Auditability Pipeline: produces per-label, per-sample supervision provenance in multi-label retrieval — tracing each label to its modality source, conflict regime, and CGD quality score.
* **B**: Modality Conflict Detection: characterizes cross-modal conflict as a structural property of image-text data and systematically predicts supervision quality degradation.
* **C**: Regime-Conditioned Label Quality: Conflict regime predicts supervision quality.

**(B $\rightarrow$ A $\rightarrow$ C):** Modality conflict is a structural dataset property (**B**). We build an auditable pipeline to detect and route it per label (**A**). We use these audits to dynamically condition downstream loss, safely integrating soft conflicts and repelling hallucinations (**C**).

To ensure computational tractability across N samples ($N >> 10**5$), we formalize our framework as a *two-phase* architecture. 
* Phase 1 performs stateless modality extraction and conflict quantification. 
* Phase 2 injects globally calibrated priors to perform stateful, per-label CGD auditing and consolidation.

This decoupling guarantees that our density metrics reflect true corpus-level statistics rather than localized batch artifacts.

## PHASE 1: Stateless Map (GPU-Heavy / Per-Sample)

### **Stage 1: Joint VLM Extraction with CoT Attribution**
*   **Goal:** Extract distinct semantic concepts while forcing the VLM to explicitly attribute their modality source.
*   **Mechanism:** A highly structured System Prompt fed to a Joint VLM (Image + Caption). It enforces strict domain rules (Anti-Scope-Creep, Proper Noun Ban) and forces the VLM to categorize outputs.
*   **Output (JSON):** Three distinct lists of raw open-vocabulary concepts: `C_text` (Coverage), `C_vis` (Grounding), and `C_fused` (Density resolution). *Crucial constraint: outputs `[]` for `C_fused` if modalities are completely disjoint.*

### **Stage 2: Modality Conflict Quantification \& Routing**
*   **Goal:** Mathematically quantify cross-modal dissonance and route the sample into a "Conflict Regime."
*   **Mechanism:** 
    1.  **Symmetric Audit (Cosine):** Uses `all-MiniLM` to find semantic overlap between `C_text` and `C_vis`. Identifies unverified *Orphans* ($O_{text}, O_{vis}$).
    2.  **Asymmetric Audit (NLI):** Uses `DeBERTa-NLI` cross-encoder to compute directional entailment. 
        * Computes **Asymmetry Gap ($\Delta_{density}$)** to prove which modality is denser (Hyponym) vs broader (Hypernym).
    3.  **Router:** Deterministically assigns the sample to:
        *   *Agreement:* High overlap, low orphans.
        *   *Soft Conflict:* Topic matches, but high $\Delta_{density}$ (abstraction mismatch).
        *   *Hard Conflict:* Disjoint modalities (high orphans or VLM `[]` short-circuit).
*   **Output:** mathematically auditable `Evidence_Receipt` JSON per sample.

---
## BRIDGE: Global Aggregation (CPU / Dataset-Level)
*   **Goal:** Target Canonical Vocabulary ($V$) and compute global dataset statistics.
*   **Mechanism:** Collect all raw concepts from Stage 1. 
    * Existing `clustering.py` engine (Agglomerative Linkage, Virtual Hypernym Synthesis, 5-Signal Canonical Assignment). 
*   **Output:** A universal `canonical_map.json`, a pre-computed `emb_cache.pt`, and global corpus frequencies for every concept.
---

## PHASE 2: Stateful Map (Fast Vector Math / Per-Sample)

### **Stage 3: The Micro-CGD Audit**
*   **Goal:** Assign a continuous quality score $[0,1]$ to every raw concept proposed in Stage 1 based on its Stage 2 receipt and Global Aggregation frequencies.
*   **Mechanism:**
    *   **Coverage $C(c)$:** How strongly the concept is supported by the historical text.
    *   **Grounding $G(c)$:** How strongly the concept is supported by the visual pixels.
    *   **Density $D(c)$:** Computed as $D_{global}$ (Corpus reusability) $\times$ $D_{local}$ (NLI Penalty). 
        *   *If NLI proved a concept is an overly broad hypernym in a Soft Conflict, its Density score is penalized.*
*   **Output:** The raw concept list, enriched with exact $C, G, D$ float scores.

### **Stage 4: Regime-Aware Consolidation \& Weight Derivation**
*   **Goal:** Map raw concepts to canonical vocabulary $V$, gate them using the Conflict Regime, and derive gradient scaling weights ($\omega_{pos}, \omega_{neg}$) for training.
*   **Mechanism:**
    *   **Agreement:** Union mapping. `w_pos = 1.0`, `w_neg = 0.0`.
    *   **Soft Conflict:** Map to $V$, but explicitly drop broad concepts failing the $D(c)$ audit. `w_pos = 1.0 - |\Delta_{density}|`, `w_neg = 0.0`.
    *   **Hard Conflict:** Block all text concepts. Map visual concepts to $V$ as positive targets (`w_pos = 0.3`). Map orphaned text concepts to $V$ as *Hard Negatives* (`w_neg = 1.0 - G(c)`).
*   **Output:** The `auditable_supervision_matrix.parquet`. 
    *(Schema: `sample_id | positive_targets | hn_targets | w_pos | w_neg | regime`)*

---
## DOWNSTREAM: Representation Learning (Fine-Tuning using PEFT)

### **Stage 5: Regime-Conditioned Dual-Encoder Training**
*   **Goal:** Train a VLR model (e.g., CLIP ViT-L/14) that safely learns from the long-tail, respects soft conflicts, and actively unlearns archival hallucinations.
*   **Mechanism:** Modulate a standard `BCEWithLogitsLoss` using two orthogonal axes:
    1.  **Class-Level Balance:** Your existing `pos_weight` mask to handle long-tail rare classes.
    2.  **Sample-Level Regime Weights:** 
        *   Multiply the positive target loss by **$\omega_{pos}$** (throttling learning from rescued/noisy labels).
        *   Multiply a repulsion loss by **$\omega_{neg}$** to explicitly push the image embedding *away* from hallucinated `hn_targets`.
*   **Output:** A fully trained, noise-resilient multimodal retrieval model achieving SOTA on the HISTORY-X4 benchmark. 