# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

from utils import *


import hashlib
import numpy as np

CUSTOM = "Instruct: Given a short label describing a historical photograph, retrieve labels that name the same concept\nQuery:"

POS = [("shell-shock", "shell shock"), ("reconnaissance aircraft", "reconnaissance plane"),
       ("observation airplane", "observation plane"), ("outhouse", "latrine"), ("Bf 109", "Bf109"),
       ("icebreaker", "ice breaker"), ("sea burial", "burial at sea"), ("railway station", "train station"),
       ("clergyman", "clergy"), ("counter-attack", "counterattack")]
NEG = [("shell-shock", "shell deflector"), ("battle tank", "fuel tank"), ("Ki-46", "Ki-61"),
       ("hospital", "hospital ship"), ("C-47", "C-46"), ("patrol bomber", "reconnaissance aircraft"),
       ("AWACS aircraft", "reconnaissance aircraft"), ("cap", "cape"), ("camp", "camera"),
       ("hospital ward", "hospital"), ("shell shock", "shock absorber"), ("Sherman tank", "M3 tank")]

def screen(model, variants):
    texts = sorted({t for p in POS + NEG for t in p})
    for name, prompt in variants.items():
        kw = {} if prompt is None else {"prompt": prompt}
        E = dict(zip(texts, model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, **kw)))
        pos = np.array([float(E[a] @ E[b]) for a, b in POS])
        neg = np.array([float(E[a] @ E[b]) for a, b in NEG])
        auc = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
        print(f"{name:38} positives {pos.mean():.3f} | negatives {neg.mean():.3f} | AUC {auc:.3f} | "
              f"positives below the best negative: {(pos < neg.max()).sum()}/{len(pos)}")

# ---- run with the real model:
# screen(model, {"no prompt (current)": None,
#                "query prompt on all labels": model.prompts["query"],
#                "custom similarity instruction on all": CUSTOM})

if __name__ == "__main__":                                     # stand-in encoder to check the code path
    class Mock:
        prompts = {"query": "Instruct: q\nQuery:"}
        def encode(self, texts, prompt=None, normalize_embeddings=True, convert_to_numpy=True):
            out = []
            for t in texts:
                s = f"  {(prompt or '') + t.lower()}  "; v = np.zeros(256)
                for i in range(len(s) - 2): v[int(hashlib.md5(s[i:i+3].encode()).hexdigest(), 16) % 256] += 1
                out.append(v / np.linalg.norm(v))
            return np.array(out)
    m = Mock()
    screen(m, {"no prompt (current)": None, "query prompt on all labels": m.prompts["query"], "custom": CUSTOM})

# # Load the model
# # model_id = "Qwen/Qwen3-Embedding-0.6B" # local
# model_id = "Qwen/Qwen3-Embedding-8B" # HPC

# model = SentenceTransformer(
# 	model_name_or_path=model_id,
# 	model_kwargs={"attn_implementation": "flash_attention_2"}, # no device_map
# 	trust_remote_code=True,
# 	device="cuda:0" if torch.cuda.is_available() else "cpu",
# 	cache_folder=cache_directory[os.getenv('USER')],
# 	token=os.getenv("HUGGINGFACE_TOKEN"),
# 	processor_kwargs={"padding_side": "left"}, # renamed from tokenizer_kwargs
# )
# print(model.default_prompt_name)
# print(model.prompts)

# print(repr(model.prompts["query"]))
# print(repr(model.prompts["document"]))

# queries = [
# 	"shell-shock",
# ]
# documents = [
# 	'shell deflector', 
# 	'shell-shock treatment',
# 	'shell shock', 
# 	'shell shocked Marine', 
# 	'shell shocked soldiers',
# ]

# # without specifying prompt_name, Sentence Transformers will not automatically apply "query". 
# # It will encode the texts with no prompt.
# query_embeddings = model.encode(
# 	queries,
# )
# document_embeddings = model.encode(documents)

# # Compute the (cosine) similarity between the query and document embeddings
# similarity = model.similarity(query_embeddings, document_embeddings)
# print(type(similarity), similarity.shape, torch.sum(similarity))
# print(similarity)
# best_idx = torch.argmax(similarity) 
# print(best_idx, documents[best_idx])

# # 1. Correct Setup (Query gets 'query' prompt, Documents get default/no prompt)
# query_emb_correct = model.encode(queries, prompt_name="query")
# doc_emb = model.encode(documents)
# similarity_correct = model.similarity(query_emb_correct, doc_emb)

# # 2. Incorrect Setup (Query mistakenly gets 'document' prompt)
# query_emb_incorrect = model.encode(queries, prompt_name="document")
# similarity_incorrect = model.similarity(query_emb_incorrect, doc_emb)

# # Compare the outputs
# print("--- Correct (Query prompt) ---")
# for doc, score in zip(documents, similarity_correct[0].tolist()):
# 		print(f"{score:.4f} : {doc}")

# print("\n--- Incorrect (Document prompt used on query) ---")
# for doc, score in zip(documents, similarity_incorrect[0].tolist()):
# 		print(f"{score:.4f} : {doc}")