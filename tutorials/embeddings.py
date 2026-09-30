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

# Load the model
# model_id = "Qwen/Qwen3-Embedding-0.6B" # local
model_id = "Qwen/Qwen3-Embedding-8B" # HPC

model = SentenceTransformer(
	model_name_or_path=model_id,
	model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"}, # no device_map
	trust_remote_code=True,
	cache_folder=cache_directory[os.getenv('USER')],
	token=os.getenv("HUGGINGFACE_TOKEN"),
	processor_kwargs={"padding_side": "left"}, # renamed from tokenizer_kwargs
)
print(model.default_prompt_name)
print(model.prompts)

print(repr(model.prompts["query"]))
print(repr(model.prompts["document"]))

queries = [
	"shell-shock",
]
documents = [
	'shell deflector', 
	'shell-shock treatment',
	'shell shock', 
	'shell shocked Marine', 
	'shell shocked soldiers',
]

# without specifying prompt_name, Sentence Transformers will not automatically apply "query". 
# It will encode the texts with no prompt.
query_embeddings = model.encode(
	queries,
)
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(type(similarity), similarity.shape, torch.sum(similarity))
print(similarity)
best_idx = torch.argmax(similarity) 
print(best_idx, documents[best_idx])

# 1. Correct Setup (Query gets 'query' prompt, Documents get default/no prompt)
query_emb_correct = model.encode(queries, prompt_name="query")
doc_emb = model.encode(documents)
similarity_correct = model.similarity(query_emb_correct, doc_emb)

# 2. Incorrect Setup (Query mistakenly gets 'document' prompt)
query_emb_incorrect = model.encode(queries, prompt_name="document")
similarity_incorrect = model.similarity(query_emb_incorrect, doc_emb)

# Compare the outputs
print("--- Correct (Query prompt) ---")
for doc, score in zip(documents, similarity_correct[0].tolist()):
		print(f"{score:.4f} : {doc}")

print("\n--- Incorrect (Document prompt used on query) ---")
for doc, score in zip(documents, similarity_incorrect[0].tolist()):
		print(f"{score:.4f} : {doc}")