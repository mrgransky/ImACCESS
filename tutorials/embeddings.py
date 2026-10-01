# import os
# import sys
# HOME, USER = os.getenv('HOME'), os.getenv('USER')
# IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

# CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
# sys.path.insert(0, CLIP_DIR)

# MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
# sys.path.insert(0, MISC_DIR)

# from utils import *

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


"""
Does the prompt used to encode labels change how Qwen3-Embedding separates concepts?

Three parts, all printed in one run:
  1. Prompt inspection      what the model's config says (default prompt, "query"/"document").
  2. Shell-shock example    your original experiment, plus the two SYMMETRIC encodings that
                            clustering would actually use (every label gets the same prompt).
  3. Pair screening         10 same-concept pairs and 12 hard negatives taken from the canonical
                            selection findings, scored under three symmetric encodings.

How to read part 3: a better encoding gives a higher AUC (positives score above negatives) and
fewer positives that fall below the best negative. Compare AUC and ranks, NOT raw cosines: adding
a long shared instruction shifts every cosine upwards.

Screening only. 22 pairs cannot prove anything on their own; if one encoding clearly wins, the
36-minute Ward linkage is worth re-running with it (remember: every encode() call in clustering.py,
the cache key, and every calibrated similarity threshold would have to change with it).
"""
import os
import sys
import time
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

from utils import *
import numpy as np

# model_id = "Qwen/Qwen3-Embedding-8B" # HPC
# if USER == "farid":
# 	# model_id = "Qwen/Qwen3-Embedding-0.6B"
# 	# model_id = "Octen/Octen-Embedding-0.6B"
# 	model_id = "nvidia/Nemotron-3-Embed-1B-BF16"

models_ids =[
	"Qwen/Qwen3-Embedding-8B",
	"Octen/Octen-Embedding-8B",
	"nvidia/Nemotron-3-Embed-8B-BF16",
]
if USER == "farid":
	models_ids =[
		"Qwen/Qwen3-Embedding-0.6B",
		"Octen/Octen-Embedding-0.6B",
		"nvidia/Nemotron-3-Embed-1B-BF16",
	]



# Task instruction in the same "Instruct: ...\nQuery:" format the model was trained with.
# The text is appended directly after "Query:" (no space), exactly like the built-in "query" prompt.
# Output dimension of each real model; used to catch a stand-in/mock encoder

CUSTOM_INSTRUCTION = (
	"Instruct: Given a label describing a historical photograph, "
	"retrieve labels that name the same concept\nQuery:"
)

# Should look alike (same concept, different surface form)
POS_PAIRS = [
	("shell-shock", "shell shock"),
	("reconnaissance aircraft", "reconnaissance plane"),
	("observation airplane", "observation plane"),
	("outhouse", "latrine"),
	("Bf 109", "Bf109"),
	("Messerschmitt Bf 109", "Bf 109"),
	("Messerschmitt Me 262", "Schwalbe"),
	("icebreaker", "ice breaker"),
	("sea burial", "burial at sea"),
	("railway station", "train station"),
	("clergyman", "clergy"),
	("counter-attack", "counterattack"),
	('Grey', 'Gray'),
	('Jagdgeschwader 53', 'Pik As'),
	('Nuuanu Pali', 'Pali'),
]
# Should look different (share words or spelling, different concept)
NEG_PAIRS = [
	("shell-shock", "shell deflector"),
	("battle tank", "fuel tank"),
	("armored tank", "Storage tank"),
	("Ki-46", "Ki-61"),
	("hospital", "hospital ship"),
	("C-47", "C-46"),
	("patrol bomber", "reconnaissance aircraft"),
	("AWACS aircraft", "reconnaissance aircraft"),
	("cap", "cape"),
	("camp", "camera"),
	("hospital ward", "hospital"),
	("shell shock", "shock absorber"),
	("Sherman tank", "M3 tank"),
	('pikes', 'Pik As'),
]


def encode(model, texts, prompt=None, prompt_name=None):
	"""L2-normalised float32 embeddings. prompt=None -> no prompt (what clustering.py does today)."""
	kw = {}
	if prompt is not None:
		kw["prompt"] = prompt
	if prompt_name is not None:
		kw["prompt_name"] = prompt_name
	return np.asarray(
		model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, **kw),
		dtype=np.float32,
	)

def cos(a, b) -> float:
	return float(np.dot(a, b))  # embeddings are L2-normalised

# ── 1. PROMPT INSPECTION ─────────────────────────────────────────────────────────────────────
def inspect_prompts(model):
	print("=" * 100)
	print("1. PROMPT INSPECTION")
	print("=" * 100)
	if hasattr(model, "get_sentence_embedding_dimension"):
		dim = model.get_sentence_embedding_dimension()
	elif hasattr(model, "get_embedding_dimension"):
		dim = model.get_embedding_dimension()
	else:
		dim = None
	# dim = (
	# 	model.get_sentence_embedding_dimension() 
	# 	if hasattr(model, "get_sentence_embedding_dimension") 
	# 	else "?"
	# )
	model_id = model.model_card_data.base_model 
	print(f"encoder: {type(model).__name__} | model_id: {model_id} | embedding dim: {dim}")

	print(f"default_prompt_name: {model.default_prompt_name}")
	print(f"prompts: {model.prompts}")
	print(f"query prompt   : {model.prompts['query']!r}")
	print(f"document prompt: {model.prompts['document']!r}")

	probe = ["shell-shock", "shell deflector"]
	e_none = encode(model, probe)
	e_doc = encode(model, probe, prompt_name="document")
	diff = float(np.abs(e_none - e_doc).max())
	verdict = "identical: no prompt == the empty 'document' prompt" if diff < 1e-3 else "DIFFERENT: check the model config"
	print(f"encode() without a prompt vs prompt_name='document': max |diff| = {diff:.2e} -> {verdict}")


# ── 2. SHELL-SHOCK EXAMPLE ───────────────────────────────────────────────────────────────────
def shell_shock_demo(model):
	print("\n" + "=" * 100)
	print("2. SHELL-SHOCK EXAMPLE: cosine of the query against each document, under four encodings")
	print("=" * 100)
	query = "shell-shock"
	docs = ["shell deflector", "shell-shock treatment", "shell shock", "shell shocked Marine", "shell shocked soldiers"]
	q_prompt = model.prompts["query"]

	q_none, d_none = encode(model, [query])[0], encode(model, docs)
	q_qry, d_qry = encode(model, [query], q_prompt)[0], encode(model, docs, q_prompt)
	q_cus, d_cus = encode(model, [query], CUSTOM_INSTRUCTION)[0], encode(model, docs, CUSTOM_INSTRUCTION)

	columns = {
		"no prompt (current)": (q_none, d_none),
		"query prompt on query only": (q_qry, d_none),   # your original 'Correct' setup, NOT what clustering does
		"query prompt on ALL": (q_qry, d_qry),
		"custom instruction on ALL": (q_cus, d_cus),
	}
	scores = {name: [cos(q, d) for d in dv] for name, (q, dv) in columns.items()}

	print(f"query: {query!r}   (* = highest score in the column)")
	print(f"  {'document':<26}" + "".join(f"{name[:30]:>34}" for name in columns))
	for i, doc in enumerate(docs):
		row = f"  {doc:<26}"
		for name in columns:
			s = scores[name]
			mark = "*" if s[i] == max(s) else " "
			row += f"{s[i]:>32.4f} {mark}"
		print(row)
	foil = docs.index("shell deflector")
	print("\n  rank of the foil 'shell deflector' among the 5 documents (5 = last = what we want):")
	for name in columns:
		s = scores[name]
		rank = 1 + sum(1 for x in s if x > s[foil])
		print(f"    {name:<34} #{rank}")


# ── 3. PAIR SCREENING ────────────────────────────────────────────────────────────────────────
def pair_screen(model):
	print("\n" + "=" * 100)
	print("3. PAIR SCREENING: every label gets the SAME prompt (this is the clustering setting)")
	print("=" * 100)
	variants = {
		"no prompt (current)": None,
		"query prompt on ALL": model.prompts["query"],
		"custom instr. on ALL": CUSTOM_INSTRUCTION,
	}
	texts = sorted({t for p in POS_PAIRS + NEG_PAIRS for t in p})
	pos_s, neg_s = {}, {}
	for name, prompt in variants.items():
		E = dict(zip(texts, encode(model, texts, prompt)))
		pos_s[name] = np.array([cos(E[a], E[b]) for a, b in POS_PAIRS])
		neg_s[name] = np.array([cos(E[a], E[b]) for a, b in NEG_PAIRS])

	print(f"  {'':2}{'pair':<52}" + "".join(f"{n:>24}" for n in variants))
	for kind, pairs, store in (("+", POS_PAIRS, pos_s), ("-", NEG_PAIRS, neg_s)):
		for i, (a, b) in enumerate(pairs):
			print(f"  {kind:<2}{(a + ' | ' + b):<52}" + "".join(f"{store[n][i]:>24.4f}" for n in variants))
		print()

	print("  SUMMARY (+ should score above -)")
	print(f"  {'encoding':<24}{'mean +':>9}{'mean -':>9}{'AUC':>8}   positives below the best negative")
	for name in variants:
		p, n = pos_s[name], neg_s[name]
		auc = float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
		below = int((p < n.max()).sum())
		print(f"  {name:<24}{p.mean():>9.3f}{n.mean():>9.3f}{auc:>8.3f}   {below}/{len(p)}")
	print("\n  Compare AUC and the last column across encodings, not the raw cosines.")


def main():
	t0 = time.time()
	for model_id in models_ids:
		model = SentenceTransformer(
			model_name_or_path=model_id,
			model_kwargs={"attn_implementation": "flash_attention_2"}, # no device_map
			trust_remote_code=True,
			device="cuda:0" if torch.cuda.is_available() else "cpu",
			cache_folder=cache_directory[os.getenv('USER')],
			token=os.getenv("HUGGINGFACE_TOKEN"),
			processor_kwargs={"padding_side": "left"}, # renamed from tokenizer_kwargs
		)
		inspect_prompts(model)
		shell_shock_demo(model)
		pair_screen(model)
		print(f"\n[DONE: {model_id}] {time.time() - t0:.1f} sec")


if __name__ == "__main__":
	main()