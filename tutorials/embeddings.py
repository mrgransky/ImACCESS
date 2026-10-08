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
		"google/embeddinggemma-2",
		"jinaai/jina-embeddings-v4",
		"jinaai/jina-embeddings-v5-text-small",
	]

# Task instruction in the same "Instruct: ...\nQuery:" format the model was trained with.
# The text is appended directly after "Query:" (no space), exactly like the built-in "query" prompt.
# Output dimension of each real model; used to catch a stand-in/mock encoder

CUSTOM_INSTRUCTION = (
	"Instruct: Given a label describing a historical photograph, "
	"retrieve labels that name the same concept\nQuery:"
)

# Synonyms, spelling variations, and highly related canonical labels that SHOULD cluster together.
POS_PAIRS = [
		# Originals
		("shell-shock", "shell shock"),
		("reconnaissance aircraft", "reconnaissance plane"),
		("observation airplane", "observation plane"),
		("outhouse", "latrine"),
		("Bf 109", "Bf109"),
		("Messerschmitt Bf 109", "Bf 109"),
		("icebreaker", "ice breaker"),
		("sea burial", "burial at sea"),
		("railway station", "train station"),
		("clergyman", "clergy"),
		("counter-attack", "counterattack"),
		('Grey', 'Gray'),
		
		# Aircraft & Vehicles
		("M4 Sherman", "Sherman tank"),
		("B-17G", "Flying Fortress"),
		("P-47 Thunderbolt", "Thunderbolt"),
		("P-51D Mustang", "Mustang"),
		("Higgins boat", "LCVP"),
		("hospital ship", "hospital boat"),
		("seaplane", "flying boat"),
		("U-Boat", "submarine"),
		('Landing Ship, Tank', 'LST'),
		
		# Equipment & Structures
		("field gun", "artillery piece"),
		("lighthouse", "Light Station"),
		("ruins", "rubble"),
		("gas mask", "respirator"),
		("barbed wire", "wire fence"),
		("spectacles", "glasses"),
		("casket", "wooden coffin"),
		("medal", "military medal"),
		("tarpaulin", "canvas cover"),
		('quagmire', 'military disaster'),
		('Corpo de Truppe Volontarie', 'CTV'),
		
		# People & Concepts
		("soldier", "troops"),
		("nurse", "Army Nurse"),
		("Artillery", "Field Artillery"),
		("Prisoners", "German prisoners"),
		('Prisoner of war', 'POW'),
]

# Hard negatives: Lexical overlap, same broad category, but fundamentally DIFFERENT concepts.
NEG_PAIRS = [
		# Originals
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
		
		# Aircraft & Vehicles (Different roles/generations)
		("B-17G", "B-24 Liberator"),
		("P-47 Thunderbolt", "P-51D Mustang"),
		("U-Boat", "destroyer"),
		("landing craft", "cargo ship"),
		("fighter plane", "bomber"),
		("glider", "parachute"),
		("aircraft carrier", "warship"),
		
		# Equipment & Weapons (Shared materials/functions)
		("machine gun", "rifle"),
		("gas mask", "helmet"),
		("artillery piece", "tank gun"),
		("barbed wire", "telegraph wire"),
		("searchlight", "lighthouse"),
		("torpedo", "depth charge"),
		("fuel truck", "fuel tank"),
		
		# Structures & Locations (Different infrastructure)
		("airfield", "shipyard"),
		("barracks", "hospital"),
		("trench", "foxhole"),
		("bridge", "dam"),
		("pillbox", "watchtower"),
		("factory", "power plant"),
		
		# People & Units (Different branches/roles)
		("infantry", "cavalry"),
		("officer", "sergeant"),
		("nurse", "medic"),
		("prisoner", "guard"),
		("pilot", "ground crew"),
		('CCA Control Board', 'CTV'),
]

def encode(
		model: Any,
		texts: Iterable[str],
		prompt: Optional[str] = None,
		prompt_name: Optional[str] = None,
		normalize: bool = True,
		**kwargs: Any,
) -> np.ndarray:
		"""
		Encode an iterable of texts into float32 NumPy embeddings.

		Encapsulates all necessary helpers:
		1. Resolves the `Normalize` class cleanly across sentence-transformers versions.
		2. Applies a one-time safety patch for rogue configuration keys (e.g., Octen).
		3. Recursively inspects the model pipeline to prevent double normalization.

		Parameters
		----------
		model : SentenceTransformer
				The instantiated SentenceTransformer model.
		texts : Iterable[str]
				Text sequences to encode.
		prompt : str, optional
				Explicit instruction prompt to prepend to each text.
		prompt_name : str, optional
				Prompt name pre-configured inside the model repository.
		normalize : bool, default=True
				Whether final embeddings should be L2-normalized. If the model
				architecture already contains an internal Normalize layer,
				`normalize_embeddings=True` is omitted to avoid duplicate operations.
		**kwargs : Any
				Additional arguments passed directly to `model.encode` (e.g.
				`batch_size`, `show_progress_bar`, `device`).

		Returns
		-------
		np.ndarray
				Float32 embeddings of shape (n_texts, embedding_dim).
		"""

		# -------------------------------------------------------------------------
		# Helper 1: Resilient import across sentence-transformers versions
		# -------------------------------------------------------------------------
		def _resolve_normalize_class():
				# >= 6.0 modern path
				try:
						from sentence_transformers.sentence_transformer.modules import Normalize
						return Normalize
				except ImportError:
						pass

				# intermediate / base path
				try:
						from sentence_transformers.base.modules import Normalize
						return Normalize
				except ImportError:
						pass

				# legacy fallback (< 6.0)
				from sentence_transformers.models import Normalize
				return Normalize

		# -------------------------------------------------------------------------
		# Helper 2: One-time patch for malformed config.json in repos like Octen
		# -------------------------------------------------------------------------
		def _ensure_normalize_patched(norm_cls):
				if not getattr(norm_cls, "_rogue_kwargs_patched", False):
						orig_init = norm_cls.__init__

						def patched_init(self, *args, **kw):
								# Discard invalid keyword arguments saved by rogue model configs
								kw.pop("normalize_embeddings", None)
								return orig_init(self, *args, **kw)

						norm_cls.__init__ = patched_init
						norm_cls._rogue_kwargs_patched = True

		# -------------------------------------------------------------------------
		# Helper 3: Recursive pipeline inspection
		# -------------------------------------------------------------------------
		def _has_internal_normalize(m, norm_cls) -> bool:
				if hasattr(m, "modules"):
						return any(isinstance(module, norm_cls) for module in m.modules())
				return False

		# -------------------------------------------------------------------------
		# Execution
		# -------------------------------------------------------------------------
		Normalize = _resolve_normalize_class()
		_ensure_normalize_patched(Normalize)

		encode_kwargs = {
				"convert_to_numpy": True,
				**kwargs,
		}

		# Only request external normalization if the model lacks an internal Normalize layer
		if normalize and not _has_internal_normalize(model, Normalize):
				encode_kwargs["normalize_embeddings"] = True

		if prompt is not None:
				encode_kwargs["prompt"] = prompt

		if prompt_name is not None:
				encode_kwargs["prompt_name"] = prompt_name

		embeddings = model.encode(list(texts), **encode_kwargs)

		return np.asarray(embeddings, dtype=np.float32)

# def encode(model, texts, prompt=None, prompt_name=None):
# 	"""L2-normalised float32 embeddings. prompt=None -> no prompt (what clustering.py does today)."""
# 	kw = {}
# 	if prompt is not None:
# 		kw["prompt"] = prompt
# 	if prompt_name is not None:
# 		kw["prompt_name"] = prompt_name
# 	return np.asarray(
# 		model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, **kw),
# 		dtype=np.float32,
# 	)

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