import transformers as tfs
import json
import torch

unmasker = tfs.pipeline('fill-mask', model='google-bert/bert-large-cased')
unmasked_result = unmasker("This Deep Reinforcement Learning framework learns from [MASK] data.")
print(json.dumps(unmasked_result, indent=2))


tokenizer = tfs.BertTokenizer.from_pretrained('google-bert/bert-large-cased')
model = tfs.BertModel.from_pretrained("google-bert/bert-large-cased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

print(type(output), output.keys(), output.last_hidden_state.shape)

corpus = [
	"A man is eating pizza with his friends",
	"Construction Worker",
	"Global Warming",
]

embeddings = list()
for i, text in enumerate(corpus):
	encoded_input = tokenizer(text, return_tensors='pt')
	output = model(**encoded_input)
	embeddings.append(output.last_hidden_state[:,0,:])

# Find similar documents
query_text = "climate change effects"
query_embedding = model(**tokenizer(query_text, return_tensors='pt')).last_hidden_state[:,0,:]
similarities = [torch.nn.functional.cosine_similarity(query_embedding, doc_emb) for doc_emb in embeddings]
best_match_idx = similarities.index(max(similarities))
print(similarities)
print(query_text)
print(corpus[best_match_idx])


# Content-based recommendations
product_descriptions = ["organic coffee beans", "strong black coffee", "coffee maker machine"]
user_profile = "I love strong coffee"

user_emb = model(**tokenizer(user_profile, return_tensors='pt')).last_hidden_state[:,0,:]
product_embs = [model(**tokenizer(desc, return_tensors='pt')).last_hidden_state[:,0,:] for desc in product_descriptions]
recommendations = sorted(zip(product_descriptions, product_embs), key=lambda x: torch.nn.functional.cosine_similarity(user_emb, x[1]), reverse=True)

print(recommendations)