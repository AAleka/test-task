import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

with open("sample_text.txt", 'r', encoding='utf-8') as f:
    input_text = f.read().strip()

with open("Standardised terms.csv", 'r', encoding='utf-8') as f:
    standard_phrases = f.read().strip().split('\n')

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/msmarco-distilbert-cos-v5')
model = AutoModel.from_pretrained('sentence-transformers/msmarco-distilbert-cos-v5')

input_ids = tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state

standard_embeddings = []
for phrase in standard_phrases:
    phrase_tokens = tokenizer.encode(phrase, return_tensors="pt")
    with torch.no_grad():
        phrase_outputs = model(phrase_tokens)
        phrase_embeddings = phrase_outputs.last_hidden_state.mean(dim=1).squeeze(0)
        standard_embeddings.append(phrase_embeddings)

similarity_scores = []
for input_embedding in embeddings[0]:
    input_embedding = input_embedding.unsqueeze(0)
    similarities = []
    for standard_embedding in standard_embeddings:
        similarity = cosine_similarity(input_embedding, standard_embedding.unsqueeze(0))
        similarities.append(similarity.item())
    similarity_scores.append(similarities)

max_similarities = np.max(similarity_scores, axis=1)

threshold = 0.3

suggestions = []
for i, score in enumerate(max_similarities):
    if score > threshold:
        input_phrase = tokenizer.decode(input_ids[0][i].item())
        best_match_index = np.argmax(similarity_scores[i])
        suggested_phrase = standard_phrases[best_match_index]
        suggestions.append((input_phrase, suggested_phrase, score))

print("Original Text:")
print(input_text)
print("\nText with Suggestions:")
if suggestions:
    suggested_text = input_text
    for input_phrase, suggested_phrase, _ in suggestions:
        suggested_text = suggested_text.replace(input_phrase, f"[{input_phrase} => {suggested_phrase}]", 1)
    print(suggested_text)
else:
    print("No suggestions found.")