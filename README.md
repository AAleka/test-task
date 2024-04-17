# Text Improvement Engine (English only)

## Usage:
 - Run the following command to install the required libraries.
```
pip install -r requirements.txt
```

## Sample output:

Original Text:                                                                                                                                                                                  In today's meeting, we discussed a variety of issues affecting our department. The weather was unusually sunny, a pleasant backdrop to our serious discussions. We came to the consensus that we need to do better in terms of performance. Sally brought doughnuts, which lightened the mood. It's important to make good use of what we have at our disposal. During the coffee break, we talked about the upcoming company picnic. We should aim to be more efficient and look for ways to be more creative in our daily tasks. Growth is essential for our future, but equally important is building strong relationships with our team members. As a reminder, the annual staff survey is due next Friday. Lastly, we agreed that we must take time to look over our plans carefully and consider all angles before moving forward. On a side note, David mentioned that his cat is recovering well from surgery.                                                                                                                                                                                                                    Text with Suggestions:                                                                                                                                                                          In today's meeting, we discussed a variety of issues affecting our department. The weather was unusually sunny, a pleasant backdrop to our serious discussions. We came to the consensus that we need to do better in terms of performance. Sally brought doughnuts, which lightened the mood. It's important to make good use of what we have at our disposal. During the coffee break, we talked about the upcoming company picnic. We should aim to be more efficient and look for ways to be more creative in our daily tasks. Growth is essential for our future, but equally important is building strong <span style="color: green"> [relationships => Facilitate collaboration] </span> with our team members. As a reminder, the annual staff survey is due next Friday. Lastly, we agreed that we must take time to look over our plans carefully and consider all angles before moving forward. On a side note, David mentioned that his cat is recovering well from surgery. 

## Objective:
### Develop a tool that analyses a given text and suggests improvements based on the similarity to a list of "standardised" phrases. These standardised phrases represent the ideal way certain concepts should be articulated, and the tool should recommend changes to align the input text closer to these standards.

#### Import necessary libraries:
 - torch: PyTorch library for tensor computations.
 - AutoTokenizer and AutoModel from transformers: These are used for loading pre-trained models and tokenizers.
 - cosine_similarity from sklearn.metrics.pairwise: This function computes the cosine similarity between vectors.
 - numpy as np: NumPy library for numerical computations.
```
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
```

#### Open the files "sample_text.txt" and "Standardised terms.csv" in read mode and read their contents.
```
with open("sample_text.txt", 'r', encoding='utf-8') as f:
    input_text = f.read().strip()

with open("Standardised terms.csv", 'r', encoding='utf-8') as f:
    standard_phrases = f.read().strip().split('\n')
```

#### Load a pre-trained tokenizer and model using the AutoTokenizer.from_pretrained() and AutoModel.from_pretrained() methods from the transformers library:
 - The tokenizer and model used here are from the Sentence Transformers library and are fine-tuned on the MS MARCO dataset.
```
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/msmarco-distilbert-cos-v5')
model = AutoModel.from_pretrained('sentence-transformers/msmarco-distilbert-cos-v5')
```

#### Tokenize the input_text using the tokenizer:
 - return_tensors="pt" returns PyTorch tensors.
 - Assign the tokenized input to the variable input_ids.
```
input_ids = tokenizer.encode(input_text, return_tensors="pt")
```

 - Disable gradient computation using torch.no_grad() context manager.
 - Pass the input_ids through the model to get the model outputs.
 - Extract the last hidden states from the outputs, representing the embeddings of each token in the input.
```
with torch.no_grad():
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state
```

 - Iterate through each standardized phrase in standard_phrases.
 - Tokenize each phrase using the tokenizer.
 - Pass the phrase tokens through the model to get the model outputs.
 - Compute the mean of the embeddings across tokens to get a single embedding for each phrase.
 - Append the phrase embeddings to the list standard_embeddings.
```
standard_embeddings = []
for phrase in standard_phrases:
    phrase_tokens = tokenizer.encode(phrase, return_tensors="pt")
    with torch.no_grad():
        phrase_outputs = model(phrase_tokens)
        phrase_embeddings = phrase_outputs.last_hidden_state.mean(dim=1).squeeze(0)
        standard_embeddings.append(phrase_embeddings)
```

 - Iterate through each token embedding in the embeddings.
 - Compute the cosine similarity between the token embedding and each standardized phrase embedding.
 - Append the similarity scores to the list similarity_scores
```
similarity_scores = []
for input_embedding in embeddings[0]:
    input_embedding = input_embedding.unsqueeze(0)
    similarities = []
    for standard_embedding in standard_embeddings:
        similarity = cosine_similarity(input_embedding, standard_embedding.unsqueeze(0))
        similarities.append(similarity.item())
    similarity_scores.append(similarities)
```

 - Compute the maximum similarity score for each token embedding across all standardized phrases.
```
max_similarities = np.max(similarity_scores, axis=1)
```

 - Set a threshold value to filter suggestions based on similarity scores.
```
threshold = 0.3
```

 - Iterate through each token embedding and its corresponding maximum similarity score.
 - If the similarity score is above the threshold, consider it a potential suggestion.
 - Decode the token to its original text using the tokenizer.
 - Find the index of the standardized phrase with the highest similarity score.
 - Append the original text, suggested phrase, and similarity score to the suggestions list.
```
suggestions = []
for i, score in enumerate(max_similarities):
    if score > threshold:
        input_phrase = tokenizer.decode(input_ids[0][i].item())
        best_match_index = np.argmax(similarity_scores[i])
        suggested_phrase = standard_phrases[best_match_index]
        suggestions.append((input_phrase, suggested_phrase, score))
```

 - Print the original input text.
 - If there are suggestions, replace each input phrase with its corresponding suggestion in the original text and print the suggested text.
 - If no suggestions are found, print a message indicating the same.
```
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
```
