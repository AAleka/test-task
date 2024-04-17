# Text Improvement Engine (English only)

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
