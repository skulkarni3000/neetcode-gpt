import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places

        K = self.key_gen(embedded) #(batch_size, context_length (#token in seq), attention_dim)
        Q = self.query_gen(embedded)
        V = self.value_gen(embedded)

        #Scores (Q @ K.T) / sqrt(d_k)
        context_length, attention_dim = K.shape[1], K.shape[2]
        scores = (Q @ torch.transpose(K,1,2)) / (attention_dim ** 0.5)

        #Causal mask: prevent attending to future tokens
        lower_triangular = torch.tril(torch.ones(context_length, context_length))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float('-inf'))
        
        #softmax and result
        scores = nn.functional.softmax(scores, dim=2)

        return torch.round(scores @ V, decimals=4)

