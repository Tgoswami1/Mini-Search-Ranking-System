import numpy as np

def ndcg_at_k(rels, k):
    rels = rels[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(sorted(rels, reverse=True)[i] / np.log2(i + 2) for i in range(len(rels)))
    return dcg / idcg if idcg else 0
