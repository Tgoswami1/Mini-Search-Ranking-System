def build_features(query: str, item, bm25_score: float):
    """
    Build ranking features for (query, item)
    """

    q_tokens = set(query.lower().split())
    title_tokens = set(item["title"].lower().split())

    features = [
        bm25_score,                         # retrieval score
        len(q_tokens & title_tokens),       # title overlap
        int(item["brand"].lower() in q_tokens),
        float(item["price"])
    ]

    return features
