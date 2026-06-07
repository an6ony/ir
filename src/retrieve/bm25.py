import math
import src.utils.config as cfg

def bm25(query_tokens, expansion_terms=None, top_k=100, k1=1.5, b=0.75, og_weight=1.0, ex_weight=0.3):
    if expansion_terms is None:
        expansion_terms = []

    N = len(cfg.IDX_DOCLEN)
    scores = {}
    avgdl = sum(cfg.IDX_DOCLEN.values()) / N if N > 0 else 0

    query_weights = {}
    for term in query_tokens:
        query_weights[term] = og_weight
    for term in expansion_terms:
        if term not in query_weights:
            query_weights[term] = ex_weight

    for term, term_weight in query_weights.items():
        if term not in cfg.IDX_INV:
            continue

        doc_matches = cfg.IDX_INV[term]
        n_q = len(doc_matches)
        idf = 0 if n_q == 0 else math.log(((N - n_q + 0.5) / (n_q + 0.5)))

        for doc_id, freq in doc_matches.items():
            doc_len = cfg.IDX_DOCLEN[doc_id]
            tf_strength = freq * (k1 + 1)
            length_norm = k1 * (1.0 - b + b * (doc_len / avgdl))
            term_score = idf * (tf_strength / (freq + length_norm)) * term_weight
            scores[doc_id] = scores.get(doc_id, 0.0) + term_score

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs[:top_k]