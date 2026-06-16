import math
from collections import Counter
import src.utils.config as cfg

def tf_idf(query_tokens, expansion_terms=[], top_k=100, og_weight=1.0, ex_weight=0.3):
    N = len(cfg.IDX_DOCLEN)
    scores = Counter()
    query_counts = Counter(query_tokens)
    expansion_counts = Counter(expansion_terms)
    all_terms = set(query_counts.keys()).union(set(expansion_counts.keys()))
    query_weights = {}

    for term in all_terms:
        if term not in cfg.IDX_INV: continue

        df = len(cfg.IDX_INV[term])
        if df == 0: continue

        idf = math.log10(N / df)

        w_t_q = 0.0
        if term in query_counts:
            w_t_q += og_weight * (1 + math.log10(query_counts[term]))
        if term in expansion_counts:
            w_t_q += ex_weight * (1 + math.log10(expansion_counts[term]))

        query_weights[term] = w_t_q * idf

    for term, combined_q_weight in query_weights.items():
        doc_postings = cfg.IDX_INV[term]

        for doc_id, tf in doc_postings.items():
            if tf > 0:
                w_t_d = 1 + math.log10(tf)
                scores[doc_id] += combined_q_weight * w_t_d

    for doc_id in scores:
        if cfg.IDX_DOCLEN[doc_id] > 0:
            scores[doc_id] /= math.sqrt(cfg.IDX_DOCLEN[doc_id])

    return scores.most_common(top_k)