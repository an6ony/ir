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
    term_idfs = {}
    query_l2_sum = 0.0

    for term in all_terms:
        if term not in cfg.IDX_INV: continue

        doc_postings = cfg.IDX_INV[term]
        df = len(doc_postings)
        if df == 0: continue

        idf = math.log10(N / df)
        term_idfs[term] = idf

        w_t_q = 0.0
        if term in query_counts:
            w_t_q += og_weight * (1 + math.log10(query_counts[term]))
        if term in expansion_counts:
            w_t_q += ex_weight * (1 + math.log10(expansion_counts[term]))

        final_q_weight = w_t_q * idf
        query_weights[term] = final_q_weight
        query_l2_sum += final_q_weight ** 2

    if not query_weights: return []

    query_norm = math.sqrt(query_l2_sum)

    for term, combined_q_weight in query_weights.items():
        doc_postings = cfg.IDX_INV[term]
        idf = term_idfs[term]

        for doc_id, tf in doc_postings.items():
            if tf > 0:
                w_t_d = (1 + math.log10(tf)) * idf
                scores[doc_id] += combined_q_weight * w_t_d

    final_results = []
    if query_norm > 0:
        for doc_id, raw_score in scores.items():
            doc_norm = cfg.IDX_DOCNORM.get(doc_id, 0.0)
            if doc_norm > 0:
                normalized_score = raw_score / (query_norm * doc_norm)
                final_results.append((doc_id, normalized_score))
                
        final_results.sort(key=lambda x: x[1], reverse=True)

    return final_results[:top_k]