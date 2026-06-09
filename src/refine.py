import math
from spellchecker import SpellChecker
from collections import defaultdict
import src.utils.config as cfg

def prf(query_tokens, bm25result, top_k=5, num_terms=5):
    initial_results = bm25result[:top_k]
    top_doc_ids = [doc_id for doc_id, _ in initial_results]

    candidate_scores = defaultdict(float)
    N = len(cfg.IDX_DOCLEN)

    for term, doc_matches in cfg.IDX_INV.items():
        n_q = len(doc_matches)
        if n_q == 0:
            continue

        idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
        r_t = sum(1 for d_id in top_doc_ids if d_id in doc_matches)

        if r_t > 0:
            candidate_scores[term] = r_t * idf

    for token in query_tokens:
        candidate_scores.pop(token, None)

    expansion_terms = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:num_terms]
    expansion_tokens = [term for term, _ in expansion_terms]

    return expansion_tokens

def correct(query_str: str):
    spell = SpellChecker()
    words = query_str.split()
    misspelled = spell.unknown(words)

    corrected_words = []
    for word in words:
        if word in misspelled:
            correct = spell.correction(word)
            corrected_words.append(correct if correct else word)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)