from src.retrieve.bert import bert

def serial(query_texts: list[str] | str, bm25_results_batch: list[list] | list, top_k=30):
    if not bm25_results_batch: return []

    doc_ids_batch = []
    if bm25_results_batch is not None and not isinstance(bm25_results_batch[0], list):
        bm25_results_batch = [bm25_results_batch]

    for query_bm25_list in bm25_results_batch:
        candidate_ids = [int(doc_id) for doc_id, _ in query_bm25_list]
        doc_ids_batch.append(candidate_ids)

    return bert(query_texts, top_k, False, doc_ids_batch)

def min_max_normalize(results: dict) -> dict:
    if not results: return {}

    scores = list(results.values())
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return {doc_id: 1.0 for doc_id in results.keys()}

    return {
        doc_id: (score - min_score) / (max_score - min_score)
        for doc_id, score in results.items()
    }

def parallel(bm25_list, bert_list, top_k=300, weight=0.5):
    bm25_results = dict(bm25_list)
    bert_results = dict(bert_list)
    norm_bm25 = min_max_normalize(bm25_results)
    norm_bert = min_max_normalize(bert_results)
    hybrid_scores = {}
    all_doc_ids = set(norm_bm25.keys()).union(set(norm_bert.keys()))

    for doc_id in all_doc_ids:
        s1 = norm_bm25.get(doc_id, 0.0)
        s2 = norm_bert.get(doc_id, 0.0)
        hybrid_scores[doc_id] = (weight * s1) + ((1.0 - weight) * s2)

    sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_hybrid[:top_k]