import src.utils.config as cfg

def bert(query_texts: list[str] | str, top_k=100):
    single_query = isinstance(query_texts, str)

    if single_query: query_texts = [query_texts]

    query_vectors = cfg.MODEL.encode(
        query_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype('float32')

    num_queries = len(query_texts)
    batch_results = []

    cfg.IDX_FAISS.nprobe = 16 # recommended

    for q_idx in range(num_queries):
        single_query_vector = query_vectors[q_idx : q_idx + 1]

        distances, indices = cfg.IDX_FAISS.search(single_query_vector, top_k)

        query_results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1:
                doc_id = cfg.IDX_MAP[int(idx)]
                query_results.append((doc_id, float(score)))

        batch_results.append(query_results)

    return batch_results[0] if single_query else batch_results