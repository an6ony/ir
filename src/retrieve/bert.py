import numpy as np
import faiss
import src.utils.config as cfg

def bert(query_texts: list[str] | str, top_k=100, compare_faiss=True, doc_ids_batch: list[list] | list = None):
    single_query = isinstance(query_texts, str)

    if single_query:
        query_texts = [query_texts]
        if doc_ids_batch is not None and not isinstance(doc_ids_batch[0], list):
            doc_ids_batch = [doc_ids_batch]

    query_vectors = cfg.MODEL.encode(
        query_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype('float32')

    num_queries = len(query_texts)
    batch_results = []

    if compare_faiss and hasattr(cfg.IDX_FAISS, "nprobe"):
        cfg.IDX_FAISS.nprobe = 16

    if compare_faiss and doc_ids_batch is None:
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

    else:
        id_to_idx = {str(doc_id): idx for idx, doc_id in enumerate(cfg.IDX_MAP)}

        for q_idx in range(num_queries):
            query_vector = query_vectors[q_idx]

            if doc_ids_batch is not None and q_idx < len(doc_ids_batch):
                specific_doc_ids = doc_ids_batch[q_idx]
                valid_pairs = [
                    (str(d_id), id_to_idx[str(d_id)])
                    for d_id in specific_doc_ids if str(d_id) in id_to_idx
                ]

                if not valid_pairs:
                    batch_results.append([])
                    continue

                subset_doc_ids, subset_indices = zip(*valid_pairs)
                embeddings = cfg.IDX_EMBED[list(subset_indices)]
            else:
                subset_doc_ids = cfg.IDX_MAP
                embeddings = cfg.IDX_EMBED

            query_scores = np.dot(embeddings, query_vector)
            actual_k = min(top_k, len(query_scores))
            top_indices = np.argsort(query_scores)[::-1][:actual_k]

            query_results = [
                (str(subset_doc_ids[idx]), float(query_scores[idx]))
                for idx in top_indices
            ]
            batch_results.append(query_results)

        return batch_results[0] if single_query else batch_results