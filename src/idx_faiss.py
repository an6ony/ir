import orjson
import pickle
import faiss
from tqdm import tqdm
import src.utils.config as cfg
import src.utils.paths as pth

def idx_faiss(batch_size=32):
    device = cfg.DEVICE
    model = cfg.MODEL
    embedding_dim = 384
    cpu_index = faiss.IndexFlatIP(embedding_dim)
    print(f"Using device: {device.upper()}")

    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else: index = cpu_index

    with open(pth.DATA_DOCS, "r", encoding="utf-8") as f:
        total_docs = sum(1 for _ in f)

    pbar = tqdm(total=total_docs, desc="Vector Indexing", unit="doc")
    doc_ids = []
    batch_texts = []

    with open(pth.DATA_DOCS, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue

            data = orjson.loads(line)
            batch_texts.append(data["text"])
            doc_ids.append(data["doc_id"])

            if len(batch_texts) >= batch_size:
                embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
                faiss.normalize_L2(embeddings)
                index.add(embeddings)
                pbar.update(len(batch_texts))
                batch_texts = []

        if batch_texts:
            embeddings = model.encode(batch_texts, batch_size=len(batch_texts), show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            pbar.update(len(batch_texts))

    pbar.close()

    if device == "cuda":
        final_cpu_index = faiss.index_gpu_to_cpu(index)
    else: final_cpu_index = index

    faiss.write_index(final_cpu_index, str(pth.IDX_FLAT))

    with open(pth.IDX_MAP, "wb") as f_map:
        pickle.dump(doc_ids, f_map)