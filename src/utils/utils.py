import time
import json
import faiss
import pickle
import sqlite3
import numpy as np
import src.utils.paths as pth

def timer(fn):
    start_time = time.perf_counter()
    result = fn()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"time:\t{execution_time:.6f} seconds\n")
    return result

def flat2ivf():
    old_index = faiss.read_index(str(pth.IDX_FLAT))
    total_vectors = old_index.ntotal
    dimension = old_index.d
    vectors = old_index.reconstruct_n(0, total_vectors)
    nlist = int(4 * np.sqrt(total_vectors))
    quantizer = faiss.IndexFlatIP(dimension)
    new_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    max_train_points = 256 * nlist
    train_vectors = vectors[:max_train_points] if total_vectors > max_train_points else vectors
    new_index.train(train_vectors)
    new_index.add(vectors)
    new_idx_path = str(pth.IDX_FAISS)
    print("Converted flat to ivf")
    faiss.write_index(new_index, new_idx_path)

def faiss2numpy():
    index = faiss.read_index(str(pth.IDX_FLAT))
    num_docs = index.ntotal
    embedding_dimension = index.d
    raw_buffer = index.get_xb()
    flat_array = faiss.rev_swig_ptr(raw_buffer, num_docs * embedding_dimension)
    embeddings_matrix = flat_array.reshape(num_docs, embedding_dimension).astype('float32')
    print("Converted flat to numpy")
    np.save(pth.IDX_EMBED, embeddings_matrix)

def db():
    conn = sqlite3.connect(pth.DATA_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            text TEXT NOT NULL
        )
    """)
    chunk = []
    chunk_size = 50000
    with open(pth.DATA_DOCS, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            chunk.append((str(doc['doc_id']), doc['text']))
            if len(chunk) >= chunk_size:
                cursor.executemany("INSERT OR REPLACE INTO documents (doc_id, text) VALUES (?, ?)", chunk)
                conn.commit()
                chunk = []
                print(f"Inserted {chunk_size} records...")
        if chunk:
            cursor.executemany("INSERT OR REPLACE INTO documents (doc_id, text) VALUES (?, ?)", chunk)
            conn.commit()
    print("Converted jsonl to sqlite")
    conn.close()


def pickle2json(pickle_path, json_path):
    with open(pickle_path, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

def convert():
    pickle2json(pth.IDX_INV, "index.json")
    # pickle2json(pth.IDX_DOCLEN, "lengths.json")
    # pickle2json(pth.IDX_MAP, "map.json")