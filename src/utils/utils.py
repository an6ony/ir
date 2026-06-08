import time
import faiss
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