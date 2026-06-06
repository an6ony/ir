import faiss
import numpy as np
import src.utils.paths as pth

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