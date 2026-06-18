import faiss
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import src.utils.paths as pth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
IDX_FAISS = None
IDX_MAP = None
IDX_INV = None
IDX_DOCLEN = None
IDX_DOCNORM = None

def init(key="q"):
    global DEVICE, MODEL, IDX_FAISS, IDX_EMBED, IDX_MAP, IDX_INV, IDX_DOCLEN, IDX_DOCNORM

    pth.init(key)

    print("init...", pth.DATASET_NAME)

    IDX_EMBED = np.load(str(pth.IDX_EMBED))
    IDX_FAISS = faiss.read_index(str(pth.IDX_FAISS))
    with open(pth.IDX_MAP, "rb") as f: IDX_MAP = pickle.load(f)

    with open(pth.IDX_INV, "rb") as f: IDX_INV = pickle.load(f)
    with open(pth.IDX_DOCLEN, "rb") as f: IDX_DOCLEN = pickle.load(f)
    with open(pth.IDX_DOCNORM, "rb") as f: IDX_DOCNORM = pickle.load(f)

    print("done init.")