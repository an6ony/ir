import pickle
import torch
from sentence_transformers import SentenceTransformer
import src.utils.paths as pth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

pth.init()
with open(pth.IDX_INV, "rb") as f: IDX_INV = pickle.load(f)
with open(pth.IDX_DOCLEN, "rb") as f: IDX_DOCLEN = pickle.load(f)
