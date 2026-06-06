import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)