import os
os.environ["HF_HUB_OFFLINE"] = "1" # uncomment after downloading
from src.download import download
from src.preprocessing_service import preprocess_docs
from src.inverted_index import build_inverted_index
from src.idx_faiss import idx_faiss
from src.retrieve.tf_idf import tf_idf
import src.utils.paths as pth
import src.utils.config as cfg

if __name__ == '__main__':
    pth.init()
    # download()
    # preprocess_docs()
    # build_inverted_index()
    # idx_faiss()
    print(tf_idf(["test", "query", "token"], top_k=10))