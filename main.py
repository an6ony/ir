import os
os.environ["HF_HUB_OFFLINE"] = "1" # uncomment after downloading
from src.download import download
from src.preprocess import preprocess
from src.index import index
from src.idx_faiss import idx_faiss
from src.retrieve.tf_idf import tf_idf
from src.retrieve.bm25 import bm25
from src.retrieve.bert import bert
from src.retrieve.hybrid import serial, parallel
from src.refine import correct, prf
from src.rank import rank
import src.utils.paths as pth
import src.utils.config as cfg
import src.utils.utils as utl

if __name__ == '__main__':
    ds = "quora"
    # ds = "cran"
    pth.init(ds)

    # download()
    # preprocess()
    # index()
    # idx_faiss()

    # utl.faiss2numpy()
    # utl.flat2ivf()
    utl.db()

    cfg.init(ds)

    rank()