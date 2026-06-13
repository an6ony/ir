import os
os.environ["HF_HUB_OFFLINE"] = "1" # uncomment after downloading
from src.download import download
from src.preprocessing_service import preprocess_docs
from src.inverted_index import build_inverted_index
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
    pth.init()
    utl.db()
    cfg.init()
    # utl.flat2ivf()
    # utl.faiss2numpy()
    # download()
    # preprocess_docs()
    # build_inverted_index()
    # idx_faiss()
    # query = "tesf singl queires tokne"
    # tokens = correct(query).split()
    # print(tf_idf(tokens, top_k=10))
    # bm = bm25(tokens, top_k=10)
    # print(utl.timer(lambda: bert([query, "test multiple queries"], top_k=5, use_faiss=False)))
    # print(utl.timer(lambda: bert([query, "test multiple queries"], top_k=5, use_faiss=True)))
    # print(utl.timer(lambda: bert(query, top_k=5, use_faiss=False)))
    # bert_list = bert(query, top_k=5, use_faiss=True)
    # print(bert_list)
    # print(serial(query, bm, top_k=5))
    # print(parallel(bm, bert_list, top_k=5))
    # print(correct(query))
    # print(prf(tokens, bm))
    # rank()
