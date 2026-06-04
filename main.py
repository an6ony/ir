from src.download import download
from src.preprocessing_service import preprocess_docs
from src.inverted_index import build_inverted_index
import src.utils.paths as pth

if __name__ == '__main__':
    pth.init()
    download()
    preprocess_docs()
    build_inverted_index()