from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
SRC_DIR = UTILS_DIR.parent
MAIN_DIR = SRC_DIR.parent

CRAN = "cranfield"
QUORA = "beir/quora/dev"
WIKI = "wikir/en1k/test"

def init(key="q"):
    global DATASET_NAME, DATASET_PATH, DATA_DIR, DATA_DIR, DATA_DOCS, DATA_DB, DATA_QRELS, DATA_QUERIES, PREP_DIR, PREP_DIR, PREP, IDX_DIR, IDX_DIR, IDX_INV, IDX_DOCLEN, IDX_FLAT, IDX_MAP, IDX_FAISS, IDX_EMBED

    if "q" in key.lower(): DATASET_NAME = QUORA
    elif "w" in key.lower(): DATASET_NAME = WIKI
    else: DATASET_NAME = CRAN

    DATASET_PATH = DATASET_NAME.strip(" /")

    DATA_DIR = MAIN_DIR / "data" / DATASET_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DB = DATA_DIR / "docs.db"
    DATA_DOCS = DATA_DIR / "docs.jsonl"
    DATA_QRELS = DATA_DIR / "qrels.tsv"
    DATA_QUERIES = DATA_DIR / "queries.tsv"

    PREP_DIR = MAIN_DIR / "preprocesses" / DATASET_PATH
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    PREP = PREP_DIR / "prep.jsonl"

    IDX_DIR = MAIN_DIR / "indexes" / DATASET_PATH
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    IDX_INV = IDX_DIR / "index.pkl"
    IDX_DOCLEN = IDX_DIR / "lengths.pkl"
    IDX_FLAT = IDX_DIR / "faiss.index"
    IDX_FAISS = IDX_DIR / "ivf.index"
    IDX_EMBED = IDX_DIR / "embed.npy"
    IDX_MAP = IDX_DIR / "map.pkl"