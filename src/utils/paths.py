from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
SRC_DIR = UTILS_DIR.parent
MAIN_DIR = SRC_DIR.parent

CRAN = "cranfield"

def init():
    global DATASET_NAME, DATASET_PATH, DATA_DIR, DATA_DIR, DATA_DOCS, DATA_QRELS, DATA_QUERIES, PREP_DIR, PREP_DIR, PREP
    
    DATASET_NAME = CRAN
    DATASET_PATH = DATASET_NAME

    DATA_DIR = MAIN_DIR / "data" / DATASET_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DOCS = DATA_DIR / "docs.jsonl"
    DATA_QRELS = DATA_DIR / "qrels.tsv"
    DATA_QUERIES = DATA_DIR / "queries.tsv"

    PREP_DIR = MAIN_DIR / "preprocesses" / DATASET_PATH
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    PREP = PREP_DIR / "prep.jsonl"