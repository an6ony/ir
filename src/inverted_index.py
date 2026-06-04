import orjson
import pickle
from tqdm import tqdm
from collections import Counter, defaultdict
import src.utils.paths as pth

def build_inverted_index():
    with open(pth.PREP, "r", encoding="utf-8") as f:
        total_docs = sum(1 for line in f if line.strip())

    inverted_index = defaultdict(dict)
    doc_lengths = {}
    pbar = tqdm(total=total_docs, desc="indexing preprocessed docs", unit="doc")

    with open(pth.PREP, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue

            doc = orjson.loads(line)
            doc_id = doc.get("doc_id")
            tokens = doc.get("tokens")

            if doc_id is None or tokens is None:
                pbar.update(1)
                continue

            doc_id_str = str(doc_id)
            doc_lengths[doc_id_str] = len(tokens)

            term_freqs = Counter(tokens)
            for term, frequency in term_freqs.items():
                inverted_index[term][doc_id_str] = frequency

            pbar.update(1)

    pbar.close()

    with open(pth.IDX_INV, "wb") as out_f:
        pickle.dump(inverted_index, out_f)

    with open(pth.IDX_DOCLEN, "wb") as out_l:
        pickle.dump(doc_lengths, out_l)