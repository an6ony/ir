import re
import json
import ir_datasets
from tqdm import tqdm
import src.utils.paths as pth

def download():
    dataset = ir_datasets.load(pth.DATASET_NAME)

    with (pth.DATA_DOCS).open("w", encoding="utf-8") as f:
        for doc in tqdm(dataset.docs_iter(), desc=f"Writing {pth.DATASET_NAME} dqq", unit=" docs"):
            doc_data = {"doc_id": doc.doc_id, "text": doc.text}
            f.write(json.dumps(doc_data, ensure_ascii=False) + "\n")

    with (pth.DATA_QUERIES).open("w", encoding="utf-8") as f:
        f.write("query_id\tquery_text\n")
        for q in dataset.queries_iter():
            text = re.sub(re.compile(r'\s+'), ' ', q.text)
            f.write(f"{q.query_id}\t{text}\n")

    with (pth.DATA_QRELS).open("w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\trelevance\n")
        for q in dataset.qrels_iter():
            f.write(f"{q.query_id}\t{q.doc_id}\t{q.relevance}\n")