import json
import queue
import threading
import spacy
from tqdm import tqdm
import src.utils.paths as pth

_nlp = None

def bg_writer(write_queue, pbar: tqdm):
    with open(pth.PREP, "w", encoding="utf-8") as f_out:
        while True:
            item = write_queue.get()
            if item is None:
                break
            f_out.write(item)
            pbar.update(1)
            write_queue.task_done()

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    return _nlp

def process_doc_tokens(doc) -> list:
    return [
        token.lemma_.lower()
        for token in doc
        if not (token.is_stop or token.is_punct or token.is_space)
    ]

def preprocess_docs():
    with open(pth.DATA_DOCS, "r", encoding="utf-8") as f:
        total_docs = sum(1 for _ in f)

    nlp = get_nlp()
    pbar = tqdm(total=total_docs, desc="Preprocessing", unit="doc")

    write_queue = queue.Queue(maxsize=2000)
    writer_thread = threading.Thread(target=bg_writer, args=(write_queue, pbar))
    writer_thread.daemon = True
    writer_thread.start()

    def doc_generator():
        with open(pth.DATA_DOCS, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield (data["text"], data["doc_id"])

    doc_stream = nlp.pipe(doc_generator(), as_tuples=True, batch_size=1048, n_process=1)

    for doc, doc_id in doc_stream:
        tokens = process_doc_tokens(doc)
        write_queue.put(json.dumps({"doc_id": doc_id, "tokens": tokens}) + "\n")

    write_queue.put(None)
    writer_thread.join()
    pbar.close()