# Information Retrieval Engine

A robust, high-performance Information Retrieval (IR) system designed for academic evaluation and real-world semantic search. This project bridges the gap between traditional keyword-based retrieval and modern transformer-based dense vector search, providing a comprehensive framework for experimentation and benchmarking.

## 🚀 Key Features

*   **Multi-Strategy Retrieval:**
    *   **Sparse:** Optimized BM25 and TF-IDF implementations.
    *   **Dense:** Semantic search using BERT embeddings (`all-MiniLM-L6-v2`).
    *   **Hybrid Pipelines:** "Serial Re-ranking" (BM25 -> BERT) and "Parallel Score Fusion" (BM25 + BERT) for superior precision.
*   **Vector Optimization:** Leverages **FAISS** (Facebook AI Similarity Search) for sub-millisecond similarity search across high-dimensional embeddings.
*   **Intelligent Query Refinement:** 
    *   **Spelling Correction:** Auto-normalizes user queries.
    *   **Pseudo Relevance Feedback (PRF):** Automatically expands queries using top-tier initial results to improve recall.
*   **Professional Evaluation Suite:** Built-in dashboard computing **MAP**, **nDCG@10**, and **P@10** using the `ir-measures` standard.
*   **Interactive Interface:** A dual-tab Streamlit UI for live searching and batch performance visualization.

## 💻 Usage

### 1. Environment Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```
*Note: Ensure `spacy` model is installed: `python -m spacy download en_core_web_sm`.*

### 2. Initialization & Pipeline Execution
To download datasets, build indexes, and run the evaluation suite:
```bash
python main.py
```
*Note: You can switch between 'quora' and 'wikipedia' datasets in `main.py`.*

### 3. Interactive UI
Launch the search engine and evaluation dashboard:
```bash
streamlit run app.py
```

## 📁 Project Structure & Flow

### Repository Architecture
```text
ir-main/
├── app.py                  # Streamlit Dashboard & UI
├── main.py                 # Core CLI Pipeline (Run-all)
└── src/
    ├── retrieve/           # Logic for TF-IDF, BM25, BERT, and Hybrid models
    ├── utils/              # Configuration and Path management
    ├── preprocess.py       # NLP pipeline (SpaCy)
    ├── index.py            # Sparse inverted index construction
    ├── idx_faiss.py        # Vector database (FAISS) construction
    ├── refine.py           # Query expansion & correction logic
    └── rank.py             # Metrics (ir-measures) & Plotting
```

### Execution & Data Flow
The system operates as a modular pipeline where data transitions through several refined states:

1.  **Data Acquisition (`download.py`):** Fetches and structures raw datasets (Quora/Wiki) into localized SQLite and JSON formats.
2.  **Preprocessing (`preprocess.py`):** 
    *   Uses **SpaCy** to perform tokenization, lemmatization, and stop-word removal.
    *   **Flow:** Implements a producer-consumer model with a background writer thread to handle large-scale text processing efficiently.
3.  **Dual-Indexing Strategy:**
    *   **Sparse Indexing (`index.py`):** Generates a global **Inverted Index** and Document Length mapping for BM25/TF-IDF.
    *   **Dense Indexing (`idx_faiss.py`):** Converts documents into 384-dimensional embeddings using **BERT**. These are normalized and stored in a **FAISS `IndexFlatIP`** for high-speed Inner Product similarity.
4.  **Retrieval & Refinement (`retrieve/`, `refine.py`):**
    *   Queries are preprocessed and optionally refined via spelling correction or **PRF expansion**.
    *   The retrieval engine fetches candidates from either the sparse inverted index (via token matching) or the dense FAISS index (via vector similarity).
5.  **Evaluation & Ranking (`rank.py`):**
    *   The system executes batch queries against ground-truth "Qrels."
    *   Calculates precision and ranking metrics.
    *   Benchmarks the **FAISS speedup** (typically 10x-50x faster) compared to brute-force NumPy matrix multiplication.
