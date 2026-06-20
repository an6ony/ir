import streamlit as st
import sqlite3
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import src.utils.config as cfg
import src.utils.paths as pth
from src.retrieve.tf_idf import tf_idf
from src.retrieve.bm25 import bm25
from src.retrieve.bert import bert
from src.retrieve.hybrid import serial, parallel
from src.preprocess import preprocess_query
from src.refine import correct, prf
from src.rank import rank_ui

@st.cache_resource(show_spinner="Parsing dataset indexes...")
def get_dataset_state(dataset_key: str):
    cfg.init(dataset_key)

    return {
        "DATASET_NAME": pth.DATASET_NAME,
        "DATA_DB": pth.DATA_DB,
        "DATA_QUERIES": pth.DATA_QUERIES,
        "DATA_QRELS": pth.DATA_QRELS,
        "IDX_FAISS": cfg.IDX_FAISS,
        "IDX_EMBED": cfg.IDX_EMBED,
        "IDX_MAP": cfg.IDX_MAP,
        "IDX_INV": cfg.IDX_INV,
        "IDX_DOCLEN": cfg.IDX_DOCLEN,
        "IDX_DOCNORM": cfg.IDX_DOCNORM
    }

def load_dataset_index(dataset_key: str):
    state_bundle = get_dataset_state(dataset_key)

    pth.DATASET_NAME = state_bundle["DATASET_NAME"]
    pth.DATA_DB      = state_bundle["DATA_DB"]
    pth.DATA_QUERIES = state_bundle["DATA_QUERIES"]
    pth.DATA_QRELS   = state_bundle["DATA_QRELS"]
    cfg.IDX_FAISS    = state_bundle["IDX_FAISS"]
    cfg.IDX_EMBED    = state_bundle["IDX_EMBED"]
    cfg.IDX_MAP      = state_bundle["IDX_MAP"]
    cfg.IDX_INV      = state_bundle["IDX_INV"]
    cfg.IDX_DOCLEN   = state_bundle["IDX_DOCLEN"]
    cfg.IDX_DOCNORM  = state_bundle["IDX_DOCNORM"]

def fetch_document_text(doc_id: str) -> str:
    try:
        db_path = str(pth.DATA_DB)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM documents WHERE doc_id = ?", (str(doc_id),))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else " Document text not found in database."
    except Exception as e:
        return f" Database Error: {str(e)}"

st.set_page_config(page_title="IR Project Engine", layout="wide")
st.title(" University Information Retrieval Engine")

tab1, tab2 = st.tabs([" Search Engine Interface", " Batch Evaluation Dashboard"])

with tab1:
    st.header("Document Retrieval System")
    st.sidebar.header(" Configuration Panel")

    dataset_choice = st.sidebar.selectbox(
        "Select Dataset Target",
        options=["Wikipedia", "Quora", "Cranfield"],
        index=0
    )
    dataset_key = "w" if "Wikipedia" in dataset_choice else "q" if "Quora" in dataset_choice else "c"

    load_dataset_index(dataset_key)

    algo_choice = st.sidebar.selectbox(
        "Retrieval Framework",
        options=["TF-IDF", "BM25", "BERT (Dense)", "Hybrid (Serial)", "Hybrid (Parallel)"]
    )

    st.sidebar.subheader(" Query Refinement Options")
    enable_spelling = st.sidebar.toggle("Enable Spelling Correction", value=False)
    enable_prf = st.sidebar.toggle("Enable Pseudo Relevance Feedback (PRF)", value=False)

    st.sidebar.subheader(" Algorithm Tuning Parameters")
    top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=50, value=5)

    og_weight = 1.0
    ex_weight = 0.3
    k1 = 1.0
    b = 0.3
    compare_faiss = True
    hybrid_weight = 0.5

    if algo_choice == "TF-IDF":
        og_weight = st.sidebar.slider("Original Weight", 0.0, 2.0, 1.0, 0.1)
        ex_weight = st.sidebar.slider("Expansion Weight", 0.0, 2.0, 0.3, 0.1)
    elif algo_choice == "BM25":
        k1 = st.sidebar.slider("Saturating Parameter (k1)", 0.1, 3.0, 1.0, 0.1)
        b = st.sidebar.slider("Length Normalization (b)", 0.0, 1.0, 0.3, 0.05)
        og_weight = st.sidebar.slider("Original Weight", 0.0, 2.0, 1.0, 0.1)
        ex_weight = st.sidebar.slider("Expansion Weight", 0.0, 2.0, 0.3, 0.1)
    elif algo_choice == "BERT (Dense)":
        compare_faiss = st.sidebar.checkbox("Use FAISS Indexing", value=True, help="Uncheck to fallback to basic numpy matrix embeddings")
    elif algo_choice in ["Hybrid (Serial)", "Hybrid (Parallel)"]:
        if algo_choice == "Hybrid (Parallel)":
            hybrid_weight = st.sidebar.slider("Parallel Score Fusion Weight", 0.0, 1.0, 0.5, 0.05)
        k1 = st.sidebar.slider("BM25: k1", 0.1, 3.0, 1.0, 0.1)
        b = st.sidebar.slider("BM25: b", 0.0, 1.0, 0.3, 0.05)
        og_weight = st.sidebar.slider("Original Weight", 0.0, 2.0, 1.0, 0.1)
        ex_weight = st.sidebar.slider("Expansion Weight", 0.0, 2.0, 0.3, 0.1)
        compare_faiss = st.sidebar.checkbox("BERT: Use FAISS Indexing", value=True)

    query_input = st.text_input(" Enter your search pipeline query:", placeholder="e.g., How do IR Engines work?")

    if query_input:
        working_query = query_input
        if enable_spelling:
            working_query = correct(query_input)
            if working_query != query_input:
                st.info(f" *Auto-corrected Query to:* **{working_query}**")

        query_tokens = preprocess_query(working_query)
        expansion_tokens = []

        if enable_prf:
            prf_base_results = bm25(query_tokens, expansion_terms=[], top_k=10, k1=k1, b=b, og_weight=og_weight, ex_weight=ex_weight)
            expansion_tokens = prf(query_tokens, prf_base_results, top_k=5, num_terms=5)
            if expansion_tokens:
                st.info(f" *PRF Expanded Tokens:* {', '.join(expansion_tokens)}")

        results = []
        with st.spinner("Processing index tables..."):
            if algo_choice == "TF-IDF":
                results = tf_idf(query_tokens, expansion_terms=expansion_tokens, top_k=top_k, og_weight=og_weight, ex_weight=ex_weight)

            elif algo_choice == "BM25":
                results = bm25(query_tokens, expansion_terms=expansion_tokens, top_k=top_k, k1=k1, b=b, og_weight=og_weight, ex_weight=ex_weight)

            elif algo_choice == "BERT (Dense)":
                results = bert(working_query, top_k=top_k, compare_faiss=compare_faiss)

            elif algo_choice == "Hybrid (Serial)":
                bm25_base = bm25(query_tokens, expansion_terms=expansion_tokens, top_k=100, k1=k1, b=b, og_weight=og_weight, ex_weight=ex_weight)
                results = serial(working_query, bm25_base, top_k=top_k)

            elif algo_choice == "Hybrid (Parallel)":
                bm25_base = bm25(query_tokens, expansion_terms=expansion_tokens, top_k=100, k1=k1, b=b, og_weight=og_weight, ex_weight=ex_weight)
                bert_base = bert(working_query, top_k=100, compare_faiss=compare_faiss)
                results = parallel(bm25_base, bert_base, top_k=top_k, weight=hybrid_weight)

        st.subheader(f" Top Results matching '{algo_choice}' Strategy")

        if not results:
            st.warning("No documents retrieved matching the constraints.")
        else:
            for idx, (doc_id, score) in enumerate(results, 1):
                with st.container(border=True):
                    col1, col2 = st.columns([1.3, 5])  # Slightly widened col1 for horizontal space

                    with col1:
                        st.markdown(f"**ID:** `{doc_id}` | **Score:** `{score:.4f}`")

                    with col2:
                        with st.expander("Click here to expand the document"):
                            doc_text = fetch_document_text(doc_id)
                            st.write(doc_text)
with tab2:
    st.header("System Performance Metrics")
    st.write("Run evaluation against a ground-truth dataset subset to analyze metric variations.")

    eval_count = st.slider("Select evaluation query sample size", min_value=5, max_value=100, value=10, step=5)

    # New UI controls to match your updated function signatures
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        use_refine = st.checkbox("Enable Query Refinement", value=False)
    with col_cfg2:
        compare_faiss = st.checkbox("Compare FAISS Indexing", value=False)

    if st.button(" Run Batch Evaluation", type="primary"):
        with st.spinner("Executing query batches across selected frameworks..."):

            results_df, fig_metrics, fig_speed = rank_ui(
                queries=eval_count,
                use_refine=use_refine,
                compare_faiss=compare_faiss
            )

            st.subheader(" Final Metrics Output")
            st.dataframe(
                results_df.style.background_gradient(cmap="Blues", subset=["MAP", "P@10", "nDCG@10"]),
                use_container_width=True
            )

            st.subheader(" Performance Visualization")
            st.plotly_chart(fig_metrics, use_container_width=True)

            st.subheader(" Index Optimization Benchmarks")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(fig_speed, use_container_width=True)
            with col2:
                bf_time = fig_speed.data[0].y[0]
                faiss_time = fig_speed.data[0].y[1]
                speedup = bf_time / max(faiss_time, 0.00001)

                st.metric(label="Brute Force Execution", value=f"{bf_time:.4f}s")
                st.metric(label="FAISS-Indexed Execution", value=f"{faiss_time:.4f}s")
                st.success(f" FAISS yields a **{speedup:.1f}x** performance boost!")
