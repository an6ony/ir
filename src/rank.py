import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ir_measures
from ir_measures import Qrel, ScoredDoc
from ir_measures import MAP, P, Recall, nDCG
import matplotlib.pyplot as plt
import src.utils.paths as pth
from src.retrieve.tf_idf import tf_idf
from src.retrieve.bm25 import bm25
from src.retrieve.bert import bert
from src.retrieve.hybrid import serial, parallel
from src.refine import prf
from src.preprocess import preprocess_query

def load_evaluation_data(count):
    queries_df = pd.read_csv(pth.DATA_QUERIES, sep="\t", names=["query_id", "query_text"], header=0)
    queries_df = queries_df.head(count)
    query_lookup = dict(zip(queries_df["query_id"].astype(str), queries_df["query_text"].astype(str)))

    qrels_df = pd.read_csv(pth.DATA_QRELS, sep="\t", names=["query_id", "doc_id", "relevance"], header=0)
    qrels_df = qrels_df[qrels_df["query_id"].astype(str).isin(query_lookup.keys())]
    qrels = [Qrel(str(q), str(d), int(r)) for q, d, r in qrels_df.itertuples(index=False)]

    queries = len(query_lookup)
    batch_texts = [None] * queries
    q_ids = list(query_lookup.keys())
    batch_texts = list(query_lookup.values())

    runs = {
        "TF-IDF": {}, "BM25": {}, "BERT": {}, "Hybrid Serial": {}, "Hybrid Parallel": {},
        "Refined TF-IDF": {}, "Refined BM25": {}, "Refined Serial": {}, "Refined Parallel": {}
    }

    for q_id, q_text in query_lookup.items():
        print(q_id, q_text)
        pre = preprocess_query(q_text)
        runs["TF-IDF"][q_id] = tf_idf(pre)
        runs["BM25"][q_id] = bm25(pre)
        srf = prf(pre, runs["BM25"][q_id])
        runs["Refined TF-IDF"][q_id] = tf_idf(pre, srf)
        runs["Refined BM25"][q_id] = bm25(pre, srf)

    start_bf = time.time()
    bert_batch = bert(batch_texts, use_faiss=False)
    time_bf = time.time() - start_bf

    start_faiss = time.time()
    _ = bert(batch_texts, use_faiss=True)
    time_faiss = time.time() - start_faiss

    speed_data = { "Brute Force": time_bf, "FAISS Index": time_faiss }

    bm25_order = [runs["BM25"][q_id] for q_id in q_ids]
    bm25_refined_order = [runs["Refined BM25"][q_id] for q_id in q_ids]
    serial_batch = serial(batch_texts, bm25_order)
    serial_refined_batch = serial(batch_texts, bm25_refined_order)

    for i, q_id in enumerate(q_ids):
        runs["BERT"][q_id] = bert_batch[i]
        runs["Hybrid Serial"][q_id] = serial_batch[i]
        runs["Refined Serial"][q_id] = serial_refined_batch[i]
        runs["Hybrid Parallel"][q_id] = parallel(runs["BM25"][q_id], runs["BERT"][q_id])
        runs["Refined Parallel"][q_id] = parallel(runs["Refined BM25"][q_id], runs["BERT"][q_id])

    formatted_runs = {}
    for model_name, run_data in runs.items():
        scored_docs = []
        for q_id, results in run_data.items():
            for d_id, score in results:
                scored_docs.append(ScoredDoc(str(q_id), str(d_id), float(score)))
        formatted_runs[model_name] = scored_docs

    return qrels, formatted_runs, speed_data

def evaluate_models(qrels, formatted_runs):
    map_m = MAP
    p10_m = P@10
    rec10_m = Recall@10
    ndcg10_m = nDCG@10
    metrics = [map_m, p10_m, rec10_m, ndcg10_m]
    results_list = []
    print("\nComputing evaluation metrics...")
    for model_name, run in formatted_runs.items():
        eval_dict = ir_measures.calc_aggregate(metrics, qrels, run)
        results_list.append({
            'Model': model_name,
            'MAP': eval_dict[map_m],
            'P@10': eval_dict[p10_m],
            'Recall@10': eval_dict[rec10_m],
            'nDCG@10': eval_dict[ndcg10_m]
        })
    return pd.DataFrame(results_list)

def plot_model_comparison(results_df):
    results_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
    plt.title('Retrieval Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_faiss_speed_comparison(speed_data):
    fig, ax = plt.subplots(figsize=(5, 5))
    methods = list(speed_data.keys())
    times = list(speed_data.values())
    ax.bar(methods, times, color=['crimson', 'lightseagreen'], width=0.5)
    ax.set_title('BERT Retrieval Time: Brute Force vs FAISS', fontsize=12, fontweight='bold')
    ax.set_xlabel('Search Method', fontsize=10)
    ax.set_ylabel('Execution Time (seconds)', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.show()

def rank(queries=100):
    qrels, formatted_runs, speed_data = load_evaluation_data(queries)
    results_df = evaluate_models(qrels, formatted_runs)
    print("\n\t=== PERFORMANCE COMPARISON TABLE ===")
    print(results_df.to_string(index=False))
    plot_model_comparison(results_df)
    plot_faiss_speed_comparison(speed_data)

def plot_model_comparison_ui(results_df):
    df_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig = px.bar(
        df_melted, x='Model', y='Score', color='Metric', barmode='group',
        title='Retrieval Model Performance Comparison',
        labels={'Score': 'Value', 'Model': 'Retrieval Model'}
    )
    fig.update_layout(xaxis_tickangle=-45, margin=dict(b=100))
    return fig

def plot_faiss_speed_comparison_ui(speed_data):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(speed_data.keys()), y=list(speed_data.values()),
        marker_color=['#DC143C', '#20B2AA'], width=0.4
    ))
    fig.update_layout(
        title='BERT Retrieval Time: Brute Force vs FAISS',
        xaxis_title='Search Method', yaxis_title='Execution Time (seconds)',
        template='plotly_white'
    )
    return fig

def rank_ui(queries=100):
    qrels, formatted_runs, speed_data = load_evaluation_data(queries)
    results_df = evaluate_models(qrels, formatted_runs)
    fig_metrics = plot_model_comparison_ui(results_df)
    fig_speed = plot_faiss_speed_comparison_ui(speed_data)
    return results_df, fig_metrics, fig_speed