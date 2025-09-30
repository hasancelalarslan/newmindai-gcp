"""
Enhanced Topic Matching with flexible retrievers + reranking

- Retriever: from config (E5-large-v2, all-mpnet, or others)
- Reranker: optional cross-encoder stage
- ANN: Torch-based cosine similarity on GPU (no FAISS dependency)
- Outputs: diagnostics CSV + embeddings into results_dir/topic_matching/
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import yaml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")


# ---------- Config ----------
def load_config(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- Data ----------
def load_data(project_root: Path):
    topics = pd.read_csv(project_root / "data/interim/topics_clean.csv")
    opinions = pd.read_csv(project_root / "data/interim/opinions_clean.csv")
    topics["topic_id"] = topics["topic_id"].astype(str)
    opinions["topic_id"] = opinions["topic_id"].astype(str)
    return topics, opinions


def build_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=DEVICE)


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype("float32")


def build_topic_bank(topics_df: pd.DataFrame, model: SentenceTransformer, batch_size: int):
    grouped = topics_df.groupby("topic_id")["text"].apply(list).reset_index()
    topic_ids = grouped["topic_id"].tolist()
    all_texts = [text for texts in grouped["text"] for text in texts]
    # Encode all topic texts in one batch (single progress bar)
    all_embs = encode_texts(model, all_texts, batch_size=batch_size)
    # Split embeddings per topic
    topic_vecs = []
    idx = 0
    for texts in grouped["text"]:
        emb = all_embs[idx:idx+len(texts)]
        topic_vecs.append(emb.mean(axis=0))
        idx += len(texts)
    topic_mat = np.vstack(topic_vecs).astype("float32")

    # renormalize
    norms = np.linalg.norm(topic_mat, axis=1, keepdims=True) + 1e-12
    topic_mat = topic_mat / norms
    return topic_ids, topic_mat


# ---------- Torch-based Retrieval ----------
def torch_search(corpus_emb: np.ndarray, query_emb: np.ndarray, k: int):
    # send to GPU if available
    corpus = torch.tensor(corpus_emb, device=DEVICE)
    queries = torch.tensor(query_emb, device=DEVICE)

    sims = torch.matmul(queries, corpus.T)  # [n_queries, n_topics]
    D, I = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)

    return D.cpu().numpy(), I.cpu().numpy()


def match_opinions(opinions_df: pd.DataFrame,
                   topic_mat: np.ndarray,
                   model: SentenceTransformer,
                   batch_size: int,
                   k: int):
    texts = opinions_df["text"].tolist()
    opin_emb = encode_texts(model, texts, batch_size=batch_size)
    D, I = torch_search(topic_mat, opin_emb, k)
    return D, I, opin_emb


# ---------- Metrics ----------
def recall_at_k(true_ids, pred_ids_ranked, k=1):
    hits = sum(1 for t, preds in zip(true_ids, pred_ids_ranked) if t in preds[:k])
    return hits / len(true_ids) if len(true_ids) > 0 else 0.0


def mean_reciprocal_rank(true_ids, pred_ids_ranked):
    total = 0.0
    for t, preds in zip(true_ids, pred_ids_ranked):
        try:
            rank = preds.index(t) + 1
            total += 1.0 / rank
        except ValueError:
            pass
    return total / len(true_ids) if len(true_ids) > 0 else 0.0


def evaluate_ks(true_ids, pred_ids_ranked, ks=(1, 5, 10, 20)):
    scores = {}
    for k in ks:
        scores[f"Recall@{k}"] = recall_at_k(true_ids, pred_ids_ranked, k=k)
    scores["MRR"] = mean_reciprocal_rank(true_ids, pred_ids_ranked)
    return scores


# ---------- Reranking ----------
def rerank(opinions_df, I, topic_ids_bank, topics_texts, reranker, topk_rerank: int, batch_size: int = 32):
    new_rankings = []
    cap = max(1, topk_rerank)
    for op_text, cand_indices in zip(opinions_df["text"], I):
        cand_ids = [topic_ids_bank[j] for j in cand_indices]
        cand_texts = [topics_texts[cid] for cid in cand_ids]

        pairs = [(op_text, t) for t in cand_texts]
        scores = reranker.predict(pairs, batch_size=batch_size)  # use batch_size from config

        ranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)
        new_rankings.append([cid for cid, _ in ranked[:cap]])
    return new_rankings


# ---------- Diagnostics ----------
def save_diagnostics_csv(opinions_df, pred_ids_ranked, out_path: Path, top_show: int = 5):
    rows = []
    true_ids = opinions_df["topic_id"].tolist()
    texts = opinions_df["text"].tolist()

    for idx, (preds_ids, t_true, text) in enumerate(zip(pred_ids_ranked, true_ids, texts)):
        try:
            true_rank = preds_ids.index(t_true) + 1
        except ValueError:
            true_rank = -1

        row = {
            "opinion_id": opinions_df.iloc[idx].get("id", idx),
            "true_topic_id": t_true,
            "true_rank_in_topK": true_rank,
            "hit@1": int(t_true in preds_ids[:1]),
            "hit@5": int(t_true in preds_ids[:5]),
            "hit@10": int(t_true in preds_ids[:10]),
            "hit@20": int(t_true in preds_ids[:20]),
            "text": text,
        }
        for n in range(top_show):
            if n < len(preds_ids):
                row[f"pred{n+1}_topic_id"] = preds_ids[n]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_df.shape[0]


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "configs" / "config.yaml")

    tm_cfg = cfg["topic_matching"]
    results_dir = Path(cfg["paths"].get("results_dir", "results"))
    out_dir = results_dir / "topic_matching"
    out_dir.mkdir(parents=True, exist_ok=True)

    retrievers = [tm_cfg["retriever_model"]] + tm_cfg.get("alt_retrievers", [])
    for retr_model in retrievers:
        print(f"\n🚀 Running topic matching with retriever: {retr_model}")
        retriever = build_model(retr_model)

        topics_df, opinions_df = load_data(project_root)

        # Encode topics
        topic_ids_bank, topic_mat = build_topic_bank(
            topics_df, retriever, batch_size=tm_cfg["batch_size"]
        )
        topics_texts = topics_df.groupby("topic_id")["text"].first().to_dict()
        print(f"✅ Topic bank ready with {topic_mat.shape[0]} topics")

        # Encode opinions + retrieve
        D, I, opin_emb = match_opinions(
            opinions_df,
            topic_mat,
            model=retriever,
            batch_size=tm_cfg["batch_size"],
            k=int(tm_cfg["topk_retrieve"]),
        )
        pred_ids_ranked = [[topic_ids_bank[j] for j in row] for row in I]

        # Optional reranker
        if tm_cfg.get("use_reranker", False):
            reranker_model = tm_cfg["reranker_model"]
            print(f"Loading reranker model: {reranker_model}")
            reranker = CrossEncoder(reranker_model, device=DEVICE)
            topk_rerank = int(tm_cfg.get("topk_rerank", min(20, int(tm_cfg["topk_retrieve"]))))
            pred_ids_ranked = rerank(
                opinions_df,
                I,
                topic_ids_bank,
                topics_texts,
                reranker,
                topk_rerank=topk_rerank,
            )
            eval_cap = topk_rerank
        else:
            eval_cap = int(tm_cfg["topk_retrieve"])

        # Evaluate
        true_ids = opinions_df["topic_id"].tolist()
        capped_preds = [preds[:eval_cap] for preds in pred_ids_ranked]
        scores = evaluate_ks(true_ids, capped_preds, ks=(1, 5, 10, 20))

        print("=== Evaluation Metrics ===")
        for k, v in scores.items():
            print(f"{k:10s}: {v:.3f}")

        # Save outputs (separate per retriever)
        retriever_name = retr_model.split("/")[-1]
        retr_out_dir = out_dir / retriever_name
        retr_out_dir.mkdir(parents=True, exist_ok=True)

        diag_path = retr_out_dir / "diagnostics_top5.csv"
        save_diagnostics_csv(opinions_df, capped_preds, diag_path, top_show=min(5, eval_cap))

        np.save(retr_out_dir / "topic_embeddings.npy", topic_mat)
        np.save(retr_out_dir / "opinions_embeddings.npy", opin_emb)

        # Save metrics
        import json
        with open(retr_out_dir / "metrics.json", "w") as f:
            json.dump(scores, f, indent=2)

        print(f"🗂 Results saved under {retr_out_dir}")



if __name__ == "__main__":
    main()
