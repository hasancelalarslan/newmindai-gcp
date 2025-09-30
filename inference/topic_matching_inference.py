from pathlib import Path
import torch
from src.topic_matching import load_config, load_data, build_model, build_topic_bank, encode_texts, torch_search

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_topic_model = None
_topic_ids_bank = None
_topic_mat = None

def match_topic(comment: str, topic_text_hint: str = None) -> str:
    """Return the most relevant topic_id for a given comment."""
    global _topic_model, _topic_ids_bank, _topic_mat
    if _topic_model is None:
        project_root = Path(__file__).resolve().parents[1]
        cfg = load_config(project_root / "configs" / "config.yaml")
        tm_cfg = cfg["topic_matching"]

        retr_model = tm_cfg["retriever_model"]
        print(f"🔄 Loading topic matcher model: {retr_model}")
        _topic_model = build_model(retr_model)

        topics_df, _ = load_data(project_root)
        _topic_ids_bank, _topic_mat = build_topic_bank(
            topics_df, _topic_model, batch_size=tm_cfg["batch_size"]
        )

    opin_emb = encode_texts(_topic_model, [comment], batch_size=1)
    D, I = torch_search(_topic_mat, opin_emb, k=1)
    return _topic_ids_bank[I[0][0]]
    