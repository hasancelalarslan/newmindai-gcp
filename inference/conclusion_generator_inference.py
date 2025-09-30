from pathlib import Path
import torch
import pandas as pd
from src.conclusion_generator import load_config, load_model, build_prompt, generate_summary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_conc_tokenizer = None
_conc_model = None
_conc_cfg = None

def generate_conclusion(topic_text: str, comment_text: str) -> str:
    """Generate a short conclusion for a topic given one comment."""
    global _conc_tokenizer, _conc_model, _conc_cfg
    if _conc_tokenizer is None or _conc_model is None:
        project_root = Path(__file__).resolve().parents[1]
        cfg = load_config(project_root / "configs" / "config.yaml")
        _conc_cfg = cfg["conclusion_generator"]

        model_id = _conc_cfg.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2")
        print(f"🔄 Loading conclusion generator model: {model_id}")
        _conc_tokenizer, _conc_model = load_model(model_id, DEVICE)

    df = pd.DataFrame([{"text": comment_text, "type": "Claim"}])
    user_prompt = build_prompt(
        topic_text,
        df,
        per_opinion_char_limit=int(_conc_cfg.get("per_opinion_char_limit", 220)),
        label_col="type"
    )
    return generate_summary(
        _conc_tokenizer,
        _conc_model,
        user_prompt,
        max_new_tokens=int(_conc_cfg.get("max_tokens", 150)),
        temperature=float(_conc_cfg.get("temperature", 0.3))
    )
