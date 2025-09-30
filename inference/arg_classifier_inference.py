from pathlib import Path
from transformers import pipeline
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
project_root = Path(__file__).resolve().parents[1]

_classifier_pipeline = None

def classify_argument(text: str) -> str:
    """Classify a comment as Claim/Evidence/Counterclaim/Rebuttal"""
    global _classifier_pipeline
    if _classifier_pipeline is None:
        model_dir = project_root / "results" / "arg_classifier" / "final_model"
        print(f"🔄 Loading classifier from {model_dir}")
        _classifier_pipeline = pipeline(
            "text-classification",
            model=str(model_dir),
            tokenizer=str(model_dir),
            device=0 if DEVICE == "cuda" else -1
        )
    result = _classifier_pipeline(text, truncation=True, max_length=256)[0]
    return result["label"]
