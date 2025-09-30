import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse

# ====== Labels ======
LABEL2ID = {"Claim": 0, "Evidence": 1, "Counterclaim": 2, "Rebuttal": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")

# ========= Config Loader =========
def load_config(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_inference(df, model, tokenizer, batch_size=16, max_len=256):
    texts = df["text"].tolist()
    preds = []
    print("🔎 Running classifier inference...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True,
                        return_tensors="pt", max_length=max_len).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "infer"], default="infer",
                        help="eval → processed/test, infer → interim/clean")
    args = parser.parse_args()

    # --- Paths ---
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "configs" / "config.yaml")

    model_dir = project_root / "results" / "arg_classifier" / "final_model"
    results_dir = project_root / "results" / "arg_classifier"

    if args.mode == "eval":
        opinions_path = project_root / "data" / "processed" / "opinions_test.csv"
        out_path = results_dir / "opinions_eval_predictions.csv"
    else:
        opinions_path = project_root / "data" / "interim" / "opinions_clean.csv"
        out_path = results_dir / "opinions_infer_predictions.csv"

    # --- Load opinions ---
    df = pd.read_csv(opinions_path)
    print(f"📄 Loaded {len(df)} opinions from {opinions_path}")

    # --- Load fine-tuned classifier ---
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    # --- Inference ---
    preds = run_inference(df, model, tokenizer)

    # --- Add predictions ---
    df["pred_type"] = [ID2LABEL[p] for p in preds]
    if args.mode == "eval" and "type" in df.columns:
        df.rename(columns={"type": "true_type"}, inplace=True)

    # --- Save ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved classified opinions → {out_path}")

    # --- Preview ---
    cols = ["text", "pred_type"] + (["true_type"] if "true_type" in df.columns else [])
    print("\n🔍 Sample predictions:")
    print(df.head(10)[cols])

if __name__ == "__main__":
    main()
