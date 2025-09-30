import pandas as pd
from pathlib import Path
import yaml
import re
import evaluate
from bert_score import score
from sklearn.metrics import confusion_matrix, classification_report
import json

# ========= Config Loader =========
def load_config(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ========= Helpers =========
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def extract_stance(text: str) -> str:
    """Extract stance from generated/ref summaries."""
    if not isinstance(text, str):
        return "UNKNOWN"
    t = text.lower()
    if re.search(r"\bsupport|\bfavor|\bagree", t):
        return "SUPPORT"
    if re.search(r"\boppose|\bagainst|\bdisagree", t):
        return "OPPOSE"
    if re.search(r"\bmixed|\bdivided|\bboth", t):
        return "MIXED"
    return "UNKNOWN"

def compute_rouge(preds, refs):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=preds, references=refs)

def compute_bleu(preds, refs):
    bleu = evaluate.load("bleu")
    references = [[r] for r in refs]
    return bleu.compute(predictions=preds, references=references)

def compute_bertscore(preds, refs):
    P, R, F1 = score(preds, refs, lang="en", rescale_with_baseline=True)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
        "f1_per_item": F1.tolist()
    }

# ========= Eval Runner =========
def run_eval(gen_path: Path, ref_path: Path, out_prefix: Path):
    print(f"\n=== Running evaluation for {gen_path.name} ===")

    gen_df = pd.read_csv(gen_path)
    ref_df = pd.read_csv(ref_path)

    if "conclusion" in gen_df.columns:
        gen_df = gen_df.rename(columns={"conclusion": "conclusion_gen"})
    if "text" in ref_df.columns:
        ref_df = ref_df.rename(columns={"text": "conclusion_ref"})

    merged = pd.merge(gen_df, ref_df, on="topic_id")
    print(f"✅ Loaded {len(merged)} aligned rows")

    preds = merged["conclusion_gen"].map(clean_text).tolist()
    refs  = merged["conclusion_ref"].map(clean_text).tolist()

    # --- Metrics ---
    rouge_results = compute_rouge(preds, refs)
    bleu_results  = compute_bleu(preds, refs)
    bert_results  = compute_bertscore(preds, refs)

    print("\n=== ROUGE ===")
    for k, v in rouge_results.items():
        print(f"{k}: {v:.4f}")

    print("\n=== BLEU ===")
    print(f"BLEU: {bleu_results['bleu']:.4f}")

    print("\n=== BERTScore ===")
    print(f"Precision: {bert_results['precision']:.4f}")
    print(f"Recall:    {bert_results['recall']:.4f}")
    print(f"F1:        {bert_results['f1']:.4f}")

    # --- Stance ---
    merged["stance_gen"] = merged["conclusion_gen"].map(extract_stance)
    merged["stance_ref"] = merged["conclusion_ref"].map(extract_stance)

    stance_acc = (merged["stance_gen"] == merged["stance_ref"]).mean()
    print("\n=== Stance Accuracy ===")
    print(f"Accuracy: {stance_acc:.4f}")

    labels = ["SUPPORT", "OPPOSE", "MIXED", "UNKNOWN"]
    cm = confusion_matrix(merged["stance_ref"], merged["stance_gen"], labels=labels)
    print("\n=== Confusion Matrix ===")
    print(pd.DataFrame(cm, index=[f"ref_{l}" for l in labels],
                          columns=[f"gen_{l}" for l in labels]))

    print("\n=== Classification Report ===")
    print(classification_report(merged["stance_ref"], merged["stance_gen"], labels=labels))

    # --- Save detailed merged CSV ---
    merged["rougeL"] = rouge_results["rougeL"]
    merged["bert_f1"] = bert_results["f1_per_item"]
    merged.to_csv(f"{out_prefix}_detailed.csv", index=False, encoding="utf-8")

    # --- Save summary JSON ---
    summary = {
        "rouge": rouge_results,
        "bleu": bleu_results,
        "bert": {k: v for k, v in bert_results.items() if k != "f1_per_item"},
        "stance_accuracy": stance_acc,
    }
    with open(f"{out_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📂 Saved detailed CSV → {out_prefix}_detailed.csv")
    print(f"📂 Saved summary JSON → {out_prefix}_summary.json")


# ========= Infer Runner =========
def run_infer(gen_path: Path, out_prefix: Path):
    print(f"\n=== Running stance-only analysis for {gen_path.name} ===")

    df = pd.read_csv(gen_path)

    if "conclusion" in df.columns:
        df = df.rename(columns={"conclusion": "conclusion_gen"})

    df["stance_gen"] = df["conclusion_gen"].map(extract_stance)

    stance_counts = df["stance_gen"].value_counts().to_dict()
    print("\n=== Stance Distribution ===")
    for stance, count in stance_counts.items():
        print(f"{stance}: {count}")

    # save
    df.to_csv(f"{out_prefix}_detailed.csv", index=False, encoding="utf-8")

    with open(f"{out_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"stance_distribution": stance_counts}, f, indent=2)

    print(f"\n📂 Saved stance-only detailed CSV → {out_prefix}_detailed.csv")
    print(f"📂 Saved stance-only summary JSON → {out_prefix}_summary.json")


# ========= Main =========
def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "configs" / "config.yaml")
    results_dir = Path(cfg["paths"]["results_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])

    ref_path = interim_dir / "conclusions_clean.csv"

    # Eval set
    run_eval(
        results_dir / "conc_generator" / "conclusions_eval_generated.csv",
        ref_path,
        results_dir / "evaluation_eval"
    )

    # Infer set
    infer_path = results_dir / "conc_generator" / "conclusions_infer_generated.csv"
    if infer_path.exists():
        run_infer(
            infer_path,
            results_dir / "evaluation_infer"
        )

if __name__ == "__main__":
    main()
