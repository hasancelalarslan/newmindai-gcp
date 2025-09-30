import pandas as pd
from pathlib import Path
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# ========= Config Loader =========
def load_config(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ========= Helpers =========
def trim(s: str, max_chars: int) -> str:
    if max_chars is None or len(s) <= max_chars:
        return s
    return (s[: max_chars - 1] + "…")

def build_prompt(topic_text: str, opinions_df: pd.DataFrame, per_opinion_char_limit: int, label_col: str = "type"):
    lines = []
    for _, row in opinions_df.iterrows():
        op = trim(str(row["text"]), per_opinion_char_limit)
        label = row[label_col] if label_col in row else "UNKNOWN"
        lines.append(f"- ({label}) {op}")
    opinions_str = "\n".join(lines)

    user_content = f"""
Task:
Summarize in 2–3 sentences the overall stance on this topic.  
- Always write a short paragraph.  
- Clearly say if the majority SUPPORTS, OPPOSES, or is MIXED.  
- Briefly explain why (mention common reasons).  
- Do not repeat the topic or opinions. Only output the conclusion summary.

Topic:
"{topic_text}"

Related opinions (labeled):
{opinions_str}
""".strip()

    return user_content

def load_model(model_id: str, device: str):
    dtype = torch.float16 if device == "cuda" else "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda" if device == "cuda" else "cpu",
    )
    model.eval()
    return tokenizer, model

def clean_summary(text: str) -> str:
    text = re.sub(r"(Topic:.*?Summary:)", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^Topic:.*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\n+", " ", text).strip()
    return text

@torch.inference_mode()
def generate_summary(tokenizer, model, user_prompt: str, max_new_tokens: int, temperature: float):
    messages = [
        {"role": "system", "content": "You are an assistant that summarizes social media opinions for analysts."},
        {"role": "user", "content": user_prompt},
    ]
    prompt_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    enc = tokenizer(
        prompt_chat,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return clean_summary(text) or "Summary not generated"

# ========= Runner =========
def run_generation(mode: str, topics, opinions, tokenizer, model, out_path, cfg):
    if out_path.exists():
        prev_df = pd.read_csv(out_path)
        done_ids = set(prev_df["topic_id"].astype(str))
        outputs = prev_df.to_dict("records")
        print(f"🔄 Resuming {mode} → {len(done_ids)} topics already saved in {out_path}")
    else:
        done_ids = set()
        outputs = []
        pd.DataFrame(columns=[
            "topic_id","topic_text","num_claims","num_evidence",
            "num_counterclaims","num_rebuttals","num_tokens","conclusion"
        ]).to_csv(out_path, index=False, encoding="utf-8")
        print(f"🆕 Starting fresh {mode} → empty CSV created at {out_path}")

    max_tokens           = int(cfg.get("max_tokens", 150))
    max_opinions         = int(cfg.get("max_opinions_per_topic", 8))
    per_opinion_char_lim = int(cfg.get("per_opinion_char_limit", 220))
    temperature          = float(cfg.get("temperature", 0.3))

    grouped = topics.groupby("topic_id")
    for idx, (topic_id, topic_group) in enumerate(tqdm(grouped, desc=f"Processing {mode} topics"), 1):
        if str(topic_id) in done_ids:
            continue

        topic_text = topic_group["text"].iloc[0]
        related = opinions[opinions["topic_id"] == topic_id]
        if related.empty:
            continue

        if len(related) > max_opinions:
            related = related.sample(n=max_opinions, random_state=42).reset_index(drop=True)

        if "type" in related.columns:
            label_col = "type"
        elif "pred_type" in related.columns:
            label_col = "pred_type"
        elif "true_type" in related.columns:   # ✅ eval dosyası için
            label_col = "true_type"
        else:
            label_col = None

        counts = related[label_col].value_counts().to_dict() if label_col else {}
        user_prompt = build_prompt(topic_text, related, per_opinion_char_lim)

        enc = tokenizer(user_prompt, return_tensors="pt", truncation=True)
        num_tokens = enc["input_ids"].shape[1]
        conclusion = generate_summary(tokenizer, model, user_prompt, max_new_tokens=max_tokens, temperature=temperature)

        outputs.append({
            "topic_id": topic_id,
            "topic_text": topic_text,
            "num_claims": counts.get("Claim", 0),
            "num_evidence": counts.get("Evidence", 0),
            "num_counterclaims": counts.get("Counterclaim", 0),
            "num_rebuttals": counts.get("Rebuttal", 0),
            "num_tokens": num_tokens,
            "conclusion": conclusion
        })

        pd.DataFrame(outputs).to_csv(out_path, index=False, encoding="utf-8")
        print(f"💾 [{mode}] Saved {len(outputs)} rows → {out_path}")

    print(f"📂 Final {mode} CSV available at: {out_path.resolve()}")


# ========= Main =========
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    # Load config
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "configs" / "config.yaml")
    gen_cfg = cfg["conclusion_generator"]

    interim_dir   = Path(cfg["paths"]["interim_dir"])
    results_dir   = Path(cfg["paths"]["results_dir"])
    conc_dir      = results_dir / "conc_generator"
    conc_dir.mkdir(parents=True, exist_ok=True)

    topics = pd.read_csv(interim_dir / "topics_clean.csv")

    # Load model once
    model_id = gen_cfg.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer, model = load_model(model_id, device)

    # --- Eval run ---
    opinions_eval = pd.read_csv(results_dir / "arg_classifier" / "opinions_eval_predictions.csv")
    out_eval = conc_dir / "conclusions_eval_generated.csv"
    run_generation("eval", topics, opinions_eval, tokenizer, model, out_eval, gen_cfg)

    # --- Infer run ---
    opinions_infer = pd.read_csv(results_dir / "arg_classifier" / "opinions_infer_predictions.csv")
    out_infer = conc_dir / "conclusions_infer_generated.csv"
    run_generation("infer", topics, opinions_infer, tokenizer, model, out_infer, gen_cfg)

if __name__ == "__main__":
    main()