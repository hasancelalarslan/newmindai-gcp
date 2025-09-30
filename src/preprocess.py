import unicodedata
import re
import json
from pathlib import Path
import pandas as pd
import yaml
from langdetect import detect, DetectorFactory, LangDetectException
import torch

DetectorFactory.seed = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")

# ---------- Config ----------
def load_config(cfg_path: Path):
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------- Regexes ----------
URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
EMOJI_RE   = re.compile(
    "[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF"
    "\u2600-\u26FF\u2700-\u27BF]+", flags=re.UNICODE)

# ---------- Helpers ----------
def lower_by_lang(text: str, lang: str) -> str:
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFC", text)
    if lang == "tr":
        s = s.replace("I", "ı").replace("İ", "i")
        return s.lower()
    return s.lower()

def clean_text(text: str, placeholders: dict, lang: str) -> str:
    s = lower_by_lang(text, lang)
    if placeholders.get("url", True): s = URL_RE.sub("[URL]", s)
    if placeholders.get("mention", True): s = MENTION_RE.sub("[MENTION]", s)
    if placeholders.get("hashtag", True): s = HASHTAG_RE.sub("[HASHTAG]", s)
    if placeholders.get("emoji", True): s = EMOJI_RE.sub("[EMOJI]", s)
    return re.sub(r"\s+", " ", s).strip()

def detect_lang(text: str) -> str:
    try:
        if not isinstance(text, str):
            return "unk"
        if len(text) < 10 or len(text.split()) < 2:
            return "unk"
        lang = detect(text)
        return lang if lang in {"tr", "en"} else "unk"
    except LangDetectException:
        return "unk"

def token_count(text: str) -> int:
    return len(str(text).split())

# ---------- Preprocessing ----------
def normalize_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.rename(columns={c.lower(): c for c in df.columns})
    if "text" not in df.columns:
        raise ValueError(f"{name} missing text column")
    return df

def preprocess_df(df: pd.DataFrame, cfg: dict, name="") -> pd.DataFrame:
    df = normalize_columns(df, name)
    orig_len = len(df)

    df = df.dropna(subset=["text", "topic_id"]) if "topic_id" in df.columns else df.dropna(subset=["text"])
    after_na = len(df)

    df["lang"] = df["text"].astype(str).apply(detect_lang)
    df["text"] = [clean_text(t, cfg["preprocess"]["placeholders"], l) for t, l in zip(df["text"], df["lang"])]
    df["tok_len"] = df["text"].apply(token_count)

    df = df[df["tok_len"] >= cfg["preprocess"]["filters"]["min_tokens"]]
    df = df[df["lang"].isin(cfg["preprocess"]["filters"]["allowed_langs"])]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"📊 {name}: {orig_len} → {after_na} (dropna) → {len(df)} (final)")
    return df

# ---------- Splits ----------
def save_splits(df: pd.DataFrame, out_dir: Path, ratios=(0.8, 0.1, 0.1), seed=42):
    topics = df["topic_id"].astype(str).unique().tolist()
    rng = pd.Series(topics).sample(frac=1, random_state=seed).tolist()

    n_train = int(len(rng) * ratios[0])
    n_val = int(len(rng) * ratios[1])
    train_ids, val_ids, test_ids = set(rng[:n_train]), set(rng[n_train:n_train+n_val]), set(rng[n_train+n_val:])

    splits = {
        "train": df[df["topic_id"].astype(str).isin(train_ids)],
        "val":   df[df["topic_id"].astype(str).isin(val_ids)],
        "test":  df[df["topic_id"].astype(str).isin(test_ids)],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in splits.items():
        v.to_csv(out_dir / f"opinions_{k}.csv", index=False)

    return {k: int(len(v)) for k, v in splits.items()}

# ---------- Stats ----------
def save_stats(dfs: dict, out_path: Path):
    stats = {}
    for name, df in dfs.items():
        stats[name] = {
            "rows": len(df),
            "avg_len": float(df["tok_len"].mean()) if "tok_len" in df else None,
            "min_len": int(df["tok_len"].min()) if "tok_len" in df else None,
            "max_len": int(df["tok_len"].max()) if "tok_len" in df else None,
            "langs": df["lang"].value_counts().to_dict() if "lang" in df else {}
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

# ---------- CLI ----------
def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "configs" / "config.yaml")

    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"]); interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(cfg["paths"]["processed_dir"]); processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"]); artifacts_dir.mkdir(parents=True, exist_ok=True)

    # load raw
    topics = pd.read_csv(raw_dir / "topics.csv")
    opinions = pd.read_csv(raw_dir / "opinions.csv")
    conclusions = pd.read_csv(raw_dir / "conclusions.csv")

    # preprocess
    topics_clean = preprocess_df(topics, cfg, "topics")
    opinions_clean = preprocess_df(opinions, cfg, "opinions")
    conclusions_clean = preprocess_df(conclusions, cfg, "conclusions")

    # save interim
    topics_clean.to_csv(interim_dir / "topics_clean.csv", index=False)
    opinions_clean.to_csv(interim_dir / "opinions_clean.csv", index=False)
    conclusions_clean.to_csv(interim_dir / "conclusions_clean.csv", index=False)

    # save processed
    split_counts = save_splits(opinions_clean, processed_dir)
    topics_clean.to_csv(processed_dir / "topics_processed.csv", index=False)
    conclusions_clean.to_csv(processed_dir / "conclusions_processed.csv", index=False)

    # artifacts
    qa_report = {
        "topics_raw": len(topics),
        "opinions_raw": len(opinions),
        "conclusions_raw": len(conclusions),
        "topics_clean": len(topics_clean),
        "opinions_clean": len(opinions_clean),
        "conclusions_clean": len(conclusions_clean),
        "split_counts": split_counts
    }
    with open(artifacts_dir / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump(qa_report, f, indent=2, ensure_ascii=False)

    save_stats(
        {"topics": topics_clean, "opinions": opinions_clean, "conclusions": conclusions_clean},
        artifacts_dir / "data_stats.json"
    )

    print("✅ Preprocessing complete.")
    print("Interim:", interim_dir)
    print("Processed:", processed_dir)
    print("Artifacts:", artifacts_dir)

if __name__ == "__main__":
    main()
