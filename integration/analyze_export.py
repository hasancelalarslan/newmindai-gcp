import pandas as pd
from pathlib import Path
import json

def analyze_export():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "final_integration_results.csv")

    metrics = {
        "total_records": len(df),
    }

    # Classification dağılımı
    if "classification" in df.columns:
        class_dist = df["classification"].value_counts().to_dict()
        metrics["classification_distribution"] = class_dist

    # Unique topic sayısı
    if "matched_topic" in df.columns:
        metrics["unique_matched_topics"] = int(df["matched_topic"].nunique())

    # Conclusion uzunluğu
    if "conclusion" in df.columns:
        df["conclusion_len"] = df["conclusion"].astype(str).apply(lambda x: len(x.split()))
        metrics["avg_conclusion_length_words"] = float(df["conclusion_len"].mean())

    # En çok yorum alan 5 topic
    if "matched_topic" in df.columns:
        top5 = df["matched_topic"].value_counts().head(5).to_dict()
        metrics["top5_topics_by_opinions"] = top5

    # JSON kaydet
    out_json = results_dir / "export_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # TXT rapor kaydet
    out_txt = results_dir / "export_metrics.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("📊 Export Metrics Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Total records: {metrics['total_records']}\n\n")

        if "classification_distribution" in metrics:
            f.write("Classification distribution:\n")
            for k,v in metrics["classification_distribution"].items():
                f.write(f"  {k:12s}: {v}\n")
            f.write("\n")

        if "unique_matched_topics" in metrics:
            f.write(f"Unique matched topics: {metrics['unique_matched_topics']}\n\n")

        if "avg_conclusion_length_words" in metrics:
            f.write(f"Average conclusion length: {metrics['avg_conclusion_length_words']:.2f} words\n\n")

        if "top5_topics_by_opinions" in metrics:
            f.write("Top 5 topics by number of matched opinions:\n")
            for k,v in metrics["top5_topics_by_opinions"].items():
                f.write(f"  {k}: {v}\n")

    print(f"✅ Metrics saved to:\n - {out_json}\n - {out_txt}")


if __name__ == "__main__":
    analyze_export()
