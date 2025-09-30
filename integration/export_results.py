import psycopg2
import pandas as pd
from pathlib import Path
import argparse


def export_results(prefix=None, last_n=None):
    project_root = Path(__file__).resolve().parents[1]

    conn = psycopg2.connect(
        dbname="commentsdb",
        user="newmindai",
        password="newmindai_pass",
        host="localhost",
        port="5432"
    )

    query = "SELECT * FROM results ORDER BY id;"
    df = pd.read_sql(query, conn)
    conn.close()

    # Filter by prefix (e.g. interim_)
    if prefix:
        df = df[df["comment_id"].astype(str).str.startswith(prefix)]
        print(f"📊 Filtered by prefix='{prefix}' → {len(df)} rows")

    # Keep only last N
    if last_n:
        df = df.tail(last_n)
        print(f"📊 Keeping only last {last_n} rows")

    out_path = project_root / "results" / "final_integration_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Export completed → {out_path}")

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default=None, help="Filter by comment_id prefix (e.g., interim_)")
    parser.add_argument("--last_n", type=int, default=None, help="Export only last N rows")
    args = parser.parse_args()

    export_results(prefix=args.prefix, last_n=args.last_n)
