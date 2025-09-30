import json
import pika
import pandas as pd
from pathlib import Path
import argparse


class Publisher:
    def __init__(self, host="localhost", queue="comments"):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host))
        self.channel = self.connection.channel()
        self.queue = queue
        self.channel.queue_declare(queue=queue)

    def publish_event(self, comment_id, topic, comment):
        msg = {"id": str(comment_id), "topic": topic, "comment": comment}
        self.channel.basic_publish(exchange="", routing_key=self.queue, body=json.dumps(msg))
        print("✅ Published:", msg)

    def close(self):
        self.connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Kaç kayıt publish edilecek")
    args = parser.parse_args()

    pub = Publisher()
    project_root = Path(__file__).resolve().parents[1]

    opinions_df = pd.read_csv(project_root / "data/interim/opinions_clean.csv")
    topics_df = pd.read_csv(project_root / "data/interim/topics_clean.csv")

    # Merge ile sadece eşleşen topic_id'ler alınır
    merged = opinions_df.merge(topics_df, on="topic_id", how="inner", suffixes=("_opinion", "_topic"))
    print(f"📊 Opinions: {len(opinions_df)}, Topics: {len(topics_df)}, Merged: {len(merged)}")

    # Limit uygula
    limited = merged.head(args.limit)

    for idx, row in limited.iterrows():
        pub.publish_event(f"interim_{idx}", row["text_topic"], row["text_opinion"])

    pub.close()
    print(f"🎉 Published {len(limited)} records (limit={args.limit})")
