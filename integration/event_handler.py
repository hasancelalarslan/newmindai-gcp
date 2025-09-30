import json
import pika
import grpc
import psycopg2

import integration.comment_service_pb2 as pb2
import integration.comment_service_pb2_grpc as pb2_grpc


class DBManager:
    """Persistent PostgreSQL connection with simple pooling."""
    def __init__(self, dbname, user, password, host="localhost", port="5432"):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cur = self.conn.cursor()
        self._ensure_table()

    def _ensure_table(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id SERIAL PRIMARY KEY,
                comment_id TEXT,
                classification TEXT,
                matched_topic TEXT,
                conclusion TEXT
            )
        """)
        self.conn.commit()

    def save_result(self, comment_id, classification, matched_topic, conclusion):
        self.cur.execute(
            """
            INSERT INTO results (comment_id, classification, matched_topic, conclusion)
            VALUES (%s, %s, %s, %s)
            """,
            (comment_id, classification, matched_topic, conclusion)
        )
        self.conn.commit()
        print("💾 Result saved to DB.")

    def close(self):
        self.cur.close()
        self.conn.close()


def consume_event():
    """RabbitMQ kuyruğunu dinle, gRPC server'a bağlan ve sonuçları kaydet."""
    # 🔹 DB bağlantısını 1 kez aç
    db = DBManager(
        dbname="commentsdb",
        user="newmindai",
        password="newmindai_pass",
        host="localhost",
        port="5432"
    )

    while True:
        try:
            # 🔹 RabbitMQ bağlantısı
            connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
            channel = connection.channel()
            channel.queue_declare(queue="comments")

            def callback(ch, method, properties, body):
                data = json.loads(body)
                print("📥 Received:", data)

                # gRPC server'a bağlan
                with grpc.insecure_channel("localhost:50051") as channel_grpc:
                    stub = pb2_grpc.CommentServiceStub(channel_grpc)
                    response = stub.AnalyzeComment(
                        pb2.CommentRequest(
                            comment_id=data["id"],
                            topic_text=data["topic"],
                            comment_text=data["comment"]
                        )
                    )

                    print("📌 Response:", response)

                    # 🔹 Tek connection üzerinden DB’ye kaydet
                    db.save_result(
                        response.comment_id,
                        response.classification,
                        response.matched_topic,
                        response.conclusion
                    )

            channel.basic_consume(queue="comments", on_message_callback=callback, auto_ack=True)
            print(" [*] Waiting for comments. To exit press CTRL+C")
            channel.start_consuming()

        except pika.exceptions.StreamLostError:
            print("⚠️ RabbitMQ connection lost. Reconnecting...")
            continue
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            db.close()
            connection.close()
            break


if __name__ == "__main__":
    consume_event()
