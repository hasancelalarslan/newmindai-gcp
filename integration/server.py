import grpc
from concurrent import futures

import integration.comment_service_pb2 as pb2
import integration.comment_service_pb2_grpc as pb2_grpc

# ✅ Real inference imports
from inference.arg_classifier_inference import classify_argument
from inference.topic_matching_inference import match_topic
from inference.conclusion_generator_inference import generate_conclusion


class CommentService(pb2_grpc.CommentServiceServicer):
    def AnalyzeComment(self, request, context):
        # Call real ML pipelines
        classification = classify_argument(request.comment_text)
        matched_topic = match_topic(request.comment_text, request.topic_text)
        conclusion = generate_conclusion(request.topic_text, request.comment_text)

        return pb2.CommentResponse(
            comment_id=request.comment_id,
            classification=classification,
            matched_topic=matched_topic,
            conclusion=conclusion
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_CommentServiceServicer_to_server(CommentService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("🚀 gRPC server running on port 50051...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
