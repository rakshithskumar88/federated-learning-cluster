import grpc
from concurrent import futures
import torch
import io
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))
import federated_pb2
import federated_pb2_grpc

class FederatedServer(federated_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.current_round = 1
        self.global_weights = {'layer_1': torch.tensor([0.0, 0.0, 0.0])}
        print("[INIT] Central Aggregator initialized. Math State: Blank.")

    def GetGlobalModel(self, request, context):
        print(f"[RPC] Node {request.client_id} requested the global model.")
        buffer = io.BytesIO()
        torch.save(self.global_weights, buffer)
        return federated_pb2.GlobalModel(
            round_number=self.current_round,
            model_weights=buffer.getvalue()
        )

    def SendLocalUpdate(self, request, context):
        buffer = io.BytesIO(request.model_weights)
        client_weights = torch.load(buffer, weights_only=True)
        n_k = request.data_size 
        print(f"[RPC] Received payload from {request.client_id}. Data points used: {n_k}")
        print(f"[MATH] Node Payload Tensors: {client_weights}")
        return federated_pb2.UpdateAck(success=True, message="Mathematical payload integrated.")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedServer(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    print("========================================================")
    print(" Global Aggregator Active (CPU-Bound execution)")
    print(" Listening on [::]:50052 for mathematical payloads...")
    print("========================================================")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
