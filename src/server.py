# src/server.py
import grpc
from concurrent import futures
import torch
import io
import sys
import os

# Ensure Python can locate our newly generated network classes
sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))
import federated_pb2
import federated_pb2_grpc

class FederatedServer(federated_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.current_round = 1
        
        # ---------------------------------------------------------
        # MATHEMATICAL STATE: The "Global Brain"
        # ---------------------------------------------------------
        # We initialize a 'blank' PyTorch Tensor. 
        # A Tensor is just a highly optimized matrix of numbers.
        # For this skeleton, we assume our AI has 3 parameters (weights).
        self.global_weights = {'layer_1': torch.tensor([0.0, 0.0, 0.0])}
        
        print("[INIT] Central Aggregator initialized. Math State: Blank.")

    # ---------------------------------------------------------
    # RPC ENDPOINT 1: Broadcast to Clients
    # ---------------------------------------------------------
    def GetGlobalModel(self, request, context):
        print(f"[RPC] Node {request.client_id} requested the global model.")
        
        # We cannot send a PyTorch Tensor directly over standard network sockets.
        # We must serialize the mathematical matrix into raw bytes.
        buffer = io.BytesIO()
        torch.save(self.global_weights, buffer)
        
        # Wrap the bytes in our strict gRPC Protocol Buffer contract
        return federated_pb2.GlobalModel(
            round_number=self.current_round,
            model_weights=buffer.getvalue()
        )

    # ---------------------------------------------------------
    # RPC ENDPOINT 2: Receive Client Gradients
    # ---------------------------------------------------------
    def SendLocalUpdate(self, request, context):
        # 1. Network to Math Translation
        # Convert the incoming byte stream back into a PyTorch mathematical matrix
        buffer = io.BytesIO(request.model_weights)
        client_weights = torch.load(buffer, weights_only=True)
        
        # 2. Extract Mathematical Weighting
        # n_k = How many data samples this specific node used
        n_k = request.data_size 
        
        print(f"[RPC] Received payload from {request.client_id}. Data points used: {n_k}")
        print(f"[MATH] Node Payload Tensors: {client_weights}")

        # Note: The actual Federated Averaging loop (multiplying these weights by n_k) 
        # will be implemented here once we have multiple clients sending data.
        
        return federated_pb2.UpdateAck(success=True, message="Mathematical payload integrated.")

# ---------------------------------------------------------
# INFRASTRUCTURE DEPLOYMENT
# ---------------------------------------------------------
def serve():
    # Define a threaded server capable of handling multiple remote nodes simultaneously
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedServer(), server)
    
    # Bind to all IPv4 and IPv6 interfaces on port 50052
    # This prepares the server to be mapped inside our Podman container later
    server.add_insecure_port('[::]:50052')
    server.start()
    
    print("========================================================")
    print(" Global Aggregator Active (CPU-Bound execution)")
    print(" Listening on [::]:50052 for mathematical payloads...")
    print("========================================================")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
