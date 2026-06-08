# src/server.py
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
        
        # NEW: State management for the FedAvg Algorithm
        self.expected_clients = 2
        self.client_updates = []
        
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
        print(f"[MATH] Node Payload Tensors: {client_weights['layer_1'].numpy()}")

        # NEW: Store the incoming weights until all expected clients report in
        self.client_updates.append((n_k, client_weights))
        
        # If both the NVIDIA and AMD nodes have reported, run the math!
        if len(self.client_updates) == self.expected_clients:
            self._aggregate_models()
            
        return federated_pb2.UpdateAck(success=True, message="Mathematical payload integrated.")

    def _aggregate_models(self):
        print("\n========================================================")
        print(f"[FED-AVG] Initiating Federated Averaging for Round {self.current_round}...")
        
        # Calculate N (Total data points across all clients)
        total_n = sum([n_k for n_k, _ in self.client_updates])
        
        # Create a blank slate for the new global brain
        new_global = {'layer_1': torch.zeros_like(self.global_weights['layer_1'])}
        
        # The FedAvg Calculus: Multiply each client's weight by its data fraction (n_k / N)
        for n_k, client_weights in self.client_updates:
            weight_fraction = n_k / total_n
            new_global['layer_1'] += client_weights['layer_1'] * weight_fraction
            
        self.global_weights = new_global
        print(f"[FED-AVG] Aggregation Complete! New Global Brain: {self.global_weights['layer_1'].numpy()}")
        print("========================================================\n")
        
        # Reset for the next round
        self.current_round += 1
        self.client_updates = []

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
