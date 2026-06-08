# src/server.py
import grpc
from concurrent import futures
import torch
import torch.nn as nn
import io
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))
import federated_pb2
import federated_pb2_grpc

# ---------------------------------------------------------
# THE NEURAL NETWORK ARCHITECTURE
# ---------------------------------------------------------
class FraudNet(nn.Module):
    def __init__(self):
        super(FraudNet, self).__init__()
        # 30 input features from the Hugging Face Fraud Dataset (V1-V28, Time, Amount)
        self.fc1 = nn.Linear(30, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return self.sigmoid(out)

class FederatedServer(federated_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.current_round = 1
        self.expected_clients = 2
        self.client_updates = []
        
        # Initialize the Global Brain
        self.global_model = FraudNet()
        print("[INIT] Central Aggregator initialized. Global FraudNet Model created.")

    def GetGlobalModel(self, request, context):
        print(f"[RPC] Node {request.client_id} requested the global model.")
        buffer = io.BytesIO()
        # Serialize the entire neural network state dictionary
        torch.save(self.global_model.state_dict(), buffer)
        return federated_pb2.GlobalModel(
            round_number=self.current_round,
            model_weights=buffer.getvalue()
        )

    def SendLocalUpdate(self, request, context):
        buffer = io.BytesIO(request.model_weights)
        client_state_dict = torch.load(buffer, weights_only=True)
        n_k = request.data_size 
        
        print(f"[RPC] Received payload from {request.client_id}. Transactions processed: {n_k}")
        self.client_updates.append((n_k, client_state_dict))
        
        if len(self.client_updates) == self.expected_clients:
            self._aggregate_models()
            
        return federated_pb2.UpdateAck(success=True, message="FraudNet payload integrated.")

    def _aggregate_models(self):
        print("\n========================================================")
        print(f"[FED-AVG] Initiating Federated Averaging for Round {self.current_round}...")
        
        total_n = sum([n_k for n_k, _ in self.client_updates])
        
        # Create a blank state dictionary based on the model structure
        new_state_dict = self.global_model.state_dict()
        for key in new_state_dict.keys():
            new_state_dict[key] = torch.zeros_like(new_state_dict[key], dtype=torch.float32)
        
        # FedAvg: Weighted average of all neural network layers
        for n_k, client_state_dict in self.client_updates:
            weight_fraction = n_k / total_n
            for key in new_state_dict.keys():
                new_state_dict[key] += client_state_dict[key] * weight_fraction
                
        # Load the newly averaged knowledge back into the global model
        self.global_model.load_state_dict(new_state_dict)
        print(f"[FED-AVG] Aggregation Complete! Global FraudNet is now smarter.")
        print("========================================================\n")
        
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
