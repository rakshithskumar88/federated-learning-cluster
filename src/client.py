import grpc
import torch
import io
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))
import federated_pb2
import federated_pb2_grpc

def train_local_model(global_weights, data_size=100, learning_rate=0.1):
    print("\n[COMPUTE] Moving weights to parallel execution space...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] Active Compute Target: {device.type.upper()}")
    
    local_weights = global_weights['layer_1'].clone().to(device)
    print(f"[MATH] Initial Local Tensors in VRAM: {local_weights.cpu().numpy()}")
    
    mock_gradient = torch.tensor([-0.5, 1.2, -0.3]).to(device)
    local_weights = local_weights - (learning_rate * mock_gradient)
    
    print(f"[MATH] Optimized Local Tensors after training: {local_weights.cpu().numpy()}")
    return {'layer_1': local_weights.to("cpu")}, data_size

def run_client():
    print("[NETWORK] Connecting to Aggregation Server via gRPC channel...")
    with grpc.insecure_channel('server:50052') as channel:
        stub = federated_pb2_grpc.FederatedLearningStub(channel)
        
        print("[RPC] Pulling global model state...")
        request = federated_pb2.ModelRequest(client_id="node_sahakarnagar_01")
        global_model_response = stub.GetGlobalModel(request)
        
        buffer = io.BytesIO(global_model_response.model_weights)
        global_weights = torch.load(buffer, weights_only=True)
        print(f"[RPC] Global model for round {global_model_response.round_number} loaded successfully.")
        
        optimized_weights, local_n = train_local_model(global_weights, data_size=250, learning_rate=0.05)
        
        out_buffer = io.BytesIO()
        torch.save(optimized_weights, out_buffer)
        
        print("[RPC] Streaming local mathematical adjustments back to server...")
        update_payload = federated_pb2.LocalUpdate(
            client_id="node_sahakarnagar_01",
            data_size=local_n,
            model_weights=out_buffer.getvalue()
        )
        
        ack = stub.SendLocalUpdate(update_payload)
        print(f"[RPC] Server Response: {ack.message}")

if __name__ == '__main__':
    run_client()
