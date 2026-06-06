# src/client.py
import grpc
import torch
import io
import sys
import os

# Ensure Python can locate our generated network classes
sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))
import federated_pb2
import federated_pb2_grpc

def train_local_model(global_weights, data_size=100, learning_rate=0.1):
    print("\n[COMPUTE] Moving weights to parallel execution space...")
    
    # ---------------------------------------------------------
    # HARDWARE ALLOCATION: Binding to NVIDIA RTX 4070 CUDA
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] Active Compute Target: {device.type.upper()}")
    
    # Clone the server's weights and push them directly into VRAM
    local_weights = global_weights['layer_1'].clone().to(device)
    
    # ---------------------------------------------------------
    # MATHEMATICAL OPTIMIZATION: Simulated Gradient Descent
    # ---------------------------------------------------------
    print(f"[MATH] Initial Local Tensors in VRAM: {local_weights.cpu().numpy()}")
    
    # Simulate turning the mathematical knobs based on local private data
    # We create a mock gradient (direction of optimization)
    mock_gradient = torch.tensor([-0.5, 1.2, -0.3]).to(device)
    
    # Formula: local_weight = local_weight - (learning_rate * gradient)
    local_weights = local_weights - (learning_rate * mock_gradient)
    
    print(f"[MATH] Optimized Local Tensors after training: {local_weights.cpu().numpy()}")
    
    # Bring the tensors back from GPU VRAM to system memory (CPU) for serialization
    return {'layer_1': local_weights.to("cpu")}, data_size

def run_client():
    # Establish high-speed TCP socket connection to our Aggregator Server
    print("[NETWORK] Connecting to Aggregation Server via gRPC channel...")
    with grpc.insecure_channel('localhost:50052') as channel:
        stub = federated_pb2_grpc.FederatedLearningStub(channel)
        
        # 1. Request the latest global model from the server
        print("[RPC] Pulling global model state...")
        request = federated_pb2.ModelRequest(client_id="node_sahakarnagar_01")
        global_model_response = stub.GetGlobalModel(request)
        
        # Deserialize the binary stream from the network back into a PyTorch Matrix
        buffer = io.BytesIO(global_model_response.model_weights)
        global_weights = torch.load(buffer, weights_only=True)
        print(f"[RPC] Global model for round {global_model_response.round_number} loaded successfully.")
        
        # 2. Trigger GPU training loop
        optimized_weights, local_n = train_local_model(
            global_weights, 
            data_size=250, 
            learning_rate=0.05
        )
        
        # 3. Serialize our optimized VRAM outputs back into bytes for transmission
        out_buffer = io.BytesIO()
        torch.save(optimized_weights, out_buffer)
        
        # 4. Beam the mathematical updates back to the server contract
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
