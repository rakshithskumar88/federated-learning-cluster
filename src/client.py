# src/client.py
import os
import sys
import io
import time  # NEW: Required for resilient network polling
import grpc
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))
import federated_pb2
import federated_pb2_grpc

CLIENT_ID = os.getenv("CLIENT_ID", "local_node")
SERVER_ADDR = os.getenv("SERVER_ADDR", "server:50052")

# ---------------------------------------------------------
# IDENTICAL NEURAL NETWORK ARCHITECTURE
# ---------------------------------------------------------
class FraudNet(nn.Module):
    def __init__(self):
        super(FraudNet, self).__init__()
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

# ---------------------------------------------------------
# DATA PIPELINE: Pulling from Hugging Face
# ---------------------------------------------------------
def fetch_and_prep_data():
    print(f"[DATA] {CLIENT_ID} connecting to Hugging Face Datasets...")
    hf_dataset = load_dataset("jyunyilin/credit-card-fraud-detection", split="train")
    df = hf_dataset.to_pandas()
    
    df = df.sample(n=2000, random_state=42)
    
    if "RTX4070" in CLIENT_ID:
        df = df.iloc[:1000]
    else:
        df = df.iloc[1000:]

    X = df.drop(columns=['Class']).values
    y = df['Class'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor

def train_local_model(global_state_dict):
    # ---------------------------------------------------------
    # STRICT HARDWARE ENFORCEMENT
    # ---------------------------------------------------------
    # 1. Verify we are running the AMD ROCm version of PyTorch
    if not torch.version.hip:
        raise RuntimeError("[FATAL] This PyTorch binary was not compiled with AMD ROCm/HIP support!")
    
    # 2. Verify the Kernel Fusion Driver (KFD) successfully mapped the GPU
    if not torch.cuda.is_available():
        raise RuntimeError("[FATAL] ROCm failed to initialize. The AMD GPU is locked by the host GUI or missing.")
    
    # 3. Lock the device to the AMD APU (PyTorch uses 'cuda' as the alias for HIP)
    device = torch.device("cuda")
    print(f"[HARDWARE] Hardware locked. Active Target: AMD ROCm ({torch.cuda.get_device_name(0)})")
    
    # Identify if we are running on NVIDIA (CUDA) or AMD (ROCm/HIP)
    compute_backend = "AMD ROCm" if torch.version.hip else ("NVIDIA CUDA" if torch.cuda.is_available() else "CPU")
    print(f"[HARDWARE] Hardware locked. Active Target: {compute_backend}")
    
    X_train, y_train = fetch_and_prep_data()
    X_train, y_train = X_train.to(device), y_train.to(device)
    
    model = FraudNet().to(device)
    model.load_state_dict(global_state_dict)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"[TRAINING] Commencing Backpropagation on {compute_backend}...")
    epochs = 5
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch == 0 or epoch == epochs - 1:
             print(f"   -> Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
             
        # MEMORY MANAGEMENT: Force VRAM cleanup after every epoch to prevent 
        # crashing GNOME or hitting APU shared memory limits.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"[TRAINING] {compute_backend} Deep Learning complete. Extracting weights.")
    
    model.to("cpu")
    return model.state_dict(), len(X_train)

# ---------------------------------------------------------
# NETWORK ORCHESTRATION: Resilient Polling
# ---------------------------------------------------------
def run_client():
    print(f"[NETWORK] Node '{CLIENT_ID}' locating Aggregation Server at {SERVER_ADDR}...")
    
    max_retries = 20
    for attempt in range(1, max_retries + 1):
        try:
            with grpc.insecure_channel(SERVER_ADDR) as channel:
                stub = federated_pb2_grpc.FederatedLearningStub(channel)
                request = federated_pb2.ModelRequest(client_id=CLIENT_ID)
                
                # The timeout prevents it from hanging infinitely if the server is offline
                print(f"[RPC] Pulling global model state (Attempt {attempt}/{max_retries})...")
                global_model_response = stub.GetGlobalModel(request, timeout=10)
                
                buffer = io.BytesIO(global_model_response.model_weights)
                global_state_dict = torch.load(buffer, weights_only=True)
                
                # Trigger actual Deep Learning on the GPU
                optimized_state_dict, local_n = train_local_model(global_state_dict)
                
                out_buffer = io.BytesIO()
                torch.save(optimized_state_dict, out_buffer)
                
                print("[RPC] Streaming optimized FraudNet weights back to server...")
                update_payload = federated_pb2.LocalUpdate(
                    client_id=CLIENT_ID,
                    data_size=local_n,
                    model_weights=out_buffer.getvalue()
                )
                
                ack = stub.SendLocalUpdate(update_payload)
                print(f"[RPC] Server Response: {ack.message}")
                
                # Success! Exit the function.
                return 
                
        except grpc.RpcError as e:
            print(f"[WARNING] Server unreachable. Sleeping for 15 seconds before retry...")
            time.sleep(15)
            
    print("[FATAL] Could not connect to Aggregator Server after maximum attempts.")

if __name__ == '__main__':
    run_client()
