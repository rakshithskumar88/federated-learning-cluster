# src/client.py
import os
import sys
import io
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
# CRITICAL FIX: Default to 'server:50052' for standard bridge networks
SERVER_ADDR = os.getenv("SERVER_ADDR", "server:50052")

# ---------------------------------------------------------
# NEURAL NETWORK ARCHITECTURE
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
    
    # We take a sample to accelerate the demonstration
    df = df.sample(n=2000, random_state=42)
    
    # Simulating Data Privacy Split (Bank A vs Bank B)
    if "RTX4070" in CLIENT_ID:
        df = df.iloc[:1000] # NVIDIA takes the first 1000 transactions
    else:
        df = df.iloc[1000:] # AMD takes the next 1000 transactions

    X = df.drop(columns=['Class']).values
    y = df['Class'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor

def train_local_model(global_state_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] Hardware locked. Active Target: {device.type.upper()}")
    
    X_train, y_train = fetch_and_prep_data()
    X_train, y_train = X_train.to(device), y_train.to(device)
    
    model = FraudNet().to(device)
    model.load_state_dict(global_state_dict)
    
    criterion = nn.BCELoss() # Binary Cross Entropy (Fraud vs Not Fraud)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"[TRAINING] Commencing Backpropagation on {device.type.upper()}...")
    epochs = 5
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch == 0 or epoch == epochs - 1:
             print(f"   -> Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    print("[TRAINING] Deep Learning complete. Extracting optimized network weights.")
    
    model.to("cpu")
    return model.state_dict(), len(X_train)

def run_client():
    print(f"[NETWORK] Node '{CLIENT_ID}' connecting to Aggregation Server at {SERVER_ADDR}...")
    with grpc.insecure_channel(SERVER_ADDR) as channel:
        stub = federated_pb2_grpc.FederatedLearningStub(channel)
        
        request = federated_pb2.ModelRequest(client_id=CLIENT_ID)
        global_model_response = stub.GetGlobalModel(request)
        
        buffer = io.BytesIO(global_model_response.model_weights)
        global_state_dict = torch.load(buffer, weights_only=True)
        
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

if __name__ == '__main__':
    run_client()
