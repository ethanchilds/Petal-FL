import torch
import torch.nn as nn
import pandas as pd
import io
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def serialize(model, buffer):
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

def deserialize(model_data):
        return torch.load(io.BytesIO(model_data))

def load_data(path):
        df = pd.read_csv(path)
        X = torch.tensor(df[["feature1", "feature2"]].values, dtype=torch.float32)
        y = torch.tensor(df["target"].values, dtype=torch.float32).unsqueeze(1)
        return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

def train(client_id, model):
        path = 'fl_app/data/dataset_' + str(client_id) + '.csv'
        dataloader = load_data(path)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(20):
                running_loss = 0.0
                for X_batch, y_batch in dataloader:
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()