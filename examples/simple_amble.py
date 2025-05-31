# Necessary for federated learning
from fl.server_app.amble_server import FedLearnServer
from fl.client_app.amble_client import FedLearnClient
from fl.build_fl.config import Config, set_config
from fl.build_fl.run_fl import run_fed_learning

# Necessary for machine learning
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_simple_dataloader(batch_size = 16):
    path = 'data/dataset_' + str(1) + '.csv'
    df = pd.read_csv(path)
    X = torch.tensor(df[["feature1", "feature2"]].values, dtype=torch.float32)
    y = torch.tensor(df["target"].values, dtype=torch.float32).unsqueeze(1)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    return dataloader


def train_simpleNN(model, dataloader, epochs, lr = 0.01):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for _ in range(epochs):
        running_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

def evaluate_simpleNN(model):
    df = pd.read_csv("data/dataset_1.csv")
    X = torch.tensor(df[["feature1", "feature2"]].values, dtype=torch.float32)
    y_true = df["target"].values

    model.eval()
    with torch.no_grad():
        y_pred = model(X).numpy().flatten()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)


    print(f"Evaluation Results — MSE: {mse:.4f}, R²: {r2:.4f}")
    return mse, r2

if __name__ == "__main__":

    config = Config(
        max_clients = 2,
        train_iterations = 20,
        epochs = 5,
        learning_rate = 0.01,
        train_function = train_simpleNN,
        dataloader = get_simple_dataloader,
        model=SimpleNN,
        evaluation_function=evaluate_simpleNN,
        # recommend zipping tuples for more advanced settings
        partition=True,

        # Delay provides each client with three parameters (x1, x2, x3)
        # x1 controls the added delay between each epoch
        # x2 controls the added delay after the model is received
        # x3 controls the added delay before the model is sent back
        delay = [(0.01,0,0),(2,0,0)]
    )

    set_config(config)

    run_fed_learning(FedLearnServer, FedLearnClient)

