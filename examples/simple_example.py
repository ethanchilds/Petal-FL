# Necessary for federated learning
from fl.server_app.simple_server import FedLearnServer
from fl.client_app.simple_client import FedLearnClient
from fl.build_fl.config import Config, set_config
from fl.build_fl.run_fl import run_fed_learning

# Necessary for machine learning
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

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
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


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

if __name__ == "__main__":

    # set proportion of clients

    config = Config(
        max_clients = 2,
        train_iterations = 3,
        epochs = 5,
        learning_rate = 0.01,
        train_function = train_simpleNN,
        dataloader = get_simple_dataloader,
        model=SimpleNN,
        # recommend zipping tuples for more advanced settings
        delay = [(0,0,0),(0,0,0)]
    )

    set_config(config)

    run_fed_learning(FedLearnServer, FedLearnClient)

