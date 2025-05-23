# Necessary for federated learning
from fl_app.server_app.server import FedLearnServer
from fl_app.client_app.client import FedLearnClient
from fl_app.config import Config, set_config
from run_all import run_fed_learning


#from fl_app.CIFAR10 import SimpleCNN_CIFAR, load_cifar10_partition, train_CIFAR

# Necessary for machine learning
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# set proportion of clients
# set base learning rate

class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_simple_dataloader(client_id):
    path = 'fl_app/data/dataset_' + str(1) + '.csv'
    df = pd.read_csv(path)
    X = torch.tensor(df[["feature1", "feature2"]].values, dtype=torch.float32)
    y = torch.tensor(df["target"].values, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)


def train_simpleNN(model, dataloader, epochs):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

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
    config = Config(
        max_clients = 2,
        train_iterations = 3,
        epochs = 5,
        train_function = train_simpleNN,
        dataloader = get_simple_dataloader,
        model=SimpleNN,
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )

    set_config(config)

    run_fed_learning(FedLearnServer, FedLearnClient, config.max_clients)

