import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error, r2_score

from collections import OrderedDict
from multiprocessing import Process, Queue
import copy
import numpy as np

class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def load_dataset(client_id):
    df = pd.read_csv("data/dataset_1.csv")  # Load full dataset
    df_len = len(df)

    # Shuffle the indices
    indices = np.arange(df_len)
    np.random.seed(42)
    np.random.shuffle(indices)

    # Split into 2 partitions
    partition_size = df_len // 2
    start = client_id * partition_size
    end = start + partition_size if client_id == 0 else df_len  # client 1 gets the remainder

    selected_indices = indices[start:end]
    df_subset = df.iloc[selected_indices]

    X = torch.tensor(df_subset[["feature1", "feature2"]].values, dtype=torch.float32)
    y = torch.tensor(df_subset["target"].values, dtype=torch.float32).unsqueeze(1)

    return DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True), len(df_subset)


def train_simpleNN(model, client_id, queue, lr = 0.01):

    dataloader, df_len = load_dataset(client_id)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for _ in range(5):
        running_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    queue.put((model.state_dict(), df_len))

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

def fed_avg(models, sizes):
    new_state = OrderedDict()
    total = sum(sizes)
    

    for key in models[0]:
        new_param = torch.zeros_like(models[0][key])
        for i, model in enumerate(models):
            new_param += ((sizes[i] / total) * model[key])
                
        new_state[key] = new_param

    return new_state

if __name__ == "__main__":
    queue = Queue()
    global_model = SimpleNN()

    for _ in range(20):
        processes = [Process(target=train_simpleNN, args=(copy.deepcopy(global_model), i, queue)) for i in range(2)]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        queue_list = [queue.get() for _ in range(2)]
        models, sizes = zip(*queue_list)
        new_model = fed_avg(models, sizes)

        # for model in models:
        #     global_model.load_state_dict(model)
        #     evaluate_simpleNN(global_model)

        global_model.load_state_dict(new_model)
        evaluate_simpleNN(global_model)



# Might be loading data incorrectly
# might be passing over length of whole data rather than partition.