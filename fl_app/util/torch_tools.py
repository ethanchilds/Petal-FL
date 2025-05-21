import torch
import pandas as pd
import io
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

def train(model, criterion, optimizer, dataloader):

        model.train()

        for _ in range(100):
                running_loss = 0.0
                for x, y in dataloader:
                        optimizer.zero_grad()
                        outputs = model(x)
                        loss = criterion(outputs, y)
                        loss.backward()

                        optimizer.step()
                        running_loss += loss.item()




def state_dicts_equal(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True

def state_dicts_close(dict1, dict2, tol=1e-5):
    for key in dict1:
        if not torch.allclose(dict1[key], dict2[key], atol=tol):
            return False
    return True