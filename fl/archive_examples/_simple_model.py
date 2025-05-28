import torch
import torch.nn as nn
import torch.optim as optim
from util import torch_tools

class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_simple_dataloader(client_id, num_clients):
    path = 'fl_app/data/dataset_' + str(client_id+1) + '.csv'
    return torch_tools.load_data(path)
    
def train_simpleNN(model, dataloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    torch_tools.train(model, criterion, optimizer, dataloader)