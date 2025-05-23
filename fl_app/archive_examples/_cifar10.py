### CHATGPT CIFAR10 loading
## consider using non-iid as an option

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from fl_app.util import torch_tools

def load_cifar10_partition(client_id, num_clients, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    data_len = len(dataset)
    indices = list(range(data_len))
    np.random.seed(42)
    np.random.shuffle(indices)

    # Equal split (non-iid will be explained below)
    partition_size = data_len // num_clients
    start = client_id * partition_size
    end = start + partition_size

    subset = Subset(dataset, indices[start:end])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return loader

class SimpleCNN_CIFAR(nn.Module):
    def __init__(self):
        super(SimpleCNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_CIFAR(model, dataloader, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    torch_tools.train(model, criterion, optimizer, dataloader)