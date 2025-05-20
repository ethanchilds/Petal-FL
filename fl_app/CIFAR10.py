### CHATGPT CIFAR10 loading
## consider using non-iid as an option

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

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