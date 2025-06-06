# Necessary for federated learning
from fl.server_app.pamble_server import FedLearnServer
from fl.client_app.amble_client import FedLearnClient
from fl.build_fl.config import Config, set_config
from fl.build_fl.run_fl import run_fed_learning

# Necessary for machine learning
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleCNN_CIFAR(nn.Module):
    def __init__(self):
        super(SimpleCNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def load_cifar10(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

def train_CIFAR(model, dataloader, epochs, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

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

def evaluate_CIFAR(model):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to('cpu'), labels.to('cpu')
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100*correct / total
    avg_loss = test_loss / len(testloader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    return accuracy

def stop_condition(value):
    if (value / 100) > 0.7:
        return True
    else:
        return False

if __name__ == "__main__":

    # set proportion of clients

    config = Config(
        max_clients = 5,
        train_iterations = 20,
        epochs = 2,
        learning_rate = 0.01,
        train_function = train_CIFAR,
        dataloader = load_cifar10,
        model=SimpleCNN_CIFAR,
        evaluation_function=evaluate_CIFAR,
        stop_condition=stop_condition,
        # recommend zipping tuples for more advanced settings
        delay = [(0.1,0,0), (0.2,0,0), (0.3, 0, 0), (0.8,0,0), (15,0,0)],
        partition=True,
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )

    set_config(config)

    run_fed_learning(FedLearnServer, FedLearnClient)

