from fl_app.CIFAR10 import SimpleCNN_CIFAR, load_cifar10_partition, train_CIFAR
from fl_app.simple_model import SimpleNN, train_simpleNN, get_simple_dataloader

class Config:

    max_clients = 2
    train_iterations = 5
    train_function = train_simpleNN
    dataloader = get_simple_dataloader
    options=[
        ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ]
    
    
    @staticmethod
    def model():
        return SimpleNN()
