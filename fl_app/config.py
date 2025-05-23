_config = None

def set_config(config_obj):
    global _config
    _config = config_obj

def get_config():
    if _config is None:
        raise RuntimeError("Config has not been set yet. Call set_config(config_obj) first.")
    return _config

class Config:

    def __init__(self, 
                 max_clients,
                 train_iterations,
                 train_function,
                 dataloader,
                 model,
                 epochs,
                 options=None):
    
        self.max_clients = max_clients
        self.train_iterations = train_iterations
        self.train_function = train_function
        self.dataloader = dataloader
        self.model_class = model
        self.epochs = epochs
        self.options = options or []

    # max_clients = 2
    # train_iterations = 5
    # train_function = train_simpleNN
    # dataloader = get_simple_dataloader
    # options=[
    #     ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
    #     ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    # ]


    def model(self):
        return self.model_class()
