_config = None

def set_config(config_obj):
    global _config
    _config = config_obj

def get_config():
    if _config is None:
        raise RuntimeError("Config has not been set yet. Call set_config(config_obj) first.")
    return _config

def false(x):
    return False


class Config:

    def __init__(self, 
                 max_clients,
                 train_iterations,
                 train_function,
                 dataloader,
                 model,
                 evaluation_function,
                 epochs,
                 learning_rate,
                 stop_condition = false,
                 partition = False,
                 delay=None,
                 options=None):
    
        self.max_clients = max_clients
        self.train_iterations = train_iterations
        self.train_function = train_function
        self.dataloader = dataloader
        self.model_class = model
        self.eval = evaluation_function
        self.epochs = epochs
        self.lr = learning_rate
        self.partition = partition
        self.stop_condition = stop_condition
        self.delay = delay or [(0,0,0)]*max_clients
        self.options = options or []


    def model(self):
        return self.model_class()
