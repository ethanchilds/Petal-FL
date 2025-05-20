import asyncio
import torch
from collections import OrderedDict
from fl_app.util import torch_tools

import copy

class FedAvg():

    def __init__(self):
        self.updated_models = []
        self.data_sizes = []
        self.lock = asyncio.Lock()
        self.has_averaged = False
    
    async def add_model(self, model, data_size):
        
        async with self.lock:
            self.updated_models.append(model)
            self.data_sizes.append(data_size)

    def fed_avg(self):
        new_state = OrderedDict()
        total = sum(self.data_sizes)

        for key in self.updated_models[0]:
            new_param = torch.zeros_like(self.updated_models[0][key])

            for i, model in enumerate(self.updated_models):
                new_param += ((self.data_sizes[i] / total) * model[key])
                    
            new_state[key] = new_param

        self.updated_models = []
        self.data_sizes = []
        return new_state