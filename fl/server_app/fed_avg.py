import asyncio
import torch
from collections import OrderedDict
from fl.util import torch_tools

import copy

class FedAvg():

    def __init__(self):
        self.updated_models = []
        self.data_sizes = []
        self.lock = asyncio.Lock()

        self.old_models = None
        self.old_state = None
        self.iter = 0
    
    async def add_model(self, model, data_size):
        
        async with self.lock:
            self.updated_models.append(model)
            self.data_sizes.append(data_size)

    def fed_avg(self):
        if self.old_models:
            same = True
            for i, old_model in enumerate(self.old_models):
                if not torch_tools.state_dicts_equal(old_model, self.updated_models[i]):
                    same = False
    
        new_state = OrderedDict()
        total = sum(self.data_sizes)

        for key in self.updated_models[0]:
            new_param = torch.zeros_like(self.updated_models[0][key])
            for i, model in enumerate(self.updated_models):
                new_param += ((self.data_sizes[i] / total) * model[key])
                    
            new_state[key] = new_param


        # if self.old_state:
        #     print("############## ITER:", self.iter, "#################")
        #     if not same:
        #         for i in range(len(self.old_models)):
        #             if not torch_tools.state_dicts_equal(self.old_models[i], self.updated_models[i]):
        #                 for key in self.old_models[i]:
        #                     if not torch.equal(self.old_models[i][key], self.updated_models[i][key]):
        #                         print(i, key)
        #     self.iter += 1 
        self.old_models = copy.deepcopy(self.updated_models)
        self.old_state = copy.deepcopy(new_state)
        self.updated_models = []
        self.data_sizes = []
        return new_state