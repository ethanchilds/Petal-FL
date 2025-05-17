import asyncio

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
        new_state = {}
        total = sum(self.data_sizes)

        for key in self.updated_models[0]:
            new_weight = 0

            for i in range(len(self.updated_models)):
                new_weight += ((self.data_sizes[i] / total)*self.updated_models[i][key])

            new_state[key] = new_weight
        
        self.updated_models = []
        self.data_sizes = []
        return new_state
    
    def print_state(self):
        print(self.updated_models)
        print(self.data_sizes)

        self.updated_models = []
        self.data_sizes = []