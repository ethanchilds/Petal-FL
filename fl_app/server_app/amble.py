import asyncio
import math

class Amble():

    def __init__(self, processor_speed, base_epochs, base_lr, base_batch_size, model_size):
        # C_k * 1e-9
        self.ghz = processor_speed
        # E
        self.base_epochs = base_epochs
        # nu
        self.base_lr = base_lr
        # M
        self.mode_size = model_size

        self.base_batch_size = base_batch_size

        self.lock = asyncio.Lock()

        self.client_info = {}
        #self.client_history = {}

        self.results = {}


    async def add_client_info(self, client_id, round_time, data_size):
        async with self.lock:
            self.client_info[client_id] = [round_time,data_size]



    def AMBLE(self):
        slowest = float('-inf')
        for client in self.client_info:
            if self.client_info[client][0] > slowest:
                slowest = self.client_info[client][0]

        for client in self.client_info:
            # First iteration before client data taken into account
            new_epoch = math.floor((slowest / self.client_info[client][0])*self.base_epochs)

            # nu_k
            new_lr = self.base_lr * (self.base_epochs/new_epoch)

            # delta_k
            est_iter_time = self.client_info[client][0] / self.base_epochs

            # s
            est_cycles = est_iter_time * (self.ghz*1e9)

            numerator = (est_iter_time*new_epoch*self.client_info[client][1]*(self.ghz*1e9))
            denominator = (slowest*(self.ghz*1e9) - (self.mode_size*est_cycles*new_epoch*self.client_info[client][1]))

            new_batch = max(math.floor(numerator/denominator), 1)

            new_lr = new_lr * (new_batch / self.base_batch_size)       

            self.results[client] = (new_lr, new_batch, new_epoch)

        return self.results