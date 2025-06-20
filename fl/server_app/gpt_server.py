import grpc
import io
import asyncio
import simple_fl_pb2 as fl_pb2
import simple_fl_pb2_grpc as fl_pb2_grpc
from fl.server_app.fed_avg import FedAvg
from fl.util import torch_tools
from fl.server_app import query

import time

from fl.logging.log_set_up import setup_logger
logger = setup_logger("server", level="INFO")

import os
os.environ["GRPC_VERBOSITY"] = "ERROR"

from fl.build_fl.config import get_config

class FedLearnServicer(fl_pb2_grpc.FedLearnServicer):

    def __init__(self, config, done_event):
        self.config = config
        self.ready_train = True
        self.model = self.config.model()
        self.lock = asyncio.Lock()
        self.iteration_ready = asyncio.Event()
        self.current_clients = 0
        self.max_clients = self.config.max_clients
        self.current_iteration = 0
        self.train_iterations = self.config.train_iterations + 1
        self.need_reset = False
        self.buffer = io.BytesIO()
        self.stop_condition = self.config.stop_condition
        self.eval = 0

        self.fed_avg = FedAvg()

        # TEMP: Only for cancelling server once fed learn is done
        self.done = done_event
        self.clients_done = 0

        self.begin = 0
        self.end = 0

        self.client_times = [None]*self.max_clients
        self.epochs = [self.config.epochs]*self.max_clients

    def get_ready_train(self):
        return self.ready_train

    async def ModelPoll(self, request, context):
        logger.info(f"Client {request.ready} connected")
        print(request.ready)
        
        msg = await asyncio.to_thread(self.get_ready_train)
        return fl_pb2.ReadyReply(ready = msg)
    
    async def GetModel(self, request_iterator, context):
        async for request in request_iterator:

            which = request.WhichOneof("response")
            if which == "model_data":
                await self.fed_avg.add_model(torch_tools.deserialize(request.model_data.model), request.model_data.data_size)
                
            async with self.lock:
                self.current_clients += 1
                self.epochs[request.client_id] = request.epoch
                self.client_times[request.client_id] = request.time
                if self.current_clients == self.max_clients:
                    self.iteration_ready.set()
                    self.need_reset = True
                    if self.current_iteration == 0:
                        self.begin = time.perf_counter()

            await self.iteration_ready.wait()


            async with self.lock:
                if self.need_reset:
                    self.need_reset = False
                    self.iteration_ready.clear()
                    self.current_clients = 0
                    self.current_iteration += 1

                    if which == "model_data":
                        updated_model = self.fed_avg.fed_avg()
                        self.model.load_state_dict(updated_model)

                        self.eval = self.config.eval(self.model)
                        client_info = query.build_client_info(
                            [x for x in range(self.max_clients)], 
                            self.client_times, 
                            self.epochs)
                        self.epochs = query.query_openai(client_info)



            if self.current_iteration == self.train_iterations or self.stop_condition(self.eval):
                yield fl_pb2.ModelReady(wait=True)
                break
            else:
                model=torch_tools.serialize(self.model, self.buffer)
                yield fl_pb2.ModelReady(model=model)

        async with self.lock:
            self.clients_done += 1
            if self.clients_done == self.max_clients:
                self.end = time.perf_counter()
                print("Process took:", self.end - self.begin)
                self.done.set()


class FedLearnServer():

    def __init__(self):
        self.config = get_config()

    async def serve(self):
        done = asyncio.Event()
        
        server = grpc.aio.server(options=self.config.options)
        fl_pb2_grpc.add_FedLearnServicer_to_server(FedLearnServicer(self.config, done), server)
        server.add_insecure_port('[::]:50051')
        await server.start()
        print('Listening on 50051...')
        
        await done.wait()
        await server.stop(0)

    async def start_server(self):
        await self.serve()
