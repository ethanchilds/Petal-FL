import grpc
import io
import asyncio
from fl_app import fl_pb2
from fl_app import fl_pb2_grpc
from fl_app.config import Config
from fl_app.server_app.fed_avg import FedAvg
from fl_app.util import torch_tools

import copy
from fl_app.logging.log_set_up import setup_logger
logger = setup_logger("server", level="INFO")

import os
os.environ["GRPC_VERBOSITY"] = "ERROR"

class FedLearnServicer(fl_pb2_grpc.FedLearnServicer):

    def __init__(self, done_event):
        self.ready_train = True
        self.model = Config.model()
        self.lock = asyncio.Lock()
        self.iteration_ready = asyncio.Event()
        self.current_clients = 0
        self.max_clients = Config.max_clients
        self.current_iteration = 0
        self.train_iterations = Config.train_iterations
        self.need_reset = False
        self.buffer = io.BytesIO()
        self.fed_avg = FedAvg()

        # TEMP: Only for cancelling server once fed learn is done
        self.done = done_event
        self.clients_done = 0

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
                if self.current_clients == self.max_clients:
                    self.iteration_ready.set()
                    self.need_reset = True

            await self.iteration_ready.wait()


            async with self.lock:
                if self.need_reset:
                    self.need_reset = False
                    self.iteration_ready.clear()
                    self.current_clients = 0
                    self.current_iteration += 1

                    if which == "model_data":
                        pretrain = copy.deepcopy(self.model.state_dict())
                        updated_model = self.fed_avg.fed_avg()
                        self.model.load_state_dict(updated_model)
                        #print(torch_tools.state_dicts_close(pretrain, updated_model))

            if self.current_iteration == self.train_iterations:
                yield fl_pb2.ModelReady(wait=True)
                break
            else:
                yield fl_pb2.ModelReady(model=torch_tools.serialize(self.model, self.buffer))

        async with self.lock:
            self.clients_done += 1
            if self.clients_done == self.max_clients:
                self.done.set()


async def serve():
    done = asyncio.Event()
    
    server = grpc.aio.server(options=Config.options)
    fl_pb2_grpc.add_FedLearnServicer_to_server(FedLearnServicer(done), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    print('Listening on 50051...')
    
    await done.wait()
    await server.stop(0)

async def start_server():
    await serve()

if __name__ == "__main__":
    asyncio.run(serve())