import grpc
import io
import torch
import asyncio
from fl_app import fl_pb2
from fl_app import fl_pb2_grpc
from fl_app.server_app.server_config import ServerConfig
from fl_app.base_model import SimpleNN
from fl_app.server_app.fed_avg import FedAvg
from fl_app.util import torch_tools

class FedLearnServicer(fl_pb2_grpc.FedLearnServicer):

    def __init__(self):
        self.ready_train = True
        self.model = SimpleNN()
        self.lock = asyncio.Lock()
        self.iteration_ready = asyncio.Event()
        self.current_clients = 0
        self.max_clients = ServerConfig.max_clients
        self.current_iteration = 0
        self.train_iterations = ServerConfig.train_iterations
        self.need_reset = False
        self.buffer = io.BytesIO()
        self.fed_avg = FedAvg()

    def get_ready_train(self):
        return self.ready_train

    async def ModelPoll(self, request, context):
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
                        updated_model = self.fed_avg.fed_avg()
                        self.model.load_state_dict(updated_model)

            if self.current_iteration == self.train_iterations:
                yield fl_pb2.ModelReady(wait=True)
                break
            else:
                yield fl_pb2.ModelReady(model=torch_tools.serialize(self.model, self.buffer))

async def serve():
    server = grpc.aio.server()
    fl_pb2_grpc.add_FedLearnServicer_to_server(FedLearnServicer(), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    print('Listening on 50051...')
    await server.wait_for_termination()

async def start_server():
    await serve()

if __name__ == "__main__":
    asyncio.run(serve())