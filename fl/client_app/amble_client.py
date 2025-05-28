import grpc
import io
import amble_fl_pb2 as fl_pb2
import amble_fl_pb2_grpc as fl_pb2_grpc
import asyncio
import time
import argparse
from fl.util import torch_tools
from fl.client_app.sleep_injector import SleepInjector

from fl.build_fl.config import get_config


class FedLearnClient():

    def __init__(self, client_id, delay):
        self.config = get_config()
        self.client_id = client_id
        self.model = self.config.model()
        self.buffer = io.BytesIO()
        self.dataloader = self.config.dataloader(client_id)
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.work_delay = delay[0]
        self.receive_delay = delay[1]
        self.send_delay = delay[2]

        if self.config.partition:
            self.dataloader = self.subset_loader(self.config.dataloader(), self.config.max_clients, client_id)
        else:
            self.dataloader = self.config.dataloader()

    async def model_poll(self, stub):
        # Assign to a daemon or have it await in future
        request = fl_pb2.Ready(ready = f"client_id: {self.client_id}")
        result = await stub.ModelPoll(request)
        return result.ready


    async def train_model(self, stub):
        response_stream = stub.GetModel()

        request = fl_pb2.ClientFetchModel(send_model = True)
        await response_stream.write(request)

        while True:

            response = await response_stream.read()
            time.sleep(self.receive_delay)
            which = response.WhichOneof("response")

            if which == "model":
                received_model = torch_tools.deserialize(response.model)
                self.model.load_state_dict(received_model)

                train_time_begin = time.perf_counter()
                wrapped_loader = SleepInjector(self.dataloader, self.work_delay)
                self.config.train_function(self.model, wrapped_loader, self.epochs, self.lr)
                train_time = time.perf_counter() - train_time_begin


                update_data = fl_pb2.UpdateData(model=torch_tools.serialize(self.model, self.buffer), data_size=len(self.dataloader.dataset))
                request = fl_pb2.ClientFetchModel(model_data = update_data)
                request.client_id = self.client_id
                request.round_time = train_time
                time.sleep(self.send_delay)
                await response_stream.write(request)

            else:
                break

        await response_stream.done_writing()
        print("Done training")


    async def run(self):
        async with grpc.aio.insecure_channel("localhost:50051", options=self.config.options) as channel:
            stub = fl_pb2_grpc.FedLearnStub(channel)
            poll_result = await self.model_poll(stub)

            if poll_result:
                await self.train_model(stub)

async def start_client(client_id, num_clients):
    client = FedLearnClient(client_id, num_clients)
    await client.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get client id")
    
    parser.add_argument(
        "-c", "--client",
        type=int,
        required=True,
        help="Your client ID"
    )

    args = parser.parse_args()

    client = FedLearnClient(args.client)
    asyncio.run(client.run())
