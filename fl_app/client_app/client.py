import grpc
import io
from fl_app import fl_pb2
from fl_app import fl_pb2_grpc
import asyncio
import argparse
from fl_app.base_model import SimpleNN
from fl_app.util import torch_tools
from fl_app.CIFAR10 import load_cifar10_partition


class FedLearnClient():

    def __init__(self, client_id, num_clients):
        self.client_id = client_id
        self.model = SimpleNN()
        self.buffer = io.BytesIO()
        self.data_loader = load_cifar10_partition(client_id, num_clients)

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
            which = response.WhichOneof("response")

            if which == "model":
                received_model = torch_tools.deserialize(response.model)
                self.model.load_state_dict(received_model)
                torch_tools.train(self.client_id, self.model)

                update_data = fl_pb2.UpdateData(model=torch_tools.serialize(self.model, self.buffer), data_size=100)
                await response_stream.write(fl_pb2.ClientFetchModel(model_data = update_data))
            else:
                break

        await response_stream.done_writing()
        print("Done training")


    async def run(self):
        async with grpc.aio.insecure_channel("localhost:50051") as channel:
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
