import grpc
from fl_app import fl_pb2
from fl_app import fl_pb2_grpc
import asyncio
import argparse



class FedLearnClient():

    def __init__(self, client_id):
        self.client_id = client_id

    async def model_poll(self, stub):
        # Assign to a daemon or have it await in future
        result = await stub.ModelPoll(fl_pb2.Ready(ready = f"client_id: {self.client_id}"))
        return result.ready


    async def train_model(self, stub):
        response_stream = stub.GetModel()

        await response_stream.write(fl_pb2.Ready(ready='Client1'))

        while True:

            response = await response_stream.read()
            which = response.WhichOneof("response")

            if which == "model":
                print(response.model)
                await response_stream.write(fl_pb2.Ready(ready=f'Client {self.client_id}'))
            else:
                break

        await response_stream.done_writing()


    async def run(self):
        async with grpc.aio.insecure_channel("localhost:50051") as channel:
            stub = fl_pb2_grpc.FedLearnStub(channel)
            poll_result = await self.model_poll(stub)

            if poll_result:
                await self.train_model(stub)

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
