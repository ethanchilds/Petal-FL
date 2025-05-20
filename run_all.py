import multiprocessing
import asyncio
from fl_app.client_app import client
from fl_app.server_app import server
import time

def launch_server():
    asyncio.run(server.start_server())

def launch_client(client_id, num_clients):
    asyncio.run(client.start_client(client_id, num_clients))

def main():
    processes = []
    
    p_sever = multiprocessing.Process(target=launch_server)
    p_sever.start()
    processes.append(p_sever)

    # Start clients
    num_clients = 2
    for client_id in range(num_clients):
        p_client = multiprocessing.Process(target=launch_client, args=(client_id,num_clients))
        p_client.start()
        processes.append(p_client)

    try:
        # Wait for processes (or implement your own logic)
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping all...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()