import multiprocessing
import asyncio
from fl.build_fl.config import get_config

def launch_server(server):
    asyncio.run(server.start_server())

def launch_client(client):
    asyncio.run(client.run())

def run_fed_learning(server_obj, client_obj): 
    config = get_config()
    processes = []
    
    server = server_obj()
    p_sever = multiprocessing.Process(target=launch_server, args=(server,))
    p_sever.start()
    processes.append(p_sever)


    for client_id in range(config.max_clients):
        client = client_obj(client_id, config.delay[client_id])
        p_client = multiprocessing.Process(target=launch_client, args=(client,))
        p_client.start()
        processes.append(p_client)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping all...")
        for p in processes:
            p.terminate()