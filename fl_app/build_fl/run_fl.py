import multiprocessing
import asyncio

def launch_server(server):
    asyncio.run(server.start_server())

def launch_client(client):
    asyncio.run(client.run())

def run_fed_learning(server_obj, client_obj, num_clients): 
    processes = []
    
    server = server_obj()
    p_sever = multiprocessing.Process(target=launch_server, args=(server, ))
    p_sever.start()
    processes.append(p_sever)


    for client_id in range(num_clients):
        client = client_obj(client_id, 1/(client_id+1))
        p_client = multiprocessing.Process(target=launch_client, args=(client,))
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