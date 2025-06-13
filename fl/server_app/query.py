import openai

# Make sure you've installed the openai library
# pip install openai

# Set your OpenAI API key
openai.api_key = "your-api-key"

def build_client_info(client_ids, training_times, epochs_trained):
    clients = []
    for cid, time, epoch in zip(client_ids, training_times, epochs_trained):
        clients.append(f'{{"id": "{cid}", "training_time": {time}, "epochs": {epoch}}}')
    return "[\n    " + ",\n    ".join(clients) + "\n]"

def create_prompt(client_info):
    return f"""
You are helping optimize federated learning training across heterogeneous clients.
Each client trains a local model for a given number of epochs, and we record how long
their training took. Some clients are slow, and some are fast. We want to reassign 
epochs so that faster clients train more and slower ones train less in order to better 
synchronize overall training rounds.

Given the following data:

clients = {client_info}

Return a list of integers representing the *new* number of epochs for each client, 
aiming to balance the clients such that they all take approximately the same amount 
of wall-clock time to train. The list should be in the same order as the input clients. 
Make sure each client trains at least 1 epoch.

Explain your reasoning and show the formula used.
"""

def query_openai(client_info, model="gpt-3.5-turbo"):
    prompt = create_prompt(client_info)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response['choices'][0]['message']['content']