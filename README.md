# Petal-FL
*A Modular Federated Learning Framework for Experimenting With Client Heterogeneity*

[![License](https://img.shields.io/github/license/ethanchilds/Petal-FL)](LICENSE)
[![Issues](https://img.shields.io/github/issues/ethanchilds/Petal-FL)](https://github.com/ethanchilds/Petal-FL/issues)
[![Stars](https://img.shields.io/github/stars/ethanchilds/Petal-FL?style=social)](https://github.com/ethanchilds/Petal-FL/stargazers)
---

## 📖 Overview

Petal-FL is a simple federated learning framework built with the express intention of providing users with out-of-the-box simulation of client device heterogeneity. This framework has been built with PyTorch for all machine learning functionalities and gRPC as a middleware service.

---

## Table of Contents


## Usage Manual

Users can quickly find example usage of the Petal-FL framework in the [examples](./examples/). While these are readily available, they are missing a large amount of information required to truly understand how this system works. Petal-FL deploys and runs federated learning through a combination of three main files, [a build file](./fl/build_fl/run_fl.py), [a config file](./fl/build_fl/run_fl.py), and finally a user defined file that ties it all together. Petal-FL is designed to mostly abstract away the details of the config and build files from the users, but has not yet fully acomplished complete abstraction of these concepts. In order to begin building your own federated learning system, please clone Petal-FL locally. 

### Imports

For the rest of this usage manual we will walk through the [simple_example](./examples/simple_example). For your own model, please create your file in any repository you wish and then continue with this guide.

In the simple example the first thing you will notice are the imports. I have marked them by necessary for federated learning and necessary for machine learning. The imported modules for machine learning will be decided at your own disgression depending on what you design with this system. The objects and functions imported that are necessary are not up for user disgression and the system will not work without these imports. However, while the system does require these imports, they can be slightly tweaked to have different effects on the system.

For the client, users can choose to import either of the following:

- from fl.client_app.simple_client import FedLearnClient
- from fl.client_app.amble_client import FedLearnClient

And for the server users can choose to import any of the following:

- from fl.server_app.simple_server import FedLearnServer
- from fl.server_app.amble_server import FedLearnServer
- from fl.server_app.pamble_server import FedLearnServer

Please note that simple server and client will work together and the AMBLE client will work with any of the AMBLE-based servers. If one would like to test their federated learning process with any different strategies, a simple change to the import is all that is needed.

```{Python}
# Necessary for federated learning
from fl.server_app.simple_server import FedLearnServer
from fl.client_app.simple_client import FedLearnClient
from fl.build_fl.config import Config, set_config
from fl.build_fl.run_fl import run_fed_learning

# Necessary for machine learning
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
```

### Model

Petal-FL requires it's users to define a PyTorch model to be trained in the federated learning process. For experienced PyTorch users, there is no change to usual functionality, simply define the layers of your model and the forward step required for training in PyTorch.

```{Python}
class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Dataloader

For clients to train, a PyTorch dataloader object must be created. This is not any different from how one would define in a normal PyTorch machine learning context, just provide the system with a path to the dataset you would like to train on. One requirement, however, is a batch_size parameter within your dataloader. Please set this to some base value that would be desirable for your training. A useful feature of Petal-FL is that partitioning of data for clients is abstracted away from the user, so all that must be defined is one global dataloader.

```{Python}
def get_simple_dataloader(batch_size = 16):
    path = 'data/dataset_' + str(1) + '.csv'
    df = pd.read_csv(path)
    X = torch.tensor(df[["feature1", "feature2"]].values, dtype=torch.float32)
    y = torch.tensor(df["target"].values, dtype=torch.float32).unsqueeze(1)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    return dataloader
```

### Training

Users must define a train function for the federated learning process. This function does not need to follow any naming convention, however, it must have the model, dataloader, epoch, and learning rate parameters. The actually functionality that must be defined is standard PyTorch training. Users can define their criterion, optimizer, and any other training related decisions within this function.

```{Python}
def train_simpleNN(model, dataloader, epochs, lr = 0.01):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for _ in range(epochs):
        running_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
```

### Evaluation

Users must define an evaluation function for the federated learning process. This function does not need to follow any naming convention, but must have the model parameter defined. This function does two things. First, it must define the path to the testing data and which values are the target. Second, it must have the model enter eval mode and then calculate some metric the users feel is apt for their model.

```{Python}
def evaluate_simpleNN(model):
    df = pd.read_csv("data/dataset_1.csv")
    X = torch.tensor(df[["feature1", "feature2"]].values, dtype=torch.float32)
    y_true = df["target"].values

    model.eval()
    with torch.no_grad():
        y_pred = model(X).numpy().flatten()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)


    print(f"Evaluation Results — MSE: {mse:.4f}, R²: {r2:.4f}")
    return r2
```

### Config and Running

Finally, the user must handle a few steps to run their code. Petal-FL makes use of a global config to help define important information pertinent to the federated learning process. Users must define these variables by instantiating a config object and setting it. Within this config, a max number of clients, number of train iteration, training epochs, learning rate, train function, data loader, model, and evaluation MUST be defined for Petal-FL to run properly. Once these values have been set, users pass their desired sever and client object into the run federated learning function.

```{Python}
if __name__ == "__main__":

    config = Config(
        max_clients = 2,
        train_iterations = 20,
        epochs = 5,
        learning_rate = 0.01,
        train_function = train_simpleNN,
        dataloader = get_simple_dataloader,
        model=SimpleNN,
        evaluation_function=evaluate_simpleNN,
        stop_condition=stop_conditon,
        # recommend zipping tuples for more advanced settings
        partition=True,
        delay = [(0,0,0),(0,0,0)]
    )

    set_config(config)

    run_fed_learning(FedLearnServer, FedLearnClient)
```

Once the entire user-defined file is complete users simply have to run their file. For the [simple example](./examples/simple_example.py), users will run

```
python -m examples.simple_example
```

### Optional Parameters and Functions

Along with the required parameters for running federated learning with Petal-FL, there are some additional parameters available. As mentioned earlier, Petal-FL offers abstraction of data partitioning to users, however, in order to do this, users must set the `partion` parameter equal to true. While not shown in the above simple example code, an example of message size being set can be found in the [CIFAR10 example](./examples/cifar_simple.py). This parameter is necessary when working with larger models as gRPC requires a larger message size to be set, or it will not allow for the flow of gradients between client and sever.

One of the most important parameters to Petal-FL is the `delay` parameter. One of the main goals of Petal-FL was to provide users with the ability to apply some degree of client device heterogeneity out of the box. This parameter is how users can control this functionality at a fine tuned level. Users pass in a list of tuples, where each tuple contatains three important delay values for its respective client.

Delay provides each client with three parameters (x1, x2, x3)
- x1 controls the added delay between each epoch.
- x2 controls the added delay after the model is received.
- x3 controls the added delay before the model is sent back.

As an example, if a user wishes the third client in their system to exhibit a longer training time than other clients, they can adjust the first value in the third tuple to be the extra length of time they would like this client to take. Similarly, if a user wishes to see the second user take extra time to receive data from the server, they can adjust the second value of the the second tuple. It is important to note that if this parameter is not defined, Petal-FL simply initializes all of these values for all clients at 0 seconds.

The only optional function available at the moment is the stop condition. Often in federated learning it is desirable to stop training once a certain accuracy is achieved rather than continue to run through many iterations without any improvement of accuracy. To define this function, please set the evaluation function to return the value you would like to check, and then define a stop condition function. There is no naming convention required for this function, but it must take a value parameter and return either true or false.


```{Python}
def stop_conditon(value):
    if value > 0.9:
        return True
    else:
        return False
```


## Virtualization

Currently on the main branch there is no true virtualization outside of some basic image set-up, however, I would dissuade users from builiding this image as installing PyTorch with CUDA enhancements on Docker is an incredibly long task. Currently in development on the `compose-up-test` branch I have been experimenting with adding in virtualization and spinning up applications within their own containers. All the work done on this branch thus far has been focused around the [simple example](./examples/simple_example.py). In this branch users can execute a similar process to the one described in the above usage manual through a containerized environment. For those interested, please run the following command from the Petal-FL directory.

```
docker-compose up --build
```


## A Quick Discussion on the Current State of Petal-FL

Petal-FL was originally designed with the express intention of being an experimentation framework. For me, this meant basing my development of Petal-FL on script-like design methodologies, however, I don't love how Petal-FL is currently executing this. As described earlier, Petal-FL has yet to fully abstract away the config and build functions from users, and I don't believe this is ideal design for a framework with the intention of being used for experimentation. At most, I feel users should only ever have to define some functionalities and a few varaibles, but not observe how those definitions are being used to complete the federated learning task. Along with this, the parameter definition is still very hard coded within Petal-FL, so if users need to veer away from the predefined paramters alloted, they do not have this functionality available.

This partially is what led me to experimenting with virtualization, as I could remove the run_fl and maybe even the config necessities, however, this approach veers away from the desired experimental framework. While it would be very nice for providing virtualization, I believe this may be more production focused than systems focused, something that I haven't dived to deep into with this project. If one were to dive into this code, they would see missing security assurances that would be expected from systems such as this, but as it was developed originally for a scientific purpose and not a production purpose, I decided against implementing it, assuming users would be running their code locally.
