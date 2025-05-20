from fl_app.base_model import SimpleNN
import torch

bias = torch.tensor([0.1, 0.2, 0.3])

# 2D tensor (weight matrix)
weight1 = torch.tensor([[0.5, -0.6],
                       [0.7,  0.8]])

weight2 = torch.tensor([[0.5, -0.6],
                       [0.7,  0.8]])

print(weight1 + weight2)

