import torch
import io

def serialize(model, buffer):
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

def deserialize(model_data):
        return torch.load(io.BytesIO(model_data))