import torch.utils.data

device = "cuda" if torch.cuda.is_available() else "cpu"

def gather(tensorVal: torch.Tensor, timeStep: torch.Tensor):
    tempTensor = tensorVal.to(device).gather(-1, timeStep[0])
    return tempTensor.reshape(-1, 1, 1, 1)