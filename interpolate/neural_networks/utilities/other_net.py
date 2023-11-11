import torch

class Other_Net(torch.nn.Module):
    def __init__(self, dimension: int, activation_function: any):
        super().__init__()
        self.net_fun = torch.nn.Sequential(
            torch.nn.Linear(1, dimension),
            activation_function,
            torch.nn.Linear(dimension, dimension),
            activation_function,
            torch.nn.Linear(dimension, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_fun(x)