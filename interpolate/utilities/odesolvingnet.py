import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net_fun = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_fun(x)