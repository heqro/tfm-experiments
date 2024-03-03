import torch


class InterpolationNet(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list = []):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.Tanh())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
