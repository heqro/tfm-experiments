import torch
from torch import exp


def franke_func(grid: torch.Tensor):
    x = grid[:, 0]
    y = grid[:, 1]
    return 0.75 * exp(-((9 * x - 2) ** 2 / 4) - ((9 * y - 2) ** 2 / 4)) \
        + 0.75 * exp(-(9 * x + 1) ** 2 / 49 - (9 * y + 1) / 10) \
        + 0.5 * exp(-(9 * x - 7) ** 2 / 4 - (9 * y - 3) ** 2 / 4) \
        - 0.2 * exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)


def norm_cos(grid: torch.Tensor):
    return torch.cos(grid.norm(p=2., dim=1))
