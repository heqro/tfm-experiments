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
    return torch.cos(grid.norm(p=2., dim=1)**2)


# 1-d functions


def plane_0(x: torch.Tensor):
    return torch.ones_like(x)


def plane_1(x: torch.Tensor):
    return x


def plane_2(x: torch.Tensor):
    return 1 - x


def v_shaped(x: torch.Tensor):
    return (x <= 0.5) * x + (x > 0.5) * (1-x)


def exp_compressed(x: torch.Tensor):
    # assume x in [0,1]
    return torch.exp(x) / 3
