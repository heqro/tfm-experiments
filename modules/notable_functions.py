import torch
from torch import exp

# 2-d functions


def franke_function(grid: torch.Tensor):
    x = grid[:, 0]
    y = grid[:, 1]
    return 0.75 * exp(-((9 * x - 2) ** 2 / 4) - ((9 * y - 2) ** 2 / 4)) \
        + 0.75 * exp(-(9 * x + 1) ** 2 / 49 - (9 * y + 1) / 10) \
        + 0.5 * exp(-(9 * x - 7) ** 2 / 4 - (9 * y - 3) ** 2 / 4) \
        - 0.2 * exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)


def sin_pi_x_y_sq(grid: torch.Tensor):
    x, y = grid[:, 0], grid[:, 1]
    return torch.sin(torch.pi * (x ** 2 + y ** 2))


def norm_cos(grid: torch.Tensor):
    return torch.cos(grid.norm(p=2., dim=1)**2)


def norm_cos_pi(grid: torch.Tensor):
    return torch.cos(torch.pi * grid.norm(p=2., dim=1)**2)

# 1-d functions


def plane_0(x: torch.Tensor):
    return torch.ones((x.shape[0], 1))


def plane_1(x: torch.Tensor):
    return x


def plane_2(x: torch.Tensor):
    return 1 - x


def v_shaped(x: torch.Tensor):
    return (x <= 0.5) * x + (x > 0.5) * (1-x)


def runge_function(x: torch.Tensor):
    return 1 / (1 + 25 * x ** 2)


def gibbs_function(x: torch.Tensor):
    return torch.atan(20 * x)


def sin_pi_x_sq(x: torch.Tensor):  # sin(pi * x**2)
    return torch.sin(torch.pi * x ** 2)


def sin_higher_oscillations(x: torch.Tensor):  # sin(2.5*(x+1)**2)
    return torch.sin(2.5 * (x+1) ** 2)


def u2(x: torch.Tensor):
    return (x-.5) ** 2


def u3(x: torch.Tensor):
    return (x-.5) ** 3
