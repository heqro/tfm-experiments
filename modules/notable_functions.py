import torch
from torch import exp
import numpy as np

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


def sin_pi_x_y_sq_laplacian(grid: torch.Tensor):
    from torch import pi, sin, cos
    x, y = grid[:, 0], grid[:, 1]
    return 4*pi*(-pi*x**2*sin(pi*(x**2 + y**2))
                 - pi*y**2*sin(pi*(x**2 + y**2))
                 + cos(pi*(x**2 + y**2)))


def norm_cos(grid: torch.Tensor):
    return torch.cos(grid.norm(p=2., dim=1)**2)


def norm_cos_pi(grid: torch.Tensor):
    return torch.cos(torch.pi * grid.norm(p=2., dim=1)**2)


def parabola(grid: torch.Tensor):
    x, y = grid[:, 0], grid[:, 1]
    return (x-1/2)**2 + (y-1/2) ** 2


def parabola_numpy(X: np.ndarray, Y: np.ndarray):
    return (X-1/2)**2+(Y-1/2)**2


def parabola_laplacian(grid: torch.Tensor):
    return 4 * torch.ones_like(grid[:, 0])

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


def arctan_paper(x: torch.Tensor):
    # Taken from "Observations on the Behavior of Radial Basis Function Approximations Near Boundaries"
    # Interval: [-1, 1]
    return -torch.arctan(5*(x+1/2))


def sin_cube_tref(x: torch.Tensor):
    # Taken from "Spectral Methods in Matlab"
    # Interval: [0, 1]
    return torch.abs(torch.sin(2*torch.pi*x))**3


def runge_2d(grid: torch.Tensor):
    # Taken from "Observations on the Behavior of Radial Basis Function Approximations Near Boundaries"
    # Domain: unit circle
    x, y = grid[:, 0], grid[:, 1]
    return 25/(25+(x-1/5)**2+2*y**2)


def runge_2d_numpy(X: np.ndarray, Y: np.ndarray):
    # Taken from "Observations on the Behavior of Radial Basis Function Approximations Near Boundaries"
    # Domain: unit circle
    return 25/(25+(X-1/5)**2+2*Y**2)


def runge_2d_laplacian(grid: torch.Tensor):
    # Computes the Laplacian of runge_2d
    # Domain: unit circle
    x, y = grid[:, 0], grid[:, 1]
    return 25*(20*y**2 - 6*(x - 0.2)**2 + (2*x - 0.4)*(4*x - 0.8) - 150)/(2*y**2 + (x - 0.2)**2 + 25)**3


def arctan_paper_2d(grid: torch.Tensor):
    # Taken from "Observations on the Behavior of Radial Basis Function Approximations Near Boundaries"
    # Domain: unit circle
    x, y = grid[:, 0], grid[:, 1]
    return torch.arctan(2*(x+3*y-1))


def arctan_paper_2d_numpy(X: np.ndarray, Y: np.ndarray):
    # Taken from "Observations on the Behavior of Radial Basis Function Approximations Near Boundaries"
    # Domain: unit circle
    return np.arctan(2*(X+3*Y-1))


def heat_1d(grid: torch.Tensor, a: float = 1/30, tau=1 / 50):
    from torch import exp, sqrt, pi
    x, t = grid[:, 0], grid[:, 1]
    return exp(-(x - 0.5)**2/(4*a*(t + tau)))/(2*sqrt(pi*a*(t + tau)))


def heat_1d_numpy(X: np.ndarray, T: np.ndarray, a: float = 1/30, tau=1 / 50):
    from numpy import sqrt, exp, pi
    return exp(-(X - 0.5)**2/(4*a*(T + tau)))/(2*sqrt(pi*a*(T + tau)))


def heat_1d_dx_dx(grid: torch.Tensor, a: float = 1/30, tau=1/50):
    from torch import exp, sqrt, pi
    x, t = grid[:, 0], grid[:, 1]
    return -(2 - (x - 0.5)**2/(a*(t + tau)))*exp(-(x - 0.5)**2/(4*a*(t + tau)))\
        / (8*a*sqrt(pi*a*(t + tau))*(t + tau))


def heat_1d_dt(grid: torch.Tensor, a: float = 1/30, tau: float = 1/50):
    from torch import exp, sqrt, pi
    x, t = grid[:, 0], grid[:, 1]
    return -exp(-(x - 0.5)**2/(4*a*(t + tau)))\
        / (4*sqrt(pi*a*(t + tau))
           * (t + tau)) \
        + (x - 0.5)**2*exp(-(x - 0.5)**2/(4*a*(t + tau)))\
        / (8 * a*sqrt(pi*a*(t + tau))*(t + tau)**2)
