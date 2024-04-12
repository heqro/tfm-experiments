import torch
from typing import Callable


def verify_input(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.shape[1] != 1 or x.shape[2] != y.shape[1]:
        raise Exception(
            f'Expected input tensor to have shape [N, 1, dim]. Received shape: {x.shape}.')

def compute_radii_squared(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, d)
    centers = centers.unsqueeze(0)  # Shape: (1, num_centers, d)
    
    return torch.sum((x - centers)**2, dim=2)  # Shape: (batch_size, num_centers)


def gaussian_kernel(eps: float | torch.Tensor):
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        radii_sq = compute_radii_squared(x, y)
        return torch.exp(-eps ** 2 * radii_sq)
    return fn


def mq_kernel_sarra(eps: float | torch.Tensor):
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        radii_sq = compute_radii_squared(x, y)
        return torch.sqrt(1 + eps ** 2 * radii_sq)
    return fn


def mq_kernel_hardy(eps: float):
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        radii = compute_radii_squared(x, y)
        return torch.sqrt(eps ** 2 + radii)
    return fn


def phs_kernel(exponent: float | torch.Tensor):
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return compute_radii_squared(x, y) ** (exponent/2)
    return fn
