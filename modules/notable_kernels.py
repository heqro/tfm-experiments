import torch


def verify_input(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.shape[1] != 1 or x.shape[2] != y.shape[1]:
        raise Exception(
            f'Expected input tensor to have shape [N, 1, dim]. Received shape: {x.shape}.')


def compute_radii(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.norm(x - y, dim=-1, p=2.0)


def gaussian_kernel(eps: float):
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        verify_input(x, y)
        radii = compute_radii(x, y)
        return torch.exp(-eps ** 2 * radii ** 2)
    return fn


def mq_kernel_sarra(eps: float):
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        verify_input(x, y)
        radii = compute_radii(x, y)
        return torch.sqrt(1 + eps ** 2 * radii ** 2)
    return fn


def mq_kernel_hardy(eps: float):
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        verify_input(x, y)
        radii = compute_radii(x, y)
        return torch.sqrt(eps ** 2 + radii ** 2)
    return fn


def phs_kernel(exponent: int):
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        verify_input(x, y)
        return compute_radii(x, y) ** exponent
    return fn
