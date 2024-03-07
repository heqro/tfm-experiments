from typing import Callable
import torch


class RBFInterpolant(torch.nn.Module):
    def __init__(self,
                 centers: torch.Tensor,
                 rbf_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 coefs: list[float] = []):

        super(RBFInterpolant, self).__init__()

        self.centers = centers.clone().detach()
        self.output_layer = torch.nn.Linear(
            in_features=centers.shape[0], out_features=1, bias=False)
        if coefs != []:
            fix_coefs(self, torch.tensor(coefs))
        self.kernel = rbf_kernel

    def forward(self, x: torch.Tensor):
        kernel_values = self.kernel(x, self.centers)
        result = self.output_layer(kernel_values)
        return result

    def get_interpolation_matrix(self, *args):
        return interpolation_matrix(self)

    def set_coefs(self, coefs: torch.Tensor):
        return fix_coefs(self, coefs)

    def get_coefs(self):
        return list_coefs(self)


class RBFInterpolantFreeCenters(torch.nn.Module):
    def __init__(self,
                 dim: int, n_centers: int,
                 rbf_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 coefs: list[float] = []):

        super(RBFInterpolantFreeCenters, self).__init__()

        self.centers = torch.nn.Parameter(
            torch.rand(n_centers, dim), requires_grad=True)
        self.output_layer = torch.nn.Linear(
            in_features=n_centers, out_features=1, bias=False)
        if coefs != []:
            fix_coefs(self, torch.tensor(coefs))
        self.kernel = rbf_kernel

    def forward(self, x: torch.Tensor):
        kernel_values = self.kernel(x, self.centers)
        result = self.output_layer(kernel_values)
        return result

    def get_interpolation_matrix(self, *args):
        return interpolation_matrix(self)

    def set_coefs(self, coefs: torch.Tensor):
        return fix_coefs(self, coefs)

    def get_coefs(self):
        return list_coefs(self)


def interpolation_matrix(interpolant: RBFInterpolant | RBFInterpolantFreeCenters, *args) -> torch.Tensor:
    return interpolant.kernel(interpolant.centers, interpolant.centers)


def fix_coefs(interpolant: RBFInterpolant | RBFInterpolantFreeCenters, coefs: torch.Tensor):
    with torch.no_grad():
        interpolant.output_layer.weight = torch.nn.Parameter(coefs)


def list_coefs(interpolant: RBFInterpolant | RBFInterpolantFreeCenters):
    return interpolant.output_layer.weight
