from typing import Callable, Union
import torch
from torch import nn, Tensor
from notable_kernels import gaussian_kernel


class RBF_Free_All(nn.Module):
    def __init__(self, input_dim: int, num_centers: int, output_dim: int,
                 kernel: Callable[[Union[float, Tensor]],
                                  Callable[[Tensor, Tensor], Tensor]] = gaussian_kernel,):

        super(RBF_Free_All, self).__init__()
        self.centers = nn.Parameter(Tensor(num_centers, input_dim))
        nn.init.normal_(tensor=self.centers, mean=0, std=1)

        self.shape = nn.Parameter(Tensor(num_centers))
        nn.init.constant_(tensor=self.shape, val=1)

        self.linear = nn.Linear(num_centers, output_dim, bias=False)
        self.kernel = kernel(self.shape)

    def forward(self, x):
        kernel_out = self.kernel(x, self.centers)
        return self.linear(kernel_out)

    def get_interpolation_matrix(self, *args):
        return interpolation_matrix(self)

    def set_coefs(self, coefs: Tensor):
        return fix_coefs(self, coefs)

    def get_coefs(self):
        return list_coefs(self)

    def set_centers(self, centers: Tensor):
        with torch.no_grad():
            self.centers.data = torch.nn.Parameter(centers)


class RBF_Fix_All(nn.Module):
    def __init__(self,
                 centers: Tensor,
                 rbf_kernel: Callable[[Tensor, Tensor], Tensor],
                 coefs: list[float] = []):

        super(RBF_Fix_All, self).__init__()

        self.centers = centers.clone().detach()
        self.output_layer = torch.nn.Linear(
            in_features=centers.shape[0], out_features=1, bias=False)
        if coefs != []:
            fix_coefs(self, torch.tensor(coefs))
        self.kernel = rbf_kernel

    def forward(self, x: Tensor):
        kernel_values = self.kernel(x, self.centers)
        result = self.output_layer(kernel_values)
        return result

    def get_interpolation_matrix(self, *args):
        return interpolation_matrix(self)

    def set_coefs(self, coefs: Tensor):
        return fix_coefs(self, coefs)

    def get_coefs(self):
        return list_coefs(self)


class RBF_Free_Centers(torch.nn.Module):
    def __init__(self,
                 dim: int, n_centers: int,
                 rbf_kernel: Callable[[Tensor, Tensor], Tensor],
                 coefs: list[float] = []):

        super(RBF_Free_Centers, self).__init__()

        self.centers = torch.nn.Parameter(
            torch.rand(n_centers, dim), requires_grad=True)
        self.output_layer = torch.nn.Linear(
            in_features=n_centers, out_features=1, bias=False)
        if coefs != []:
            fix_coefs(self, torch.tensor(coefs))
        self.kernel = rbf_kernel

    def forward(self, x: Tensor):
        kernel_values = self.kernel(x, self.centers)
        result = self.output_layer(kernel_values)
        return result

    def get_interpolation_matrix(self, *args):
        return interpolation_matrix(self)

    def set_coefs(self, coefs: Tensor):
        return fix_coefs(self, coefs)

    def get_coefs(self):
        return list_coefs(self)

    def set_centers(self, centers: Tensor):
        with torch.no_grad():
            self.centers.data = torch.nn.Parameter(centers)


def interpolation_matrix(interpolant: RBF_Free_All | RBF_Fix_All | RBF_Free_Centers, *args) -> Tensor:
    return interpolant.kernel(interpolant.centers, interpolant.centers)


def fix_coefs(interpolant: RBF_Free_All | RBF_Fix_All | RBF_Free_Centers, coefs: Tensor):
    with torch.no_grad():
        interpolant.output_layer.weight = torch.nn.Parameter(coefs)


def list_coefs(interpolant: RBF_Free_All | RBF_Fix_All | RBF_Free_Centers):
    return interpolant.output_layer.weight
