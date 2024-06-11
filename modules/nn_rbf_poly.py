import torch
from torch import Tensor
from typing import Union
from notable_kernels import *
from nn_poly import PolynomialInterpolant as Poly


class RBF_Poly_Free_All(torch.nn.Module):
    def __init__(self, input_dim: int, num_centers: int, output_dim: int,
                 kernel: Callable[[Union[float, Tensor]],
                                  Callable[[Tensor, Tensor], Tensor]],
                 dev: torch.device | str = 'cpu', degree: int = 1, starting_shape=1.,
                 left_lim=0., right_lim=1.):
        super(RBF_Poly_Free_All, self).__init__()

        from nn_rbf import RBF_Free_All as RBF
        self.rbf = RBF(input_dim, num_centers, output_dim,
                       kernel, starting_shape, left_lim, right_lim).to(dev)
        self.poly = Poly(degree, dim=input_dim).to(dev)

    def forward(self, x: torch.Tensor):
        return self.rbf(x) + self.poly(x)

    def get_interpolation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        return interpolation_matrix(self, x)

    def get_coefs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return list_coefs(self)

    def set_coefs(self, rbf_coefs: torch.Tensor, poly_coefs: torch.Tensor):
        fix_coefs(self, rbf_coefs, poly_coefs)

    def get_centers(self):
        return list_centers(self)

    def set_centers(self, centers: torch.Tensor):
        self.rbf.set_centers(centers)


class RBF_Poly(torch.nn.Module):
    def __init__(self, centers: torch.Tensor, degree: int,
                 dev: torch.device | str = 'cpu', coefs_rbf: list[float] = [],
                 coefs_poly: list[float] = [], dim=1,
                 kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = phs_kernel(3)):

        super(RBF_Poly, self).__init__()

        from nn_rbf import RBF_Fix_All as RBF
        self.rbf = RBF(centers, kernel, coefs_rbf).to(dev)
        self.poly = Poly(degree, coefs_poly, dim).to(dev)

    def forward(self, x: torch.Tensor):
        return self.rbf(x) + self.poly(x)

    def get_interpolation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        return interpolation_matrix(self, x)

    def get_coefs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return list_coefs(self)

    def set_coefs(self, rbf_coefs: torch.Tensor, poly_coefs: torch.Tensor):
        fix_coefs(self, rbf_coefs, poly_coefs)

    def get_centers(self):
        return list_centers(self)


class RBFInterpolantFreeCenters(torch.nn.Module):
    def __init__(self, n_centers: int, degree: int,
                 dev: torch.device | str = 'cpu', coefs_rbf: list[float] = [],
                 coefs_poly: list[float] = [], dim=1,
                 kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = phs_kernel(3)):

        super(RBFInterpolantFreeCenters, self).__init__()

        from nn_rbf import RBF_Free_Centers as RBF
        self.rbf = RBF(dim, n_centers, kernel, coefs_rbf).to(dev)
        self.poly = Poly(degree, coefs_poly, dim).to(dev)

    def forward(self, x: torch.Tensor):
        return self.rbf(x) + self.poly(x)

    def get_interpolation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        return interpolation_matrix(self, x)

    def get_coefs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return list_coefs(self)

    def set_coefs(self, rbf_coefs: torch.Tensor, poly_coefs: torch.Tensor):
        fix_coefs(self, rbf_coefs, poly_coefs)

    def get_centers(self):
        return list_centers(self)


def interpolation_matrix(interpolant: RBF_Poly_Free_All | RBF_Poly | RBFInterpolantFreeCenters, x: torch.Tensor) -> torch.Tensor:
    A = interpolant.rbf.kernel(x, interpolant.get_centers())
    P = interpolant.poly.get_interpolation_matrix(x)
    aux = torch.cat((A, P), dim=1)
    padding = torch.zeros(
        size=(P.T.size(0), aux.size(1) - P.T.size(1)), device=aux.device)
    P_t_padded = torch.cat((P.T, padding), dim=1)
    return torch.cat((aux, P_t_padded), dim=0)


def list_coefs(interpolant: RBF_Poly_Free_All | RBF_Poly | RBFInterpolantFreeCenters) -> tuple[torch.Tensor, torch.Tensor]:
    return interpolant.rbf.get_coefs(), interpolant.poly.get_coefs()


def fix_coefs(interpolant:  RBF_Poly_Free_All | RBF_Poly | RBFInterpolantFreeCenters,
              rbf_coefs: torch.Tensor,
              poly_coefs: torch.Tensor):
    interpolant.rbf.set_coefs(rbf_coefs)
    interpolant.poly.set_coefs(poly_coefs)


def list_centers(interpolant: RBF_Poly_Free_All | RBF_Poly | RBFInterpolantFreeCenters):
    return interpolant.rbf.centers
