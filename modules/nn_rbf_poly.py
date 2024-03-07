import torch

from notable_kernels import *
from nn_poly import PolynomialInterpolant as Poly


class RBFInterpolant(torch.nn.Module):
    def __init__(self, centers: torch.Tensor, degree: int,
                 dev: torch.device | str = 'cpu', coefs_rbf: list[float] = [],
                 coefs_poly: list[float] = [], dim=1,
                 kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = phs_kernel(3)):

        super(RBFInterpolant, self).__init__()

        from nn_rbf import RBFInterpolant as RBF
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

        from nn_rbf import RBFInterpolantFreeCenters as RBF
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


def interpolation_matrix(interpolant: RBFInterpolant | RBFInterpolantFreeCenters, x: torch.Tensor) -> torch.Tensor:
    A = interpolant.rbf.get_interpolation_matrix()
    P = interpolant.poly.get_interpolation_matrix(x)
    aux = torch.cat((A, P), dim=1)
    padding = torch.zeros(
        size=(P.t().size(0), aux.size(1) - P.t().size(1)))
    P_t_padded = torch.cat((P.t(), padding), dim=1)
    return torch.cat((aux, P_t_padded), dim=0)


def list_coefs(interpolant: RBFInterpolant | RBFInterpolantFreeCenters) -> tuple[torch.Tensor, torch.Tensor]:
    return interpolant.rbf.get_coefs(), interpolant.poly.get_coefs()


def fix_coefs(interpolant: RBFInterpolant | RBFInterpolantFreeCenters,
              rbf_coefs: torch.Tensor,
              poly_coefs: torch.Tensor):
    interpolant.rbf.set_coefs(rbf_coefs)
    interpolant.poly.set_coefs(poly_coefs)


def list_centers(interpolant: RBFInterpolant | RBFInterpolantFreeCenters):
    return interpolant.rbf.centers
