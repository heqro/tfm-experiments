import torch

from notable_kernels import phs_kernel
from nn_poly import PolynomialInterpolant as Poly


class RBFInterpolant(torch.nn.Module):
    def __init__(self, k: int, centers: torch.Tensor, degree: int,
                 dev: torch.device | str = 'cpu', coefs_rbf: list[float] = [],
                 coefs_poly: list[float] = [], dim=1):
        super(RBFInterpolant, self).__init__()
        from nn_rbf import RBFInterpolant as RBF
        self.rbf = RBF(centers, phs_kernel(2*k+1), coefs_rbf).to(dev)
        self.poly = Poly(degree, coefs_poly, dim).to(dev)

    def forward(self, x: torch.Tensor):
        return self.rbf(x) + self.poly(x)

    def get_interpolation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        return interpolation_matrix(self, x)

    def get_coefs(self) -> torch.Tensor:
        return list_coefs(self)

    def set_coefs(self, coefs: torch.Tensor | list[float]):
        fix_coefs(self, coefs)


class RBFInterpolantFreeCenters(torch.nn.Module):
    def __init__(self, k: int, n_centers: int, degree: int,
                 dev: torch.device | str = 'cpu', coefs_rbf: list[float] = [],
                 coefs_poly: list[float] = [], dim=1):

        super(RBFInterpolantFreeCenters, self).__init__()

        from nn_rbf import RBFInterpolantFreeCenters as RBF
        self.rbf = RBF(dim, n_centers, phs_kernel(2*k+1), coefs_rbf).to(dev)
        self.poly = Poly(degree, coefs_poly, dim).to(dev)

    def forward(self, x: torch.Tensor):
        return self.rbf(x) + self.poly(x)

    def get_interpolation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        return interpolation_matrix(self, x)

    def get_coefs(self) -> torch.Tensor:
        return list_coefs(self)

    def set_coefs(self, coefs: torch.Tensor | list[float]):
        fix_coefs(self, coefs)


def interpolation_matrix(interpolant: RBFInterpolant | RBFInterpolantFreeCenters, x: torch.Tensor) -> torch.Tensor:
    A = interpolant.rbf.get_interpolation_matrix()
    P = interpolant.poly.get_interpolation_matrix(x)
    aux = torch.cat((A, P), dim=1)
    padding = torch.zeros(
        size=(P.t().size(0), aux.size(1) - P.t().size(1)))
    P_t_padded = torch.cat((P.t(), padding), dim=1)
    return torch.cat((aux, P_t_padded), dim=0)


def list_coefs(interpolant: RBFInterpolant | RBFInterpolantFreeCenters) -> torch.Tensor:
    return torch.cat((interpolant.rbf.coefs, interpolant.poly.coefs))


def fix_coefs(interpolant: RBFInterpolant | RBFInterpolantFreeCenters, coefs: torch.Tensor | list[float]):
    idx = len(interpolant.rbf.get_coefs())
    interpolant.rbf.set_coefs(coefs[:idx])
    interpolant.poly.set_coefs(coefs[idx:])
