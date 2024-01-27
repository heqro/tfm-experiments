import torch
from nn_rbf_phs import RBFInterpolant as RBF
from nn_poly import PolynomialInterpolant as Poly


class RBFInterpolant(torch.nn.Module):
    def __init__(self, k: int, centers: torch.Tensor, degree: int, coefs_rbf: list[float] = [], coefs_poly: list[float] = []):
        super(RBFInterpolant, self).__init__()
        self.rbf = RBF(k, centers, coefs_rbf)
        self.poly = Poly(degree, coefs_poly)

    def forward(self, x: torch.Tensor):
        return self.rbf(x) + self.poly(x)

    def get_interpolation_matrix(self, x: torch.Tensor):
        A = self.rbf.get_interpolation_matrix()
        P = self.poly.get_interpolation_matrix(x)
        aux = torch.cat((A, P), dim=1)
        padding = torch.zeros(
            size=(P.t().size(0), aux.size(1) - P.t().size(1)))
        P_t_padded = torch.cat((P.t(), padding), dim=1)
        return torch.cat((aux, P_t_padded), dim=0)

    def get_coefs(self):
        return torch.cat((self.rbf.coefs, self.poly.coefs))
