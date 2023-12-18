import torch

# Define the RBF kernel


def rbf_kernel(radius: torch.Tensor, eps: float):
    return torch.exp(-eps ** 2 * radius ** 2)

# Define the RBF interpolant model


class RBFInterpolant(torch.nn.Module):
    def __init__(self, centers: torch.Tensor, eps: float = 1.0, alphas=None):
        super(RBFInterpolant, self).__init__()
        self.eps = eps
        self.alphas = torch.nn.Parameter(
            (torch.rand_like(centers) - 0.5)) if alphas is None else torch.nn.Parameter(
                torch.tensor(alphas))
        self.centers = centers

    def forward(self, x: torch.Tensor):
        radius = torch.stack([torch.abs(x_input - self.centers)
                             for x_input in x])
        products_list = self.alphas * rbf_kernel(radius, self.eps)
        return torch.sum(products_list, dim=1)


class PolynomialRBFInterpolant(torch.nn.Module):
    def __init__(self,
                 centers: torch.Tensor, eps: float = 1.0, alphas: list[float] = None,
                 cte: float = None, line: float = None, quad: float = None):
        super(PolynomialRBFInterpolant, self).__init__()
        self.eps = eps
        self.alphas = torch.nn.Parameter(
            (torch.rand_like(centers) - 0.5)) \
            if alphas is None else torch.nn.Parameter(
            torch.tensor(alphas))
        self.centers = centers
        self.cte = torch.nn.Parameter(torch.rand(1)) if cte is None \
            else torch.nn.Parameter(torch.tensor(cte))
        self.line = torch.nn.Parameter(torch.rand(1)) if line is None \
            else torch.nn.Parameter(torch.tensor(line))
        self.quad = torch.nn.Parameter(torch.rand(1)) if quad is None \
            else torch.nn.Parameter(torch.tensor(quad))

    def forward(self, x: torch.Tensor):
        radius = torch.stack([torch.abs(x_input - self.centers)
                             for x_input in x])
        products_list = self.alphas * rbf_kernel(radius, self.eps)
        return self.quad * x ** 2 + self.line * x + self.cte + torch.sum(products_list, dim=1)
