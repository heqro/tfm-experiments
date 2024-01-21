import torch


def rbf_kernel(radius: torch.Tensor, eps: float):
    return torch.exp(-eps ** 2 * radius ** 2)


class RBFInterpolant(torch.nn.Module):
    def __init__(self, centers: torch.Tensor, eps: float = 1.0, alphas: list[float] = None):
        super(RBFInterpolant, self).__init__()
        self.eps = eps
        self.alphas = \
            torch.nn.Parameter((torch.rand_like(centers) - 0.5)) if alphas is None \
            else torch.nn.Parameter(torch.tensor(alphas))
        self.centers = centers

    def forward(self, x: torch.Tensor):
        radius = torch.stack([torch.abs(x_input - self.centers)
                             for x_input in x])
        products_list = self.alphas * rbf_kernel(radius, self.eps)
        return torch.sum(products_list, dim=1)


class PolynomialRBFInterpolant(torch.nn.Module):
    def __init__(self,
                 centers: torch.Tensor, eps: float = 1.0,
                 alphas: list[float] = None, degree: int = 2, coefficients: list[float] = None):
        super(PolynomialRBFInterpolant, self).__init__()

        # RBFs
        self.eps = eps
        self.alphas = torch.nn.Parameter(
            (torch.rand_like(centers) - 0.5)) \
            if alphas is None else torch.nn.Parameter(
            torch.tensor(alphas))
        self.centers = centers

        # Polynomials
        self.coefficients = torch.nn.Parameter(torch.rand(
            degree + 1)) if coefficients is None else torch.nn.Parameter(torch.tensor(coefficients))

        self.powers = torch.arange(
            self.coefficients.size(0), dtype=torch.float32)
        self.powers = self.powers.view(-1, 1)

    def forward(self, x: torch.Tensor):
        # RBF Section
        radius = torch.stack([torch.abs(x_input - self.centers)
                             for x_input in x])
        products_list = self.alphas * rbf_kernel(radius, self.eps)
        rbf_result = torch.sum(products_list, dim=1)

        # Polynomial Section
        poly_result = torch.matmul(self.coefficients, x ** self.powers)

        return poly_result + rbf_result
