import torch
from itertools import product


class PolynomialInterpolant(torch.nn.Module):
    def __init__(self, degree: int = 1, coefs: list[float] = [], dim: int = 1):
        def set_coefs(coefs: torch.Tensor | list[float]):
            coefs = torch.nn.Parameter(torch.tensor(coefs))

        def compute_products(dim: int, degree: int):
            products = []
            for exponents in product(range(degree + 1), repeat=dim):
                if sum(exponents) <= degree:
                    products.append(exponents)
            return products

        super(PolynomialInterpolant, self).__init__()
        if dim < 1:
            raise Exception(f'Input dimensions equal to {dim}!')
        self.dim = dim
        products_list = compute_products(dim=self.dim, degree=degree)
        self.exponents = torch.nn.Parameter(
            torch.tensor(products_list), requires_grad=False)
        self.coefs = torch.nn.Parameter(
            torch.tensor(torch.rand(len(products_list))))
        if coefs != []:
            if len(coefs) != len(products_list):
                raise Exception(f"Coefficients vector of length ({len(coefs)})  \
                        does not fit prescribed degree ({degree}).")
            else:
                set_coefs(coefs)

    def forward(self, x: torch.Tensor):
        if x.shape[1] != 1 or x.shape[2] != self.dim:
            raise Exception('Expected input tensor to have shape [N, 1, dim].')
        poly_matrix = torch.prod(x ** self.exponents, dim=-1)
        return (poly_matrix @ self.coefs).reshape([-1, 1])

    def get_interpolation_matrix(self, x: torch.Tensor):
        if x.shape[1] != 1 or x.shape[2] != self.dim:
            raise Exception('Expected input tensor to have shape [N, 1, dim].')
        return (x ** self.exponents).squeeze()

    def set_coefs(self, coefs: torch.Tensor | list[float]):
        self.coefs = torch.nn.Parameter(torch.tensor(coefs))
