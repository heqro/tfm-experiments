import torch
from itertools import product
from math import comb


class PolynomialInterpolant(torch.nn.Module):
    def __init__(self, degree: int = 1, coefs: list[float] = [], dim: int = 1):
        def compute_products(dim: int, degree: int):
            products = []
            for exponents in product(range(degree + 1), repeat=dim):
                if sum(exponents) <= degree:
                    products.append(exponents)
            return products

        if dim < 1:
            raise Exception(f'Input dimensions equal to {dim}!')

        super(PolynomialInterpolant, self).__init__()

        self.dim = dim

        products_list = compute_products(dim=self.dim, degree=degree)
        self.exponents = torch.nn.Parameter(
            torch.tensor(products_list), requires_grad=False)

        self.output_layer = torch.nn.Linear(
            in_features=comb(degree + dim, dim), out_features=1, bias=False)

        if coefs != []:
            self.set_coefs(torch.tensor(coefs))

    def get_products_list(self, x: torch.Tensor):
        if x.shape[1] != self.dim:
            raise Exception(f'Input tensor should have shape [N, {self.dim}].')
        return x.unsqueeze(1) ** self.exponents

    def forward(self, x: torch.Tensor):
        poly_matrix = torch.prod(self.get_products_list(x), dim=-1)
        return self.output_layer(poly_matrix)

    def get_interpolation_matrix(self, x: torch.Tensor):
        return self.get_products_list(x).reshape(-1, len(self.exponents))

    def set_coefs(self, coefs: torch.Tensor):
        with torch.no_grad():
            self.output_layer.weight = torch.nn.Parameter(coefs)

    def get_coefs(self):
        return self.output_layer.weight
