import torch


class PolynomialInterpolant(torch.nn.Module):
    def __init__(self, degree: int = 1, coefs: list[float] = []):
        super(PolynomialInterpolant, self).__init__()
        if coefs != []:
            if len(coefs) != degree + 1:
                raise Exception(f"Coefficients vector of length ({len(coefs)})  \
                        does not fit prescribed degree ({degree}).")
            else:
                self.coefs = torch.nn.Parameter(torch.tensor(coefs))
        else:
            self.coefs = torch.nn.Parameter(torch.rand(degree + 1))
        self.powers = torch.arange(start=0, end=degree + 1)

    def forward(self, x: torch.Tensor):
        input_vector = x.view(-1, 1)
        result = self.coefs @ (input_vector ** self.powers).t()
        return result.view_as(x)
