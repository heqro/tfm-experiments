import torch


class RBFInterpolant(torch.nn.Module):
    def __init__(self, k: int, centers: torch.Tensor, coefs: list[float] = []):
        super(RBFInterpolant, self).__init__()
        self.coefs = \
            torch.nn.Parameter((torch.rand_like(centers) - 0.5)) if coefs == [] \
            else torch.nn.Parameter(torch.tensor(coefs))
        self.exponent = 2 * k + 1
        self.centers = centers

    def forward(self, x: torch.Tensor):
        input_vector = x.view(-1, 1)
        result = self.coefs @ (torch.abs(input_vector - self.centers)
                               ** self.exponent).t()
        return result.view_as(x)

    def get_interpolation_matrix(self, *args):
        auxiliary_vector = self.centers.view(-1, 1)
        return torch.abs(auxiliary_vector - self.centers) ** self.exponent

    def get_coefs(self):
        return self.coefs

    def set_coefs(self, coefs: torch.Tensor):
        self.coefs = torch.nn.Parameter(torch.tensor(coefs))
