import torch


class RBFInterpolant(torch.nn.Module):
    def __init__(self, k: int, centers: torch.Tensor, coefs: list[float] = []):
        super(RBFInterpolant, self).__init__()
        self.centers = centers
        if coefs == []:
            self.coefs = torch.nn.Parameter(
                (torch.rand(centers.shape[0]) - 0.5))
        else:
            self.set_coefs(coefs)
        self.exponent = 2 * k + 1
        self.expected_shape = centers.shape[1]

    def forward(self, x: torch.Tensor):
        if self.expected_shape != x.shape[1]:
            raise Exception(
                f'Expected input tensor of shape[1] equal to: {self.expected_shape}')
        radii = torch.norm(x[:, None, :] - self.centers, dim=-1, p=2.0)
        radii_exp = radii ** self.exponent
        radii_transposed = radii_exp.t()
        result = self.coefs @ radii_transposed
        return result.reshape([x.shape[0], 1])

    def get_interpolation_matrix(self, *args):
        return torch.norm(self.centers[:, None, :] - self.centers, dim=-1, p=2.0) ** self.exponent

    def get_coefs(self):
        return self.coefs

    def set_coefs(self, coefs: torch.Tensor | list[float]):
        if len(coefs) != self.centers.shape[0]:
            raise Exception(
                f'Mismatched dimensions: coefs {len(coefs)} w.r.t. centers {self.centers.shape[0]}')
        self.coefs = torch.nn.Parameter(torch.tensor(coefs))
