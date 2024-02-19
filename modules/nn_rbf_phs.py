import torch


class RBFInterpolant(torch.nn.Module):
    def __init__(self, k: int, centers: torch.Tensor, coefs: list[float] = []):
        def set_coefs(coefs: torch.Tensor | list[float]):
            if len(coefs) != self.centers.shape[0]:
                raise Exception(
                    f'Mismatched dimensions: coefs {len(coefs)} w.r.t. centers {self.centers.shape[0]}')
            self.coefs = torch.nn.Parameter(torch.tensor(coefs))
        super(RBFInterpolant, self).__init__()
        self.centers = centers
        if coefs == []:
            self.coefs = torch.nn.Parameter(
                (torch.rand(centers.shape[0]) - 0.5))
        else:
            set_coefs(coefs)
        self.exponent = 2 * k + 1
        self.dim = centers.shape[1]

    def forward(self, x: torch.Tensor):
        if x.shape[1] != 1 or x.shape[2] != self.dim:
            raise Exception(
                f'Expected input tensor to have shape [N, 1, dim]. Received shape: {x.shape}.')
        result = self.coefs @ (torch.norm(x - self.centers,
                                          dim=-1, p=2.0) ** self.exponent).t()
        return result.reshape([-1, 1])

    def get_interpolation_matrix(self, *args):
        return torch.norm(self.centers[:, None, :] - self.centers, dim=-1, p=2.0) ** self.exponent

    def get_coefs(self):
        return self.coefs

    def set_coefs(self, coefs: torch.Tensor | list[float]):
        if len(coefs) != self.centers.shape[0]:
            raise Exception(
                f'Mismatched dimensions: coefs {len(coefs)} w.r.t. centers {self.centers.shape[0]}')
        self.coefs = torch.nn.Parameter(torch.tensor(coefs))
