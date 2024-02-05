import torch


class RBFInterpolant(torch.nn.Module):
    def __init__(self, k: int, centers: torch.Tensor, coefs: list[float] = []):
        super(RBFInterpolant, self).__init__()
        if coefs == []:
            self.coefs = torch.nn.Parameter(
                (torch.rand(centers.shape[0]) - 0.5))
        else:
            if len(coefs) != centers.shape[0]:
                raise Exception(
                    f'Mismatched dimensions: coefs {len(coefs)} w.r.t. centers {centers.shape[0]}')
            else:
                torch.nn.Parameter(torch.tensor(coefs))
        self.exponent = 2 * k + 1
        self.centers = centers

    def forward(self, x: torch.Tensor):
        return (self.coefs @ (torch.norm(x[:, None, :] - self.centers, dim=-1, p=2.0)
                              ** self.exponent).t()).reshape([x.shape[0], 1])

    def get_interpolation_matrix(self, *args):  # TODO REVISE
        auxiliary_vector = self.centers.view(-1, 1)
        return torch.abs(auxiliary_vector - self.centers) ** self.exponent

    def get_coefs(self):  # TODO REVISE
        return self.coefs

    def set_coefs(self, coefs: torch.Tensor | list[float]):  # TODO REVISE
        if len(coefs) != self.centers.shape[0]:
            raise Exception(
                f'Mismatched dimensions: coefs {len(coefs)} w.r.t. centers {self.centers.shape[0]}')
        self.coefs = torch.nn.Parameter(torch.tensor(coefs))
