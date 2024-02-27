from typing import Callable
import torch


class RBFInterpolant(torch.nn.Module):
    def set_coefs(self, coefs: torch.Tensor | list[float]):
        if len(coefs) != self.centers.shape[0]:
            raise Exception(
                f'Mismatched dimensions: coefs {len(coefs)} w.r.t. centers {self.centers.shape[0]}')
        self.coefs = torch.nn.Parameter(torch.tensor(coefs))

    def __init__(self,
                 centers: torch.Tensor,
                 rbf_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 coefs: list[float] = []):
        super(RBFInterpolant, self).__init__()

        self.centers = centers
        if coefs == []:
            self.set_coefs(torch.nn.Parameter(
                (torch.rand(centers.shape[0]) - 0.5)))
        else:
            self.set_coefs(coefs)
        self.dim = centers.shape[1]
        self.kernel = rbf_kernel

    def forward(self, x: torch.Tensor):
        kernel_values = self.kernel(x, self.centers)
        result = kernel_values @ self.coefs
        return result

    def get_interpolation_matrix(self, *args) -> torch.Tensor:
        return self.kernel(self.centers[:, None, :], self.centers)
