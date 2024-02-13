import matplotlib.pyplot as plt
import torch
import sys
sys.path.insert(0, '../../../modules')
if True:
    from nn_rbf_phs import RBFInterpolant
    from notable_functions import franke_func


def d_meshgrid(*args):
    # Create grid coordinates along each dimension
    grid = torch.meshgrid(*args)
    # Convert the grid coordinates to tensors
    grid_tensors = [coord.flatten() for coord in grid]
    # Stack the grid tensors along a new dimension to create the d-dimensional meshgrid
    meshgrid = torch.stack(grid_tensors, dim=-1)
    return meshgrid


conditioning_vector = []
for N in range(95, 120, 5):
    grid = d_meshgrid(*[torch.linspace(0, 1, N) for _ in range(2)])
    nn = RBFInterpolant(1, grid)
    conditioning_vector += [torch.linalg.cond(nn.get_interpolation_matrix())]
    print(N)

print(conditioning_vector)
