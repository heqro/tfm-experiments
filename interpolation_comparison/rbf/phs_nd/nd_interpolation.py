

import os
import torch
import sys
import csv

sys.path.insert(0, '../../../modules')
if True:
    from notable_functions import *
    from nn_rbf_phs import RBFInterpolant
# Usage example: python nd_interpolation.py 5 1 -1 'a' 3
N = int(sys.argv[1])
k = int(sys.argv[2])
degree = int(sys.argv[3])
if degree > 0:
    raise Exception('Not implemented yet')
degree = -1  # if degree < 0 else degree
m = 2*k+1
fn = franke_func if sys.argv[4] == 'franke' else norm_cos
dim = 2
if fn != franke_func:
    dim = int(sys.argv[5])

experiment_string = f'{fn.__name__}_interpolation_dim{dim}_N{N}_r{m}_deg{degree}'


def d_meshgrid(*args):
    # Create grid coordinates along each dimension
    grid = torch.meshgrid(*args)
    # Convert the grid coordinates to tensors
    grid_tensors = [coord.flatten() for coord in grid]
    # Stack the grid tensors along a new dimension to create the d-dimensional meshgrid
    meshgrid = torch.stack(grid_tensors, dim=-1)
    return meshgrid


grid = d_meshgrid(*[torch.linspace(0, 1, N) for _ in range(dim)]).to('cuda')

nn = RBFInterpolant(k, grid).to('cuda')
if os.path.exists(f'experiments_csv/{experiment_string}.csv'):
    with open(f'experiments_csv/{experiment_string}.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Read the last line of the file
        last_line = None
        for row in csv_reader:
            last_line = row
        if last_line is not None:
            coefficients = [float(val) for val in last_line[:-1]]
            nn = RBFInterpolant(k, grid, coefficients).to('cuda')

target = (fn((grid - 1/2) * 10) / 2 + 1/2).reshape(-1, 1)

best_loss = torch.inf
best_coefs = torch.zeros_like(nn.get_coefs()).to('cuda')

optimizer = torch.optim.Adam(nn.parameters(), lr=.75e-1)

batch_size = 64
num_batches = len(target) // batch_size

while True:
    for i in range(100000):
        loss = torch.mean(torch.sum((nn(grid) - target) ** 2))

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_coefs = nn.get_coefs()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with open(f'experiments_csv/{experiment_string}.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            best_coefs.cpu().detach().numpy().tolist() + [best_loss])