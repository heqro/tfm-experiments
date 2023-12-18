# Usage: $ python coefficients_optimizer.py N eps
import csv
import torch
import nn_rbf
import sys
import os

N = int(sys.argv[1])
eps = float(sys.argv[2])

# Generate input points
x = torch.linspace(0, 2, N)
y = torch.exp(x)


file_path = f'coefs/{N}-{eps}.csv'
if os.path.exists(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Read the last line of the file
        last_line = None
        for row in csv_reader:
            last_line = row
        if last_line is not None:
            # Extract alphas from the last line
            alphas = [float(val) for val in last_line[:-1]]
else:
    alphas = None


rbf_interpolant = nn_rbf.RBFInterpolant(centers=x, eps=eps, alphas=alphas)
optimizer = torch.optim.Adam(rbf_interpolant.parameters(), lr=1e-3)

save_every = 30000
it = 1

progress = torch.zeros(save_every, len(x) + 1)  # alphas + loss
while True:
    loss = torch.mean(torch.sum((rbf_interpolant(x) - y) ** 2))
    progress[it - 1, -1] = loss.item()
    progress[it - 1, :-1] = rbf_interpolant.alphas

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if it == save_every:
        with open(file_path, 'a') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(progress.detach().numpy())
        progress = torch.zeros(save_every, len(x) + 1)
        # break
        it = 0
    it = it + 1
