from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from torch import Tensor
import torch
import sys
import os
from matplotlib.backends.backend_pdf import PdfPages
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Parameters
N = int(sys.argv[1])
k = int(sys.argv[2])
degree = int(sys.argv[3])
m = 2*k+1

# Function to intterpolate


def fn_interpolate(x: Tensor):
    return torch.exp(x)


x = torch.linspace(0, 2, N)
y = fn_interpolate(x)


# Interpolator
f = torch.cat((y, torch.zeros(degree + 1)))
sys.path.insert(0, '../../../modules')
if degree < 0:
    from nn_rbf_phs import RBFInterpolant
    nn_interpolator = RBFInterpolant(k=1, centers=x)
    Interpolation_Matrix = nn_interpolator.get_interpolation_matrix()
    lda_optimum = torch.linalg.solve(Interpolation_Matrix, f)
    degree = -1
    nn_interpolator_untrained = RBFInterpolant(k=1, centers=x,
                                               coefs=lda_optimum)
else:
    from nn_rbf_phs_poly import RBFInterpolant
    nn_interpolator = RBFInterpolant(k=1, centers=x, degree=degree)
    Interpolation_Matrix = nn_interpolator.get_interpolation_matrix(x)
    lda_optimum = torch.linalg.solve(Interpolation_Matrix, f)
    nn_interpolator_untrained = RBFInterpolant(k=1, centers=x, degree=degree,
                                               coefs_rbf=lda_optimum[:N],
                                               coefs_poly=lda_optimum[N:])

# Compute parameters by matrix inversion
params_number = sum(p.numel() for p in nn_interpolator.parameters()
                    if p.requires_grad)  # number of parameters

# Iteratively find parameters
max_iterations = 200000
parameters_progress_mse = torch.zeros(
    size=(1, params_number + 1), requires_grad=False)

best_loss_index = 1
optimizer = torch.optim.Adam(params=nn_interpolator.parameters(), lr=1e-3)
# Use ReduceLROnPlateau to adjust the learning rate
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=100, threshold=1e-8)


for i in range(max_iterations):
    loss = torch.sum((nn_interpolator(x) - y) ** 2) / N

    # print(f'{i}: {loss.item()}', end='\r')
    if loss < 1e-16:
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update lr based on loss
    scheduler.step(loss)

    # Update log
    parameters_progress_mse = torch.cat(
        (parameters_progress_mse, torch.zeros(1, params_number + 1)))
    parameters_progress_mse[-1, :-1] = nn_interpolator.get_coefs()
    parameters_progress_mse[-1, -1] = loss.item()

    if loss < parameters_progress_mse[best_loss_index, -1]:
        best_loss_index = i


nn_interpolator.set_coefs(parameters_progress_mse[best_loss_index, :-1])
print(f'Final LR: {optimizer.param_groups[0]["lr"]}')

# Interpolation error plot
with torch.no_grad():
    x_verification = torch.linspace(0, 2, 1000)
    y_verification = fn_interpolate(x_verification)
    fig = plt.figure()
    for vertical in x:
        plt.axvline(vertical, color="grey", linestyle='--')

    plt.semilogy(x_verification,
                 torch.abs(nn_interpolator(x_verification) - y_verification),
                 label="Trained", linewidth='3.5')
    plt.semilogy(x_verification,
                 torch.abs(nn_interpolator_untrained(
                     x_verification) - y_verification),
                 label="Untrained", linestyle='--')

    plt.legend()
    plt.title("Log-10 interpolation error")
    # Save plot
    pp = PdfPages(f'r{m}_N{N}_phs_deg_{degree}_interpolation_error.pdf')
    pp.savefig(fig)
    pp.close()
    plt.close()

# Approximation towards exact parameters
colors = cm.rainbow(torch.linspace(0, 1, params_number))
fig = plt.figure()
for i in range(params_number):
    plt.plot(parameters_progress_mse[:best_loss_index+1, i].detach().numpy(),
             label=f"Lambda {i}", color=colors[i])
    plt.axhline(lda_optimum[i], color=colors[i],
                linestyle='dashed', label=f'Exact lambda {i}')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.ylabel('Parameter values')
plt.xlabel('Iterations')
plt.title('Approximation towards exact parameters (matrix expression)')
# Save plot
pp = PdfPages(f'r{m}_N{N}_phs_deg_{degree}_params.pdf')
pp.savefig(fig, bbox_inches='tight')
pp.close()
plt.close()

# Loss function
fig = plt.figure()
plt.semilogy(parameters_progress_mse[:best_loss_index, -1].detach().numpy())
plt.title('Loss function')
# Save plot
pp = PdfPages(f'r{m}_N{N}_phs_deg_{degree}_loss.pdf')
pp.savefig(fig)
pp.close()
plt.close()
