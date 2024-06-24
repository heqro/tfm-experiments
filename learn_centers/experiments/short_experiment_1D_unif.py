
import csv

from typing import Any, Callable, Union
from scipy.stats import qmc
import torch
import argparse
import time
from datetime import datetime
import sys
sys.path.append('../../modules')
if True:
    from notable_functions import *
    from notable_kernels import *
    from torch import sin as torch_sin, cos as torch_cos, sign as torch_sign
    from plot_utils import *
    from grid_utils import cheb_points
    from nn_rbf import RBF_Free_All as RBF
    from nn_rbf_poly import RBF_Poly_Free_All as RBF_Poly
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from macros import save_curves, save_to_csv
dim = 1
n_verification = 400


def launch_experiment(train_size: int, centers_size: int, fn_name: str,
                      save_every: int,
                      show_interpolation_nodes: bool, kernel_name: str,
                      poly_degree: int,
                      shape: float, tag: str, xmin: float, xmax: float, lr: float, use_scheduler: bool):
    folder = 'dim_1/numerics_uniform_boundary'
    # Load function
    fn = globals().get(fn_name)
    if fn is None or not callable(fn):
        raise ValueError(f"No function named '{fn_name}' found.")
    kernel_fn = globals().get(kernel_name)
    if kernel_fn is None or not callable(kernel_fn):
        raise ValueError(f"No function named '{kernel_name}' found.")

    # Generate experiment name
    poly_suffix = f'-Poly{poly_degree}' if poly_degree > -1 else ''
    experiment_file = f'{folder}/{fn_name}-TR{train_size}-C{centers_size}' +\
        f'-K{kernel_name}{poly_suffix}-Sh{shape}-{tag}'

    # Experiment's input and target
    x_train = ((xmax - xmin) * torch.rand(train_size) +
               xmin).reshape(-1, dim).to('cuda')
    # Explicitly boundary points
    x_train[0, :] = torch.tensor(xmin)
    x_train[1, :] = torch.tensor(xmax)
    y_train = fn(x_train).reshape((-1, 1)).to('cuda')

    x_validate = torch.linspace(
        xmin, xmax, n_verification).reshape(-1, dim).to('cuda')
    y_validate = fn(x_validate).reshape((-1, 1)).to('cuda')

    if poly_degree <= -1:
        nn = RBF(input_dim=dim, num_centers=centers_size,
                 output_dim=1, kernel=kernel_fn, starting_shape=shape).to('cuda')
        nn.set_centers(torch.linspace(
            xmin, xmax, centers_size, device='cuda').reshape(-1, 1))

        def get_shape(): return nn.shape
    else:
        nn = RBF_Poly(input_dim=dim, num_centers=centers_size, output_dim=1,
                      degree=poly_degree, starting_shape=shape, kernel=kernel_fn).to('cuda')
        nn.rbf.set_centers(torch.linspace(
            xmin, xmax, centers_size, device='cuda').reshape(-1, 1))

        def get_shape(): return nn.rbf.shape

    # Training setup
    loss_fn = torch.nn.MSELoss().to('cuda')
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
    min_lr = 1e-6
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=100, threshold=1e-4, verbose=True, min_lr=min_lr)

    # Training loop
    L_inf_norm_epoch = [-1, torch.tensor(torch.inf), torch.nan, torch.nan]
    max_error_point = torch.nan

    loss_list, linf_list, l2_list, lr_list,  = [], [], [], [],

    start_time = time.time()
    epoch = 0
    while True:
        # Train
        loss = loss_fn(nn(x_train), y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        validation_error = torch.abs(nn(x_validate) - y_validate)
        max_error_idx = torch.argmax(validation_error)
        max_error_point = x_validate[max_error_idx].item()

        # Update logs
        linf_list += [validation_error[max_error_idx].item()]
        loss_list += [loss.item()]
        l2_list += [torch.sqrt(
            torch.sum(validation_error ** 2) / n_verification).item()]
        lr_list += [scheduler.optimizer.param_groups[0]['lr']]

        if use_scheduler:
            scheduler.step(loss)

        if linf_list[epoch] < L_inf_norm_epoch[1]:
            L_inf_norm_epoch = (
                epoch, linf_list[epoch], max_error_point, l2_list[epoch])

        if (epoch > 0 and epoch % save_every == 0 and save_every > 0) or (math.isclose(lr_list[epoch], min_lr, abs_tol=min_lr/10)):
            plot_figure_free_shape(centers=nn.get_centers(), fn=fn, nn=nn, nodes=x_train, dim=dim, xmin=xmin, xmax=xmax,
                                   path=f'{experiment_file}-E{epoch}', shape=get_shape(), extension='pdf',
                                   show_nodes=show_interpolation_nodes, epoch=epoch, loss=loss_list, linf=linf_list, l2=l2_list, lr=lr_list, l2_rel=None, linf_rel=None)
            if (math.isclose(lr_list[epoch], min_lr, abs_tol=min_lr/10)):
                break
        epoch += 1
    end_time = time.time()
    save_to_csv(experiment_file=f'{experiment_file}-E{epoch}.csv', kernel_name=kernel_name, nn=nn,
                shape=shape, L_inf_norm_epoch=L_inf_norm_epoch, get_shape=get_shape, elapsed_time=end_time - start_time,
                last_epoch=epoch, last_l2=l2_list[-1], last_linf=linf_list[-1], last_linf_location=max_error_point)
    save_curves(f'{folder}/curves-{fn_name}-TR{train_size}-C{centers_size}{poly_suffix}-tag{tag}-E{epoch}-curves.csv',
                loss_list, l2_list, linf_list, lr_list, l2_rel=None, linf_rel=None)


is_required = False
parser = argparse.ArgumentParser(
    description="Process parameters for training")
parser.add_argument("--train_size", type=int,
                    help="Size of training data", required=is_required, default=33)
parser.add_argument("--center_size", type=int,
                    help="Number of equispaced centers", required=is_required, default=13)
parser.add_argument("--fn_name", type=str,
                    help="Name of the function to interpolate",
                    required=is_required, default='u2')
parser.add_argument('--kernel_name', type=str, default='gaussian_kernel')
parser.add_argument('--save_every', type=int,
                    default=-1, required=False)
parser.add_argument('--show_interpolation_nodes',
                    action=argparse.BooleanOptionalAction, default=True, required=False)
parser.add_argument('--polynomial_degree', type=int,
                    default=-1, required=False)
parser.add_argument('--shape', type=float, default=2.03125)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--xmin', type=float, default=-1)
parser.add_argument('--xmax', type=float, default=1)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument('--use_scheduler',
                    action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
launch_experiment(train_size=args.train_size, centers_size=args.center_size,
                  fn_name=args.fn_name, save_every=args.save_every,
                  show_interpolation_nodes=args.show_interpolation_nodes,
                  kernel_name=args.kernel_name, shape=args.shape, poly_degree=args.polynomial_degree,
                  xmin=args.xmin, xmax=args.xmax,
                  tag=args.tag, lr=args.lr, use_scheduler=args.use_scheduler)
