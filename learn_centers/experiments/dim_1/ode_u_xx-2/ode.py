import math
import sys
from numpy import ndenumerate
import torch
import argparse
import time
if True:
    sys.path.append('../../../../modules')

from notable_kernels import *
from nn_rbf import RBF_Free_All as RBF
from nn_rbf_poly import RBF_Poly_Free_All as RBF_Poly
from macros import save_to_csv, save_curves
from torch.optim.lr_scheduler import ReduceLROnPlateau
from plot_utils import *


def diff(y: torch.Tensor, xs: list[torch.Tensor]):
    grad = y
    ones = torch.ones_like(y)
    for x in xs:
        grad = torch.autograd.grad(
            grad, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return grad


def ode(model: RBF | RBF_Poly, x: torch.Tensor) -> torch.Tensor:
    return torch.mean((diff(model(x), [x, x]) - 2) ** 2)


def bc(model: RBF | RBF_Poly) -> torch.Tensor:
    u_0 = model(torch.tensor([[-1.]], requires_grad=True, device='cuda'))
    u_1 = model(torch.tensor([[1.]], requires_grad=True, device='cuda'))
    loss = (((u_0 - 2.25) ** 2) + ((u_1 - 0.25) ** 2))/2
    return loss


def total_loss(model: RBF | RBF_Poly, x: torch.Tensor):
    return ode(model, x), bc(model)


def sol(x):
    return (x - 1/2) ** 2


dim = 1
n_verification = 400


def launch_experiment(train_size: int, centers_size: int, save_every: int,
                      show_interpolation_nodes: bool, kernel_name: str, poly_degree: int,
                      tag: str, shape: float, lr: float, use_scheduler: bool):

    xmax = 1.
    xmin = -1.

    x_train = ((xmax - xmin) * torch.rand(train_size-2, requires_grad=True) +
               xmin).reshape(-1, dim).to('cuda')

    kernel_fn = globals().get(kernel_name)
    if kernel_fn is None or not callable(kernel_fn):
        raise ValueError(f"No function named '{kernel_name}' found.")

    # Generate experiment name
    poly_suffix = f'-Poly{poly_degree}' if poly_degree > -1 else ''
    experiment_file = f'TR{train_size}-C{centers_size}' +\
        f'-K{kernel_name}{poly_suffix}-Sh{shape}-{tag}.csv'

    if poly_degree <= -1:
        nn = RBF(input_dim=dim, num_centers=centers_size,
                 output_dim=1, kernel=kernel_fn, starting_shape=shape).to('cuda')
        nn.set_centers(torch.linspace(xmin, xmax, centers_size,
                       device='cuda').reshape(-1, 1))

        def get_shape(): return nn.shape
    else:
        nn = RBF_Poly(input_dim=dim, num_centers=centers_size, output_dim=1,
                      degree=poly_degree, starting_shape=shape, kernel=kernel_fn).to('cuda')
        nn.rbf.set_centers(torch.linspace(
            xmin, xmax, centers_size, device='cuda').reshape(-1, 1))

        def get_shape(): return nn.rbf.shape

    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)

    x_ver = torch.linspace(xmin, xmax, 400).reshape(-1, 1).to('cuda')
    y_ver = sol(x_ver).reshape(-1, 1)

    min_lr = 1e-6
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=100, threshold=1e-4, verbose=True, min_lr=min_lr)

    L_inf_norm_epoch = [-1, torch.tensor(torch.inf), torch.nan, torch.nan]
    max_error_point = torch.nan

    loss_int_list, loss_bd_list, linf_list, l2_list, lr_list = [], [], [], [], []
    start_time = time.time()
    epoch = 0

    while True:

        optimizer.zero_grad()
        loss_int, loss_bd = total_loss(nn, x_train)
        loss = loss_int + loss_bd
        loss.backward(retain_graph=True)
        optimizer.step()

        validation_error = torch.abs(nn(x_ver) - y_ver)
        max_error_idx = torch.argmax(validation_error)
        max_error_point = x_ver[max_error_idx].item()

        # Update logs
        linf_list += [validation_error[max_error_idx].item()]
        loss_int_list += [loss_int.item()]
        loss_bd_list += [loss_bd.item()]
        l2_list += [torch.sqrt(
            torch.sum(validation_error ** 2) / n_verification).item()]
        lr_list += [scheduler.optimizer.param_groups[0]['lr']]

        if use_scheduler:
            scheduler.step(loss)

        if linf_list[epoch] < L_inf_norm_epoch[1]:
            L_inf_norm_epoch = (
                epoch, linf_list[epoch], max_error_point, l2_list[epoch])

        if (epoch > 0 and epoch % save_every == 0 and save_every > 0) or (math.isclose(lr_list[epoch], min_lr, abs_tol=min_lr/10)):
            lb, rb = torch.tensor([[-1.]]), torch.tensor([[1.]])
            nodes_boundary = torch.cat((lb, rb), dim=0).to('cuda')
            plot_figure_free_shape(centers=torch.cat((nn.get_centers(), nodes_boundary), dim=0), fn=sol, nn=nn,
                                   nodes=torch.cat((x_train, nodes_boundary), dim=0), dim=dim,
                                   path=f'{experiment_file}-E{epoch}', shape=get_shape(),
                                   extension='pdf', xmin=xmin, xmax=xmax, linf_rel=None, l2_rel=None,
                                   show_nodes=show_interpolation_nodes, epoch=epoch,
                                   loss_int=loss_int_list, linf=linf_list, l2=l2_list, lr=lr_list,
                                   loss_boundary=loss_bd_list,
                                   loss=None)

            if (math.isclose(lr_list[epoch], min_lr, abs_tol=min_lr/10)):
                break
        epoch += 1
    end_time = time.time()
    save_to_csv(experiment_file=f'{experiment_file}-E{epoch}.csv', kernel_name=kernel_name, nn=nn,
                shape=shape, L_inf_norm_epoch=L_inf_norm_epoch, get_shape=get_shape, elapsed_time=end_time - start_time,
                last_epoch=epoch, last_l2=l2_list[-1], last_linf=linf_list[-1], last_linf_location=max_error_point)
    save_curves(f'curves-TR{train_size}-C{centers_size}{poly_suffix}-tag{tag}-E{epoch}-curves.csv',
                loss_int_list, l2_list, linf_list, lr_list, None, None)


is_required = False
parser = argparse.ArgumentParser(
    description="Process parameters for training")
parser.add_argument("--train_size", type=int,
                    help="Size of training data", required=is_required, default=15)
parser.add_argument("--center_size", type=int,
                    help="Number of equispaced centers", required=is_required, default=5)
parser.add_argument('--kernel_name', type=str, default='gaussian_kernel')
parser.add_argument('--save_every', type=int,
                    default=-1, required=False)
parser.add_argument('--show_interpolation_nodes',
                    action=argparse.BooleanOptionalAction, default=True, required=False)
parser.add_argument('--polynomial_degree', type=int,
                    default=-1, required=False)
parser.add_argument('--shape', type=float, default=.78)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument('--use_scheduler',
                    action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--tag', type=str, default='')

args = parser.parse_args()
launch_experiment(train_size=args.train_size, centers_size=args.center_size,
                  save_every=args.save_every, show_interpolation_nodes=args.show_interpolation_nodes,
                  kernel_name=args.kernel_name, poly_degree=args.polynomial_degree,
                  tag=args.tag, shape=args.shape, lr=args.lr, use_scheduler=args.use_scheduler)
