
from math import floor, sqrt
from typing import Callable, Union
from scipy.stats import qmc
import torch
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import time
sys.path.append('../../modules')
if True:
    from notable_functions import *
    from notable_kernels import *
    from plot_utils import *
    from nn_rbf import RBF_Free_All as RBF
    from nn_rbf_poly import RBF_Poly_Free_All as RBF_Poly
    from grid_utils import get_ball
    from macros import save_curves


dim = 2
n_verification = 50


def save_to_csv(experiment_file: str, kernel_name: str, nn: RBF | RBF_Poly, shape: float, L_inf_norm_epoch: list,
                get_shape: Callable[[], torch.nn.Parameter], elapsed_time: float,
                last_epoch: int, last_l2: float, last_linf: float, last_linf_location: float):
    with open(experiment_file, 'w') as file:
        file.write(
            'Best_epoch,Linf_norm_verification,Linf_max_location,L2_norm_verification,Last_epoch,Last_linf,Last_linf_location,Last_L2\n')
        file.write(
            f'{L_inf_norm_epoch[0]},{L_inf_norm_epoch[1]},{L_inf_norm_epoch[2]},{L_inf_norm_epoch[3]},{last_epoch},{last_linf},{last_linf_location},{last_l2}\n')
        file.write('centers_list\n')
        centers_list = str(nn.get_centers().T.cpu(
        ).detach().numpy().squeeze().tolist())
        file.write(f'{centers_list[1:-1]}\n')
        file.write('shapes_list\n')
        if kernel_name == 'phs_kernel':
            file.write(str(get_shape()))
        else:
            shapes_list = str(get_shape().cpu().detach().numpy().tolist())
            file.write(shapes_list[1:-1])
        file.write('\nrbf_and_poly_coefs_list\n')
        if isinstance(nn, RBF):
            file.write(
                str(nn.output_layer.weight.cpu().detach().numpy().tolist()) + '\n')
        if isinstance(nn, RBF_Poly):
            c1, c2 = nn.get_coefs()
            file.write(
                f'{str(c1.cpu().detach().numpy().flatten().tolist())[1:-1]}\n')
            file.write(
                f'{str(c2.cpu().detach().numpy().flatten().tolist())[1:-1]}\n')
        file.write('starting_shape\n')
        file.write(f'{str(shape)}\n')
        file.write('elapsed_time\n')
        file.write(f'{elapsed_time}')


def launch_experiment(train_size: int, centers_size: int, fn_name: str,
                      mb_size: int, save_every: int,
                      show_interpolation_nodes: bool, kernel_name: str,
                      poly_degree: int,  tag: str, patience: int,
                      shape: float,  lr: float,
                      radius: float,
                      use_scheduler: bool):
    # Load function
    fn = globals().get(fn_name)
    if fn is None or not callable(fn):
        raise ValueError(f"No function named '{fn_name}' found.")
    kernel_fn = globals().get(kernel_name)
    if kernel_fn is None or not callable(kernel_fn):
        raise ValueError(f"No function named '{kernel_name}' found.")
    fn_callable_numpy = globals().get(fn_name+'_numpy')
    # Generate experiment name
    poly_suffix = f'-Poly{poly_degree}' if poly_degree > -1 else ''
    folder = 'dim_2/numerics_circle'
    experiment_file = f'{folder}/{fn_name}-TR{train_size}-C{centers_size}' +\
        f'-K{kernel_name}{poly_suffix}-Sh{shape}-{tag}'

    x_validate_raw = torch.cartesian_prod(
        *[torch.linspace(-radius, radius, n_verification) for _ in range(dim)]).reshape(-1, dim)
    y_validate_raw = fn(x_validate_raw).reshape(-1, 1)
    x_validate_indices = torch.sum(x_validate_raw**2, dim=1) <= radius**2
    x_validate = x_validate_raw[x_validate_indices].to('cuda')
    y_validate = fn(x_validate).reshape(-1, 1).to('cuda')

    # Experiment's input and target
    n_boundary_points = 4 * (train_size - 2) + 4  # deform a square into a ball
    interior_train, boundary_train = get_ball(radius=radius, n_boundary_points=n_boundary_points,
                                              n_interior_points=train_size**dim - n_boundary_points)
    x_train = torch.cat((interior_train, boundary_train), dim=0).to('cuda')
    y_train = fn(x_train).reshape(-1, 1).to('cuda')

    if poly_degree > -1:
        nn = RBF_Poly(input_dim=dim, num_centers=centers_size, output_dim=1, kernel=kernel_fn,
                      left_lim=-radius, right_lim=radius, starting_shape=shape, degree=poly_degree,
                      dev='cuda').to('cuda')
        nn.rbf.set_centers(torch.cartesian_prod(
            *[torch.linspace(-radius, radius, floor(sqrt(centers_size)), device='cuda'),
              torch.linspace(-radius, radius, floor(sqrt(centers_size)), device='cuda')]).reshape(-1, 2))

        def get_shape(): return nn.rbf.shape
    else:
        nn = RBF(input_dim=dim, num_centers=centers_size,
                 output_dim=1, kernel=kernel_fn, left_lim=-radius, right_lim=radius, starting_shape=shape).to('cuda')
        nn.set_centers(torch.cartesian_prod(
            *[torch.linspace(-radius, radius, floor(sqrt(centers_size)), device='cuda'),
              torch.linspace(-radius, radius, floor(sqrt(centers_size)), device='cuda')]).reshape(-1, 2))

        def get_shape(): return nn.shape

    # Training setup
    loss_fn = torch.nn.MSELoss().to('cuda')
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)

    min_lr = 1e-6
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=patience, threshold=1e-4, verbose=True, min_lr=min_lr)

    L_inf_norm_epoch = [-1, torch.tensor(torch.inf), torch.nan, torch.nan]
    max_error_point = torch.nan

    loss_list, linf_list, l2_list, lr_list = [], [], [], []
    linf_relative_list, l2_relative_list = [], []

    start_time = time.time()
    epoch = -1

    while True:
        epoch += 1

        loss = loss_fn(nn(x_train), y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compare performance against verification dataset
        nn_validation_output = nn(x_validate)
        validation_error = torch.abs(nn_validation_output - y_validate)
        max_error_idx = torch.argmax(validation_error)
        max_error_point = x_validate[max_error_idx, :].cpu().numpy().tolist()

        # Update logs
        loss_list += [loss.item()]
        linf_list += [validation_error[max_error_idx].item()]
        linf_relative_list += [(validation_error[max_error_idx] /
                               torch.abs(y_validate[max_error_idx])).item()]
        l2_list += [torch.sqrt(
            torch.sum(validation_error ** 2) / (n_verification ** 2)).item()]
        l2_relative_list += [torch.sqrt(
            torch.sum(validation_error ** 2) / (torch.sum(y_validate ** 2))).item()]
        lr_list += [scheduler.optimizer.param_groups[0]['lr']]

        if use_scheduler:
            scheduler.step(loss)

        if linf_list[-1] < L_inf_norm_epoch[1]:
            L_inf_norm_epoch = (
                epoch, linf_list[-1], max_error_point, l2_list[-1])

        if (epoch > 0 and epoch % save_every == 0 and save_every > 0) or (math.isclose(lr_list[-1], min_lr, abs_tol=min_lr/10)):
            approximation_square = nn(x_validate_raw.to('cuda')).cpu()
            approximation_domain = torch.where(
                x_validate_indices.unsqueeze(1), approximation_square, torch.nan)
            y_validate_domain = torch.where(
                x_validate_indices.unsqueeze(1), y_validate_raw, torch.nan)

            # plot_data_3d(centers=nn.get_centers().cpu(), xmin=-radius, xmax=radius,
            #              approx=approximation_domain,
            #              target=y_validate_domain, res=n_verification,
            #              filename=f'{experiment_file}-E{epoch}-all-curves.pdf', linf_rel=linf_relative_list,
            #              l2_rel=l2_relative_list,
            #              loss=loss_list, linf=linf_list, l2=l2_list, lr=lr_list)
            plot_data_3d(centers=nn.get_centers().cpu(), xmin=-radius, xmax=radius,
                         approx=approximation_domain,
                         target=y_validate_domain, res=n_verification,
                         filename=f'{experiment_file}-E{epoch}.pdf', fn_callable=fn_callable_numpy,
                         loss=loss_list, linf=linf_list, l2=l2_list, lr=lr_list)
            if (math.isclose(lr_list[-1], min_lr, abs_tol=min_lr/10)):
                break
    end_time = time.time()
    save_to_csv(experiment_file=f'{experiment_file}-E{epoch}.csv', kernel_name=kernel_name, nn=nn,
                shape=shape, L_inf_norm_epoch=L_inf_norm_epoch, get_shape=get_shape, elapsed_time=end_time - start_time,
                last_epoch=epoch, last_l2=l2_list[-1], last_linf=linf_list[-1], last_linf_location=max_error_point)
    save_curves(f'{folder}/curves-{fn_name}-TR{train_size}-C{centers_size}{poly_suffix}-tag{tag}-E{epoch}-curves.csv',
                loss_list, l2_list, linf_list, lr_list, l2_relative_list, linf_relative_list, x_train.cpu().detach().numpy().tolist())


is_required = False
parser = argparse.ArgumentParser(
    description="Process parameters for training")
parser.add_argument("--train_size", type=int,
                    help="Size of training data", required=is_required, default=11)
parser.add_argument("--center_size", type=int,
                    help="Number of equispaced centers", required=is_required, default=9)
parser.add_argument("--fn_name", type=str,
                    help="Name of the function to interpolate",
                    required=is_required, default='parabola')
parser.add_argument("--mb_size", type=int,
                    help="Mini batch size", default=144)
parser.add_argument('--kernel_name', type=str, default='gaussian_kernel')
parser.add_argument('--save_every', type=int,
                    default=-1, required=False)
parser.add_argument('--show_interpolation_nodes',
                    action=argparse.BooleanOptionalAction, default=True, required=False)
parser.add_argument('--polynomial_degree', type=int,
                    default=-1, required=False)
parser.add_argument('--shape', type=float, default=1)
# parser.add_argument('--xmin', type=float, default=0)
# parser.add_argument('--xmax', type=float, default=1)
parser.add_argument('--radius', type=float, default=1.)
parser.add_argument('--tag', type=str, required=False, default='')
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument('--use_scheduler',
                    action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--patience", type=int, default=30)
args = parser.parse_args()
launch_experiment(train_size=args.train_size, centers_size=args.center_size,
                  fn_name=args.fn_name, mb_size=args.mb_size, save_every=args.save_every,
                  show_interpolation_nodes=args.show_interpolation_nodes,
                  kernel_name=args.kernel_name, shape=args.shape,
                  poly_degree=args.polynomial_degree, patience=args.patience,
                  use_scheduler=args.use_scheduler, radius=args.radius,
                  tag=args.tag, lr=args.lr)
