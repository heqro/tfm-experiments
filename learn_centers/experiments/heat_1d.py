
import argparse
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys


sys.path.append('../../modules')
if True:
    from notable_kernels import *
    from nn_rbf import RBF_Free_All as RBF
    from plot_utils import *
    from macros import *
    from grid_utils import get_rectangle
    from math import sqrt, floor
    from notable_functions import *

dim = 2
n_verification = 50


def launch_experiment(n_training_points_1d: int, n_total_centers: int, tag: str, save_every: int,
                      shape: float, weight_boundary: float, weight_interior: float,
                      kernel_name: str = 'gaussian_kernel',
                      sol: str = 'heat_1d', xmax: float = 1., use_scheduler: bool = True):

    def interior_expr(model: RBF,
                      x_pts: torch.Tensor, t_pts: torch.Tensor,
                      alpha: float) -> torch.Tensor:
        # Unsqueeze and put together the x and y vectors to allow the engine to keep track of derivatives
        u = model(torch.cat((x_pts.unsqueeze(1), t_pts.unsqueeze(1)), dim=1))
        u_x = torch.autograd.grad(u, x_pts,
                                  grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_t = torch.autograd.grad(u, t_pts,
                                  grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x_pts,
                                   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]
        return ((alpha * u_xx - u_t) ** 2).mean()

    def boundary_expr(model: RBF, boundary_pts: torch.Tensor, target_boundary: torch.Tensor):
        # in here, we consider both the boundary and initial conditions
        return ((model(boundary_pts) - target_boundary)**2).mean()

    def total_loss(model: RBF, x_int: torch.Tensor, y_int: torch.Tensor,
                   xy_boundary: torch.Tensor, f_boundary: torch.Tensor,
                   alpha: float):
        loss_interior = weight_interior * \
            interior_expr(model, x_int, y_int, alpha)
        loss_boundary = weight_boundary * \
            boundary_expr(model, xy_boundary, f_boundary)
        return loss_interior, loss_boundary

    # Load function
    fn = globals().get(sol)
    if fn is None or not callable(fn):
        raise ValueError(f"No function named '{sol}' found.")
    kernel_fn = globals().get(kernel_name)
    if kernel_fn is None or not callable(kernel_fn):
        raise ValueError(f"No function named '{kernel_name}' found.")

    fn_callable_numpy = globals().get(sol+'_numpy')
    # Generate experiment name
    folder = f'dim_2/{sol}'
    experiment_file = f'{folder}/{sol}-TR{n_training_points_1d}-C{n_total_centers}-{tag}'

    # prepare verification dataset
    x_validate = torch.cartesian_prod(
        *[torch.linspace(0, xmax, n_verification) for _ in range(dim)]).reshape(-1, dim).to('cuda')
    y_validate = fn(x_validate).reshape(-1, 1)

    # Experiment's input and target
    n_boundary_points = 4 * (n_training_points_1d - 2) + 4
    rectangle_dict = get_rectangle(bottom_left=(0., 0.), width=1., height=1.,
                                   n_points_per_boundary=n_boundary_points // 4,
                                   n_interior_points=n_training_points_1d**dim - n_boundary_points,
                                   requires_grad=True, separate_into_sets=True)

    int_top_train = rectangle_dict['boundary']['top']  # type: ignore
    int_train = rectangle_dict['interior']  # type: ignore
    int_train = torch.cat((int_train, int_top_train),  # type: ignore
                          dim=0).to('cuda')
    if not isinstance(int_train, torch.Tensor):
        raise Exception('The interior points should be torch.Tensor')

    bd_train_left = rectangle_dict['boundary']['left']  # type: ignore
    bd_train_right = rectangle_dict['boundary']['right']  # type: ignore
    bd_train_bottom = rectangle_dict['boundary']['bottom']  # type: ignore
    bd_train = torch.cat((bd_train_left, bd_train_right, bd_train_bottom),
                         dim=0).to('cuda')
    bd_target = fn(bd_train).reshape(-1, 1)

    nn = RBF(input_dim=dim, num_centers=n_total_centers,
             output_dim=1, kernel=kernel_fn, left_lim=0, right_lim=xmax, starting_shape=shape).to('cuda')
    nn.set_centers(torch.cartesian_prod(
        *[torch.linspace(0, xmax, floor(sqrt(n_total_centers)), device='cuda'),
            torch.linspace(0, xmax, floor(sqrt(n_total_centers)), device='cuda')]).reshape(-1, 2))

    def get_shape(): return nn.shape

    optimizer = torch.optim.Adam(nn.parameters(), lr=.01)

    min_lr = 1e-6
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=200, threshold=1e-4, verbose=True, min_lr=min_lr)

    start_time = time.time()
    epoch = -1

    # Logging utilities 🌲
    L_inf_norm_epoch = [-1, torch.tensor(torch.inf), torch.nan, torch.nan]
    max_error_point = torch.nan
    loss_int_list, loss_bd_list, linf_list, l2_list, lr_list = [], [], [], [], []

    while True:
        epoch += 1
        optimizer.zero_grad()
        loss_int, loss_bd = total_loss(model=nn, x_int=int_train[:, 0], y_int=int_train[:, 1],
                                       xy_boundary=bd_train, f_boundary=bd_target, alpha=1/30)

        loss = loss_int + loss_bd
        if epoch % 500 == 0:
            print(f'{epoch}: I - {loss_int.item():.2e}, B - {loss_bd.item():.2e}')
        if torch.isnan(loss):
            raise Exception('NANs appeared!')

        loss.backward(retain_graph=True)
        optimizer.step()

        nn_validation_output = nn(x_validate)
        validation_error = torch.abs(nn_validation_output - y_validate)
        max_error_idx = torch.argmax(validation_error)
        max_error_point = x_validate[max_error_idx, :].cpu().numpy().tolist()

        # Update logs
        linf_list += [validation_error[max_error_idx].item()]
        loss_int_list += [loss_int.item()]
        loss_bd_list += [loss_bd.item()]
        l2_list += [torch.sqrt(
            torch.sum(validation_error ** 2) / (n_verification ** 2)).item()]
        lr_list += [scheduler.optimizer.param_groups[0]['lr']]

        if use_scheduler:
            scheduler.step(loss)

        if linf_list[-1] < L_inf_norm_epoch[1]:
            L_inf_norm_epoch = (
                epoch, linf_list[-1], max_error_point, l2_list[-1])

        if (epoch > 0 and epoch % save_every == 0 and save_every > 0) or (math.isclose(lr_list[-1], min_lr, abs_tol=min_lr/10)):
            approximation_square = nn(x_validate).cpu()

            plot_data_3d(centers=nn.get_centers().cpu(), xmin=0, xmax=xmax,
                         approx=approximation_square,
                         target=y_validate.cpu(), res=n_verification, is_circle=False,
                         filename=f'{experiment_file}-E{epoch}.pdf', fn_callable=fn_callable_numpy, use_larger_domain=False,
                         loss_boundary=loss_bd_list, loss_int=loss_int_list, linf=linf_list, l2=l2_list, lr=lr_list, loss=None)
            if (math.isclose(lr_list[-1], min_lr, abs_tol=min_lr/10)):
                break

    end_time = time.time()
    save_to_csv(experiment_file=f'{experiment_file}-E{epoch}.csv', kernel_name=kernel_name, nn=nn,
                shape=shape, L_inf_norm_epoch=L_inf_norm_epoch, get_shape=get_shape, elapsed_time=end_time - start_time,
                last_epoch=epoch, last_l2=l2_list[-1], last_linf=linf_list[-1], last_linf_location=max_error_point)


is_required = False
parser = argparse.ArgumentParser(
    description="Process parameters for training")
parser.add_argument("--train_size", type=int,
                    help="Number of 1d training points", required=is_required, default=400)
parser.add_argument("--center_size", type=int,
                    help="Total number of centers", required=is_required, default=49)
parser.add_argument('--tag', type=str, required=False,
                    default='test')
parser.add_argument('--shape', type=float, default=2.6855)
parser.add_argument('--save_every', type=int,
                    default=-1, required=False)
parser.add_argument("--fn_name", type=str,
                    required=is_required, default='heat_1d')
parser.add_argument('--weight_boundary', type=float, default=1.)
parser.add_argument('--weight_interior', type=float, default=1.)

args = parser.parse_args()
launch_experiment(n_training_points_1d=args.train_size, n_total_centers=args.center_size,
                  tag=args.tag, save_every=args.save_every, shape=args.shape,
                  sol=args.fn_name,
                  weight_boundary=args.weight_boundary, weight_interior=args.weight_interior)
