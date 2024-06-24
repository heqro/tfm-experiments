
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
    from grid_utils import get_ball
    from math import sqrt, floor
    from notable_functions import *
dim = 2
n_verification = 50


def launch_experiment(n_training_points_1d: int, n_total_centers: int, tag: str, save_every: int,
                      poly_degree: int, shape: float, weight_boundary: float, weight_interior: float,
                      kernel_name: str = 'gaussian_kernel',
                      sol: str = 'parabola', radius: float = 1., use_scheduler: bool = True):

    def pde(model: RBF, x_pts: torch.Tensor, y_pts: torch.Tensor, target: torch.Tensor | float) -> torch.Tensor:
        # Unsqueeze and put together the x and y vectors to allow the engine to keep track of derivatives
        u = model(torch.cat((x_pts.unsqueeze(1), y_pts.unsqueeze(1)), dim=1))
        grad_u_x = torch.autograd.grad(u, x_pts,
                                       grad_outputs=torch.ones_like(u),
                                       create_graph=True, retain_graph=True)[0]
        grad_u_y = torch.autograd.grad(u, y_pts,
                                       grad_outputs=torch.ones_like(u),
                                       create_graph=True, retain_graph=True)[0]

        u_xx = torch.autograd.grad(grad_u_x, x_pts,
                                   grad_outputs=torch.ones_like(grad_u_x),
                                   create_graph=True)[0]
        u_yy = torch.autograd.grad(grad_u_y, y_pts,
                                   grad_outputs=torch.ones_like(grad_u_y),
                                   create_graph=True)[0]
        return ((u_xx + u_yy - target) ** 2).mean()

    def bc(model: RBF, boundary_pts: torch.Tensor, target_boundary: torch.Tensor):
        return ((model(boundary_pts) - target_boundary)**2).mean()

    def total_loss(model, x_int, y_int, f_int, xy_boundary, f_boundary):
        loss_interior = weight_interior * pde(model, x_int, y_int, f_int)
        loss_boundary = weight_boundary * bc(model, xy_boundary, f_boundary)
        return loss_interior, loss_boundary

    # Load function
    fn = globals().get(sol)
    if fn is None or not callable(fn):
        raise ValueError(f"No function named '{sol}' found.")
    kernel_fn = globals().get(kernel_name)
    if kernel_fn is None or not callable(kernel_fn):
        raise ValueError(f"No function named '{kernel_name}' found.")
    lap = globals().get(f'{sol}_laplacian')
    if lap is None or not callable(lap):
        raise ValueError(f"No function named '{sol}_laplacian' found.")
    fn_callable_numpy = globals().get(sol+'_numpy')
    # Generate experiment name
    folder = f'dim_2/pde_{sol}'
    poly_suffix = f'-Poly{poly_degree}' if poly_degree > -1 else ''
    experiment_file = f'{folder}/{sol}-TR{n_training_points_1d}-C{n_total_centers}{poly_suffix}-{tag}'

    # prepare verification dataset
    x_validate_raw = torch.cartesian_prod(
        *[torch.linspace(-radius, radius, n_verification) for _ in range(dim)]).reshape(-1, dim)
    y_validate_raw = fn(x_validate_raw).reshape(-1, 1)
    x_validate_indices = torch.sum(x_validate_raw**2, dim=1) <= radius**2
    x_validate = x_validate_raw[x_validate_indices].to('cuda')
    y_validate = fn(x_validate).reshape(-1, 1).to('cuda')

    # Experiment's input and target
    n_boundary_points = 4 * (n_training_points_1d - 2) + 4
    interior_train, boundary_train = get_ball(radius=radius, n_boundary_points=n_boundary_points,
                                              n_interior_points=n_training_points_1d**dim - n_boundary_points,
                                              requires_grad=True)
    interior_train = interior_train.to('cuda')
    interior_target = lap(interior_train).reshape(-1, 1)

    boundary_train = boundary_train.to('cuda')
    boundary_target = fn(boundary_train).reshape(-1, 1)

    if poly_degree > -1:
        nn = RBF_Poly(input_dim=dim, num_centers=n_total_centers, output_dim=1, kernel=kernel_fn,
                      left_lim=-radius, right_lim=radius, starting_shape=shape, degree=poly_degree,
                      dev='cuda').to('cuda')
        nn.rbf.set_centers(torch.cartesian_prod(
            *[torch.linspace(-radius, radius, floor(sqrt(n_total_centers)), device='cuda'),
              torch.linspace(-radius, radius, floor(sqrt(n_total_centers)), device='cuda')]).reshape(-1, 2))

        def get_shape(): return nn.rbf.shape
    else:
        nn = RBF(input_dim=dim, num_centers=n_total_centers,
                 output_dim=1, kernel=kernel_fn, left_lim=-radius, right_lim=radius, starting_shape=shape).to('cuda')
        nn.set_centers(torch.cartesian_prod(
            *[torch.linspace(-radius, radius, floor(sqrt(n_total_centers)), device='cuda'),
              torch.linspace(-radius, radius, floor(sqrt(n_total_centers)), device='cuda')]).reshape(-1, 2))

        def get_shape(): return nn.shape

    optimizer = torch.optim.Adam(nn.parameters(), lr=.01)

    min_lr = 1e-6
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=30, threshold=1e-4, verbose=True, min_lr=min_lr)

    start_time = time.time()
    epoch = -1

    # Logging utilities 🌲
    L_inf_norm_epoch = [-1, torch.tensor(torch.inf), torch.nan, torch.nan]
    max_error_point = torch.nan
    loss_list, linf_list, l2_list, lr_list = [], [], [], []

    while True:
        epoch += 1
        optimizer.zero_grad()
        loss_int, loss_bd = total_loss(
            nn, interior_train[:, 0], interior_train[:, 1], interior_target,
            boundary_train, boundary_target)
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
        loss_list += [loss.item()]
        l2_list += [torch.sqrt(
            torch.sum(validation_error ** 2) / (n_verification ** 2)).item()]
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
    save_curves(f'{folder}/curves-{sol}-TR{n_training_points_1d}-C{n_total_centers}{poly_suffix}-tag{tag}-E{epoch}-curves.csv',
                loss_list, l2_list, linf_list, lr_list, l2_rel=None, linf_rel=None,
                training_pts=torch.cat(
                    (interior_train, boundary_train), dim=0).cpu().detach().numpy().tolist(),
                )


is_required = False
parser = argparse.ArgumentParser(
    description="Process parameters for training")
parser.add_argument("--train_size", type=int,
                    help="Number of 1d training points", required=is_required, default=30)
parser.add_argument("--center_size", type=int,
                    help="Total number of centers", required=is_required, default=225)
parser.add_argument('--tag', type=str, required=False,
                    default='test')
parser.add_argument('--shape', type=float, default=3.6621)
parser.add_argument('--save_every', type=int,
                    default=-1, required=False)
parser.add_argument('--polynomial_degree', type=int,
                    default=-1, required=False)
parser.add_argument("--fn_name", type=str,
                    required=is_required, default='runge_2d')
parser.add_argument('--weight_boundary', type=float, default=1)
parser.add_argument('--weight_interior', type=float, default=1e2)

args = parser.parse_args()
launch_experiment(n_training_points_1d=args.train_size, n_total_centers=args.center_size,
                  tag=args.tag, save_every=args.save_every, shape=args.shape,
                  poly_degree=args.polynomial_degree, sol=args.fn_name,
                  weight_boundary=args.weight_boundary, weight_interior=args.weight_interior)
