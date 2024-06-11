from typing import Callable
import torch
from grid_utils import d_meshgrid
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from nn_rbf import RBF_Free_All
from nn_rbf_poly import RBF_Poly_Free_All, RBFInterpolantFreeCenters
import numpy as np
import seaborn as sns
import math


def show_interpolation(x, target_points, approx_points, centers, show_centers, xmin, xmax, show_nodes, nodes, fn, epoch):
    # For legend
    lines = []  # To keep track of lines for the legend
    labels = []  # To keep track of labels for the legend

    plt.plot(x, target_points, color='blue')
    plt.plot(x, approx_points, color='red')

    target_lines = mlines.Line2D([], [], color='blue')
    actual_lines = mlines.Line2D([], [], color='red')

    lines.append(target_lines)
    labels.append('Target')
    lines.append(actual_lines)
    labels.append('Approx')

    if show_centers:  # show RBF centers
        valid_centers = centers[(centers >= xmin) & (centers <= xmax)]
        centers_line = plt.vlines(valid_centers,
                                  ymin=min(min(target_points),
                                           min(approx_points)),
                                  ymax=max(max(target_points),
                                           max(approx_points)),
                                  linestyles='dashed', label='Centers', colors='gray')
        lines.append(centers_line)
        labels.append('Centers')
    if show_nodes:  # show interpolation nodes
        nodes_scatter = plt.scatter(
            nodes.cpu(), fn(nodes).cpu(), label='Nodes', color='green')
        lines.append(nodes_scatter)
        labels.append('Nodes')

    plt.title(
        f'Approximation (it {epoch})' if epoch > -1 else 'Approximation')

    return lines, labels


def show_linf_norm(lines, labels, target_points, approx_points, nodes, x):
    absolute_error = (target_points - approx_points).abs()
    plt.semilogy(x, absolute_error)
    plt.title('Log-10 absolute error and interpolation nodes')

    # if show_nodes:  # show interpolation nodes
    nodes_vline = plt.vlines(nodes.cpu(), ymin=min(absolute_error), ymax=max(absolute_error),
                             linestyles='dashed', colors='green', alpha=.6)
    lines.append(nodes_vline)
    labels.append('Nodes')

    handles, legend_labels = plt.gca().get_legend_handles_labels()
    handles.extend(lines)  # Add the lines for centers and nodes
    legend_labels.extend(labels)  # Add the labels for centers and nodes
    plt.legend(handles, legend_labels, loc='upper left',
               bbox_to_anchor=(1, 1))


def show_shape_values(shapes_list):
    with torch.no_grad():
        size = int(math.ceil(math.sqrt(shapes_list.shape[0])))
        shapes_list_np = shapes_list.cpu().numpy()
        shapes_list_np = np.append(shapes_list_np, [np.nan] *
                                   (size ** 2 - shapes_list.shape[0]))
        sns.heatmap(np.reshape(shapes_list_np, (size, size)), annot=True)
        plt.title('Values of shape parameter')


def show_loss_curves(loss: list[float] | None, linf_norm: list[float] | None, loss_int: list[float] | None,
                     loss_bd: list[float] | None, l2: list[float] | None, lr: list[float] | None,
                     linf_rel: list[float] | None, l2_rel: list[float] | None, show_legend=True):
    with torch.no_grad():
        if loss is not None:
            plt.semilogy(loss, label='Loss\n(train) ', color='blue', alpha=0.7)
        if linf_norm is not None:
            plt.semilogy(linf_norm, label=r'$L^\infty$',
                         color='orange', alpha=0.7)
        if l2 is not None:
            plt.semilogy(l2, label=r'$L^2$', color='red', alpha=0.7)
        if lr is not None:
            plt.semilogy(lr, label='LR', color='gray', alpha=0.7)
        if linf_rel is not None:
            plt.semilogy(linf_rel, label=r'$L^\infty$ (rel.)',
                         alpha=0.7, color='#00A488')
        if l2_rel is not None:
            plt.semilogy(l2_rel, label=r'$L^2$ (rel.)',
                         alpha=0.7, color='#BBD5E8')
        if loss_bd is not None:
            plt.semilogy(loss_bd, label='Loss\n(boundary)',
                         color='indigo', alpha=0.7)
        if loss_int is not None:
            plt.semilogy(loss_int, label='Loss\n(interior)',
                         color='blue', alpha=0.7)
        plt.title(r'Model curves', fontsize=10)
        if show_legend:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


def plot_figure_free_shape(centers, nodes, fn, nn: RBF_Free_All | RBF_Poly_Free_All | RBFInterpolantFreeCenters, dim: int,
                           loss, linf, l2, lr,
                           xmin: float, xmax: float,
                           linf_rel: list[float] | None, l2_rel: list[float] | None,
                           epoch: int = -1, path: str | None = None,  dev='cuda', extension='png',
                           show_nodes=False, show_centers=True, resolution=300, shape: float | torch.Tensor = -1):

    with torch.no_grad():
        x = d_meshgrid(*[torch.linspace(xmin, xmax, resolution)
                         for _ in range(dim)]).reshape(-1, dim).to(dev)
        target_points, approx_points = fn(x).cpu(), nn(x).cpu()
        centers = centers.cpu()
        x = x.cpu()

        plt.figure(figsize=(8, 7))

        # Showing interpolation
        plt.subplot(2, 2, 1)
        lines, labels = show_interpolation(
            x, target_points, approx_points, centers, show_centers, xmin, xmax, show_nodes, nodes, fn, epoch)

        # Relative error
        plt.subplot(2, 2, 2)
        show_linf_norm(lines, labels, target_points, approx_points, nodes, x)

        # Values of shape parameter
        plt.subplot(2, 2, 3)
        if not isinstance(shape, torch.Tensor):
            shape = torch.ones_like(centers) * shape
        show_shape_values(shape)

        # L^inf and loss curves
        plt.subplot(2, 2, 4)
        show_loss_curves(loss, linf, None, None, l2, lr, linf_rel, l2_rel)

        if path is None:
            plt.show()
        else:
            plt.savefig(f'{path}.{extension}', bbox_inches='tight')
        plt.close()


def plot_figure(centers, nodes, fn, nn, dim: int, epoch: int = -1, path: str = 'output',  extension='png',
                show_nodes=False, show_centers=True, resolution=300, xmin=-1, xmax=1):
    dev = centers.device
    with torch.no_grad():
        x = d_meshgrid(*[torch.linspace(xmin, xmax, resolution)
                         for _ in range(dim)]).reshape(-1, dim).to(dev)
        target_points, approx_points = fn(x).cpu(), nn(x).cpu()
        centers = centers.cpu()
        x = x.cpu()

        plt.figure(figsize=(8, 3))

        # Showing interpolation
        plt.subplot(1, 2, 1)
        lines, labels = show_interpolation(
            x, target_points, approx_points, centers, show_centers, xmin, xmax, show_nodes, nodes, fn, epoch)

        # Relative error
        plt.subplot(1, 2, 2)
        show_linf_norm(lines, labels, target_points, approx_points, nodes, x)

        plt.savefig(f'{path}.{extension}', bbox_inches='tight')
        plt.close()


def plot_error_figure(centers, fn, nn, dim: int, path: str = 'output', dev='cuda', extension='png',
                      show_nodes=False, show_centers=True, resolution=300, xmin=-1, xmax=1):

    with torch.no_grad():
        x = d_meshgrid(*[torch.linspace(xmin, xmax, resolution)
                         for _ in range(dim)]).reshape(-1, dim).to(dev)
        plt.figure()
        target_points, approx_points = fn(x).cpu(), nn(x).cpu()
        x = x.cpu()
        plt.semilogy(x, (target_points - approx_points).abs() /
                     (1e-10 + target_points).abs(), label='Error')
        if show_centers:  # show RBF centers
            plt.vlines(centers.cpu(), ymin=min(target_points), ymax=max(target_points),
                       linestyles='dashed', label='Centers', colors='black')
        if show_nodes:  # show interpolation nodes
            plt.scatter(x, target_points, label='Nodes')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(f'{path}.{extension}', bbox_inches='tight')
        plt.close()


def print_approximation(approx_points, target_points, xmax, xmin, res,
                        centers,
                        fn_callable: Callable[[
                            np.ndarray, np.ndarray], np.ndarray]
                        | Callable[[np.ndarray], np.ndarray]
                        | None,
                        use_larger_domain: bool,
                        use_circle: float, show_legend=False):

    # Add heatmap
    if use_circle:
        circ = patches.Circle(xy=(0, 0), radius=xmax, linewidth=1,
                              edgecolor='black', facecolor='none', zorder=10)
        plt.gca().add_patch(circ)
    else:
        rect = patches.Rectangle(
            xy=(0., 0.), width=xmax-xmin, height=xmax-xmin, zorder=10,
            facecolor='none', edgecolor='black', linewidth=1)
        plt.gca().add_patch(rect)
    plt.imshow(X=torch.abs(approx_points - target_points).T,
               extent=(xmin, xmax, xmin, xmax), cmap="Oranges_r", norm='log',
               origin='lower')
    plt.colorbar(location='left', pad=.16)

    xmin = xmin if not use_larger_domain else xmin-.5
    xmax = xmax if not use_larger_domain else xmax+.5
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.xticks(np.unique([xmin, 0, xmax]))
    plt.yticks(np.unique([xmin, 0, xmax]))
    # Add centers
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], edgecolors='black',
                    facecolors='none', s=5, linewidths=.5)
    # Add contour plot with ground-truth function
    # Generate X and Y coordinates
    if fn_callable is not None:
        x_vals = np.linspace(xmin, xmax, res)
        y_vals = np.linspace(xmin, xmax, res)
        X, Y = np.meshgrid(x_vals, y_vals)
        contours = plt.contour(X, Y, fn_callable(X, Y), levels=20,
                               colors='gray', linewidths=1., alpha=.7)
        plt.clabel(contours, inline=True, fontsize=3)

    plt.title('Absolute error (verification)', fontsize=10)
    if show_legend:
        plt.legend(loc='lower left', bbox_to_anchor=(-1, 1))


def plot_data_3d(centers, xmin: float, xmax: float, approx: torch.Tensor, target: torch.Tensor,
                 res: int, loss: list[float] | None, linf: list[float] | None, l2: list[float] | None, lr: list[float] | None, is_circle: bool,
                 loss_int: list[float] | None, loss_boundary: list[float] | None,
                 fn_callable: Callable[[np.ndarray], np.ndarray] | None = None,
                 filename: str = 'output', title: str | None = None, use_larger_domain: bool = True,
                 linf_rel: list[float] | None = None, l2_rel: list[float] | None = None):

    with torch.no_grad():

        plt.figure(figsize=(6, 2))

        gs = gridspec.GridSpec(1, 2, width_ratios=[6, 4])

        # plt.subplot(1, 2, 1)
        plt.subplot(gs[0])
        print_approximation(approx.cpu().reshape(res, res), target.cpu().reshape(res, res),
                            xmax, xmin, res, centers.cpu(), fn_callable, use_circle=is_circle, show_legend=False,
                            use_larger_domain=use_larger_domain)

        # plt.subplot(1, 2, 2)
        plt.subplot(gs[1])
        show_loss_curves(loss=loss, linf_norm=linf, loss_int=loss_int, loss_bd=loss_boundary,
                         l2=l2, lr=lr, linf_rel=linf_rel, l2_rel=l2_rel)

        # Create legend handles
        legend_handles = [
            mlines.Line2D([], [], color='orange', label=r'$L^\infty$'),
            mlines.Line2D([], [], color='red', label=r'$L^2$'),
            mlines.Line2D([], [], color='gray', label=r'LR'),
        ]
        if l2_rel is not None and linf_rel is not None:
            legend_handles += [mlines.Line2D([], [],
                                             color='#BBD5E8', label=r'$L^2$ (rel.)')]
            legend_handles += [mlines.Line2D([], [],
                                             color='#00A488', label=r'$L^\infty$ (rel.)')]
        if loss is not None:
            legend_handles += [mlines.Line2D([], [],
                                             color='blue', label='Loss\n(train)')]
        else:
            if loss_boundary is None or loss_int is None:
                raise Exception(
                    'loss is None, loss_boundary is None, loss_int is None')
            legend_handles += [mlines.Line2D([], [],
                                             color='blue', label='Loss\n(interior)')]
            legend_handles += [mlines.Line2D([], [],
                                             color='indigo', label='Loss\n(boundary)')]
        # Create the legend
        plt.legend(handles=legend_handles,
                   loc='upper left', bbox_to_anchor=(1, 1))

        plt.savefig(filename, bbox_inches='tight')
        plt.close()
