import torch
from grid_utils import d_meshgrid
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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
                             linestyles='dashed', colors='green')
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


def show_loss_curves(loss: list[float], linf_norm: list[float], l2: list[float], show_legend=True):
    with torch.no_grad():
        plt.semilogy(loss, label='Loss\n(train) ', color='blue')
        plt.semilogy(linf_norm, label=r'$L^\infty$', color='orange')
        plt.semilogy(l2, label=r'$L^2$', color='red')
        plt.title(r'Model curves', fontsize=10)
        if show_legend:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


def plot_figure_free_shape(centers, nodes, fn, nn: RBF_Free_All | RBF_Poly_Free_All | RBFInterpolantFreeCenters, dim: int, loss, linf, l2, epoch: int = -1, path: str | None = None, dev='cuda', extension='png',
                           show_nodes=False, show_centers=True, resolution=300, xmin=-1, xmax=1, shape: float | torch.Tensor = -1):

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
        show_loss_curves(loss, linf, l2)

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


def print_approximation(approx_points, target_points, xmax, xmin, res, centers, show_legend=False):

    # Add heatmap
    plt.imshow(X=torch.abs(approx_points - target_points),
               extent=(xmin, xmax, xmin, xmax), cmap="cool", norm='log')
    plt.colorbar(location='left', pad=.16)

    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    # Add centers
    plt.scatter(centers[:, 0], centers[:, 1],
                label='Centers', color='#2cc914', s=16)
    # Add contour plot with ground-truth function
    # Generate X and Y coordinates
    x_vals = np.linspace(xmin, xmax, res)
    y_vals = np.linspace(xmin, xmax, res)
    X, Y = np.meshgrid(x_vals, y_vals)
    plt.contour(X, Y,
                target_points, cmap='bone', linewidths=1., alpha=.7)
    plt.title('Absolute error (verification)', fontsize=10)
    if show_legend:
        plt.legend(loc='lower left', bbox_to_anchor=(-1, 1))


def plot_data_3d(centers, xmin: float, xmax: float, approx: torch.Tensor, target: torch.Tensor,
                 res: int, loss: list[float], linf: list[float],
                 filename: str = 'output', title: str | None = None):

    with torch.no_grad():

        plt.figure(figsize=(6, 2))

        gs = gridspec.GridSpec(1, 2, width_ratios=[6, 4])

        # plt.subplot(1, 2, 1)
        plt.subplot(gs[0])
        print_approximation(approx.cpu().reshape(res, res).T, target.cpu().reshape(res, res).T,
                            xmax, xmin, res, centers.cpu(), show_legend=False)

        # plt.subplot(1, 2, 2)
        plt.subplot(gs[1])
        show_loss_curves(loss, linf, show_legend=False)

        # Create legend handles
        legend_handles = [
            mlines.Line2D([], [], color='orange', label=r'$L^\infty$'),
            mlines.Line2D([], [], color='blue', label='Loss'),
            mlines.Line2D([], [], linestyle='', marker='o',
                          color='#2cc914', label='Center', markersize=4),
        ]

        # Create the legend
        plt.legend(handles=legend_handles,
                   loc='upper left', bbox_to_anchor=(1, 1))

        plt.savefig(filename, bbox_inches='tight')
        plt.close()
