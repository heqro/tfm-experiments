from logging import warning
from math import sqrt
import math
from typing import Callable, Literal, Tuple, Union
from pandas import DataFrame
from nn_rbf import RBF_Free_All as RBF
from nn_rbf_poly import RBF_Poly_Free_All as RBF_Poly
from notable_kernels import gaussian_kernel
import torch
import numpy as np
from notable_functions import *

colors = ['olive', 'orange', 'purple', 'red', 'blue', 'black', 'pink', 'green']


def save_curves(experiment_file: str, loss: list[float], l2: list[float], linf: list[float], lr: list[float], l2_rel: list[float] | None, linf_rel: list[float] | None, training_pts=None):
    with open(f'{experiment_file}', 'w') as file:
        file.write(f'Loss:{loss}\n')
        file.write(f'L2:{l2}\n')
        file.write(f'Linf:{linf}\n')
        file.write(f'LR:{lr}\n')
        if l2_rel is not None and linf_rel is not None:
            file.write(f'L2r:{l2_rel}\n')
            file.write(f'Linfr:{linf_rel}\n')
        if training_pts is not None:
            file.write(f'Training_pts:{training_pts}\n')


def histogram_of_shapes(center_sizes: list[int], k_values: list[float], poly_degs: list[int], df: DataFrame,
                        get_train_sizes: Callable[[int], list[int]], starting_shapes: list[float],
                        kernel: str, fn: str | None, folder: str = 'plots', colors: list[str] = colors, n_bins: int = 30):
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import numpy as np
    shapes_list = [[] for _ in k_values]
    for poly_deg in poly_degs:

        for c_idx in range(len(center_sizes)):
            c = center_sizes[c_idx]
            for k_idx in range(len(k_values)):
                k = k_values[k_idx]
                tr = get_train_sizes(c)[k_idx]
                entries = df[(df.Kernel == kernel) &
                             (df.Poly_degree == poly_deg) &
                             (df.Center_size == c) &
                             (df.Train_size == tr) &
                             (df.Points_distribution == 'Equi') &
                             (df.Trained_centers.notnull())]
                if fn is not None:
                    entries = entries[entries.Function == fn]
                for starting_shape_idx in range(len(starting_shapes)):
                    starting_shape = starting_shapes[starting_shape_idx]
                    entries_to_parse = entries[entries.Starting_shape ==
                                               starting_shape]['Trained_shapes'].values
                    for list_of_shapes in entries_to_parse:
                        shapes_list[starting_shape_idx] += [float(a)
                                                            for a in list_of_shapes.split(',')]

    for starting_shape_idx in (range(len(starting_shapes))):
        data = shapes_list[starting_shape_idx]
        count, bins, ignored = plt.hist(data, bins=n_bins, label=r'$\varepsilon=$' +
                                        str(starting_shapes[starting_shape_idx]), alpha=1, histtype='step',
                                        color=colors[starting_shape_idx],
                                        # range=(0, 7)
                                        )

        mu, std = stats.norm.fit(data)
        x = np.linspace(0, max(data), 200)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p*len(data) * (bins[1] - bins[0]),
                 linewidth=1, label=f'Fit: $\mu={mu:.2f}, \sigma={std:.2f}$', color=colors[starting_shape_idx], alpha=.8,
                 linestyle='--')
    # plt.yscale('log')
    plt.ylabel('Incidence')
    plt.xlabel(r'$\varepsilon$')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('Distribution of shape parameters at the end of training')
    plt.savefig(
        f'{folder}/distribution_of_shape_parameters_at_end_of_training.pdf', bbox_inches='tight')
    plt.close()


def histogram_of_shapes_endgame(center_sizes: list[int], df: DataFrame,
                                get_train_sizes: Callable[[int], list[int]], starting_shapes: list[float],
                                kernel: str, fn: str | None, folder: str = 'plots', colors: list[str] = colors, n_bins: int = 30):
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(len(center_sizes), 1, figsize=(7, 8))
    for c_idx in range(len(center_sizes)):
        shapes_list = []
        c = center_sizes[c_idx]

        tr = get_train_sizes(c)[-1]
        entries = df[(df.Kernel == kernel) &
                     (df.Center_size == c) &
                     (df.Train_size == tr) &
                     (df.Points_distribution == 'Equi') &
                     (df.Trained_centers.notnull())]
        if fn is not None:
            entries = entries[entries.Function == fn]
        if len(entries) == 0:
            raise Exception(f'Empty query. C: {c}, TR: {tr}')

        # for starting_shape_idx in range(len(starting_shapes)):

        entries_to_parse = entries[entries.Starting_shape ==
                                   starting_shapes[c_idx]]['Trained_shapes'].values
        if len(entries_to_parse) == 0:
            raise Exception(
                f'Empty query. C: {c}, TR: {tr}, Shape: {starting_shapes[c_idx]}')
        for list_of_shapes in entries_to_parse:
            shapes_list += [float(a)for a in list_of_shapes.split(',')]

        count, bins, ignored = axes[c_idx].hist(shapes_list, bins=n_bins, label=r'$\varepsilon=$' + str(starting_shapes[c_idx]) + f'\n{c} centers', alpha=1, histtype='step', color=colors[c_idx],
                                                # range=(0, 7)
                                                )

        mu, std = stats.norm.fit(shapes_list)
        x = np.linspace(0, max(shapes_list), 200)
        p = stats.norm.pdf(x, mu, std)

        axes[c_idx].plot(x, p*len(shapes_list) * (bins[1] - bins[0]),
                         linewidth=1, label=f'Fit: $\mu={mu:.2f}, \sigma={std:.2f}$',
                         color=colors[c_idx], alpha=.8,
                         linestyle='--')
        axes[c_idx].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axes[c_idx].set_ylabel('Incidence')
        axes[c_idx].set_xlabel(r'$\varepsilon$')
    # plt.yscale('log')

    # plt.title('Distribution of shape parameters at the end of training')
    plt.savefig(
        f'{folder}/distribution_of_shape_parameters-{fn}.pdf', bbox_inches='tight')
    plt.close()


def get_train_sizes_1d(C: int):
    from math import ceil
    import numpy as np
    return [ceil(C*k) for k in np.arange(1, 4.5, 0.5)]


def histogram_of_centers_1d(center_sizes: list[int], df: DataFrame, kernel: str, domain: Tuple[float, float], hist_dom: Tuple[float, float],
                            fn_name: str | None, fn_callable, folder: str = 'plots', n_bins: int = 30, poly_deg: int | None = None):
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import numpy as np

    entries = df[(df.Kernel == kernel) &
                 (df.Points_distribution == 'Equi') &
                 (df.Trained_centers.notnull())]
    if fn_name is not None:
        entries = entries[entries.Function == fn_name]
    if poly_deg is not None:
        entries = entries[entries.Poly_degree == poly_deg]
    for c_idx in range(len(center_sizes)):
        c = center_sizes[c_idx]
        centers_lists = entries[entries.Center_size ==
                                c]['Trained_centers'].values.tolist()
        centers_parsed = []
        for center_list in centers_lists:
            centers = center_list.replace(' ', '').split(',')
            centers_parsed += [float(center) for center in centers]

        fig, ax1 = plt.subplots()

        # Plot the histogram
        counts, bin_edges, patches = ax1.hist(
            centers_parsed, bins=n_bins, alpha=0.6, color='gray')
        ax1.set_xlabel('Center Values')
        ax1.set_ylabel('Frequency', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        # Add vertical lines
        ax1.axvline(domain[0], color='black', ls='--')
        ax1.axvline(domain[1], color='black', ls='--')

        # Plot the function on the secondary y-axis
        ax2 = ax1.twinx()
        x_plot = torch.linspace(hist_dom[0], hist_dom[1], 300)
        ax2.plot(x_plot, fn_callable(x_plot),
                 'black', linewidth=2)  # type: ignore
        ax2.set_ylabel('Function Value', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        plt.title(f'Histogram of centers ({c})')
        plt.savefig(
            f'{folder}/histogram-centers-C{c}{"_"+fn_name if fn_name is not None else ""}.pdf',
            bbox_inches='tight')
        plt.close()


def incidence_of_largest_errors_1d(center_sizes: list[int], df: DataFrame, kernel: str, fn: str | None, folder: str = 'plots', colors: list[str] = colors):
    import matplotlib.pyplot as plt
    errors_list = [[] for _ in center_sizes]
    entries = df[(df.Kernel == kernel) &
                 (df.Points_distribution == 'Equi') &
                 (df.Trained_centers.notnull())]
    if fn is not None:
        entries = entries[entries.Function == fn]
    plt.figure(figsize=(4, 4))
    for c_idx in range(len(center_sizes)):
        c = center_sizes[c_idx]
        errors_list[c_idx] = entries[entries.Center_size ==
                                     c]['Last_linf_location'].values.tolist()
        plots = plt.violinplot(
            errors_list[c_idx], showmedians=True, vert=False, positions=[c_idx])
        for pc, color in zip(plots['bodies'], colors):  # type: ignore
            pc.set_facecolor('#1F77B4')

        plots['cmedians'].set_colors('#1F77B4')  # type: ignore
        plots['cbars'].set_colors('#1F77B4')  # type: ignore
        plots['cmins'].set_colors('#1F77B4')  # type: ignore
        plots['cmaxes'].set_colors('#1F77B4')  # type: ignore
        # plt.hist(errors_list[c_idx], range=(-1, 1), bins=45,
        #          label=f'{c} centers', histtype='step', color=colors[c_idx])
    # plt.yscale('log')
    plt.ylabel('Number of centers')
    plt.xlabel('Domain')
    # plt.legend()
    plt.gca().set_yticks(list(range(len(center_sizes))))
    plt.gca().set_yticklabels(center_sizes)

    # plt.title(
    #     'Distribution of the largest error')
    plt.savefig(f'{folder}/distribution_of_errors_L_inf_end_{fn}.pdf',
                bbox_inches='tight')
    plt.close()


def incidence_of_l(df: DataFrame, kernel: str, fn: str | None, xlabel_text: str, ylabel_text: str, loss_key: str, folder: str = 'plots'):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    plt.figure(figsize=(4, 3))
    entries = df[(df.Kernel == kernel) &
                 (df.Points_distribution == 'Equi') &
                 (df.Trained_centers.notnull())
                 ]
    if fn is not None:
        entries = entries[entries.Function == fn]
    best, last = entries[f'L{loss_key}_loss'].values, entries[f'Last_l{loss_key}'].values
    min_val = min(min(best), min(last))
    max_val = max(max(best), max(last))
    x = np.linspace(min_val, max_val, 2)
    plt.plot(x, x, color='lime', linestyle='--', alpha=.4)
    sns.histplot(x=best, y=last, bins=50,
                 log_scale=(True, True), cbar=True)
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    # plt.title(f'Distribution of ' + r'$L^2-$norms approximating' +
    #           f'\n {fn} with {kernel} kernel')
    plt.tight_layout()
    plt.savefig(
        f'{folder}/incidence_of_l{loss_key}_{fn}.pdf', bbox_inches='tight')
    plt.close()


def plot_l_medians_all(center_sizes: list[int], poly_deg: int, l_medians, ylabel_text, file_start_name: str,
                       k_values: list[float], fn: str, folder: str = 'plots', colors: list[str] = colors):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    for c_idx in range(len(center_sizes)):
        c = center_sizes[c_idx]
        plt.plot(k_values, l_medians[c_idx],
                 label=f'{c} centers', marker='o', linestyle='--', linewidth='.5', color=colors[c_idx])

    # plt.xlabel(
    #     r'Scaling factor $k$ in $|training\_points| = \lceil \sqrt{|centers| \cdot k} \rceil^2$')
    plt.ylabel(ylabel_text)
    # plt.title(
    #     f'Median results for interpolation\n of {fn}\n with {kernel} kernel' + (f' + deg {poly_deg}' if poly_deg > -1 else ''))
    # plt.legend(loc='upper left', bbox_to_anchor=(5, 1), ncol=5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'{folder}/medians_{file_start_name}_{fn}_Poly{poly_deg}.pdf', bbox_inches='tight')
    plt.close()


def boxplot_of_l(center_sizes: list[int], poly_deg: int, l_data, ylabel_text, file_start_name: str,
                 k_values: list[float], fn: str, ylim: tuple[float, float] | None = None, colors: list[str] = colors, folder: str = 'plots'):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    for c_idx in range(len(center_sizes)):
        # c = center_sizes[c_idx]
        bp = plt.boxplot(l_data[c_idx], positions=k_values, flierprops=dict(
            markeredgecolor=colors[c_idx]), widths=.3 - .055 * c_idx)
        for box in bp['boxes']:
            box.set(color=colors[c_idx],
                    linewidth=.5)

        for whisker in bp['whiskers']:
            whisker.set(color=colors[c_idx], linewidth=0.05)
        for cap in bp['caps']:
            cap.set(color=colors[c_idx])
        for median in bp['medians']:
            median.set(color=colors[c_idx])
        for flier in bp['fliers']:
            flier.set(color=colors[c_idx], markersize=3)

    # plt.xlabel(
    #     r'Scaling factor $k$ in $|training\_points| = \lceil \sqrt{|centers| \cdot k} \rceil^2$')
    plt.ylabel(ylabel_text)
    plt.xlabel('k')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    # plt.title(
    #     f'Median results for interpolation\n of {fn}\n with {kernel} kernel' + (f' + deg {poly_deg}' if poly_deg > -1 else ''))
    # plt.legend(loc='upper left', bbox_to_anchor=(5, 1), ncol=5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'{folder}/boxplot_{file_start_name}_{fn}_Poly{poly_deg}.pdf', bbox_inches='tight')
    plt.close()

# Helper function to calculate the mean of a list


def mean(lst):
    return sum(lst) / len(lst)

# Helper function to calculate the variance of a list


def variance(lst):
    m = mean(lst)
    return sum((x-m)**2 for x in lst) / (len(lst) - 1)


def do_stuff_with_curves(losses: list[list[float]], curve_type: Literal['Loss', 'L2', 'Linf'], folder: str = 'plots', format: str = 'png'):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    max_length = max(len(loss_curve) for loss_curve in losses)
    result = [0.] * max_length
    maxima = [-1.] * max_length
    minima = [100.] * max_length

    plt.figure(figsize=(4, 2))

    for loss_curves in losses:

        # plt.semilogy(loss_curves, linewidth=.3, alpha=.2, color='blue')

        for i, value in enumerate(loss_curves):
            result[i] += value
            maxima[i] = max(value, maxima[i])
            minima[i] = min(value, minima[i])

    for i in range(max_length):
        count = sum(1 for sublist in losses if i < len(sublist))
        if count > 0:
            result[i] /= count
    plt.fill_between(list(range(max_length)), minima,
                     maxima, color='blue', alpha=0.2)
    plt.semilogy(
        result, color='orange', linewidth=.25, alpha=1)
    loss_line = mlines.Line2D([], [], color='blue')
    mean_line = mlines.Line2D([], [], color='orange')
    label_curve = f'$L^2$' if curve_type == 'L2' else f'$L^\infty$' if curve_type == 'Linf' else 'Loss'
    plt.legend(handles=[loss_line, mean_line], labels=[label_curve, 'Mean'])
    plt.savefig(f'{folder}/{curve_type}_curves_semilogy_all.{format}',
                bbox_inches='tight')
    plt.close()


shape_params_dict_2d_1_1: dict[int, float] = {49: 1.3428, 81: 1.9531,
                                              121: 2.5635, 169: 3.1738, 225: 3.6621, 400: 5.1270, 900: 7.8125}
shape_params_dict_2d_0_1: dict[int, float] = {49: 2.6855, 81: 3.9063,
                                              121: 5.1269, 169: 6.3477, 225: 7.3242, 400: 10.2539, 900: 15.6250}
shape_params_dict_1d_1_1: dict[int, float] = {7: 0.78125, 9: 1.25, 11: 1.5625,
                                              13: 2.03125, 15: 2.5, 20: 3.6621, 30: 5.6152, 40: 7.8125,
                                              50: 9.7656, 60: 11.7188}
shape_params_dict_1d_0_1: dict[int, float] = {7: 1.4648, 9: 2.4414, 11: 3.4180,
                                              13: 3.9063, 15: 4.8828, 20: 7.3242, 30: 11.2305, 40: 15.6250,
                                              50: 19.5313, 60: 23.4375}


def get_train_sizes_2d(C: int):
    if C == 49:
        return [7, 9, 10, 12, 13, 14, 15]
    if C == 81:
        return [9, 12, 13, 15, 16, 17, 18]
    if C == 121:
        return [11, 14, 16, 18, 20, 21, 22]
    if C == 169:
        return [13, 16, 19, 21, 23, 25, 26]
    if C == 225:
        return [15, 19, 22, 24, 26, 29, 30]
    if C == 400:
        return [20, 25, 29, 32, 35, 38, 40]
    if C == 900:
        return [30, 37, 43, 48, 52, 57, 60]
    raise Exception()


def aggregate_curves(poly_degs: list[int], fn: str, curve_type: Literal['Loss', 'L2', 'Linf'], format: str = 'png'):
    import re

    def get_list_of_files(folder_path, poly_degs: list[int]):
        list_of_files = []
        import os
        for poly_deg in poly_degs:
            poly_text = f'Poly{poly_deg}' if poly_deg > -1 else ''
            pattern = rf'curves{fn}-TR\d+-C\d+{poly_text}-tag(\d+)\-E(\d+)\-curves.csv'
            for file_name in os.listdir(folder_path):
                match = re.match(pattern, file_name)
                if match:
                    list_of_files += [file_name]
            return list_of_files
    losses_list = []

    starting_idx = 5 if curve_type == 'Loss' or curve_type == 'Linf' else 3
    for path_to_curves_file in get_list_of_files('./curves-of-executions', poly_degs):
        with open(f'curves-of-executions/{path_to_curves_file}', 'r') as file:
            if (curve_type == 'L2'):
                file.readline()
            if (curve_type == 'Linf'):
                file.readline()
                file.readline()
            loss_str = (file.readline().strip('\n')[starting_idx:]).replace(
                ' ', '').replace('[', '').replace(']', '')
            loss_list = [float(x) for x in loss_str.split(',')]
            losses_list += [loss_list]
    do_stuff_with_curves(losses_list, curve_type, format='png')


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


def plot_l_norms(l_values_per_centers_and_k, fn: str, poly_deg, ylabel_text: str,
                 file_start_name: str, starting_shapes: list[float], k_values: list[float],
                 center_sizes: list[int], kernel: str,
                 ylim: tuple[float, float] | None = None):
    # overall_min = np.min(
    #     [np.min(lst) for row in l_values_per_centers_and_k for lst in row])
    import matplotlib.pyplot as plt
    for c_idx in reversed(range(len(l_values_per_centers_and_k))):
        starting_shape = starting_shapes[c_idx]
        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)

        ax.violinplot(
            l_values_per_centers_and_k[c_idx], positions=k_values, showmedians=True)

        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        else:
            plt.ylim(-7., 0.)
        plt.ylabel(ylabel_text)

        plt.grid(visible=True)

        plt.title(f'{center_sizes[c_idx]} centers')
        plt.xlabel('k')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False)  # labels along the bottom edge are off

        plt.savefig(
            f'plots/{file_start_name}_{fn}_C{center_sizes[c_idx]}_{kernel}_shape_{starting_shape}_Poly{poly_deg}.pdf', bbox_inches='tight')
        plt.close()


def plot_l_norms_endgame(l_values_per_centers, fn: str, ylabel_text: str,
                         file_start_name: str, starting_shapes: list[float],
                         center_sizes: list[int], kernel: str,
                         ylim: tuple[float, float] | None = None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(111)

    ax.set_xticks(list(range(1, len(center_sizes)+1)))
    ax.set_xticklabels(center_sizes)
    ax.violinplot(
        l_values_per_centers,  showmedians=True)

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    else:
        plt.ylim(-7., 0.)
    plt.ylabel(ylabel_text)
    plt.xlabel('Number of centers')

    plt.grid(visible=True)
    plt.savefig(
        f'plots/{file_start_name}_{fn}_endgame.pdf', bbox_inches='tight')
    plt.close()

    # for c_idx in reversed(range(len(l_values_per_centers_and_k))):
    #     starting_shape = starting_shapes[c_idx]

    #     plt.title(f'{center_sizes[c_idx]} centers')
    #     plt.xlabel('k')
    #     plt.tick_params(
    #         axis='x',          # changes apply to the x-axis
    #         which='both',      # both major and minor ticks are affected
    #         left=False,      # ticks along the bottom edge are off
    #         right=False,         # ticks along the top edge are off
    #         labelleft=False)  # labels along the bottom edge are off

    #     plt.savefig(
    #         f'plots/{file_start_name}_{fn}_C{center_sizes[c_idx]}_{kernel}_shape_{starting_shape}_Poly{poly_deg}.pdf', bbox_inches='tight')
    #     plt.close()


def incidence_of_largest_errors(print_all_together: bool, df: DataFrame, kernel: str,
                                fn: str, centers: list[int], range_of_hist: list[list[float]],
                                use_int_formatter=False,
                                fn_callable: None | Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker as ticker
    import matplotlib as mpl
    import matplotlib.patches as patches
    # errors_list = []

    x_list, y_list = [], []
    entries = df[(df.Kernel == kernel) &
                 (df.Function == fn) &
                 (df.Points_distribution == 'Equi') &
                 (df.Trained_centers.notnull())]
    for c_idx in range(len(centers)):
        c = centers[c_idx]
        list_to_parse = entries[entries.Center_size ==
                                c]['Last_linf_location'].values.tolist()
        if len(list_to_parse) == 0:
            raise Exception(f'List is empty for C {c}')
        x_partial_list, y_partial_list = [], []
        for entry in list_to_parse:
            x_entry, y_entry = entry.replace(' ', '').split(',')
            x_partial_list += [float(x_entry)]
            y_partial_list += [float(y_entry)]

        if not print_all_together:
            plt.figure(figsize=(3, 2))
            plt.hist2d(
                x_partial_list, y_partial_list, bins=30, cmap='Oranges_r',
                norm=mpl.colors.LogNorm(),
                range=range_of_hist
            )
            cbar = plt.colorbar(label='Count')
            if use_int_formatter:
                cbar.ax.yaxis.set_minor_formatter(
                    ticker.ScalarFormatter())
                cbar.ax.yaxis.set_major_formatter(
                    ticker.ScalarFormatter())
                cbar.ax.yaxis.set_minor_locator(
                    ticker.MaxNLocator(nbins=3))
            plt.ylabel('Incidence')
            plt.title(f'{c} centers')

            if fn_callable is not None:
                import numpy as np
                x_interval, y_interval = range_of_hist
                X = np.linspace(x_interval[0]-.5, x_interval[1]+.5, 100)
                Y = np.linspace(y_interval[0]-.5, y_interval[1]+.5, 100)
                x, y = np.meshgrid(X, Y)
                Z = fn_callable(x, y)
                contours = plt.contour(x, y, Z, colors='gray',
                                       linewidths=1., levels=10, alpha=0.7)
                plt.clabel(contours, inline=True, fontsize=3)
                plt.xlim(x_interval[0]-.1, x_interval[1]+.1)
                plt.ylim(y_interval[0]-.1, y_interval[1]+.1)
            circ = patches.Circle(xy=(0, 0), radius=1, linewidth=2,  zorder=-1,
                                  edgecolor='black', facecolor='none')
            plt.gca().add_patch(circ)
            # plt.axis('equal')
            # plt.tight_layout()
            plt.gca().set_aspect('equal')
            plt.savefig(
                f'plots/distribution_of_errors_L_inf_'+f'{c if not print_all_together else ""}'+f'_{fn}.pdf', bbox_inches='tight')
            plt.close()
        else:
            x_list += x_partial_list
            y_list += y_partial_list
    if print_all_together:
        plt.figure(figsize=(3, 2))
        plt.hist2d(
            x_list, y_list, bins=30, cmap='Oranges',
            norm=mpl.colors.LogNorm(),
            range=range_of_hist
        )
        cbar = plt.colorbar(label='Count')

        if use_int_formatter:
            cbar.ax.yaxis.set_minor_formatter(
                ticker.ScalarFormatter())
            cbar.ax.yaxis.set_major_formatter(
                ticker.ScalarFormatter())
            cbar.ax.yaxis.set_minor_locator(
                ticker.MaxNLocator(nbins=3))
        plt.ylabel('Incidence')
        plt.title(f'Overall')

        if fn_callable is not None:
            import numpy as np
            x_interval, y_interval = range_of_hist
            X = np.linspace(x_interval[0]-.5, x_interval[1]+.5, 100)
            Y = np.linspace(y_interval[0]-.5, y_interval[1]+.5, 100)
            x, y = np.meshgrid(X, Y)
            Z = fn_callable(x, y)
            contours = plt.contour(x, y, Z, colors='gray',
                                   linewidths=1., levels=10, alpha=0.7)
            plt.clabel(contours, inline=True, fontsize=3)
            plt.xlim(x_interval[0]-.1, x_interval[1]+.1)
            plt.ylim(y_interval[0]-.1, y_interval[1]+.1)
        circ = patches.Circle(xy=(0, 0), radius=1, linewidth=2,  zorder=-1,
                              edgecolor='black', facecolor='none')
        plt.gca().add_patch(circ)
        # plt.axis('equal')
        # plt.tight_layout()
        plt.gca().set_aspect('equal')
        plt.savefig(
            f'plots/distribution_of_errors_L_inf_'+f'{c if not print_all_together else ""}'+f'_{fn}.pdf', bbox_inches='tight')
        plt.close()


def get_average_error_1d(df: DataFrame, kernel: str, fn: str | None, fn_callable: Callable[[torch.Tensor], torch.Tensor],
                         kernel_callable: Callable[[Union[float, torch.Tensor]],
                                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                         xmin: float = -1., xmax: float = 1.,
                         poly_deg: int | None = None,
                         k=4.):
    # Get verification dataset
    n_verification = 400

    x_validate = torch.linspace(
        xmin, xmax, n_verification).reshape(-1, 1)
    y_validate = fn_callable(x_validate).reshape(-1, 1)
    centers_list = [7, 15, 30, 60]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 3))
    for c_idx in range(len(centers_list)):
        C = centers_list[c_idx]
        errors_accumulated = torch.zeros_like(y_validate)
        entries = df[(df.Kernel == kernel) &
                     (df.Center_size == C) &
                     (df.Train_size == math.ceil(k*C)) &
                     (df.Starting_shape == (shape_params_dict_1d_1_1[C] if fn != 'sin_cube_tref' else shape_params_dict_1d_0_1[C])) &
                     (df.Points_distribution == 'Equi') &
                     (df.Trained_centers.notnull())]
        if fn is not None:
            entries = entries[entries.Function == fn]
        if poly_deg is not None:
            entries = entries[entries.Poly_degree == poly_deg]
        if len(entries) == 0:
            raise Exception(f'TR: {math.ceil(k*C)}, C: {C}')

        for idx in range(len(entries)):
            data_row = entries.iloc[idx]
            if data_row['Poly_degree'] == -1:
                nn = RBF(1, num_centers=C, output_dim=1,
                         kernel=kernel_callable)
            else:
                nn = RBF_Poly(1, num_centers=C, output_dim=1,
                              kernel=kernel_callable, degree=data_row['Poly_degree'])

            # Set centers
            list_to_convert = data_row['Trained_centers'].split(',')
            floating_list = [float(entry) for entry in list_to_convert]
            nn.set_centers(torch.tensor(floating_list).reshape(-1, 1))

            list_to_convert = data_row['Coefs_list'].split(',')
            floating_list = [float(entry) for entry in list_to_convert]
            if data_row['Poly_degree'] == -1:
                nn.set_coefs(torch.tensor(floating_list).reshape(1, -1))
            else:
                rbf_coefs = torch.tensor(
                    floating_list[:(-data_row['Poly_degree'] - 1)]).reshape(1, -1)
                poly_coefs = torch.tensor(
                    floating_list[(-data_row['Poly_degree'] - 1):]).reshape(1, -1)
                nn.set_coefs(rbf_coefs, poly_coefs)

            list_to_convert = data_row['Trained_shapes'].split(',')
            floating_list = [float(entry) for entry in list_to_convert]
            if data_row['Poly_degree'] == -1:
                nn.set_shapes(torch.tensor(floating_list))
            else:
                nn.rbf.set_shapes(torch.tensor(floating_list))

            errors_accumulated += torch.abs(nn(x_validate) - y_validate)
            if torch.max(torch.abs(nn(x_validate) - y_validate)) > 1e-3:
                err = torch.max(torch.abs(nn(x_validate) - y_validate))
                print(f'C: {C}, TR: {math.ceil(C*k)}, err: {err}')

        plt.semilogy(x_validate.detach().numpy(),
                     errors_accumulated.detach().numpy() / len(entries),
                     color=colors[c_idx],
                     label=f'{C} centers',
                     linewidth=.5)

    plt.legend(loc='upper left', bbox_to_anchor=(1.15, 1))
    plt.ylabel('Absolute error')
    plt.twinx()
    plt.plot(x_validate.detach().numpy(), y_validate.detach(
    ).numpy(), color='gray', linestyle='--', alpha=.8)
    plt.ylabel('Function Value', color='gray')
    plt.tick_params(axis='y', labelcolor='gray')
    text_title = f', degree = {poly_deg}' if poly_deg is not None else ''
    plt.title(f'k={k}' + text_title)
    plt.savefig(f'plots/{fn}_{C}_{k}.pdf', bbox_inches='tight')
    plt.close()


def get_average_error_2d(df: DataFrame, kernel: str,
                         fn: Callable[[torch.Tensor], torch.Tensor],
                         kernel_callable: Callable[[Union[float, torch.Tensor]], Callable[[
                             torch.Tensor, torch.Tensor], torch.Tensor]] = gaussian_kernel,
                         radius: float = 1., vmin: float | None = None, vmax: float | None = None,
                         poly_deg: int | None = None
                         ):
    import matplotlib.pyplot as plt
    import re
    import math
    import matplotlib.patches as patches

    with torch.no_grad():
        dim = 2
        n_verification = 50

        x_validate_raw = torch.cartesian_prod(
            *[torch.linspace(-radius, radius, n_verification) for _ in range(dim)]).reshape(-1, dim)
        y_validate_raw = fn(x_validate_raw).reshape(-1, 1)
        x_validate_indices = torch.sum(x_validate_raw**2, dim=1) <= radius**2
        x_validate = x_validate_raw[x_validate_indices]
        y_validate = fn(x_validate).reshape(-1, 1)
        y_validate_domain = torch.where(
            x_validate_indices.unsqueeze(1), y_validate_raw, torch.nan).reshape(n_verification, n_verification)

        centers_list = [7**2, 9**2, 11**2, 13**2, 15**2]

        for C in centers_list:
            # for TR in get_train_sizes_2d(C):
            TR = get_train_sizes_2d(C)[-1]
            approximations_square = torch.zeros_like(y_validate_raw)
            entries = df[(df.Kernel == kernel) &
                         (df.Function == fn.__name__) &
                         (df.Center_size == C) &
                         (df.Train_size == TR) &
                         (df.Starting_shape == shape_params_dict_2d_1_1[C]) &
                         (df.Points_distribution == 'Equi') &
                         (df.Trained_centers.notnull())]
            if poly_deg is not None:
                entries = entries[entries.Poly_degree == poly_deg]
            if len(entries) == 0:
                raise Exception(
                    f'Holy shit!. TR {TR}, C {C}, Shape: {shape_params_dict_2d_1_1[C]}')

            for idx in range(len(entries)):
                data_row = entries.iloc[idx]

                if data_row['Poly_degree'] == -1:
                    nn = RBF(dim, num_centers=C, output_dim=1,
                             kernel=kernel_callable)
                else:
                    nn = RBF_Poly(2, num_centers=C, output_dim=1,
                                  kernel=kernel_callable, degree=data_row['Poly_degree'])

                # Set centers
                pattern = re.compile(r'\[(.*?)\]')
                lists = pattern.findall(data_row['Trained_centers'])

                centers_x = [float(num.strip())
                             for num in lists[0].split(',')]
                centers_x = torch.tensor(centers_x).reshape(-1, 1)
                centers_y = [float(num.strip())
                             for num in lists[1].split(',')]
                centers_y = torch.tensor(centers_y).reshape(-1, 1)
                centers = torch.cat((centers_x, centers_y), dim=1)

                nn.set_centers(centers)

                list_to_convert = data_row['Coefs_list'].split(',')
                floating_list = [float(entry) for entry in list_to_convert]
                if data_row['Poly_degree'] == -1:
                    nn.set_coefs(torch.tensor(floating_list).reshape(1, -1))
                else:
                    rbf_coefs = torch.tensor(
                        floating_list[:-6]).reshape(1, -1)
                    poly_coefs = torch.tensor(
                        floating_list[-6:]).reshape(1, -1)
                    nn.set_coefs(rbf_coefs, poly_coefs)

                list_to_convert = data_row['Trained_shapes'].split(',')
                floating_list = [float(entry) for entry in list_to_convert]
                if data_row['Poly_degree'] == -1:
                    nn.set_shapes(torch.tensor(floating_list))
                else:
                    nn.rbf.set_shapes(torch.tensor(floating_list))

                approximations_square += nn(x_validate_raw)

            approximation_domain = torch.where(
                x_validate_indices.unsqueeze(1), approximations_square, torch.nan).reshape(n_verification, n_verification) / len(entries)
            plt.figure(figsize=(4, 2))

            plt.xlim(-radius-.5, radius+.5)
            plt.ylim(-radius-.5, radius+.5)
            plt.xticks([-radius-.5, 0, radius+.5])
            plt.yticks([-radius-.5, 0, radius+.5])

            # Add contour plot with ground-truth function
            # Generate X and Y coordinates
            x_vals = np.linspace(-radius-.5, radius+.5, n_verification)
            y_vals = np.linspace(-radius-.5, radius+.5, n_verification)
            X, Y = np.meshgrid(x_vals, y_vals)
            contours = plt.contour(X, Y, globals().get(fn.__name__ + '_numpy')(X, Y),
                                   colors='gray', linewidths=1., alpha=.7)
            plt.clabel(contours, inline=True, fontsize=3)

            circ = patches.Circle(xy=(0, 0), radius=1, linewidth=1,
                                  edgecolor='black', facecolor='none', zorder=10)
            plt.gca().add_patch(circ)

            avg_err = torch.abs(approximation_domain - y_validate_domain).T
            plt.imshow(X=avg_err,
                       extent=(-radius, radius, -radius, radius),
                       cmap="Oranges_r", norm='log', origin='lower',
                       vmin=vmin, vmax=vmax)
            plt.colorbar(label='Absolute error')
            text_title = f', degree = {poly_deg}' if poly_deg is not None else ''
            plt.title(f'{C} centers{text_title}', fontsize=10)
            # if show_legend:
            #     plt.legend(loc='lower left', bbox_to_anchor=(-1, 1))
            plt.savefig(
                f'plots/{fn.__name__}_{C}_{poly_deg}.pdf', bbox_inches='tight')
            plt.close()


def histogram_of_centers_1d_extended(n_train_size: int, df: DataFrame, kernel: str,
                                     domain: Tuple[float, float], hist_dom: Tuple[float, float],
                                     fn_name: str | None, fn_callable, n_centers: int, folder: str = 'plots',
                                     poly_deg: int | None = None,):

    import matplotlib.pyplot as plt
    import numpy as np

    entries = df[(df.Kernel == kernel) &
                 (df.Points_distribution == 'Equi') &
                 (df.Center_size == n_centers) &
                 (df.Train_size == n_train_size)]
    if fn_name is not None:
        entries = entries[entries.Function == fn_name]
    if poly_deg is not None:
        entries = entries[entries.Poly_degree == poly_deg]
    if len(entries) == 0:
        raise Exception('Wrong query')
    centers_lists = entries['Trained_centers'].values.tolist()
    centers_parsed = []
    for center_list in centers_lists:
        centers = center_list.replace(' ', '').split(',')
        centers_parsed += [float(center) for center in centers]
    n_bins = 50
    hist, bin_edges = np.histogram(centers_parsed, bins=n_bins)
    bins_for_shapes = [[] for _ in range(n_bins)]
    bins_for_coefficients = [[] for _ in range(n_bins)]
    for idx in range(len(entries)):
        row = entries.iloc[idx]
        list_of_centers = [float(x)
                           for x in row['Trained_centers'].split(',')]
        list_of_shapes = [
            float(x) for x in row['Trained_shapes'].split(',')]
        list_of_coefs = [
            float(x) for x in row['Coefs_list'].split(',')]
        # trim in case we have polynomials
        list_of_coefs = list_of_coefs[:len(list_of_centers)]

        for center_idx in range(len(list_of_centers)):
            center_bin = np.digitize(
                list_of_centers[center_idx], bin_edges, right=True) - 1
            bins_for_shapes[center_bin] += [list_of_shapes[center_idx]]
            bins_for_coefficients[center_bin] += [list_of_coefs[center_idx]]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    ax0, ax1, ax2 = axes
    ax0.set_title(f'{n_centers} centers, {n_train_size} training points')

    ax1.set_xlim(hist_dom[0], hist_dom[1])
    ax2.set_xlim(hist_dom[0], hist_dom[1])

    ax0.set_xlim(hist_dom[0], hist_dom[1])
    ax0.bar(bin_edges[:-1], hist, width=np.diff(bin_edges),
            edgecolor='black')
    ax0.set_ylabel('Incidence')

    # Add vertical lines
    ax0.axvline(domain[0], color='black', ls='--')
    ax0.axvline(domain[1], color='black', ls='--')

    # Plot the function on the secondary y-axis
    ax0t = ax0.twinx()
    x_plot = torch.linspace(hist_dom[0], hist_dom[1], 300)
    ax0t.plot(x_plot, fn_callable(x_plot),
              'red', linewidth=2)  # type: ignore
    ax0t.set_ylabel('Function Value', color='red')
    ax0t.tick_params(axis='y', labelcolor='red')

    positions_to_parse = (bin_edges[:-1] + bin_edges[1:])/2
    bins_parsed = []
    positions_parsed = []
    for idx in range(len(bins_for_shapes)):
        if len(bins_for_shapes[idx]) > 0:
            bins_parsed += [bins_for_shapes[idx]]
            positions_parsed += [positions_to_parse[idx]]
    ax1.violinplot(bins_parsed,
                   positions=positions_parsed, showmedians=True, widths=.05)
    ax1.set_ylabel('Shapes')

    bins_parsed = []
    positions_parsed = []
    for idx in range(len(bins_for_coefficients)):
        if len(bins_for_coefficients[idx]) > 0:
            bins_parsed += [bins_for_coefficients[idx]]
            positions_parsed += [positions_to_parse[idx]]
    ax2.violinplot(bins_parsed,
                   positions=positions_parsed, showmedians=True, widths=.05)

    ax2.set_ylabel('Coefficients')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig(f'{folder}/C{n_centers}-TR{n_train_size}-{fn_name}.pdf',
                bbox_inches='tight')
    plt.close()


def gather_data_endgame(df: DataFrame, fn_names: list[str], center_sizes: list[int],
                        shapes: list[float],
                        ylims_left: list[float] | None, ylims_right: list[float] | None,
                        kernel: str = 'gaussian_kernel'):
    # for poly_deg in poly_degs:
    for fn_idx in range(len(fn_names)):
        fn = fn_names[fn_idx]
        linf_values = np.empty(len(center_sizes), dtype=np.ndarray)
        l2_values = np.empty(len(center_sizes), dtype=np.ndarray)
        elapsed_times_values = np.empty(len(center_sizes), dtype=np.ndarray)

        for c_idx in range(len(center_sizes)):
            c = center_sizes[c_idx]
            starting_shape = float(shapes[c_idx])
            tr = get_train_sizes_2d(c)[-1]

            entries = df[(df.Kernel == kernel) &
                         (df.Center_size == c) &
                         (df.Function == fn) &
                         (df.Train_size == tr) &
                         #  (df.Starting_shape == starting_shape) &
                         (df.Points_distribution == 'Equi') &
                         (df.Trained_centers.notnull())]
            if len(entries) == 0:
                raise Exception(
                    f'Something went wrong, empty query: fn {fn}, TR {tr}, C {c}')
            linf_loss = entries['Linf_loss']
            l2_loss = entries['L2_loss']
            elapsed_time = entries['Time_elapsed']

            l2_values[c_idx] = np.log10(l2_loss.values)
            linf_values[c_idx] = np.log10(linf_loss.values)
            elapsed_times_values[c_idx] = elapsed_time.values / 3600
        plot_l_norms_endgame(linf_values, fn,
                             ylabel_text=r'$\log_{10}(L^\infty-\text{norm})$', file_start_name='violins_linf',
                             starting_shapes=shapes, center_sizes=center_sizes,
                             kernel=kernel,
                             #  ylim=(    (ylims_left[fn_idx], ylims_right[fn_idx]))
                             #  if ylims_left is not None and ylims_right is not None else None)
                             ylim=(-2.8, -1.5))
        plot_l_norms_endgame(l2_values, fn,
                             ylabel_text=r'$\log_{10}(L^2-\text{norm})$', file_start_name='violins_l2',
                             starting_shapes=shapes, center_sizes=center_sizes,
                             kernel=kernel,
                             #  ylim=(    (ylims_left[fn_idx]-1, ylims_right[fn_idx]))
                             #  if ylims_left is not None and ylims_right is not None else None)
                             ylim=(-3.1, -2.4))
