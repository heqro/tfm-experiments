from math import sqrt
from tkinter.ttk import LabeledScale
from typing import Callable, Literal
from pandas import DataFrame
from nn_rbf import RBF_Free_All as RBF
from nn_rbf_poly import RBF_Poly_Free_All as RBF_Poly
import torch
colors = ['olive', 'orange', 'purple', 'red', 'blue', 'black', 'pink']


def save_curves(experiment_file: str, loss: list[float], l2: list[float], linf: list[float], lr: list[float], training_pts=None):
    with open(f'{experiment_file}', 'w') as file:
        file.write(f'Loss:{loss}\n')
        file.write(f'L2:{l2}\n')
        file.write(f'Linf:{linf}\n')
        file.write(f'LR:{lr}\n')
        if training_pts is not None:
            file.write(f'Training_pts:{training_pts}\n')


def histogram_of_shapes(center_sizes: list[int], k_values: list[float], poly_degs: list[int], df: DataFrame,
                        get_train_sizes: Callable[[int], list[int]], starting_shapes: list[float],
                        kernel: str, fn: str | None, folder: str = 'plots', colors: list[str] = colors, n_bins: int = 30):
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import numpy as np
    shapes_list = [[], [], [], [], []]
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
                                        str(starting_shapes[starting_shape_idx]), alpha=1, histtype='step', color=colors[starting_shape_idx])

        mu, std = stats.norm.fit(data)
        x = np.linspace(0, 5, 100)
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


def get_train_sizes_1d(C: int):
    from math import ceil
    import numpy as np
    return [ceil(C*k) for k in np.arange(1, 4.5, 0.5)]


def histogram_of_centers_1d(center_sizes: list[int], df: DataFrame, kernel: str,
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
            centers_parsed, bins=n_bins, alpha=0.6, color='g')
        ax1.set_xlabel('Center Values')
        ax1.set_ylabel('Frequency', color='g')
        ax1.tick_params(axis='y', labelcolor='g')

        xmin, xmax = min(centers_parsed), max(centers_parsed)

        # Add vertical lines
        ax1.axvline(-1, color='b', ls='--')
        ax1.axvline(1, color='b', ls='--')

        # Plot the function on the secondary y-axis
        ax2 = ax1.twinx()
        x_plot = np.linspace(xmin, xmax, 100)
        ax2.plot(x_plot, fn_callable(x_plot), 'r', linewidth=2)  # type: ignore
        ax2.set_ylabel('Function Value', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title(f'Histogram of centers ({c})')
        plt.savefig(f'{folder}/histogram-centers-C{c}.pdf')
        plt.close()


def incidence_of_largest_errors_1d(center_sizes: list[int], df: DataFrame, kernel: str, fn: str, folder: str = 'plots', colors: list[str] = colors):
    import matplotlib.pyplot as plt
    errors_list = [[], [], [], [], []]
    entries = df[(df.Kernel == kernel) &
                 #  (df.Function == fn) &
                 (df.Points_distribution == 'Equi') &
                 (df.Trained_centers.notnull())]
    for c_idx in range(len(center_sizes)):
        c = center_sizes[c_idx]
        errors_list[c_idx] = entries[entries.Center_size ==
                                     c]['Last_linf_location'].values.tolist()
        plt.hist(errors_list[c_idx], range=(-1, 1), bins=45,
                 label=f'{c} centers', histtype='step', color=colors[c_idx])
    plt.yscale('log')
    plt.ylabel('Incidence')
    plt.legend()
    plt.title(
        'Distribution of the largest error (end of training process)')
    plt.savefig(f'{folder}/distribution_of_errors_L_inf_end.pdf',
                bbox_inches='tight')
    plt.close()


def incidence_of_l(df: DataFrame, kernel: str, fn: str, xlabel_text: str, ylabel_text: str, loss_key: str, folder: str = 'plots'):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    plt.figure(figsize=(4, 3))
    entries = df[(df.Kernel == kernel) &
                 (df.Function == fn) &
                 (df.Points_distribution == 'Equi') &
                 (df.Trained_centers.notnull())
                 ]
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
        f'{folder}/incidence_of_l{loss_key}.pdf', bbox_inches='tight')
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


def do_stuff_with_curves(losses: list[list[float]], curve_type: Literal['Loss', 'L2', 'Linf'], folder: str = 'plots'):
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
    plt.savefig(f'{folder}/{curve_type}_curves_semilogy_all.pdf',
                bbox_inches='tight')
    plt.close()


center_sizes_2d = [49, 81, 121, 169, 225]
starting_shapes_2d_1_1 = [1.3428, 1.9531, 2.5635, 3.1738, 3.6621]


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
    raise Exception()


def aggregate_curves(poly_degs: list[int], fn: str, curve_type: Literal['Loss', 'L2', 'Linf']):
    import re

    def get_list_of_files(folder_path, poly_degs: list[int]):
        list_of_files = []
        import os
        for poly_deg in poly_degs:
            poly_text = f'Poly{poly_deg}' if poly_deg > -1 else ''
            pattern = rf'curves-{fn}-TR\d+-C\d+{poly_text}-tag(\d+)\-E(\d+)\-curves.csv'
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
    do_stuff_with_curves(losses_list, curve_type)


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
