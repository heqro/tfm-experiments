from math import sqrt
from tkinter.ttk import LabeledScale
from typing import Callable
from pandas import DataFrame
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
                        kernel: str, fn: str, folder: str = 'plots', colors: list[str] = colors):
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
                             (df.Function == fn) &
                             (df.Train_size == tr) &
                             (df.Points_distribution == 'Equi') &
                             (df.Trained_centers.notnull())]

                for starting_shape_idx in range(len(starting_shapes)):
                    starting_shape = starting_shapes[starting_shape_idx]
                    entries_to_parse = entries[entries.Starting_shape ==
                                               starting_shape]['Trained_shapes'].values
                    for list_of_shapes in entries_to_parse:
                        shapes_list[starting_shape_idx] += [float(a)
                                                            for a in list_of_shapes.split(',')]

    for starting_shape_idx in (range(len(starting_shapes))):
        data = shapes_list[starting_shape_idx]
        count, bins, ignored = plt.hist(data, bins=30, label=r'$\varepsilon=$' +
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
                            fn_name: str, fn_callable, folder: str = 'plots'):
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import numpy as np

    entries = df[(df.Kernel == kernel) &
                 (df.Function == fn_name) &
                 (df.Points_distribution == 'Equi') &
                 (df.Trained_centers.notnull())]
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
            centers_parsed, bins=30, alpha=0.6, color='g')
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
                 (df.Function == fn) &
                 (df.Points_distribution == 'Equi') &
                 (df.Trained_centers.notnull())]
    for c_idx in range(len(center_sizes)):
        c = center_sizes[c_idx]
        errors_list[c_idx] = entries[entries.Center_size ==
                                     c]['Last_linf_location'].values.tolist()
        plt.hist(errors_list[c_idx], range=(-1, 1),
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


def do_stuff_with_curves(losses: list[list[float]], folder: str = 'plots'):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    max_length = max(len(loss_curve) for loss_curve in losses)
    result = [0.] * max_length

    sample_variance = [0.] * max_length
    plt.figure(figsize=(4, 2))

    for loss_curves in losses:

        plt.semilogy(loss_curves, linewidth=.3, alpha=.2, color='blue')

        for i, value in enumerate(loss_curves):
            result[i] += value

    for i in range(max_length):
        count = sum(1 for sublist in losses if i < len(sublist))
        if count > 0:
            result[i] /= count
        # sum_iterator = 0.
        # for loss_curve in losses:
        #     if i < len(loss_curve):
        #         sum_iterator += (loss_curve[i] - result[i]) ** 2
        # sample_variance[i] = sum_iterator / max((count - 1), 1)
    plt.semilogy(
        result, color='orange', linewidth=.25, alpha=.2)
    loss_line = mlines.Line2D([], [], color='blue')
    mean_line = mlines.Line2D([], [], color='orange')
    plt.legend(handles=[loss_line, mean_line], labels=['Loss', 'Mean'])
    plt.savefig(f'{folder}/loss_curves_semilogy_all_ALL.pdf',
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4, 2))
    n_last = 300
    for loss_curves in losses:
        plt.semilogy(loss_curves[-n_last:],
                     linewidth=.02, alpha=.5, color='blue')
        plt.xlabel(f"Last {n_last} iterations")
    plt.savefig(f'{folder}/last_few_all_ALL.pdf', bbox_inches='tight')
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
