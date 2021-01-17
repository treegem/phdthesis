import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as s_optimize

from util.inches import cm_to_inch
from util.tum_jet import tum_raw, tum_jet


def tum_color(index):
    color = tum_raw[index]
    norm_color = (color[0] / 256, color[1] / 256, color[2] / 256)
    return norm_color


def main(path):
    taus = np.loadtxt(os.path.join(path, 'taus.txt'))
    zs = np.loadtxt(os.path.join(path, 'zs_mat.txt'))
    zs_x = zs[:, 0::2]

    offset_x = calculate_offset(path)

    rabi_amplitude = 0.11

    matrix_x = shifted_matrix(zs_x, offset_x)

    save_triangle_plots(matrix_x, taus, rabi_amplitude, path)


def zs_to_probability(random_data, offset):
    rabi_amplitude = 0.11
    random_data -= offset
    random_data /= rabi_amplitude
    random_data = random_data / 2 + 0.5
    return random_data


def perform_fit(p0, trimmed, x_data):
    p_opt, _ = s_optimize.curve_fit(fit_func, x_data, trimmed, p0=p0,
                                    bounds=([p0[0] / 2, 0.35, -np.pi, 0.5], [p0[0] * 2, 2, 2.25 * np.pi, 1]))
    return p_opt


def fit_func(t, T, A, phi, C):
    return A * np.cos(t * 2 * np.pi / T + phi) + C


def compress_matrix_vertical(path):
    return np.loadtxt(os.path.join(path, 'sine_fit_zs.txt'))


def save_triangle_plots(matrix_x, taus, rabi_amplitude, path):
    plt.close('all')
    axes, fig = create_figure_and_axes()

    im_one = plot_triangular_plot(axes, matrix_x, rabi_amplitude, taus)
    plot_phase_oscillation(axes, matrix_x, rabi_amplitude)
    plot_echo_decay(axes, path, taus)

    fig.delaxes(axes.flat[3])
    fig.subplots_adjust(right=0.8, left=0.12, bottom=0.1, top=0.95, wspace=0.0, hspace=0.0)
    cbar_ax = fig.add_axes([0.85, 0.55, 0.05, 0.4])
    fig.colorbar(im_one, cax=cbar_ax, label=r'$1-\left\langle S_z \right\rangle$', ticks=[0, 0.5, 1])
    plt.savefig(os.path.join(path, 'self_calibrated_x_wrong_colors.jpg'), dpi=500)


def create_figure_and_axes():
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figwidth(cm_to_inch(15))
    fig.set_figheight(cm_to_inch(12))
    return axes, fig


def plot_echo_decay(axes, path, taus):
    echo_decay = compress_matrix_vertical(path)
    axes.flat[2].plot(taus, echo_decay, color=tum_color(0))
    axes.flat[2].set_xlabel(r'$\tau$ (ns)')
    axes.flat[2].set_ylabel(r'$1-\left\langle S_z \right\rangle$')
    axes.flat[2].set_xlim([0, 500])
    axes.flat[2].axes.get_xaxis().set_ticks([0, 250, 500])
    axes.flat[2].axes.get_yaxis().set_ticks([0.5, 0.6, 0.7])


def plot_phase_oscillation(axes, matrix_x, rabi_amplitude):
    c_matrix_x = compress_matrix_horizontal(matrix_x) / rabi_amplitude + 0.5
    axes.flat[1].plot(c_matrix_x, np.linspace(c_matrix_x.min(), c_matrix_x.max(), len(c_matrix_x)), color=tum_color(0))
    axes.flat[1].tick_params(axis='both', direction='in', top=True, right=True)
    axes.flat[1].axes.get_yaxis().set_ticklabels([])
    axes.flat[1].set_xlim([0, 1])
    axes.flat[1].set_ylim([c_matrix_x.min(), c_matrix_x.max()])
    axes.flat[1].axes.get_xaxis().set_ticks([0.5, 1])
    axes.flat[1].axes.get_xaxis().set_ticklabels([0.5, 1])
    amplitude = c_matrix_x.max() - c_matrix_x.min()
    axes.flat[1].axes.get_yaxis().set_ticks(
        [c_matrix_x.min(), c_matrix_x.min() + 0.2 * amplitude, c_matrix_x.min() + 0.4 * amplitude,
         c_matrix_x.min() + 0.6 * amplitude, c_matrix_x.min() + 0.8 * amplitude, c_matrix_x.min() + amplitude])
    axes.flat[1].set_xlabel(r'$1-\left\langle S_z \right\rangle$')


def plot_triangular_plot(axes, matrix_x, rabi_amplitude, taus):
    im_one = axes.flat[0].imshow(matrix_x / rabi_amplitude + 0.5, aspect='auto', vmin=0, vmax=1, cmap=tum_jet,
                                 interpolation='bicubic', extent=[taus[0], taus[-1], 0, 0.5], origin='lower')
    axes.flat[0].set_ylabel(r'$\int I \cdot \mathrm{d}t$' + r' (20 mA$\cdot \mu$s)')
    axes.flat[0].tick_params(axis='both', direction='in', top=True, right=True)
    axes.flat[0].axes.get_yaxis().set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axes.flat[0].axes.get_xaxis().set_ticks([255])
    return im_one


def compress_matrix_horizontal(matrix):
    compressed_matrix = np.zeros(matrix.shape[0])
    last_appendix_length = 0
    for row in range(matrix.shape[1]):
        appendix_length = len(np.trim_zeros(matrix[:, row])) - 5
        if appendix_length < last_appendix_length:
            continue
        compressed_matrix[last_appendix_length:appendix_length] = np.mean(
            matrix[last_appendix_length:appendix_length, row:], axis=1)
        last_appendix_length = appendix_length
    return compressed_matrix


def normalize_matrix(matrix):
    for row in range(matrix.shape[1]):
        trimmed = np.trim_zeros(matrix[:, row])
        std = np.std(trimmed)
        ampl = std * np.sqrt(2)
        matrix[:, row] /= ampl


def shifted_matrix(matrix, offset):
    final = trim_matrix(matrix)
    zero_ind = np.where(final == 0)
    zero_ind = zip(zero_ind[0], zero_ind[1])
    matrix = final - offset
    for ind in zero_ind:
        matrix[ind] = 0
    return matrix


def trim_matrix(zs_x):
    max_len = len(np.trim_zeros(zs_x[:, -1]))
    final = np.zeros((max_len, zs_x.shape[1]))
    for row in range(zs_x.shape[1]):
        trimmed = trim_row(row, zs_x)
        final[:len(trimmed), row] = trimmed
    return final


def calculate_offset(path):
    offsets_x = np.loadtxt(os.path.join(path, 'offsets_x.txt'))
    offsets_y = np.loadtxt(os.path.join(path, 'offsets_y.txt'))
    offsets = (offsets_x + offsets_y) / 2
    offsets = np.average(offsets)
    return offsets


def trim_row(row, matrix):
    trimmed = matrix[:, row]
    while trimmed.min() == 0:
        t = -1
        trimmed = np.trim_zeros(trimmed[:t])
        t -= 1
    return trimmed


if __name__ == '__main__':
    main('self_calibrated')
