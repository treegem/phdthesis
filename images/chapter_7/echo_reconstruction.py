import os

import matplotlib.pyplot as plt
import numpy as np

from util import tum_jet
from util.inches import cm_to_inch


def tum_color(index):
    color = tum_jet.tum_raw[index]
    norm_color = (color[0] / 256, color[1] / 256, color[2] / 256)
    return norm_color


def mirror_y_graphs(estimated_middle, yminus, yplus):
    # mirroring yminus and yplus into each other
    # because average_plus_minus expects yplus to be longer

    newplus = 2 * estimated_middle - yminus
    newminus = 2 * estimated_middle - yplus

    return newplus, newminus


def main():
    data_directory = 'echo_reconstruction'
    path_with_current = os.path.join(data_directory, '007_random_current_echo')
    xminus, xplus, yminus, yplus = load_random_data(path_with_current)
    taus = np.loadtxt(os.path.join(path_with_current, 'taus.txt'))
    regular_data = load_regular_data(path_with_current)
    pure_data = load_pure_data(data_directory)

    estimated_middle = (xplus[-1] + xminus[-1] + yplus[-1] + yminus[-1]) / 4

    average_plus_minus(estimated_middle, minus=xminus, plus=xplus, start=6)
    yplus, yminus = mirror_y_graphs(estimated_middle, yminus, yplus)
    average_plus_minus(estimated_middle, minus=yminus, plus=yplus, start=10)
    plt.close('all')
    fig = plt.figure(figsize=(cm_to_inch(12), cm_to_inch(7)))
    pure_color = tum_color(1)
    plot_everything(pure_color, pure_data, regular_data, taus, xplus, yplus)
    fig.tight_layout()
    plt.savefig(os.path.join(data_directory, 'echo_reconstruction.png'), dpi=500)


def plot_everything(pure_color, pure_data, regular_data, taus, xplus, yplus):
    plt.plot(taus, pure_data, label=r'$\sigma_I = 0$', color=pure_color)
    plt.plot(taus, regular_data, label='$\sigma_I = 3.5$ mA', color=tum_color(0))
    x_color = tum_color(5)
    plt.plot(taus, xplus, label=r'qff$_x$', color=x_color)
    y_color = tum_color(2)
    yplus_start = 1
    plt.plot(taus[yplus_start:], yplus[yplus_start:], label=r'qff$_y$', color=y_color)
    plt.ylabel(r'$1-\left\langle S_z \right\rangle$')
    plt.xlabel(r'$\tau$ (ns)')
    plt.ylim(bottom=plt.ylim()[0] - 0.2)
    plt.legend(loc='lower center', ncol=2, frameon=False)


def load_pure_data(data_directory):
    path_pure = os.path.join(data_directory, '008_pure_echo')
    pure_data = np.loadtxt(os.path.join(path_pure, 'zs_regular.txt'))
    pure_data = zs_to_probability(pure_data)
    return pure_data


def load_regular_data(path_with_current):
    regular_data = np.loadtxt(os.path.join(path_with_current, 'zs_regular.txt'))
    regular_data = zs_to_probability(regular_data)
    return regular_data


def load_random_data(path_with_current):
    random_data = np.loadtxt(os.path.join(path_with_current, 'zs_post_selected.txt'))
    random_data = zs_to_probability(random_data)
    xplus = random_data[0]
    yminus = random_data[1]
    xminus = random_data[2]
    yplus = random_data[3]
    return xminus, xplus, yminus, yplus


def zs_to_probability(random_data):
    rabi_amplitude = 0.0900362701808  # calculated in phi_oscillation.py
    random_data -= 1 - rabi_amplitude * 2
    random_data /= 2 * rabi_amplitude
    return random_data


def average_plus_minus(estimated_middle, minus, plus, start):
    for i, z in enumerate(minus[start:]):
        mirrored_z = 2 * estimated_middle - z
        plus[start + i] = (plus[start + i] + mirrored_z) / 2


if __name__ == '__main__':
    main()
