import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from util import graph_transformation
from util.inches import cm_to_inch


def main():
    cts_broad, freqs_broad = load_raw_data('p_odmr.000.mat')
    cts_narrow, freqs_narrow = load_raw_data('p_odmr.002.mat')

    plot_and_save_odmr_spectrum(cts_broad, cts_narrow, freqs_broad, freqs_narrow)


def plot_and_save_odmr_spectrum(cts_broad, cts_narrow, freqs_broad, freqs_narrow):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(cm_to_inch(5.5))
    fig.set_figwidth(cm_to_inch(15))
    plot_one_axis(ax1, cts_broad, freqs_broad)
    plot_one_axis(ax2, cts_narrow, freqs_narrow)
    plt.tight_layout()
    plt.savefig('pulsed_odmr_spectrum.png', dpi=500)


def plot_one_axis(axis, cts, freqs):
    axis.plot_for_thesis(freqs * 1e-9, graph_transformation.smooth_array_by_rolling_average(cts, 2) * 1e-6, '.')
    axis.set_xlabel('mw frequency (GHz)')
    axis.set_ylabel('luminescence (Mcts/s)')


def load_raw_data(file_name):
    file_name_and_path = os.path.join(
        "//file/e24/Projects/ReinhardLab/data_setup_nv1/190306_timo_cam+p_odmr/pulsed_odmr_1_detuned", file_name)
    odmr_data = sio.loadmat(file_name_and_path)
    cts = odmr_data['ci'][0]
    freqs = np.linspace(odmr_data['fstart'][0][0], odmr_data['fend'][0][0], cts.shape[0])
    return cts, freqs


if __name__ == '__main__':
    main()
