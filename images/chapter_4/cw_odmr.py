import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from util import inches, smoothing


def main():
    cts, freqs = load_raw_data()

    plot_and_save_odmr_spectrum(cts, freqs)


def plot_and_save_odmr_spectrum(cts, freqs):
    plt.figure(figsize=(inches.cm_to_inch(15), inches.cm_to_inch(5.5)))
    plt.plot(freqs * 1e-9, smoothing.smooth_array_by_rolling_average(cts, 2) * 1e-3, '.')
    plt.xlabel('mw frequency (GHz)')
    plt.ylabel('luminescence (kcts/s)')
    plt.tight_layout()
    plt.savefig('cw_odmr_spectrum.png', dpi=500)


def load_raw_data():
    file_name_and_path = "//file/e24/Projects/ReinhardLab/data_setup_nv1/190709_D02C_AlOx/odmr.003.mat"
    odmr_data = sio.loadmat(file_name_and_path)
    cts = odmr_data['ci'][0]
    freqs = np.linspace(odmr_data['fstart'][0][0], odmr_data['fend'][0][0], cts.shape[0])
    return cts, freqs


if __name__ == '__main__':
    main()
