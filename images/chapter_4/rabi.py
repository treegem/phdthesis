import os

import matplotlib.pyplot as plt
import scipy.io as sio

from util.graph_transformation import normalize
from util.graph_transformation import smooth_array_by_rolling_average as smoothen
from util.inches import cm_to_inch
from util.tum_jet import tum_color


def beautify(array):
    return 0.25 * normalize(smoothen(array)) + 0.7125


class RabiPlotter:
    def __init__(self):
        self.path = "//file/e24/Projects/ReinhardLab/data_setup_nv1/180704_hf_setup_deer_tests"
        self.fig = plt.figure(figsize=(cm_to_inch(15), cm_to_inch(6.5)))
        self.slow_rabis = self.load_file('pulsed.000.mat')
        self.fast_rabis = self.load_file('pulsed.001.mat')

    def plot_rabis(self):
        plt.plot(self.slow_rabis.x_data, beautify(self.slow_rabis.y_data), '.', label="0 dBm", color=tum_color(0))
        plt.plot(self.fast_rabis.x_data, beautify(self.fast_rabis.y_data), '.', label="6 dBm", color=tum_color(5))
        plt.xlabel("pulse duration (ns)")
        plt.ylabel("luminescence (norm.)")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('rabi.png', dpi=500)

    def load_file(self, filename: str):
        full_path = os.path.join(self.path, filename)
        data = sio.loadmat(full_path)
        return RabiData(data['power'][0][0], data['taus'][0], data['zs'][0])


class RabiData:
    def __init__(self, power, x_data, y_data):
        self.power = power
        self.x_data = x_data
        self.y_data = y_data

    def __str__(self):
        return 'power: {0}, x_data: {1}, y_data: {2}'.format(self.power, len(self.x_data), len(self.y_data))


if __name__ == '__main__':
    rabi_plotter = RabiPlotter()
    rabi_plotter.plot_rabis()
