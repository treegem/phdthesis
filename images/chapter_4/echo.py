import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from util.inches import cm_to_inch
from util.tum_jet import tum_color


class EchoPlotter:
    max_contrast = 0.15
    expected_asymptote = 1 - max_contrast

    def __init__(self):
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{siunitx}"]
        self.data = load_echo_data()
        self.fig = plt.figure(figsize=(cm_to_inch(15), cm_to_inch(6.5)))

    def plot(self):
        self.plot_echo_data()
        self.plot_asymptote()
        self.plot_t2_marker()
        plt.xlabel(r"$\tau_1 + \tau_2$ ($\si{\micro \second}$)")
        plt.ylabel("luminescence (norm.)")
        plt.tight_layout()
        plt.savefig('echo.png', dpi=500)

    def plot_t2_marker(self):
        tum_red = tum_color(5)
        x_t2 = 3.1
        self.fig.axes[0].axvline(x_t2, color=tum_red, linestyle='--')
        self.fig.axes[0].axhline(self.max_contrast * np.exp(-1) + self.expected_asymptote, color=tum_red,
                                 alpha=.3, linestyle='--')
        plt.text(x_t2 + 0.3, 0.96, r'$T_2$', color=tum_red)

    def plot_asymptote(self):
        self.fig.axes[0].axhline(self.expected_asymptote, color='k', alpha=.3, linestyle='--')

    def plot_echo_data(self):
        plt.plot(2 * self.data.taus * 1e-3, self.beautify(self.data.upper_zs - self.data.lower_zs), '.',
                 color=tum_color(0))  # time two, because sum of free evolution times is the relevant factor

    def beautify(self, array):
        asymptote = np.average(array[-5:])
        amplitude = array[0] - asymptote
        return array / amplitude * self.max_contrast + self.expected_asymptote - asymptote


class EchoData:

    def __init__(self, zs, taus):
        self.upper_zs = zs[0]
        self.lower_zs = zs[1]
        self.taus = taus

    def __str__(self):
        return "upper_zs: {}, \nlower_zs: {}, \ntaus: {}".format(self.upper_zs, self.lower_zs, self.taus)


def load_echo_data():
    data = sio.loadmat(
        '//file/e24/Projects/ReinhardLab/data_setup_nv1/190606_D45_T2Measurement/nv36/long_echo_nv36.mat')
    return EchoData(data['zs'], data['taus'][0])


if __name__ == '__main__':
    echo_plotter = EchoPlotter()
    echo_plotter.plot()
