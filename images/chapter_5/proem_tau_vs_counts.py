import matplotlib.pyplot as plt
import numpy as np

from util import tum_jet
from util.inches import cm_to_inch


class ProemTauVsCountsPlotter:
    def __init__(self):
        self.__create_and_scale_figure_and_axes()
        self.taus = np.linspace(0, 1, 100)

    def plot(self):
        self.__plot_counts()
        self.__set_labels()
        plt.tight_layout()
        plt.savefig('proem_tau_vs_counts.png', dpi=500)

    def __set_labels(self):
        self.axes[0].set_xlabel(r'$\tau$ (normalized)')
        self.axes[1].set_xlabel(r'$\tau$ (normalized)')
        self.axes[0].set_ylabel('counts (normalized)')
        self.axes[1].set_ylabel('counts (normalized)')

    def __plot_counts(self):
        self.axes[0].plot(self.taus, self.__decaying_counts_with_tau(), color=tum_jet.tum_color(0))
        self.axes[1].plot(self.taus, self.__constant_counts_with_tau(), color=tum_jet.tum_color(0))
        self.axes[1].set_ylim(self.axes[0].get_ylim())

    def __decaying_counts_with_tau(self):
        time_outside_of_sequence = 1
        # assuming that tau_max = 1
        return 1 / (time_outside_of_sequence + self.taus) * time_outside_of_sequence

    def __constant_counts_with_tau(self):
        return np.ones_like(self.taus)

    def __create_and_scale_figure_and_axes(self):
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2)
        self.fig.set_figheight(cm_to_inch(5))
        self.fig.set_figwidth(cm_to_inch(15))


if __name__ == '__main__':
    plotter = ProemTauVsCountsPlotter()
    plotter.plot()
