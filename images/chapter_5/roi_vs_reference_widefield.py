import os

import matplotlib.pyplot as plt
import numpy as np

from util import tum_jet
from util.inches import cm_to_inch


class RoiVsReferencePlotter:
    def __init__(self, folder_corrected, folder_uncorrected):
        self.fig, self.axes = self.__create_and_scale_figure_and_axes()
        self.folder_corrected = folder_corrected
        self.folder_uncorrected = folder_uncorrected
        self.corrected_data = self.__load_corrected_data()
        self.uncorrected_data = self.__load_uncorrected_data()

    def create_and_save_plot(self):
        self.__create_uncorrected_plot()
        self.__create_corrected_plot()
        plt.tight_layout()
        plt.savefig('corrected_vs_uncorrected.jpg', dpi=500)

    def __create_uncorrected_plot(self):
        self.__create_plot(self.__calc_uncorrected_taus(), 0, self.uncorrected_data)

    def __create_corrected_plot(self):
        self.__create_plot(self.__calc_corrected_taus(), 1, self.corrected_data)

    def __create_plot(self, taus, axis_index, data):
        axis = self.axes[axis_index]
        axis.plot(taus, data / data.max(), '.', color=tum_jet.tum_color(0))
        axis.set_xlabel('pulse duration (ns)')
        axis.set_ylabel('luminescence (norm.)')

    def __load_corrected_data(self):
        data = np.loadtxt(os.path.join(self.folder_corrected, 'result.txt'))
        return data - 3.8e10  # to increase contrast to level of NV 577

    def __load_uncorrected_data(self):
        filenames = os.listdir(self.folder_uncorrected)
        luminescences = np.zeros_like(filenames, dtype=np.float)
        for i, filename in enumerate(filenames):
            frame_data = np.loadtxt(os.path.join(self.folder_uncorrected, filename))[51:63, 91:103]
            luminescences[i] = np.average(frame_data)
        return luminescences

    def __create_and_scale_figure_and_axes(self):
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2)
        self.fig.set_figheight(cm_to_inch(6))
        self.fig.set_figwidth(cm_to_inch(15))
        return self.fig, self.axes

    def __calc_uncorrected_taus(self):
        n_taus = len(os.listdir(self.folder_uncorrected))
        return np.linspace(20, 350, n_taus)

    def __calc_corrected_taus(self):
        n_taus = len(list(filter(lambda x: x.startswith('frame'), os.listdir(self.folder_corrected))))
        return np.linspace(2, 350, n_taus)


if __name__ == '__main__':
    folder_with_corrected_data = '//file/e24/Projects/ReinhardLab/data_setup_nv1/170502_01B_echo_after_annel/rabi_002'
    folder_with_uncorrected_data = '//file/e24/Projects/ReinhardLab/data_setup_nv1/161109_background_rabis/rabi_001'
    plotter = RoiVsReferencePlotter(folder_with_corrected_data, folder_with_uncorrected_data)
    plotter.create_and_save_plot()
