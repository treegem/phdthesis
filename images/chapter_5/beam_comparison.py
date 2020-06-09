import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util.camera_plotting import cam_imshow
from util.inches import cm_to_inch

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}']


class BeamComparison:
    def __init__(self):
        self.gaussian_data = self.__load_data('proem_009.txt')
        self.square_data = self.__load_data('proem_010.txt')
        self.fig = None
        self.axes = (None, None)

    @staticmethod
    def __load_data(filename):
        return np.loadtxt(
            os.path.join('//file/e24/Projects/ReinhardLab/data_setup_nv1/170306_phase_plates_rabi_echo_pro', filename))

    def plot(self):
        self.__create_and_scale_figure_and_axes()
        self.__imshow_both_axes()
        self.__add_colorbar()
        plt.savefig('beam_comparison.png', dpi=500)

    def __create_and_scale_figure_and_axes(self):
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2)
        self.fig.set_figheight(cm_to_inch(8))
        self.fig.set_figwidth(cm_to_inch(15))

    def __imshow_both_axes(self):
        self.__imshow_single_axis(0, self.gaussian_data)
        self.__imshow_single_axis(1, self.square_data)
        self.__set_labels()
        self.__set_y_ticks_and_labels()

    def __set_y_ticks_and_labels(self):
        self.axes[0].set_yticks([0, 5, 10, 15])
        self.axes[1].set_yticks([0, 5, 10, 15])
        self.axes[1].set_yticklabels(['', '', '', ''])

    def __imshow_single_axis(self, axis_index, data):
        vmin, vmax = self.__define_min_max_for_plot()
        cam_imshow(data, self.axes[axis_index], vmin=vmin, vmax=vmax)

    def __define_min_max_for_plot(self):
        gaussian_min = self.gaussian_data.min()
        gaussian_max = self.gaussian_data.max()
        square_min = self.gaussian_data.min()
        square_max = self.gaussian_data.max()
        return min(gaussian_min, square_min), max(gaussian_max, square_max)

    def __set_labels(self):
        self.axes[0].set_xlabel(r'$x$ ($\si{\micro \meter}$)')
        self.axes[0].set_ylabel(r'$y$ ($\si{\micro \meter}$)')
        self.axes[1].set_xlabel(r'$x$ ($\si{\micro \meter}$)')

    def __add_colorbar(self):
        bottom, top, right = 0.15, 0.9, 0.83
        self.fig.subplots_adjust(bottom=bottom, top=0.9, left=0.05, right=right, wspace=-0.1)
        cbar_ax = self.fig.add_axes([right + 0.01, bottom, 0.05, top - bottom])
        self.fig.colorbar(self.axes[0].images[0], cax=cbar_ax, label=r'luminescence (cts/s)')


if __name__ == '__main__':
    beam_comparison = BeamComparison()
    beam_comparison.plot()
