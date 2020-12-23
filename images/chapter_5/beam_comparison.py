import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from util.camera_plotting import cam_imshow, convert_pixels_to_um
from util.inches import cm_to_inch
from util.tum_jet import tum_color

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}']


class BeamComparison:
    def __init__(self):
        self.gaussian_data = self.__load_data('proem_009.txt')
        self.square_data = self.__load_data('proem_010.txt')
        self.projected_data_scaling_factor = 1e3
        self.fig = None
        self.axes = (None, None)

    def plot(self):
        self.__create_and_scale_figure_and_axes()
        self.__imshow_both_confocal_scans()
        self.__add_colorbar()
        self.__plot_both_projections()
        plt.savefig('beam_comparison.png', dpi=500)

    @staticmethod
    def __load_data(filename):
        return np.loadtxt(
            os.path.join('//nas.ads.mwn.de/TUZE/wsi/e24/ReinhardLab/data_setup_nv1/170306_phase_plates_rabi_echo_pro',
                         filename))

    def __create_and_scale_figure_and_axes(self):
        self.fig, self.axes = plt.subplots(nrows=2, ncols=2)
        self.fig.set_figheight(cm_to_inch(15))
        self.fig.set_figwidth(cm_to_inch(15))

    def __imshow_both_confocal_scans(self):
        self.__imshow_confocal_scan_single_axis(0, self.gaussian_data)
        self.__imshow_confocal_scan_single_axis(1, self.square_data)
        self.__set_labels_for_confocal_scans()
        self.__set_y_ticks_and_tick_labels_for_confocal_scans()

    def __set_y_ticks_and_tick_labels_for_confocal_scans(self):
        self.axes[0][0].set_yticks([0, 5, 10, 15])
        self.axes[0][1].set_yticks([0, 5, 10, 15])
        self.axes[0][1].set_yticklabels(['', '', '', ''])

    def __imshow_confocal_scan_single_axis(self, axis_index, data):
        vmin, vmax = self.__define_min_max_of_raw_data()
        cam_imshow(data, self.axes[0][axis_index], vmin=vmin, vmax=vmax)

    def __define_min_max_of_raw_data(self):
        gaussian_min = self.gaussian_data.min()
        gaussian_max = self.gaussian_data.max()
        square_min = self.gaussian_data.min()
        square_max = self.gaussian_data.max()
        return min(gaussian_min, square_min), max(gaussian_max, square_max)

    def __define_min_max_of_projected_data(self):
        gaussian_data_projected = self.gaussian_data.sum(axis=0)
        gaussian_min = gaussian_data_projected.min()
        gaussian_max = gaussian_data_projected.max()
        square_data_projected = self.square_data.sum(axis=0)
        square_min = square_data_projected.min()
        square_max = square_data_projected.max()
        return min(gaussian_min, square_min) / self.projected_data_scaling_factor, \
               max(gaussian_max, square_max) / self.projected_data_scaling_factor

    def __set_labels_for_confocal_scans(self):
        self.axes[0][0].set_xlabel(r'$x$ ($\si{\micro \meter}$)')
        self.axes[0][0].set_ylabel(r'$y$ ($\si{\micro \meter}$)')
        self.axes[0][1].set_xlabel(r'$x$ ($\si{\micro \meter}$)')

    def __plot_both_projections(self):
        self.__plot_projection_single_axis(0, self.gaussian_data / self.projected_data_scaling_factor)
        self.__plot_projection_single_axis(1, self.square_data / self.projected_data_scaling_factor)
        self.__set_labels_and_tick_labels_for_projections()

    def __plot_projection_single_axis(self, axis_index, data):
        projected_data = data.sum(axis=0)
        xs = np.linspace(0, convert_pixels_to_um(200), len(projected_data))
        axis = self.axes[1][axis_index]
        axis.plot(xs, projected_data, color=tum_color(0))
        self.__plot_projection_fit(xs, projected_data, axis)
        axis.set_aspect(1 / 7)
        vmin, vmax = self.__define_min_max_of_projected_data()
        axis.set_ylim(0.9 * vmin, vmax * 1.05)

    def __set_labels_and_tick_labels_for_projections(self):
        self.axes[1][0].set_xlabel(r'$x$ ($\si{\micro \meter}$)')
        self.axes[1][0].set_ylabel(r'luminescence (kcts)')
        self.axes[1][1].set_xlabel(r'$x$ ($\si{\micro \meter}$)')
        self.axes[1][1].set_yticklabels(['', '', '', ''])

    def __add_colorbar(self):
        bottom, top, right = 0.1, 0.95, 0.9
        self.fig.subplots_adjust(bottom=bottom, top=0.9, left=0., right=right, hspace=0.25, wspace=-0.2)
        colorbar_bottom = bottom + 0.446
        cbar_ax = self.fig.add_axes([right - 0.05, colorbar_bottom, 0.05, top - colorbar_bottom - 0.05])
        self.fig.colorbar(self.axes[0][0].images[0], cax=cbar_ax, label=r'luminescence (cts)')

    @staticmethod
    def __gaussian(x, mu, sig, c, a):
        return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + c

    def __plot_projection_fit(self, xs, projected_data, axis):
        fit_params, err = curve_fit(self.__gaussian, xs, projected_data, p0=[7., 1., 120., 80.])
        err_sigma = np.sqrt(err[1, 1])
        print('sigma: ', fit_params[1])
        print('d_sigma: ', err_sigma)
        axis.plot(xs, self.__gaussian(xs, *fit_params), '--', color=tum_color(5))


if __name__ == '__main__':
    beam_comparison = BeamComparison()
    beam_comparison.plot()
