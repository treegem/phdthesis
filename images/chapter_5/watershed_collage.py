import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util.inches import cm_to_inch
from util.tum_jet import tum_jet


class WatershedPlotter:
    def __init__(self, original_file, smoothed_frame_file, all_contours_file, roi_file):
        matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}']
        self.__create_and_scale_figure_and_axes()
        self.original_data = np.loadtxt(original_file)
        self.smoothed_data = np.loadtxt(smoothed_frame_file)
        self.watershed_data = np.loadtxt(all_contours_file)
        self.roi_data = np.loadtxt(roi_file)

    def plot(self):
        self.__plot_original()
        self.__plot_contour()
        self.__plot_roi()

        self.__set_ticks()
        self.__set_labels()

        plt.tight_layout()
        plt.savefig('watershed_collage.png', dpi=500)

    def __create_and_scale_figure_and_axes(self):
        self.fig, self.axes = plt.subplots(nrows=1, ncols=3)
        self.fig.set_figheight(cm_to_inch(6.5))
        self.fig.set_figwidth(cm_to_inch(15))

    def __set_labels(self):
        self.axes[0].set_ylabel(r'$y$ ($\si{\micro \meter}$)')
        for axis in self.axes:
            axis.set_xlabel(r'$x$ ($\si{\micro \meter}$)')

    def __set_ticks(self):
        for axis in self.axes:
            axis.set_yticks([0, 5, 10, 15])
            axis.set_xticks([0, 5, 10])
        for axis in self.axes[1:]:
            axis.set_yticklabels(['', '', '', ''])

    def __plot_roi(self):
        self.__imshow_single_axis(2, self.roi_data)

    def __plot_contour(self):
        self.__imshow_single_axis(1, self.smoothed_data, color_map='gray')
        self.axes[1].contour(np.flipud(self.watershed_data), levels=np.arange(self.watershed_data.max()),
                             linewidths=.4, origin='upper', cmap=tum_jet,
                             extent=self.__convert_extent_from_pixels_to_um())

    def __plot_original(self):
        self.__imshow_single_axis(0, self.original_data)

    def __imshow_single_axis(self, axis, data, color_map=tum_jet):
        self.axes[axis].imshow(data, cmap=color_map, origin='lower', aspect=1,
                               interpolation='bilinear', extent=self.__convert_extent_from_pixels_to_um())

    def __convert_extent_from_pixels_to_um(self):
        return [0, self.__convert_pixels_to_um(200), 0, self.__convert_pixels_to_um(250)]

    @staticmethod
    def __convert_pixels_to_um(pixels):
        um_per_pixels = 10 / 150
        return pixels * um_per_pixels


if __name__ == '__main__':
    watershed_plotter = WatershedPlotter(
        original_file='original.txt',
        smoothed_frame_file='smoothed_frame.txt',
        all_contours_file='all_ws.txt',
        roi_file='roi.txt'
    )
    watershed_plotter.plot()
