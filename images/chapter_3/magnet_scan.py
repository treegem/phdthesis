import os

import matplotlib.pyplot as plt
import scipy.io as sio

from util.inches import cm_to_inch
from util.tum_jet import tum_jet


def main():
    plotter = MagnetScanPlotter()
    plotter.plot_magnet_scan()


class MagnetScanPlotter:

    def __init__(self):
        self.coordinates = 'coord_array'
        self.counts = 'count_array'
        self.wide_scan_data, self.zoom_scan_data = self.load_data()
        self.x_offset, self.y_offset = self.calc_offsets()
        self.fig, self.ax1, self.ax2 = self.create_figure_with_two_axes()
        self.vmin, self.vmax = self.calc_min_max()

    def plot_magnet_scan(self):
        image = self.plot_one_axis(axis=self.ax1, data=self.wide_scan_data)
        self.ax1.set_yticks([0, 10, 20])
        self.plot_one_axis(axis=self.ax2, data=self.zoom_scan_data, y_offset=self.y_offset - 0.023)
        self.ax2.set_yticks([12, 12.5, 13])

        self.add_colorbar(image)

        plt.savefig('magnet_scan.png', dpi=300)

    def add_colorbar(self, image):
        self.fig.subplots_adjust(bottom=0.2, top=0.92, left=0.08, right=0.85, wspace=0.2)
        cbar_ax = self.fig.add_axes([0.85, 0.2, 0.04, 0.72])
        self.fig.colorbar(image, cax=cbar_ax, label='luminescence (kcts/s)')

    def calc_offsets(self):
        x_offset = self.wide_scan_data[self.coordinates][0][0][0]
        y_offset = self.wide_scan_data[self.coordinates][0][0][0]
        return x_offset, y_offset

    def plot_one_axis(self, axis, data, x_offset=None, y_offset=None):
        x_offset, y_offset = self.ensure_offset_is_set(x_offset, y_offset)
        norm_factor = 0.5e-4
        image = axis.imshow(data[self.counts] * norm_factor, cmap=tum_jet, interpolation='bilinear', origin='lower',
                            vmin=self.vmin * norm_factor, vmax=self.vmax * norm_factor,
                            extent=[
                                data[self.coordinates][0][0][0] - x_offset,
                                data[self.coordinates][-1][-1][0] - x_offset,
                                data[self.coordinates][0][0][1] - y_offset,
                                data[self.coordinates][-1][-1][1] - y_offset
                            ])
        axis.set_xlabel(r'$x$-position (mm)')
        axis.set_ylabel(r'$y$-position (mm)')
        return image

    def ensure_offset_is_set(self, x_offset, y_offset):
        if x_offset is None:
            x_offset = self.x_offset
        if y_offset is None:
            y_offset = self.y_offset
        return x_offset, y_offset

    @staticmethod
    def create_figure_with_two_axes():
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(cm_to_inch(5.5))
        fig.set_figwidth(cm_to_inch(15))
        return fig, ax1, ax2

    @staticmethod
    def load_data():
        directory_path = '//file/e24/Projects/ReinhardLab/data_setup_nv1/190524_sample_N_40mA_highfield'
        wide_scan_file = 'bscan.000.mat'
        zoom_scan_file = 'bscan.005.mat'
        wide_scan_data = sio.loadmat(os.path.join(directory_path, wide_scan_file))
        zoom_scan_data = sio.loadmat(os.path.join(directory_path, zoom_scan_file))
        return wide_scan_data, zoom_scan_data

    def calc_min_max(self):
        wide_min = self.wide_scan_data[self.counts].min()
        wide_max = self.wide_scan_data[self.counts].max()
        zoom_min = self.zoom_scan_data[self.counts].min()
        zoom_max = self.zoom_scan_data[self.counts].max()
        return min(wide_min, zoom_min) * 1.01, max(wide_max, zoom_max) * 0.93


if __name__ == '__main__':
    main()
