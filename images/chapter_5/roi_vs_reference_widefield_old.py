import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.patches import Circle
from scipy.ndimage import center_of_mass

from util import tum_jet
from util.camera_plotting import cam_imshow
from util.inches import cm_to_inch


class RoiVsReferencePlotter:
    # noinspection PyShadowingNames
    def __init__(self, folder):
        self.fig, self.axes = self.__create_and_scale_figure_and_axes()
        self.folder = folder
        self.ws = self.__load_data()['ws']
        self.origin = self.__load_data()['origin']
        self.nvlist = self.__load_data()['nvlist']

    def create_plot(self):
        cam_imshow(data=self.origin, axis=self.axes[0])
        self.__plot_target_circles()
        self.__plot_reference_circles()

        plt.show()

    def __plot_reference_circles(self):
        target_nv_ids = np.unique(self.ws)[1:]
        for nv_id in np.setdiff1d(self.nvlist, target_nv_ids):
            self.__draw_circle(nv_id, tum_jet.tum_color(5))

    def __plot_target_circles(self):
        for nv_id in np.unique(self.ws)[1:]:
            self.__draw_circle(nv_id, tum_jet.tum_color(2))

    def __load_data(self):
        return sio.loadmat(join(self.folder, 'mes_pulsed.mat'))

    def __draw_circle(self, nv_index, color):
        zipped_coordinates = zip(*np.where(self.ws == nv_index))
        relevant_roi = np.zeros_like(self.origin)
        for coordinate in zipped_coordinates:
            relevant_roi[coordinate] = 1
        center_coordinates = self.__center_of_mass_scaled(relevant_roi)
        self.axes[0].add_patch(Circle(center_coordinates, color=color, fc='none', radius=.25))

    @staticmethod
    def __center_of_mass_scaled(relevant_roi):
        center_of_mass_coords = center_of_mass(relevant_roi)
        return center_of_mass_coords[1] * 10 / 150, center_of_mass_coords[0] * 10 / 150

    def __calc_center_of_mass_and_plot_circle(self, origin, ws):
        zipped_coordinates = zip(*np.where(ws == 173))
        relevant_roi = np.zeros_like(origin)
        for coordinate in zipped_coordinates:
            relevant_roi[coordinate] = 1
        self.axes[0].imshow(relevant_roi)
        self.axes[0].add_patch(Circle((50, 50), color='b', fc='none', radius=.25))

    def __create_and_scale_figure_and_axes(self):
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2)
        self.fig.set_figheight(cm_to_inch(6))
        self.fig.set_figwidth(cm_to_inch(15))
        return self.fig, self.axes


def join(*paths):
    return os.path.join(*paths)


if __name__ == '__main__':
    folder = '//file/e24/Projects/ReinhardLab/data_setup_nv1/170315_echoes_proem_03C_3/rabi_006'
    plotter = RoiVsReferencePlotter(folder)
    plotter.create_plot()
