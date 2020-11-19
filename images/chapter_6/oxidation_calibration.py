# -*- coding: utf-8 -*-
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy

from util import tum_jet
from util.inches import cm_to_inch


class OxidationCalibrationPlotter:

    def __init__(self):
        self.data_folder = "oxidation_calibration"
        self.files = self.__get_file_names_ordered_chronologically()
        self.upper_limit = 60
        self.lower_limit = 1.5
        self.box_color = tum_jet.tum_color(4)
        self.median_color = tum_jet.tum_color(5)

    def plot(self):

        fig = plt.figure(figsize=(cm_to_inch(15), cm_to_inch(7)))
        t2s = self.__load_t2s()
        xs = self.__create_random_x_offsets(t2s)

        self.__plot_boxes_and_datapoints(t2s, xs)

        plt.ylabel('$T_2$ (µs)')
        plt.xlim([-0.6, 3.6])
        plt.xticks(range(0, 4), ['acid \ncleaned', '465°C', '520°C', '560°C'])
        plt.tight_layout()
        fig.savefig(os.path.join(self.data_folder, 'oxidation_calibration.jpg'), dpi=600)

    def __plot_boxes_and_datapoints(self, t2s, xs):
        for t2_index, t2s_of_current_step in enumerate(t2s):
            plt.plot(xs[t2_index][:] + t2_index, t2s_of_current_step[:], '.k', alpha=0.7)
            plt.boxplot(t2s_of_current_step[:], positions=[t2_index], widths=0.4, patch_artist=True,
                        boxprops={'color': self.box_color, 'alpha': 0.2, 'facecolor': self.box_color},
                        whiskerprops={'color': self.box_color},
                        capprops={'color': self.box_color},
                        medianprops={'color': self.median_color, 'linewidth': 2},
                        showfliers=False)

    @staticmethod
    def __create_random_x_offsets(t2s):
        xs = []
        for t2 in t2s:
            xs_of_this_step = numpy.zeros_like(t2)
            for x_index in range(len(xs_of_this_step)):
                xs_of_this_step[x_index] = random.uniform(-0.2, 0.2)
            xs.append(xs_of_this_step)
        return xs

    def __load_t2s(self):
        t2s = []
        for preparation_step in self.files:
            t2s_of_this_step = self.__load_t2s_in_us(preparation_step)
            self.__remove_average(t2s_of_this_step)
            t2s_of_this_step = self.__remove_outliers(t2s_of_this_step)
            t2s.append(t2s_of_this_step)
        return t2s

    def __remove_outliers(self, t2):
        t2 *= (t2 < self.upper_limit)
        t2 *= (t2 > self.lower_limit)
        t2 = numpy.trim_zeros(numpy.sort(t2))
        return t2

    @staticmethod
    def __remove_average(t2):
        t2[-1] = 0

    @staticmethod
    def __load_t2s_in_us(preparation_step):
        t2 = numpy.loadtxt(preparation_step + ".txt")
        t2 /= 1000
        return t2

    def __get_file_names_ordered_chronologically(self):
        files = []
        for file in glob.glob(os.path.join(self.data_folder, "*.txt")):
            files.append(file.split('.')[0])
        new_order = [3, 0, 1, 2]
        files = [files[i] for i in new_order]
        return files


if __name__ == '__main__':
    plotter = OxidationCalibrationPlotter()
    plotter.plot()
