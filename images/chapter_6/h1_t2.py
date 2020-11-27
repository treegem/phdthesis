import glob
import os

import matplotlib.pyplot as plt
import numpy

from images.chapter_6.box_plot_functions import load_t2s_in_us, remove_average, create_random_x_offsets
from util import tum_jet
from util.inches import cm_to_inch


class H1T2Plotter:

    def __init__(self):
        self.data_folder = "h1_t2"
        self.files = self.__get_file_names()
        self.upper_limit = 60
        self.lower_limit = 1.5
        self.box_color = tum_jet.tum_color(4)
        self.median_color = tum_jet.tum_color(5)
        self.box_width = 0.25

    def plot(self):
        fig = plt.figure(figsize=(cm_to_inch(15), cm_to_inch(7)))
        t2s = self.__load_t2s()
        xs = create_random_x_offsets(t2s, self.box_width)

        self.__plot_boxes_and_datapoints(t2s, xs)

        plt.ylabel('$T_2$ (µs)')
        plt.xlim([-0.6, 1.6])
        plt.xticks(range(0, 2), ['acid \ncleaned', '520°C'])
        plt.tight_layout()
        fig.savefig(os.path.join(self.data_folder, 'h1_t2.jpg'), dpi=600)

    def __get_file_names(self):
        files = []
        for file in glob.glob(os.path.join(self.data_folder, "*.txt")):
            files.append(file.split('.')[0])
        return files

    def __load_t2s(self):
        t2s = []
        for preparation_step in self.files:
            t2s_of_this_step = load_t2s_in_us(preparation_step)
            remove_average(t2s_of_this_step)
            t2s_of_this_step = self.__remove_outliers(t2s_of_this_step)
            t2s.append(t2s_of_this_step)
        return t2s

    def __remove_outliers(self, t2):
        t2 *= (t2 < self.upper_limit)
        t2 *= (t2 > self.lower_limit)
        t2 = numpy.trim_zeros(numpy.sort(t2))
        return t2

    def __plot_boxes_and_datapoints(self, t2s, xs):
        for t2_index, t2s_of_current_step in enumerate(t2s):
            print(numpy.median(t2s_of_current_step))
            plt.plot(xs[t2_index][:] + t2_index, t2s_of_current_step[:], '.k', alpha=0.7)
            plt.boxplot(t2s_of_current_step[:], positions=[t2_index], widths=self.box_width, patch_artist=True,
                        boxprops={'color': self.box_color, 'alpha': 0.2, 'facecolor': self.box_color},
                        whiskerprops={'color': self.box_color},
                        capprops={'color': self.box_color},
                        medianprops={'color': self.median_color, 'linewidth': 2},
                        showfliers=False)


if __name__ == '__main__':
    plotter = H1T2Plotter()
    plotter.plot()
