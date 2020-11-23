# -*- coding: utf-8 -*-

import glob
import os
import random

import matplotlib.pylab as pylab
import numpy

from util.inches import cm_to_inch
from util.tum_jet import tum_color


class ComparisonPlotter:

    def __init__(self):
        self.data_folder = 't2_comparison'
        self.upper_limit = 60

    def plot(self):
        random.seed(40)

        files = self.__load_filenames_without_n_in_corrected_order()
        files.pop(-2)

        all_t2s_without_n = self.__load_all_t2s(files)

        xs = self.__create_random_xs(all_t2s_without_n)

        fig = pylab.figure(figsize=(cm_to_inch(15), cm_to_inch(6)))
        pylab.subplot(211)
        for i, t2s in enumerate(all_t2s_without_n):
            if i == 0:
                color = tum_color(0)
            if i == 2:
                color = tum_color(2)
            if i == 4:
                color = tum_color(4)
            if i == 6:
                color = tum_color(1)
            if i == 8:
                color = tum_color(6)
            if i == 9:
                color = 'grey'
            pylab.plot(xs[i][:], t2s[:], '.k', alpha=0.7)
            pylab.boxplot(t2s[:], positions=[i + 1], widths=0.65, patch_artist=True,
                          boxprops={'color': color, 'alpha': 0.2, 'facecolor': color},
                          whiskerprops={'color': color}, capprops={'color': color}, showfliers=False, medianprops={
                    'color': tum_color(5),
                    'linewidth': 2})

        pylab.text(0.4, 45, 'acid \ncleaned',
                   bbox={'boxstyle': 'square', 'facecolor': 'white', 'alpha': 0.0})

        for i, f in enumerate(files):
            if f == '01C':
                files[i] = 'Cmp1'
            if f == '03D':
                files[i] = 'Cmp2'
            if f == '01D':
                files[i] = 'Uv1'
            if f == '02D':
                files[i] = 'Uv2'
            if f == '02B':
                files[i] = 'Scf1'
            if f == '03C':
                files[i] = 'Scf2'
            if f == '020':
                files[i] = 'Std3'
            if f == '01Df':
                files[i] = 'Std1'
            if f == '03Df':
                files[i] = 'Std2'
            if f == '01B':
                files[i] = 'H1'

        pylab.ylabel('$T_2$ (µs)')
        pylab.yticks()

        pylab.xlim([0.1, len(files) + 0.9])
        pylab.yticks([10, 20, 30, 40, 50], [10, 20, 30, 40, 50])

        pylab.subplots_adjust(left=0.08, right=0.95, hspace=.001)

        # get all files ending with txt info 'files' after 520
        nfiles = []
        for file in glob.glob(os.path.join(self.data_folder, "*.txt")):
            if file.startswith(os.path.join(self.data_folder, 'n')):
                self.__strip_prefix_and_affix_from_file(file)
                nfiles.append(file.split('.')[0])

        nnew_order = [1, 8, 2, 6, 5, 7, 3, 9, 0, 4]
        nfiles = [nfiles[i] for i in nnew_order]

        nt2s = []

        nupper_limit = 60

        for sample in nfiles:
            t2s = numpy.loadtxt(sample + ".txt")
            t2s /= 1000
            t2s[-1] = 0
            t2s *= (t2s < nupper_limit)
            t2s *= (t2s > 1.500)

            t2s = numpy.trim_zeros(numpy.sort(t2s))

            nt2s.append(t2s)

        nt2s.pop(-2)

        xs = self.__create_random_xs(nt2s)

        pylab.subplot(212)
        color = 'blue'
        for i, t2s in enumerate(nt2s):
            if i == 0:
                color = tum_color(0)
            if i == 2:
                color = tum_color(2)
            if i == 4:
                color = tum_color(4)
            if i == 6:
                color = tum_color(1)
            if i == 8:
                color = tum_color(6)
            if i == 9:
                color = 'grey'
            pylab.plot(xs[i][:], t2s[:], '.k', alpha=0.7)
            pylab.boxplot(t2s[:], positions=[i + 1], widths=0.65, patch_artist=True,
                          boxprops={'color': color, 'alpha': 0.2, 'facecolor': color},
                          whiskerprops={'color': color}, capprops={'color': color}, showfliers=False, medianprops={
                    'color': tum_color(5),
                    'linewidth': 2})

        pylab.xlim([0.1, len(files) + 0.9])
        pylab.xticks(range(1, len(files) + 1), files)
        pylab.ylim([0, 60])
        pylab.yticks([0, 10, 20, 30, 40, 50, ], [0, 10, 20, 30, 40, 50, ])
        pylab.ylabel('$T_2$ (µs)')
        pylab.text(0.4, 47, '520 °C', bbox={'boxstyle': 'square', 'facecolor': 'white', 'alpha': 0.0})

        fig.savefig(os.path.join(self.data_folder, 't2comparisonquartil.jpg'), dpi=1000)

    @staticmethod
    def __create_random_xs(all_t2s):
        xs = []
        for i, t2s in enumerate(all_t2s):
            x = numpy.zeros_like(t2s) + (i + 1)
            for j, t in enumerate(x):
                x[j] = t + random.uniform(-0.25, 0.25)
            xs.append(x)
        return xs

    def __load_all_t2s(self, files):
        all_t2s = []
        for sample in files:
            t2s = self.__load_t2s_in_us(sample)
            t2s = self.filter_outliers_and_average(t2s)

            all_t2s.append(t2s)
        return all_t2s

    def filter_outliers_and_average(self, t2s):
        t2s[-1] = 0
        t2s *= (t2s < self.upper_limit)
        t2s *= (t2s > 1.500)
        t2s = numpy.trim_zeros(numpy.sort(t2s))
        return t2s

    def __load_t2s_in_us(self, samples):
        t2s = numpy.loadtxt(os.path.join(self.data_folder, samples + ".txt"))
        t2s /= 1000
        return t2s

    def __load_filenames_without_n_in_corrected_order(self):
        files = []
        text_files = glob.glob(os.path.join(self.data_folder, "*.txt"))
        for file in text_files:
            if not file.startswith(os.path.join(self.data_folder, 'n')):
                blank_filename = self.__strip_prefix_and_affix_from_file(file)
                files.append(blank_filename)
        new_order = [1, 8, 2, 6, 5, 7, 3, 9, 0, 4]
        files = [files[i] for i in new_order]
        return files

    def __strip_prefix_and_affix_from_file(self, file):
        file_without_extension = file.split('.')[0]
        blank_filename = file_without_extension.split(os.path.join(self.data_folder, ''))[-1]
        return blank_filename


if __name__ == '__main__':
    plotter = ComparisonPlotter()
    plotter.plot()
