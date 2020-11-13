# -*- coding: utf-8 -*-

import glob
import os

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pylab as pylab
import numpy
from matplotlib_scalebar.scalebar import ScaleBar

import util.tum_jet as tum_jet
from util.inches import cm_to_inch


class AfmPlotter:

    def __init__(self):
        self.base_path = 'afm_plots_data'
        self.profile_folder = 'profiles'
        self.afm_folder = 'data'
        self.output_folder = 'output'
        self.vmin = 0
        self.vmax = 21.471
        self.colormap = 'gist_gray'
        self.norm = colors.PowerNorm(gamma=1. / 2., vmin=self.vmin, vmax=self.vmax)
        self.color = (tum_jet.tum_raw[0][0] / 255., tum_jet.tum_raw[0][1] / 255., tum_jet.tum_raw[0][2] / 255.)
        self.Ras = {'Std1': '1.29 nm', 'Std3': '2.07 nm', 'Cmp1': '2.68 nm', 'Cmp2': '0.51 nm', 'Uv2': '0.56 nm',
                    'Scf2': '1.07 nm'}
        self.profile_file_stumps = self.get_profile_txt_file_stumps()
        self.afm_file_stumps = self.get_afm_txt_file_stumps()
        self.n_ticks = 6

    def plot(self):
        self.plot_three_columns('Std1', 'Std3', 'Uv2')
        self.plot_three_columns('Cmp1', 'Cmp2', 'Scf2')
        self.plot_colorbar()

    def plot_colorbar(self):
        pylab.close('all')
        fig = pylab.figure(figsize=(cm_to_inch(15), cm_to_inch(1.3)))
        ax = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        # noinspection PyUnresolvedReferences
        cb1 = mpl.colorbar.ColorbarBase(ax, cmap=mpl.cm.gist_gray, norm=self.norm, orientation='horizontal',
                                        ticks=numpy.arange(self.vmin, self.vmax, 3))
        cb1.set_label('h (nm)')
        cb1.ax.tick_params(labelsize=9)
        pylab.savefig(os.path.join(self.base_path, self.output_folder, 'colorbar.jpg'), dpi=1000)
        pylab.savefig(os.path.join(self.base_path, self.output_folder, 'colorbar.pdf'), dpi=1000)

    def plot_three_columns(self, left_sample, center_sample, right_sample):
        pylab.close('all')
        pylab.figure(figsize=(cm_to_inch(15), cm_to_inch(7)))
        self.plot_column(1, left_sample)
        self.plot_column(2, center_sample)
        self.plot_column(3, right_sample)
        pylab.subplots_adjust(hspace=-0.65)
        pylab.tight_layout()
        pylab.savefig(os.path.join(self.base_path, self.output_folder,
                                   '{}_{}_{}.pdf'.format(left_sample, center_sample, right_sample)), dpi=1000)
        pylab.savefig(os.path.join(self.base_path, self.output_folder,
                                   '{}_{}_{}.jpg'.format(left_sample, center_sample, right_sample)), dpi=1000)

    def plot_column(self, column, sample):
        self.plot_profile(column, sample)
        self.plot_afm_scan(column, sample)

    def plot_afm_scan(self, column, sample):
        pylab.subplot(233 + column)
        data = numpy.loadtxt(self.afm_file_stumps[sample] + ".txt") * 1e9
        pylab.imshow(numpy.flipud(data), cmap=self.colormap, origin='lower', vmin=self.vmin, vmax=self.vmax,
                     norm=self.norm)
        pylab.text(40, 50, r'$rms$' + ' = ' + self.Ras['Std1'],
                   bbox={'boxstyle': 'square', 'facecolor': 'white', 'alpha': 0.9})
        pylab.text(18, 452, sample, bbox={'boxstyle': 'square', 'facecolor': 'white', 'alpha': 0.9})
        scalebar = ScaleBar(2500. / 512. * 1e-9)
        pylab.gca().add_artist(scalebar)
        pylab.xticks(numpy.linspace(0, 511, self.n_ticks), [''] * self.n_ticks)
        pylab.yticks(numpy.linspace(0, 511, self.n_ticks), [''] * self.n_ticks)

    def plot_profile(self, column, sample):
        pylab.subplot(230 + column, aspect=30)
        data = numpy.loadtxt(self.profile_file_stumps[sample] + ".txt", skiprows=9) * 1e9
        pylab.plot(data[:, 0], data[:, 1] - data[:, 1].min(), color=self.color)
        pylab.ylim((0, 20))
        pylab.xlim((0, 2500))
        pylab.xticks(numpy.linspace(0, 2500, self.n_ticks), [''] * self.n_ticks)
        pylab.yticks([0, 10, 20])
        pylab.ylabel("h (nm)")

    @staticmethod
    def get_txt_file_stumps(path):
        file_stumps = []
        text_files = glob.glob(os.path.join(path, "*.txt"))
        for file in text_files:
            file_stumps.append(file.split('.')[0])
        return file_stumps

    def get_afm_txt_file_stumps(self):
        afm_path = os.path.join(self.base_path, self.afm_folder)
        stumps_list = self.get_txt_file_stumps(afm_path)
        stumps_dict = {
            'Std1': stumps_list[1],
            'Std3': stumps_list[2],
            'Cmp1': stumps_list[0],
            'Cmp2': stumps_list[-1],
            'Uv2': stumps_list[3],
            'Scf2': stumps_list[-2]
        }
        return stumps_dict

    def get_profile_txt_file_stumps(self):
        profile_path = os.path.join(self.base_path, self.profile_folder)
        stumps_list = self.get_txt_file_stumps(profile_path)
        stumps_dict = {
            'Std1': stumps_list[2],
            'Std3': stumps_list[-3],
            'Cmp1': stumps_list[-4],
            'Cmp2': stumps_list[4],
            'Uv2': stumps_list[-1],
            'Scf2': stumps_list[3]
        }
        return stumps_dict


if __name__ == '__main__':
    plotter = AfmPlotter()
    plotter.plot()
