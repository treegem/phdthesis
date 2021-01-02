#  adapted code from Niklas

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns

from util.inches import cm_to_inch


def load_t2_data(file_path: str):
    data_dir = os.path.join('spin_properties', file_path) + 'T2.mat'
    mat = scipy.io.loadmat(data_dir)
    t2_data = mat['csvDat']
    return t2_data


def load_odmr_data(file_path: str):
    data_dir = os.path.join('spin_properties', file_path) + 'ODMR.mat'
    mat = scipy.io.loadmat(data_dir)
    odmrh_data = mat['relODMRH'][0]
    odmrwid_data = mat['ODMRwid'][0] / np.sqrt(0.5 * (1 + 1.26 ** 2))
    return odmrh_data, odmrwid_data


class SpinPropertiesPlotter:

    def __init__(self):
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

    def plot(self):
        DNi = r'D$_\text{Ni}$'
        DPd = r'D$_\text{Pd}$'
        DUV = r'D$_\text{UV}$'
        DAlOx = r'D$_\text{AlOx}$'

        csvDatD45_2 = load_t2_data('2019-06-06D45/')
        csvDatD01B_1 = load_t2_data('2019-06-08/')
        csvDatD44_1 = load_t2_data('2019-06-12/')
        csvDatD02C = load_t2_data('2019-07-9/')

        spin_property_fig, [[ODMRAmpax, ODMRwidax], [T2Ax, boxAx]] = \
            plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2, 'hspace': 0.4},
                         figsize=(cm_to_inch(15), cm_to_inch(10)))

        nicer_ax(boxAx)
        nicer_ax(T2Ax)
        nicer_ax(ODMRAmpax)
        nicer_ax(ODMRwidax)

        T2Ax.set_xlim(0, 101)
        ODMRAmpax.set_xlim(0, 101)
        ODMRwidax.set_xlim(0, 101)

        boxAx.set_ylim(-3, 105)
        boxAx.tick_params(axis=u'x', which=u'both', length=0)

        n = pd.DataFrame({'Probe': np.repeat(DNi, csvDatD45_2.shape[0]), 'Fit': np.repeat('Var', csvDatD45_2.shape[0]),
                          'T2': csvDatD45_2[:, 12]})
        o = pd.DataFrame(
            {'Probe': np.repeat(DUV, csvDatD01B_1.shape[0]), 'Fit': np.repeat('Var', csvDatD01B_1.shape[0]),
             'T2': csvDatD01B_1[:, 12]})
        p = pd.DataFrame({'Probe': np.repeat(DPd, csvDatD44_1.shape[0]), 'Fit': np.repeat('Var', csvDatD44_1.shape[0]),
                          'T2': csvDatD44_1[:, 12]})
        q = pd.DataFrame({'Probe': np.repeat(DAlOx, csvDatD02C.shape[0]), 'Fit': np.repeat('Var', csvDatD02C.shape[0]),
                          'T2': csvDatD02C[:, 12]})

        df = pd.concat([n, p, o, q])

        colors = [(0, 101 / 256, 189 / 256),  # TUM Blue
                  (145 / 256, 172 / 256, 107 / 256),  # TUM Green
                  (255 / 256, 128 / 256, 0),  # TUM Orange
                  (229 / 256, 52 / 256, 24 / 256),  # TUM Red
                  "#884EA1",
                  (156 / 255, 157 / 255, 159 / 255),  # TUM Yellow
                  ]

        scatter_point_size = 2.5
        sns.stripplot(x='Probe', y='T2', data=df, palette=colors, jitter=0.2, dodge=False, size=scatter_point_size,
                      alpha=1,
                      linewidth=0,
                      ax=boxAx)

        sns.boxplot(x='Probe', y='T2', data=df, palette=[(0, 0, 0, 0.75)],
                    boxprops={'linewidth': 0.5, 'color': [0.8] * 3, 'edgecolor': 'k', 'alpha': 0.7},
                    medianprops={'linewidth': 1, 'color': 'k'}, meanprops={'linestyle': '-', "color": "xkcd:dark gray"},
                    meanline=False, showmeans=False, showfliers=False, saturation=1, ax=boxAx, linewidth=0.7)

        plt.legend().remove()
        plt.legend().remove()
        boxAx.set_xlabel('')
        boxAx.set_ylabel(r'$T_2$ [\textmu s]')
        boxAx.yaxis.set_label_position("right")
        boxAx.yaxis.tick_right()

        T2SortD44 = csvDatD44_1[csvDatD44_1[:, 12].argsort()]
        T2SortD45 = csvDatD45_2[csvDatD45_2[:, 12].argsort()]
        T2SortD01B = csvDatD01B_1[csvDatD01B_1[:, 12].argsort()]
        T2SortD02C = csvDatD02C[csvDatD02C[:, 12].argsort()]

        T2Ax.scatter(np.linspace(0, 100, T2SortD45.shape[0]), T2SortD45[:, 12], s=scatter_point_size, color=colors[0],
                     label=DNi)
        T2Ax.scatter(np.linspace(0, 100, T2SortD44.shape[0]), T2SortD44[:, 12], s=scatter_point_size, color=colors[1],
                     label=DPd)
        T2Ax.scatter(np.linspace(0, 100, T2SortD01B.shape[0]), T2SortD01B[:, 12], s=scatter_point_size, color=colors[2],
                     label=DUV)
        T2Ax.scatter(np.linspace(0, 100, T2SortD02C.shape[0]), T2SortD02C[:, 12], s=scatter_point_size, color=colors[3],
                     label=DAlOx)

        lw = 0.5
        T2Ax.axvline(50, color='xkcd:gray', ls='--', linewidth=lw)
        T2Ax.axvline(75, color='xkcd:gray', ls=':', linewidth=lw)
        T2Ax.axvline(25, color='xkcd:gray', ls=':', linewidth=lw)
        T2Ax.axvline(100, color='xkcd:gray', ls=(0, (1, 3)), linewidth=lw)
        T2Ax.set_ylabel(r'$T_2$ [\textmu s]')
        T2Ax.xaxis.set_major_locator(ticker.MultipleLocator(25.00))
        T2Ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(r"$%d\%%$"))
        T2Ax.set_ylim(-5, 180)

        plt.sca(ODMRAmpax)
        plt.ylabel(r'$A_\mathrm{rel}$ [\%]')

        ODMRH_45, ODMRWid_45 = load_odmr_data('2019-06-06D45/')
        ODMRH_44, ODMRWid_44 = load_odmr_data('2019-06-12/')
        ODMRH_01b, ODMRWid_01b = load_odmr_data('2019-06-08/')
        ODMRH_02c, ODMRWid_02c = load_odmr_data('2019-07-9/')

        SortODMRH_45 = ODMRH_45[ODMRH_45.argsort()] * 1E2
        SortODMRH_44 = ODMRH_44[ODMRH_44.argsort()] * 1E2
        SortODMRH_01b = ODMRH_01b[ODMRH_01b.argsort()] * 1E2
        SortODMRH_02c = ODMRH_02c[ODMRH_02c.argsort()] * 1E2

        plt.scatter(np.linspace(0, 100, ODMRH_45.shape[0]), SortODMRH_45, s=scatter_point_size, color=colors[0])
        plt.scatter(np.linspace(0, 100, ODMRH_44.shape[0]), SortODMRH_44, s=scatter_point_size, color=colors[1])
        plt.scatter(np.linspace(0, 100, ODMRH_01b.shape[0]), SortODMRH_01b, s=scatter_point_size, color=colors[2])
        plt.scatter(np.linspace(0, 100, ODMRH_02c.shape[0]), SortODMRH_02c, s=scatter_point_size, color=colors[3])

        plt.axvline(50, color='xkcd:gray', ls='--', linewidth=lw)
        plt.axvline(75, color='xkcd:gray', ls=':', linewidth=lw)
        plt.axvline(25, color='xkcd:gray', ls=':', linewidth=lw)
        plt.axvline(100, color='xkcd:gray', ls=(0, (1, 3)), linewidth=lw)

        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(25.00))
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter(r"$%d\%%$"))

        plt.sca(ODMRwidax)

        plt.ylabel(r'$\Gamma$ [MHz]')
        SortODMRWid_45 = ODMRWid_45[ODMRWid_45.argsort()] * 1E3
        SortODMRWid_44 = ODMRWid_44[ODMRWid_44.argsort()] * 1E3
        SortODMRWid_01b = ODMRWid_01b[ODMRWid_01b.argsort()] * 1E3
        SortODMRWid_02c = ODMRWid_02c[ODMRWid_02c.argsort()] * 1E3

        plt.scatter(np.linspace(0, 100, ODMRWid_45.shape[0]), SortODMRWid_45, s=scatter_point_size, color=colors[0],
                    label=DNi)
        plt.scatter(np.linspace(0, 100, ODMRWid_44.shape[0]), SortODMRWid_44, s=scatter_point_size, color=colors[1],
                    label=DPd)
        plt.scatter(np.linspace(0, 100, ODMRWid_01b.shape[0]), SortODMRWid_01b, s=scatter_point_size, color=colors[2],
                    label=DUV)
        plt.scatter(np.linspace(0, 100, ODMRWid_02c.shape[0]), SortODMRWid_02c, s=scatter_point_size, color=colors[3],
                    label=DAlOx)

        plt.legend(markerscale=2)

        plt.axvline(50, color='xkcd:gray', ls='--', linewidth=lw)
        plt.axvline(75, color='xkcd:gray', ls=':', linewidth=lw)
        plt.axvline(25, color='xkcd:gray', ls=':', linewidth=lw)
        plt.axvline(100, color='xkcd:gray', ls=(0, (1, 3)), linewidth=lw)

        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(25.00))
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter(r"$%d\%%$"))
        plt.gca().yaxis.set_label_position("right")
        plt.gca().yaxis.tick_right()

        plt.figtext(0.02, 0.9, 'a)', fontsize=8)
        plt.figtext(0.55, 0.9, 'b)', fontsize=8)
        plt.figtext(0.02, 0.455, 'c)', fontsize=8)
        plt.figtext(0.55, 0.455, 'd)', fontsize=8)

        # plt.tight_layout(pad=0.1, w_pad=2, h_pad=2)
        spin_property_fig.savefig('spin_properties/spin_properties.png', dpi=600, pad_inches=0.1, bbox_inches='tight')


def nicer_ax(ax):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.yaxis.tick_left()


if __name__ == '__main__':
    plotter = SpinPropertiesPlotter()
    plotter.plot()
