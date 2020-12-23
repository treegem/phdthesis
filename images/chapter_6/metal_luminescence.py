#  adapted code from Niklas

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.morphology import flood
from skimage.segmentation import watershed
from skimage.transform import resize

from util.inches import cm_to_inch
from util.tum_jet import tum_jet


class MetalLuminescencePlotter:

    def __init__(self):
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
        self.min_v = 0.00e2
        self.max_v = 1.4e2
        self.map_c = tum_jet

    def plot(self):
        title_pad = 3
        interpolation = 'gaussian'
        fig, axs = plt.subplots(3, 3, gridspec_kw={'width_ratios': [1, 1, 0.2], 'wspace': 0.10, 'hspace': 0.3},
                                figsize=(cm_to_inch(15) * 0.8, cm_to_inch(14)))
        plt.sca(axs[0, 0])
        nicer_ax2(plt.gca())
        d_ni = r'D$_\text{Ni}$'
        d_pd = r'D$_\text{Pd}$'

        d_uv = r'D$_\text{UV}$'
        d_al_ox = r'D$_\text{AlOx}$'
        plt.title(d_ni, fontsize=8, pad=title_pad)
        plt.tick_params(direction='inout', left=False, bottom=False)
        plt.xlim(0.5, 20)
        plt.ylim(0, 19.5)
        plt.gca().yaxis.set_ticks([])
        plt.gca().xaxis.set_ticks([])
        filepath = 'metal_luminescence/2019-06-06D45/scan.005.mat'
        mat = scipy.io.loadmat(filepath)
        x45 = mat['x'][0]
        y45 = mat['y'][0]
        z45 = mat['result'] / 1e3
        filepath = 'metal_luminescence/2019-06-08/scan.003.mat'
        mat = scipy.io.loadmat(filepath)
        x01b = mat['x'][0]
        y01b = mat['y'][0]
        z01b = mat['result'] / 1e3
        filepath = 'metal_luminescence/2019-06-12/scan.020.mat'
        mat = scipy.io.loadmat(filepath)
        x44 = mat['x'][0]
        y44 = mat['y'][0]
        z44 = mat['result'] / 1e3
        filepath = 'metal_luminescence/2019-07-9/scan.010.mat'
        mat = scipy.io.loadmat(filepath)
        x02c = mat['x'][0]
        y02c = mat['y'][0]
        z02c = mat['result'] / 1e3
        plt.imshow(z45.copy(), vmin=self.min_v, vmax=self.max_v, cmap=self.map_c, interpolation=interpolation,
                   extent=[0, x45[-1] - x45[0], 0, y45[-1] - y45[0]])
        scalebar45 = ScaleBar(1E-6)
        plt.gca().add_artist(scalebar45)

        plt.sca(axs[0, 1])
        nicer_ax2(plt.gca())
        plt.title(d_pd, fontsize=8, pad=title_pad)
        plt.tick_params(direction='inout', right=True, top=True)
        plt.xlim(0.5, 20)
        plt.ylim(0, 19.5)
        plt.gca().yaxis.set_ticks([])
        plt.gca().xaxis.set_ticks([])
        plt.imshow(z44, vmin=self.min_v, vmax=self.max_v, cmap=self.map_c, interpolation=interpolation,
                   extent=[0, x44[-1] - x44[0], 0, y44[-1] - y44[0]])
        scalebar44 = ScaleBar(1E-6)
        plt.gca().add_artist(scalebar44)

        plt.sca(axs[1, 0])
        nicer_ax2(plt.gca())
        plt.title(d_uv, fontsize=8, pad=title_pad)
        plt.tick_params(direction='inout', left=True, bottom=True)
        plt.xlim(0, 19.5)
        plt.ylim(0, 19.5)
        plt.gca().yaxis.set_ticks([])
        plt.gca().xaxis.set_ticks([])
        plt.imshow(z01b, vmin=self.min_v, vmax=self.max_v, cmap=self.map_c, interpolation=interpolation,
                   extent=[0, x01b[-1] - x01b[0], 0, y01b[-1] - y01b[0]])
        scalebar01b = ScaleBar(1E-6)
        plt.gca().add_artist(scalebar01b)

        plt.sca(axs[1, 1])
        nicer_ax2(plt.gca())
        plt.title(d_al_ox, fontsize=8, pad=title_pad)
        plt.tick_params(direction='inout', right=True, top=True)
        plt.xlim(0.5, 20)
        plt.ylim(0, 19.5)
        plt.gca().yaxis.set_ticks([])
        plt.gca().xaxis.set_ticks([])
        plt.imshow(z02c, vmin=self.min_v, vmax=self.max_v, cmap=self.map_c, interpolation=interpolation,
                   extent=[0, x02c[-1] - x02c[0], 0, y02c[-1] - y02c[0]])
        scalebar02c = ScaleBar(1E-6)
        plt.gca().add_artist(scalebar02c)

        plt.sca(axs[2, 0])
        imat = 1
        mats = [z45 - 15.29, z44 - 8.74, z01b - 10.16, z02c - 1.93]
        titles = ['Ni', 'Pd', 'UV', 'AlOx']
        thresholds = [50 - 15.29, 50 - 8.74, 40 - 10.16, 30 - 1.93]
        areas = [(4, 4), (4, 4), (4, 4), (4, 4)]
        hist_dat_mat = []
        for mat, title, threshold, area in zip(mats, titles, thresholds, areas):
            hist_dat_mat.append(find_peaks_simple(mat, threshold=threshold, area=area, sigma=1.5))
        wid = np.mean(hist_dat_mat[imat][0][:-1] - hist_dat_mat[imat][0][1:])
        plt.bar(hist_dat_mat[imat][0], hist_dat_mat[imat][1], wid, color='xkcd:gray')
        plt.plot(hist_dat_mat[imat][0], hist_dat_mat[imat][2], lw=0.7, color='xkcd:black')
        plt.plot(hist_dat_mat[imat][0], hist_dat_mat[imat][3], lw=0.7, color='#f5ea6a', ls='--')
        plt.plot(hist_dat_mat[imat][0], hist_dat_mat[imat][4], lw=0.7, color='#f5ea6a', ls='--')

        plt.gca().set_aspect(165 / 45)
        plt.xlim(45, 210)
        plt.ylim(0, 45)
        plt.xlabel(r'$I^{(k)}$ (kcps)')
        plt.ylabel('\#')
        plt.annotate('bunched', xy=(150, 24), fontsize=6, ha='center', )
        plt.annotate("", xy=(138, 15), xytext=(150, 23), arrowprops=dict(arrowstyle="->"), fontsize=6)

        plt.annotate('individual', xy=(74, 33), xytext=(110, 38.5), fontsize=6, ha='left', va='center')
        plt.annotate("", xy=(74, 36), xytext=(110, 39), arrowprops=dict(arrowstyle="->"), fontsize=6)
        plt.title(d_pd, fontsize=8, pad=0)
        plt.sca(axs[2, 1])

        yield_val = [5.612, 5.016, 3.054, 2.888]
        brightness_val = [hist_dat_mat[i][5] for i in range(len(hist_dat_mat))]
        brightness_sigma = [hist_dat_mat[i][6] for i in range(len(hist_dat_mat))]
        yield_error = [0.423, 0.385, 0.248, 0.234]
        plt.errorbar(yield_val, brightness_val, xerr=yield_error, yerr=brightness_sigma, fmt='s', ms=2, lw=1, color='k')
        plt.annotate('Ni', (yield_val[0], brightness_val[0]), xytext=(yield_val[0] - 0.07, brightness_val[0] + 2),
                     ha='right', fontsize=6, va='bottom')
        plt.annotate('Pd', (yield_val[1], brightness_val[1]), xytext=(yield_val[1] + 0.1, brightness_val[1] - 4),
                     ha='left', fontsize=6, va='top')
        plt.annotate('UV', (yield_val[2], brightness_val[2]), xytext=(yield_val[2] + 0.08, brightness_val[2] + 2),
                     ha='left', fontsize=6, va='bottom')
        plt.annotate('AlOx', (yield_val[3], brightness_val[3]), xytext=(yield_val[3] - 0.07, brightness_val[3] - 3),
                     ha='right', fontsize=6, va='top')
        nicer_ax(plt.gca())
        plt.gca().set_aspect(4 / 70)
        axs[2, 1].set_xlim(1.9, 5.9)
        axs[2, 1].set_ylim(30, 100)
        axs[2, 1].set_xlabel(r'yield  $\eta$ (\%)')
        axs[2, 1].set_ylabel(r'$I$ (kcps)')
        axs[2, 1].set_xticks([3, 5])
        axs[2, 1].set_yticks([50, 75])
        axs[2, 1].yaxis.set_label_position("right")
        axs[2, 1].yaxis.tick_right()
        axs[2, 1].spines['left'].set_linewidth(0.5)
        axs[2, 1].spines['bottom'].set_linewidth(0.5)
        axs[2, 1].spines['left'].set_linewidth(0.5)
        axs[2, 1].spines['bottom'].set_linewidth(0.5)
        plt.grid(True, lw=0.5, ls=':')

        axs[1, 2].axis('off')
        axs[2, 2].axis('off')

        gs = axs[0, 2].get_gridspec()
        # remove the underlying axes
        for ax in axs[:2, 2]:
            ax.remove()
        axcolorbar = fig.add_subplot(gs[:2, 2])
        axcolorbar.axis('off')
        plt.sca(axcolorbar)
        divider = make_axes_locatable(axcolorbar)
        cax1 = divider.append_axes("left", size="80%", pad=0.00)
        norm = mpl.colors.Normalize(vmin=self.min_v, vmax=self.max_v)
        mpl.colorbar.ColorbarBase(cax1, cmap=tum_jet,
                                  norm=norm,
                                  orientation='vertical',
                                  ticks=np.arange(0, 150, 40))

        cax1.set_title('kcps', loc='left', fontsize=6.5)

        plt.rc('scalebar', sep=1)
        plt.rc('scalebar', frameon=False)
        plt.rc('scalebar', box_alpha=0.2)
        plt.rc('scalebar', border_pad=0)
        plt.rc('scalebar', length_fraction=0.3)
        plt.rc('scalebar', label_loc='top')
        plt.rc('scalebar', location='lower right')
        plt.rc('scalebar', color='w')

        plt.figtext(0.165, 0.895, 'a)', fontsize=8)
        plt.figtext(0.52, 0.895, 'b)', fontsize=8)
        plt.figtext(0.165, 0.623, 'c)', fontsize=8)
        plt.figtext(0.52, 0.623, 'd)', fontsize=8)
        plt.figtext(0.165, 0.345, 'e)', fontsize=8)
        plt.figtext(0.52, 0.345, 'f)', fontsize=8)

        fig.savefig('metal_luminescence' + '/confocal_pic.jpg', dpi=600, pad_inches=0, bbox_inches='tight')


def nicer_ax2(ax):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.yaxis.tick_left()


def find_peaks_simple(mat_in, threshold=5., area=(4, 4), sigma=1., borders=None):
    if borders is None:
        borders = [0.305, 1.33489786, 1.42829031, 1.80438237, 2.79750851]
    local_max, local_max_val, marks, ws_area, flood_sections, flood_values = get_flood_array(mat_in, threshold, area)

    resize_factor = 3
    upsampled = resize(mat_in.copy(), np.array(mat_in.shape) * resize_factor, order=0)
    upsampled = gaussian(upsampled, sigma=sigma, cval=0)
    local_max_up, local_max_val_up, marks_up, ws_area_up, flood_sections_up, flood_values_up = get_flood_array(
        upsampled, threshold, np.array(area) * resize_factor)

    bins = 50
    values, bin_edges = np.histogram(local_max_val, bins=bins, density=False)
    bin_mid = (bin_edges[1:] + bin_edges[:-1]) / 2
    single_value = bin_mid[np.argmax(values)]

    values_up, binedges_up = np.histogram(local_max_val_up, bins=120, density=False)
    bin_mid_up = (binedges_up[1:] + binedges_up[:-1]) / 2
    single_value_up = bin_mid_up[np.argmax(values_up)]

    values_up, binedges_up = np.histogram(local_max_val_up / single_value_up, bins=bins, density=False, range=(0, 3))
    bin_mid_up = (binedges_up[1:] + binedges_up[:-1]) / 2

    p0 = [0.1, 1, 0.4, 0.21, 1.7, 0.5]
    bounds = ([0, 0, 0, 0, 1.3, .38], [np.inf, 1.3, 60, np.inf, np.inf, 20])
    try:
        coeff, b = curve_fit(double_peak, bin_mid_up, values_up, p0=p0, bounds=bounds)
    except:
        coeff = p0

    coeff_pd = pd.DataFrame()
    count_range = []
    count_range_up = []
    for ni, (i, j) in enumerate(
            [(borders[0], borders[1]), (borders[1], borders[2]), (borders[2], borders[3]), (borders[3], borders[4]),
             (borders[4], 1000)]):
        count_mask = (local_max_val > (i) * single_value) & (local_max_val <= (j) * single_value).astype('int')
        count_mask_up = (local_max_val_up > (i) * single_value_up) & (local_max_val_up <= (j) * single_value_up).astype(
            'int')
        count_range.append(np.count_nonzero(count_mask) * (ni + 1))
        count_range_up.append(np.count_nonzero(count_mask_up) * (ni + 1))
    coeff_pd['CountRange'] = count_range
    coeff_pd['CountRange_up'] = count_range_up

    rng = bin_mid_up
    peak_d = double_peak(rng, *coeff)
    peak1, peak2 = double_peak_ind(rng, *coeff)

    return bin_mid_up * single_value_up, values_up, peak_d, peak1, peak2, coeff[1] * single_value_up, coeff[
        2] * single_value_up / 2, len(local_max_val_up), len(local_max_val)


def nicer_ax(ax):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.yaxis.tick_left()


def get_flood_array(matIn, threshold, area):
    local_maxi = peak_local_max(matIn, indices=False, footprint=np.ones(area), threshold_abs=threshold)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-matIn, markers)

    flood_sections = np.zeros_like(labels)
    flood_sections -= 1
    local_maxi_val = []
    for iii in range(1, np.max(markers) + 1):
        local_maxiS = np.argwhere(markers == iii)
        local_maxi_val.append(matIn[tuple(local_maxiS[0])])
        mask = (labels == iii) & flood(-matIn, tuple(local_maxiS[0]),
                                       tolerance=0.6 * (matIn[local_maxiS[0, 0], local_maxiS[0, 1]]))
        flood_sections[mask] = iii

    flood_seq_values = []
    for i in range(1, np.max(labels) + 1):
        val = np.sum(matIn[(flood_sections == i)])
        flood_seq_values.append(val)

    return local_maxi, local_maxi_val, markers, labels, flood_sections, flood_seq_values


def peak(x, *p):
    a1, b1, c1 = p
    return a1 * 1 / 2 / ((c1 / 2) ** 2 + (x - b1) ** 2)


def double_peak(x, *p):
    return peak(x, *p[:3]) + peak(x, *p[3:])


def double_peak_ind(x, *p):
    return peak(x, *p[:3]), peak(x, *p[3:])


if __name__ == '__main__':
    plotter = MetalLuminescencePlotter()
    plotter.plot()
