import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.morphology import flood
from skimage.segmentation import watershed
from skimage.transform import resize

from util.tum_jet import tum_jet


class MetalLuminescencePlotter:

    def plot(self):
        title_fontsize = 7.5
        minv = 0.00e2
        maxv = 1.4e2
        mapc = tum_jet
        title_pad = 3
        intp = 'gaussian'
        fig, axs = plt.subplots(3, 3, gridspec_kw={'width_ratios': [1, 1, 0.2], 'wspace': 0.10, 'hspace': 0.3},
                                figsize=(3.37, 4))
        fig.tight_layout()
        plt.sca(axs[0, 0])
        nicer_ax2(plt.gca())
        DNi = r'D$_\text{Ni}$'
        DPd = r'D$_\text{Pd}$'

        DUV = r'D$_\text{UV}$'
        DAlOx = r'D$_\text{AlOx}$'
        plt.title(DNi, fontsize=title_fontsize, pad=title_pad)
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
        filepath = 'metal_luminescence/2019-07-10_D01C1200/scan.016.mat'
        mat = scipy.io.loadmat(filepath)
        plt.imshow(z45.copy(), vmin=minv, vmax=maxv, cmap=mapc, interpolation=intp,
                   extent=[0, x45[-1] - x45[0], 0, y45[-1] - y45[0]])
        print(len(x45) / (x45[-1] - x45[0]), x45[-1] - x45[0])
        scalebar45 = ScaleBar(1E-6)
        plt.gca().add_artist(scalebar45)

        plt.sca(axs[0, 1])
        nicer_ax2(plt.gca())
        plt.title(DPd, fontsize=title_fontsize, pad=title_pad)
        plt.tick_params(direction='inout', right=True, top=True)
        plt.xlim(0.5, 20)
        plt.ylim(0, 19.5)
        plt.gca().yaxis.set_ticks([])
        plt.gca().xaxis.set_ticks([])
        plt.imshow(z44, vmin=minv, vmax=maxv, cmap=mapc, interpolation=intp,
                   extent=[0, x44[-1] - x44[0], 0, y44[-1] - y44[0]])
        scalebar44 = ScaleBar(1E-6)
        plt.gca().add_artist(scalebar44)

        plt.sca(axs[1, 0])
        nicer_ax2(plt.gca())
        plt.title(DUV, fontsize=title_fontsize, pad=title_pad)
        plt.tick_params(direction='inout', left=True, bottom=True)
        plt.xlim(0, 19.5)
        plt.ylim(0, 19.5)
        plt.gca().yaxis.set_ticks([])
        plt.gca().xaxis.set_ticks([])
        plt.imshow(z01b, vmin=minv, vmax=maxv, cmap=mapc, interpolation=intp,
                   extent=[0, x01b[-1] - x01b[0], 0, y01b[-1] - y01b[0]])
        scalebar01b = ScaleBar(1E-6)
        plt.gca().add_artist(scalebar01b)

        plt.sca(axs[1, 1])
        nicer_ax2(plt.gca())
        plt.title(DAlOx, fontsize=title_fontsize, pad=title_pad)
        plt.tick_params(direction='inout', right=True, top=True)
        plt.xlim(0.5, 20)
        plt.ylim(0, 19.5)
        plt.gca().yaxis.set_ticks([])
        plt.gca().xaxis.set_ticks([])
        plt.imshow(z02c, vmin=minv, vmax=maxv, cmap=mapc, interpolation=intp,
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
            hist_dat_mat.append(find_peaks_simple(mat, threshold=threshold, area=area, sigma=1.5, title=title))
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
        plt.title(DPd, fontsize=title_fontsize, pad=0)
        plt.sca(axs[2, 1])

        yield_val = [5.612, 5.016, 3.054, 2.888]
        brightness_val = [hist_dat_mat[i][5] for i in range(len(hist_dat_mat))]
        brightness_sigma = [hist_dat_mat[i][6] for i in range(len(hist_dat_mat))]
        yield_error = [0.423, 0.385, 0.248, 0.234]
        plt.errorbar(yield_val, brightness_val, xerr=yield_error, yerr=brightness_sigma, fmt='s', ms=2, lw=1,
                     color='k')  # c = [colors[0], colors[1], colors[2], colors[3], colors[5]])
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
        norm = mpl.colors.Normalize(vmin=minv, vmax=maxv)
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

        plt.figtext(0.165, 0.895, 'a', fontsize=8)
        plt.figtext(0.52, 0.895, 'b', fontsize=8)
        plt.figtext(0.165, 0.623, 'c', fontsize=8)
        plt.figtext(0.52, 0.623, 'd', fontsize=8)
        plt.figtext(0.165, 0.345, 'e', fontsize=8)
        plt.figtext(0.52, 0.345, 'f', fontsize=8)

        fig.savefig('metal_luminescence' + '/confocal_pic.pdf', dpi=600, pad_inches=0, bbox_inches='tight')

        plt.show()


def nicer_ax2(ax):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.yaxis.tick_left()


def find_peaks_simple(matIn, threshold=5, area=(4, 4), sigma=1, title='None',
                      borders=[0.305, 1.33489786, 1.42829031, 1.80438237, 2.79750851]):
    local_max, local_max_val, marks, ws_area, flood_sections, flood_values = get_flood_array(matIn, threshold, area)

    resize_factor = 3
    upsampled = resize(matIn.copy(), np.array(matIn.shape) * resize_factor, order=0)
    upsampled = gaussian(upsampled, sigma=sigma, cval=0)
    local_max_up, local_max_val_up, marks_up, ws_area_up, flood_sections_up, flood_values_up = get_flood_array(
        upsampled, threshold, np.array(area) * resize_factor)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12 * 2, 4.5 * 2), sharex=False, sharey=False, )
    ax = axes.ravel()

    bins = 50
    values, binedges = np.histogram(local_max_val, bins=bins, density=False)
    bin_mid = (binedges[1:] + binedges[:-1]) / 2
    singleValue = bin_mid[np.argmax(values)]

    values, binedges = np.histogram(local_max_val / singleValue, bins=bins, density=False, range=(0, 3))
    bin_mid = (binedges[1:] + binedges[:-1]) / 2

    values_up, binedges_up = np.histogram(local_max_val_up, bins=120, density=False)
    bin_mid_up = (binedges_up[1:] + binedges_up[:-1]) / 2
    singleValue_up = bin_mid_up[np.argmax(values_up)]

    values_up, binedges_up = np.histogram(local_max_val_up / singleValue_up, bins=bins, density=False, range=(0, 3))
    bin_mid_up = (binedges_up[1:] + binedges_up[:-1]) / 2

    p0 = [0.1, 1, 0.4, 0.21, 1.7, 0.5]
    bounds = ([0, 0, 0, 0, 1.3, .38], [np.inf, 1.3, 60, np.inf, np.inf, 20])
    #     bounds = ([0,0,0,0,1.3,.38],[np.inf,1.3,5,np.inf,np.inf,5])
    try:
        coeff, b = curve_fit(double_peak, bin_mid_up, values_up, p0=p0, bounds=bounds)
    except:
        coeff = p0

    coeffPd = pd.DataFrame()
    countRange = []
    countRange_up = []
    #     1,1.43,1.629
    #     print('border', borders)
    for ni, (i, j) in enumerate(
            [(borders[0], borders[1]), (borders[1], borders[2]), (borders[2], borders[3]), (borders[3], borders[4]),
             (borders[4], 1000)]):
        #         print(ni, i,j)
        count_mask = (local_max_val > (i) * singleValue) & (local_max_val <= (j) * singleValue).astype('int')
        count_mask_up = (local_max_val_up > (i) * singleValue_up) & (local_max_val_up <= (j) * singleValue_up).astype(
            'int')
        #         print(count_mask)
        countRange.append(np.count_nonzero(count_mask) * (ni + 1))
        countRange_up.append(np.count_nonzero(count_mask_up) * (ni + 1))
    coeffPd['CountRange'] = countRange
    coeffPd['CountRange_up'] = countRange_up
    # display(coeffPd)

    fig.suptitle(title)
    local_maxi_indices = np.array(np.argwhere(local_max))
    ax[0].imshow(matIn, vmax=4 * threshold)
    ax[0].scatter(local_maxi_indices[:, 1], local_maxi_indices[:, 0], color='r', s=0.3)
    ax[1].set_title('Maxima')
    ax[1].imshow(local_max)
    ax[2].imshow(flood_sections, cmap='jet')
    ax[3].bar(bin_mid, values, width=0.03, align='center')
    #     ax[4].hist(flood_values, bins = bins, range = (0, 200* singleValue))

    local_maxi_Indices_up = np.array(np.argwhere(local_max_up))
    ax[4].imshow(upsampled, vmax=4 * threshold)
    ax[4].scatter(local_maxi_Indices_up[:, 1], local_maxi_Indices_up[:, 0], color='r', s=0.3)
    ax[5].set_title('Maxima')
    ax[5].imshow(local_max_up, )
    ax[6].imshow(flood_sections_up, cmap='jet')
    ax[7].bar(bin_mid_up, values_up, width=0.03, align='center')
    #     ax[7].hist(local_max_val_up / singleValue_up, bins = bins, range = (0,3))
    ax[7].axvline(borders[0], lw=0.3, c='r')
    ax[7].axvline(borders[1], lw=0.3, c='r')
    ax[7].axvline(borders[2], lw=0.3, c='r')
    ax[7].axvline(borders[3], lw=0.3, c='r')
    ax[7].axvline(borders[4], lw=0.3, c='r')

    rng = bin_mid_up
    peakD = double_peak(rng, *coeff)
    ax[7].plot(rng, peakD, 'k')
    peak1, peak2 = double_peak_ind(rng, *coeff)
    ax[7].plot(rng, peak1, ':', color='xkcd:light gray')
    ax[7].plot(rng, peak2, ':', color='xkcd:light gray')
    #     ax[9].hist(flood_values_up, bins = bins, range = (0, 200 * singleValue_up))

    plt.savefig('metal_luminescence' + '/simple_' + title + '.jpg')
    plt.show()

    return bin_mid_up * singleValue_up, values_up, peakD, peak1, peak2, coeff[1] * singleValue_up, coeff[
        2] * singleValue_up / 2, len(local_max_val_up), len(local_max_val)


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
