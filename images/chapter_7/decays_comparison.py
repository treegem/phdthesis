import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from util.inches import cm_to_inch
from util.tum_jet import tum_color


def stretched(t, T2, n, A):
    return A * np.exp(-np.power(t / T2, n))


def main(path):
    decay_comparison(path)

    oscillation(path)


def oscillation(path):
    #  fit parameters taken from pulsed.013.mat in the corresponding folder
    amplitude = 7.67478639e-02
    tau_zs_0, taus_0, zs_0, taus_b_0 = load_txts(name='002_phase_oscillation_5_5000', sequence_length_taus=50,
                                                 sequence_length_bs=40, zs_shift=3, path=path)
    tau_zs_1, taus_1, zs_1, taus_b_1 = load_txts(name='20000_25000', sequence_length_taus=50,
                                                 sequence_length_bs=40, zs_shift=3, path=path)
    tau_zs_2, taus_2, zs_2, taus_b_2 = load_txts(name='25000_30000', sequence_length_taus=50,
                                                 sequence_length_bs=40, zs_shift=3, path=path)
    plt.close('all')
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='all')
    fig.set_figwidth(cm_to_inch(12))
    fig.set_figheight(cm_to_inch(5.5))
    fig.text(0.55, 0.04, 'current duration(µs)', ha='center')
    fig.text(0.03, 0.55, r'$1-\left\langle S_z \right\rangle$', va='center', rotation='vertical')
    plt.subplots_adjust(hspace=0, wspace=0.2, bottom=0.2, top=0.95, right=0.95, left=0.15)
    plot_column(amplitude, axes, 0, tau_zs_0, taus_0, taus_b_0, zs_0)
    plot_column(amplitude, axes, 1, tau_zs_1, taus_1, taus_b_1, zs_1)
    plot_column(amplitude, axes, 2, tau_zs_2, taus_2, taus_b_2, zs_2)
    plt.savefig(os.path.join(path, 'oscillations.jpg'), dpi=500)


def plot_column(amplitude, axes, column, tau_zs, taus, taus_b, zs):
    axes[0, column].plot(taus_b, (zs - np.average(zs)) / amplitude * 0.5 + 0.5, color=tum_color(0), label='test')
    axes[1, column].plot(taus, (tau_zs - np.average(tau_zs[1:])) / amplitude * 0.5 + 0.5, color=tum_color(5))
    if column == 2:
        axes[0, column].text(25.35, 0.85, 'qff')
        axes[1, column].text(25.2, 0.85, 'no qff')


def load_txts(name, sequence_length_taus, sequence_length_bs, zs_shift, path):
    taus = np.loadtxt(os.path.join(path, '{}_taus.txt'.format(name)))[:sequence_length_taus]
    taus_b = np.loadtxt(os.path.join(path, '{}_taus.txt'.format(name)))[:sequence_length_bs]
    taus_b = rescale(taus, taus_b)
    taus_b = taus_b * 1e-3
    taus = taus * 1e-3
    zs = np.loadtxt(os.path.join(path, '{}_zs.txt'.format(name)))[zs_shift:zs_shift + sequence_length_bs]
    tau_zs = np.loadtxt(os.path.join(path, '{}_tau_zs.txt'.format(name)))[:sequence_length_taus]
    return tau_zs, taus, zs, taus_b


def rescale(taus, taus_b):
    start = taus_b[0]
    taus_b = taus_b - start
    taus_b = taus_b * (taus - taus[0]).max() / taus_b.max()
    taus_b = taus_b + start
    return taus_b


def decay_comparison(path):
    qff_average, tau_average = calc_average(path)
    qff = np.loadtxt(os.path.join(path, 'decays_qff.txt'))
    tau = np.loadtxt(os.path.join(path, 'decays_tau.txt'))
    qff = qff / qff_average
    tau = tau / tau_average
    xs = np.linspace(5e-3, 30, len(qff))
    popt_qff, pcov_qff = curve_fit(stretched, xdata=xs, ydata=qff, p0=[20, 1, 1])
    popt_tau, pcov_tau = curve_fit(stretched, xdata=xs, ydata=tau, p0=[20, 1, 1])
    plt.close('all')
    plt.figure(figsize=(cm_to_inch(12), cm_to_inch(7.5)))
    plt.plot(xs, tau / 2 + 0.5, '.',
             label=r'uncorr., $T_\rho=\left( {:.1f} \pm {:.1f} \right) $ µs'
             .format(popt_tau[0], np.sqrt(pcov_tau[0, 0])),
             color=tum_color(5))
    plt.plot(xs[::2], stretched(xs, *popt_tau)[::2] / 2 + 0.5, color=tum_color(5))
    plt.plot(xs, qff / 2 + 0.5, '.',
             label=r'corr., $T_\rho=\left( {:.1f} \pm {:.1f} \right) $ µs'
             .format(popt_qff[0], np.sqrt(pcov_qff[0, 0])),
             color=tum_color(0))
    plt.plot(xs[::2], stretched(xs, *popt_qff)[::2] / 2 + 0.5, color=tum_color(0))
    plt.xlabel(r'$\int I \cdot \mathrm{d}t$' + ' (40 mA' + r'$\cdot $' + 'µs)')
    plt.ylabel(r'$1 - \left\langle S_z \right\rangle$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'decays_comparison.jpg'), dpi=500)


def calc_average(path):
    qff = np.loadtxt(os.path.join(path, 'decays_qff.txt'))
    tau = np.loadtxt(os.path.join(path, 'decays_tau.txt'))
    qff_average = np.average(qff[:5])
    tau_average = np.average(tau[:1])
    return qff_average, tau_average


if __name__ == '__main__':
    main('decays_comparison')
