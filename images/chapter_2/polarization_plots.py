import matplotlib.pyplot as plt
import numpy as np

from calcs.optical_polarization import calc_polarization, percentage_of_polarized_states
from util.inches import cm_to_inch
from util.tum_jet import tum_color

plt.rcParams['text.latex.preamble'] = [
    r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{braket}"]


def main():
    ks = np.arange(0, 11, 1)

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(cm_to_inch(15), cm_to_inch(6)))

    polarizations = calc_polarization(ks)
    ax1.plot(ks, polarizations, '.', color=tum_color(0))
    ax1.set_xlabel(r'$n$')
    ax1.set_xticks(ks[::2])
    ax1.set_ylabel(r'$ P_n $')
    ax1.__set_ticks(np.arange(0, 1.1, 0.25))

    ax2.plot(ks, percentage_of_polarized_states(polarizations), '.', color=tum_color(0))
    ax2.set_xlabel(r'$n$')
    ax2.set_xticks(ks[::2])
    ax2.set_ylabel(r'$ \dfrac{N_0}{N_0 + N_{\pm 1}} $')
    ax2.__set_ticks(np.arange(0.5, 1.01, 0.1))

    plt.tight_layout()
    plt.savefig('polarization.png', dpi=300)


if __name__ == '__main__':
    main()
