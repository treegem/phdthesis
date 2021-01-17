import matplotlib.pyplot as plt
import numpy as np

from util import tum_jet
from util.inches import cm_to_inch


def sensitivity(n):
    return np.sqrt(n / 2) / np.sinc(1 / n)  # the sinc function automatically multiplies pi to the argument


def create_and_save_plot():
    plt.figure(figsize=(cm_to_inch(12), cm_to_inch(7)))
    plt.plot(ns_continuous, sensitivity(ns_continuous), '--',
             color=tum_jet.tum_color(0))
    plt.plot(ns_discrete, sensitivity(ns_discrete), 'o', color=tum_jet.tum_color(0))
    plt.xlabel(r'$N_\phi$')
    plt.ylabel(r'$\sigma_B / \sigma_{B,0}$')
    plt.tight_layout()
    plt.savefig('sensitivity_vs_sectors.png', dpi=500)


def print_index_of_minimum():
    min_index = ns_continuous[np.argmin(sensitivity(ns_continuous))]
    print('minimum at N =', min_index)


def print_time_increase():
    best_sensitivity = sensitivity(2)
    print('best case (N=2,4):', best_sensitivity)
    time_increase = np.power(best_sensitivity, 2)
    print('time increase:', time_increase)


if __name__ == '__main__':
    ns_continuous = np.linspace(1.8, 8.2, 150)
    ns_discrete = np.arange(2, 8.1, 2)

    print_index_of_minimum()
    print_time_increase()

    create_and_save_plot()
