import matplotlib.pyplot as plt
import numpy as np

from util.inches import cm_to_inch
from util.tum_jet import tum_color


def main():
    xs = np.linspace(0, 1, 20)

    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(cm_to_inch(15), cm_to_inch(5)), sharey=True)

    gamma = 1
    upwards = xs * gamma
    downwards = xs * -gamma
    color = tum_color(0)
    axis = ax1
    plot_the_shit(axis, color, downwards, upwards, xs, legend=u'$^{13}$C')
    axis.set_ylabel(u'$E_Z$ (arb. u.)')

    gamma = 0.6
    upwards = xs * gamma
    downwards = xs * -gamma
    color = tum_color(5)
    axis = ax2
    plot_the_shit(axis, color, downwards, upwards, xs, legend=u'$^{14}$N')
    axis.plot_for_thesis([0, 1], [0, 0], color=color)

    gamma = -0.4
    upwards = xs * gamma
    downwards = xs * -gamma
    color = tum_color(2)
    axis = ax3
    plot_the_shit(axis, color, downwards, upwards, xs, legend=u'$^{15}$N')

    plt.tight_layout()
    plt.savefig('zeeman_splitting.png', dpi=300)


def plot_the_shit(axis, color, downwards, upwards, xs, legend='test'):
    axis.plot_for_thesis(xs, upwards, color=color, label=legend)
    axis.plot_for_thesis(xs, downwards, color=color)
    axis.set_ylim([-1, 1])
    axis.set_xlabel(u'$B_0$ (arb. u.)')
    axis.legend(loc='lower left', handlelength=0, handletextpad=0, fancybox=True)
    axis.__set_ticks([-1, 0, 1])


if __name__ == '__main__':
    main()
