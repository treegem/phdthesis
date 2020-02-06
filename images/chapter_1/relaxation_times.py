import matplotlib.pyplot as plt
import numpy as np

import util.inches as inch
import util.tum_jet as tum

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [
    r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{braket}"]


def plot_t1():
    xs = np.linspace(0, 3, 100)
    ys = np.exp(-xs)

    plt.close('all')
    plt.figure(figsize=(inch.cm_to_inch(7.5), inch.cm_to_inch(5)))
    plt.plot(xs, ys, color=tum.tum_color(0))
    plt.ylim((-0.1, 1.1))
    plt.yticks([0, 1], [r'$\braket{\boldsymbol{\mu}_0}_z$', r'$\left| \braket{\boldsymbol{\mu}_S} \right|$'])
    plt.ylabel(r'$z$-component')
    plt.xlabel('time (arb.u.)')
    plt.tight_layout()
    plt.savefig('t1_decay.png', dpi=300)


def plot_t2():
    xs = np.linspace(0, 3, 1000)
    damping = 0.4
    ys = np.exp(-xs * damping) * np.cos(10 * xs)
    upper_envelope = np.exp(-xs * damping)
    lower_envelope = -np.exp(-xs * damping)

    plt.close('all')
    plt.figure(figsize=(inch.cm_to_inch(7.5), inch.cm_to_inch(5)))
    plt.plot(xs, ys, color=tum.tum_color(0))
    alpha = 0.4
    plt.plot(xs, upper_envelope, '--', color=tum.tum_color(0), alpha=alpha)
    plt.plot(xs, lower_envelope, '--', color=tum.tum_color(0), alpha=alpha)
    plt.ylim((-1.1, 1.1))
    plt.yticks([-1, 0, 1],
               [r'$-\left| \braket{\boldsymbol{\mu}_S} \right|$', 0, r'$\left| \braket{\boldsymbol{\mu}_S} \right|$'])
    plt.ylabel(r'$y$-component')
    plt.xlabel('time (arb.u.)')
    plt.tight_layout()
    plt.savefig('t2_decay.png', dpi=300)


def main():
    plot_t1()
    plot_t2()


if __name__ == '__main__':
    main()
