import numpy as np


def main():
    n_one_over_e = 130  # number of oscillations to reach 1/e
    phi = np.arccos(1 / np.e)
    phi_per_pi = phi / np.sqrt(2 * n_one_over_e)  # 2 because one oscillation is two pi pulses, sqrt wegen random walk
    phi_per_pi = 2 * np.pi / 2**14 * np.sqrt(25)

    a_0 = np.cos(phi_per_pi)  # factor of |0>, a_1 not needed for calculation
    fidelity = np.power(a_0, 2)

    print('fidelity: ', fidelity)
    print('infidelity: ', 1 - fidelity)


if __name__ == '__main__':
    main()
