import matplotlib.colors as clrs
import matplotlib.pyplot as plt
import numpy as np

import util.tum_jet as tum_jet


def calc_dipole_field(x, y):
    r = np.array([x, y])
    mu = np.array([0, 1])
    abs_r = np.linalg.norm(r)
    unity = r / abs_r
    dipole_field = (3 * np.dot(unity, np.dot(mu, unity)) - mu) / np.power(abs_r, 3)
    return dipole_field


def main():
    mesh_resolution = 100
    xs = np.linspace(-1, 1, mesh_resolution)
    ys = np.linspace(-1, 1, mesh_resolution)
    dipole_field = np.zeros((mesh_resolution, mesh_resolution, 2))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            dipole_field[i, j] = calc_dipole_field(x, y)
    dipole_field = np.clip(dipole_field, -1000, 1000)
    starting_points = []
    for i in np.linspace(-0.5, 0.5, 10):
        starting_points.append([i, 0])

    x_field = dipole_field[:, ::1, 0]
    y_field = dipole_field[::, ::, 1]
    plt.streamplot(xs, ys, x_field, -y_field, density=[10, 10],
                   start_points=starting_points, norm=clrs.Normalize(vmin=-200, vmax=200, clip=True),
                   color=-y_field, cmap=tum_jet.tum_jet, linewidth=2)
    plt.savefig('dipole_field.svg')


if __name__ == '__main__':
    main()
