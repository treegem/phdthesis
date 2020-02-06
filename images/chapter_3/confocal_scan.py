import os

import matplotlib.pyplot as plt
from scipy.io import loadmat

from util.inches import cm_to_inch
from util.tum_jet import tum_jet


def main():
    file_data = load_file_data()

    plt.close('all')
    plt.figure(figsize=(cm_to_inch(7.5), cm_to_inch(5)))
    plt.imshow(file_data['result'], vmax=8e4, cmap=tum_jet, interpolation='bilinear', origin='lower',
               extent=[0, x_extent(file_data), 0, y_extent(file_data)])
    plt.xlabel(r'$x$ (nm)')
    plt.ylabel(r'$y$ (nm)')
    plt.yticks([0, 10, 20])
    cbar = plt.colorbar(aspect=10)
    cbar.ax.set_yticklabels(['20', '40', '60', '80'])
    cbar.set_label(r'luminescence (kcts/s)')
    plt.tight_layout()
    plt.savefig('confocal_scan.png', dpi=500)


def x_extent(file_data):
    return file_data['x'][0][-1] - file_data['x'][0][0]


def y_extent(file_data):
    return file_data['y'][0][-1] - file_data['y'][0][0]


def load_file_data():
    file_path = "//file/e24/Projects/ReinhardLab/data_setup_nv1/151127_5keV_after12h@450C"
    file_name = "5keV.1.mat"
    file_data = loadmat(os.path.join(file_path, file_name))
    return file_data


if __name__ == '__main__':
    main()
