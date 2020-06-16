import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from util.inches import cm_to_inch
from util.tum_jet import tum_color


class LuminescenceFluctuationPlotter:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def plot(self):
        plt.figure(figsize=(cm_to_inch(15), cm_to_inch(5.5)))
        luminescences = self.__load_luminescences()
        average = np.average(luminescences[:20])
        plt.ylabel('luminescence (normalized)')
        plt.xlabel('time (h)')
        plt.plot(1.5 * np.array(list(range(len(luminescences)))) / 600, luminescences / average, color=tum_color(0))
        plt.tight_layout()
        plt.savefig('luminescence_fluctuation.png', dpi=500)

    def __load_luminescence_from_image_file(self, fname):
        opened_image = Image.open(os.path.join(self.folder_path, fname))
        number_of_channels = 3
        return (np.asarray(opened_image).sum(axis=2) / number_of_channels)[28:-45, 117:-80]

    def __load_luminescences(self):
        file_names = self.__assemble_file_names(sweeps=12, taus=50)
        luminescences = np.zeros(len(file_names))
        for i, file_name in enumerate(file_names):
            self.__print_image_loading_progress(file_names, i)
            luminescences[i] = self.__load_luminescence_from_image_file(file_name).sum()
        return luminescences

    @staticmethod
    def __print_image_loading_progress(file_names, i):
        if (i + 1) % 10 == 0:
            print('\rloading images: {:.02f} %    '.format((i + 1) * 100 / len(file_names)), end='')

    @staticmethod
    def __assemble_file_names(sweeps, taus):
        file_names = []
        for sweep in range(sweeps):
            for tau in range(taus):
                file_names.append('zzz_frame_sweep{:03d}_tau{:03d}_sequence000.jpg'.format(sweep, tau))
        return file_names


if __name__ == '__main__':
    folder = '//file/e24/Projects/ReinhardLab/data_setup_nv1/161122_watershed_proem'
    plotter = LuminescenceFluctuationPlotter(folder)
    plotter.plot()

# sweeps go from 000 to 011
# taus go from 000 to 049
