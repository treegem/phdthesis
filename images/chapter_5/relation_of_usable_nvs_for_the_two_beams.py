import os

import matplotlib.pyplot as plt
import numpy as np


class RelationDeterminer:
    def __init__(self):
        self.gaussian_data = self.__load_data('proem_009.txt')
        self.square_data = self.__load_data('proem_010.txt')
        self.fig = None
        self.axes = (None, None)

    @staticmethod
    def __load_data(filename):
        return np.loadtxt(
            os.path.join('//file/e24/Projects/ReinhardLab/data_setup_nv1/170306_phase_plates_rabi_echo_pro', filename))

    def calc(self):
        threshold_min = 900
        threshold_max = 2500
        gaussian_data_mask_above = np.where(self.gaussian_data > threshold_min, 1, 0)
        gaussian_data_mask_below = np.where(self.gaussian_data < threshold_max, 1, 0)
        plt.imshow(gaussian_data_mask_above * self.gaussian_data * gaussian_data_mask_below)
        plt.show()
        square_data_mask_above = np.where(self.square_data > threshold_min, 1, 0)
        square_data_mask_below = np.where(self.square_data < threshold_max, 1, 0)
        plt.imshow(square_data_mask_above * self.square_data * square_data_mask_below)
        plt.show()

        pixels_per_nv = 35  # roughly
        nvs_in_gaussian_data = (gaussian_data_mask_above * gaussian_data_mask_below).sum() / pixels_per_nv
        print('nvs_in_gaussian_data', nvs_in_gaussian_data)
        nvs_in_square_data = (square_data_mask_above * square_data_mask_below).sum() / pixels_per_nv
        print('nvs_in_square_data', nvs_in_square_data)


if __name__ == '__main__':
    relation_determiner = RelationDeterminer()
    relation_determiner.calc()
