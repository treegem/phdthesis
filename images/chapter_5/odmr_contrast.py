import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


class ContrastPlotter:

    def __init__(self, file_name, create_overview_plots, nv_with_contrast, nv_without_contrast):
        self.raw_data = sio.loadmat(file_name)
        self.ws = self.raw_data['ws']
        self.frames = self.raw_data['frames']
        self.results = self.raw_data['result']
        self.test_results = self.raw_data['test_result']
        self.sums = self.raw_data['sums'][0]
        self.create_overview_plots = create_overview_plots
        self.overview_plots_folder = 'odmr_contrast__single_nv_odmrs'
        self.nv_with_contrast = nv_with_contrast
        self.nv_without_contrast = nv_without_contrast

    def plot_for_thesis(self):
        self.__print_keys_with_shapes()
        # print(self.raw_data['freqs'][0])
        nvs = self.__get_automatically_determined_nvs()
        contrasts = np.zeros(len(nvs))
        for i, nv in enumerate(nvs):
            luminescences = self.__calculate_luminescences(nv)
            averaged_max_luminescence = np.average(np.hstack((luminescences[1:4], luminescences[-3:])))
            contrasts[i] = max(0, 1 - luminescences[14] / averaged_max_luminescence)
        plt.plot(sorted(contrasts), '.')
        plt.show()

    def __calculate_luminescences(self, nv):
        luminescences = np.zeros(len(self.frames))
        for i, frame in enumerate(self.frames):
            nv_frame = np.where(self.ws == nv, frame, np.zeros_like(frame))
            luminescences[i] = nv_frame.sum()
        return luminescences

    def __get_automatically_determined_nvs(self):
        return sorted(set(self.ws.flatten()))[1:]

    def __print_keys_with_shapes(self):
        for key in self.raw_data.keys():
            try:
                print(key, self.raw_data[key].shape)
            except AttributeError:
                pass

    def plot_for_overview(self):
        if self.create_overview_plots:
            self.__plot_ws()
            self.__plot_all_single_nv_odmrs()
            self.__plot_array_list_for_overview(self.frames, 'frame')
            self.__plot_array_list_for_overview(self.results, 'result')
            self.__plot_array_list_for_overview(self.test_results, 'test_result')

    def __plot_all_single_nv_odmrs(self):
        for nv in self.__get_automatically_determined_nvs():
            luminescences = self.__calculate_luminescences(nv)
            plt.close('all')
            plt.plot(luminescences[1:])
            plt.savefig(os.path.join(self.overview_plots_folder, 'odmr_nv_{:03d}.png'.format(nv)), dpi=500)

    def __plot_ws(self):
        plt.imshow(self.ws)
        plt.savefig(os.path.join(self.overview_plots_folder, 'ws.png'), dpi=500)

    def __plot_array_list_for_overview(self, array_list, name):
        for i, frame in enumerate(array_list):
            print('\rplotting {}_{:03d}'.format(name, i), end='')
            plt.imshow(frame)
            plt.colorbar()
            plt.savefig(os.path.join(self.overview_plots_folder, '{}_{:03d}.png'.format(name, i)), dpi=500)
            plt.clf()
        print('\nall {}s plotted'.format(name))


if __name__ == '__main__':
    fname = '//file/e24/Projects/ReinhardLab/data_setup_nv1/170306_phase_plates_rabi_echo_pro/odmr_measurement_002.mat'
    contrast_plotter = ContrastPlotter(
        file_name=fname,
        create_overview_plots=False,
        nv_with_contrast=279,
        nv_without_contrast=307
    )
    contrast_plotter.plot_for_overview()
    contrast_plotter.plot_for_thesis()
