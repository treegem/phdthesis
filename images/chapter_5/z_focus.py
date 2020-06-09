import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from util.camera_plotting import cam_imshow, convert_pixels_to_um
from util.inches import cm_to_inch
from util.tum_jet import tum_jet, tum_color


class ZFocusPlotter:
    def __init__(self, folder, z_values, index_unsharp, index_sharp):
        matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}']
        self.zs = z_values
        self.proem_images = self.__load_images(folder)
        self.sharp_image = self.proem_images[index_sharp]
        self.unsharp_image = self.proem_images[index_unsharp]
        self.__create_figures_and_axes()

    def plot(self):
        self.__plot_original_images()

        self.__plot_fft_images()

        all_sharpnesses = self.__sharpness_of_all_images()
        sharpness_axis = self.axes[4]
        sharpness_axis.plot(self.zs, all_sharpnesses / all_sharpnesses.max(), '.', color=tum_color(0))
        sharpness_axis.set_xlabel(r'focus depth ($\si{\micro \meter}$)')
        sharpness_axis.set_ylabel('sharpness (normalized)')

        plt.tight_layout()
        plt.savefig('z_focus_collage.png', dpi=500)

    def __plot_fft_images(self):
        fft_extent = self.calculate_fft_extent()

        fft_unsharp = np.abs(np.fft.fft2(self.unsharp_image))
        image2 = self.axes[2].imshow(fft_unsharp / 2e5, cmap=tum_jet, origin='lower', vmax=1, extent=fft_extent)
        self.fig.colorbar(image2, ax=self.axes[2], label='fourier component (arb. u.)')

        fft_sharp = np.abs(np.fft.fft2(self.sharp_image))
        image3 = self.axes[3].imshow(fft_sharp / 2e5, cmap=tum_jet, origin='lower', vmax=1, extent=fft_extent)
        self.fig.colorbar(image3, ax=self.axes[3], label='fourier component (arb. u.)')

        for axis in [self.axes[2], self.axes[3]]:
            axis.set_xlabel(r'$\omega_x$ ($\si{2 \pi \per \micro \meter}$)')
            axis.set_ylabel(r'$\omega_y$ ($\si{2 \pi \per \micro \meter}$)')

    def calculate_fft_extent(self):
        fft_freq_x = np.fft.fftfreq(self.unsharp_image.shape[1],
                                    convert_pixels_to_um(200) / self.unsharp_image.shape[1])
        fft_freq_y = np.fft.fftfreq(self.unsharp_image.shape[0],
                                    convert_pixels_to_um(250) / self.unsharp_image.shape[0])
        fft_extent = [fft_freq_x.min(), fft_freq_x.max(), fft_freq_y.min(), fft_freq_y.max()]
        return fft_extent

    def __plot_original_images(self):
        image0 = cam_imshow(self.unsharp_image, self.axes[0])
        self.fig.colorbar(image0, ax=self.axes[0], label='counts')
        image1 = cam_imshow(self.sharp_image, self.axes[1])
        self.fig.colorbar(image1, ax=self.axes[1], label='counts')
        for axis in [self.axes[0], self.axes[1]]:
            axis.set_xlabel(r'$x$ ($\si{\micro \meter}$)')
            axis.set_ylabel(r'$y$ ($\si{\micro \meter}$)')

    def __create_figures_and_axes(self):
        self.fig = plt.figure(figsize=(cm_to_inch(15), cm_to_inch(16.5)))
        gs = self.fig.add_gridspec(3, 2)
        self.axes = []
        self.axes.append(self.fig.add_subplot(gs[0, 0]))
        self.axes.append(self.fig.add_subplot(gs[0, 1]))
        self.axes.append(self.fig.add_subplot(gs[1, 0]))
        self.axes.append(self.fig.add_subplot(gs[1, 1]))
        self.axes.append(self.fig.add_subplot(gs[2, :]))

    def __load_images(self, folder):
        image_list = []
        for z in self.zs:
            image = np.loadtxt(os.path.join(folder, 'proem_{:.03f}_000.txt'.format(z)))
            image_list.append(image)
        return image_list

    def __sharpness_of_all_images(self):
        sharpnesses = np.zeros(len(self.zs))
        for i, image in enumerate(self.proem_images):
            sharpnesses[i] = self.__sharpness_of_single_image(image)
        return sharpnesses

    @staticmethod
    def __sharpness_of_single_image(image):
        fft = np.abs(np.fft.fft2(image))
        return np.linalg.norm(np.array(scipy.ndimage.center_of_mass(fft)))


if __name__ == '__main__':
    zs = np.arange(27.805, 31.805 + 0.01, 0.1)
    data_folder = '//file/e24/Projects/ReinhardLab/data_setup_nv1/161122_improving_rois'
    z_focus_plotter = ZFocusPlotter(data_folder, zs, 15, 30)
    z_focus_plotter.plot()
