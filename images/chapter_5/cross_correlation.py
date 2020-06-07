import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from util.graph_transformation import normalize
from util.inches import cm_to_inch
from util.tum_jet import tum_jet


class CrossCorrelator:
    def __init__(self, original_file):
        self.__create_and_scale_figure_and_axes()
        self.original_data = np.loadtxt(original_file)
        self.shifted_data = self.__create_shifted_data()

    def plot(self):
        self.__plot_cross_correlation_and_colorbar(0, self.original_data, self.original_data, False)
        self.__plot_cross_correlation_and_colorbar(1, self.original_data, self.shifted_data, True)

        self.__set_labels()

        plt.tight_layout()
        plt.savefig('cross_correlation.png', dpi=500)

    def __set_labels(self):
        for axis in self.axes:
            axis.set_xlabel(r'$x$ shift (pixel)')
        self.axes[0].set_ylabel(r'$y$ shift (pixel)')

    def __plot_cross_correlation_and_colorbar(self, axis_index, data1, data2, add_colorbar):
        data1_minus_mean = self.__subtract_mean(data1)
        data2_minus_mean = self.__subtract_mean(data2)
        correlated_data = scipy.signal.fftconvolve(data1_minus_mean, data2_minus_mean[::-1, ::-1], mode='same')
        extent = [-data1.shape[1] / 2, data1.shape[1] / 2, -data1.shape[0] / 2, data1.shape[0] / 2]
        image = self.axes[axis_index].imshow(normalize(correlated_data), cmap=tum_jet, origin='lower',
                                             interpolation='bilinear', extent=extent)
        if add_colorbar:
            self.fig.colorbar(image, ax=self.axes[axis_index], label='convolution (normalized)')

    @staticmethod
    def __subtract_mean(data):
        data_mean = data - data.mean()
        return data_mean

    def __create_and_scale_figure_and_axes(self):
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2)
        self.fig.set_figheight(cm_to_inch(7.5))
        self.fig.set_figwidth(cm_to_inch(15))

    def __create_shifted_data(self):
        return np.roll(self.original_data, [50, 50], [0, 1])


if __name__ == '__main__':
    correlator = CrossCorrelator('original.txt')
    correlator.plot()
