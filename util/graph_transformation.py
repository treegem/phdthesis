import numpy as np


def smooth_array_by_rolling_average(array: np.ndarray, neighbors: int = 1):
    smoothed_array = np.zeros_like(array)
    for i, value in enumerate(array):
        if i - neighbors < 0:
            smoothed_array[i] = __smooth_single_value_by_rolling_average(array, i, neighbors=i)
        elif i + neighbors >= array.shape[0]:
            max_neighbors = array.shape[0] - i - 1
            smoothed_array[i] = __smooth_single_value_by_rolling_average(array, i, neighbors=max_neighbors)
        else:
            smoothed_array[i] = __smooth_single_value_by_rolling_average(array, i, neighbors)
    return smoothed_array


def normalize(array: np.ndarray):
    array -= array.min()
    max_ = array / array.max()
    return max_


def __smooth_single_value_by_rolling_average(array: np.ndarray, i: int, neighbors: int = 1):
    summed_values = np.sum(array[i - neighbors:i + neighbors + 1])
    return summed_values / (2 * neighbors + 1)
