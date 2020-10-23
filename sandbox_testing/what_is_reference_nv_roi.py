import matplotlib.pyplot as plt
import numpy as np


def main():
    reference_data = np.loadtxt('reference.txt')
    plt.imshow(reference_data)
    plt.show()

    nvs_roi_data = np.loadtxt('nvs_roi.txt')
    plt.imshow(nvs_roi_data)
    plt.show()


if __name__ == '__main__':
    main()
