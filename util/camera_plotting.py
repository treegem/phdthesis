import matplotlib.pyplot as plt

from util.tum_jet import tum_jet


def cam_imshow(data, axis=None, color_map=tum_jet, vmin=None, vmax=None):
    if axis is None:
        return plt.imshow(data, cmap=color_map, origin='lower', aspect=1, interpolation='bilinear',
                          extent=convert_extent_from_pixels_to_um(), vmin=vmin, vmax=vmax)
    else:
        return axis.imshow(data, cmap=color_map, origin='lower', aspect=1, interpolation='bilinear',
                           extent=convert_extent_from_pixels_to_um(), vmin=vmin, vmax=vmax)


def convert_extent_from_pixels_to_um():
    return [0, convert_pixels_to_um(200), 0, convert_pixels_to_um(250)]


def convert_pixels_to_um(pixels):
    um_per_pixels = 10 / 150
    return pixels * um_per_pixels
