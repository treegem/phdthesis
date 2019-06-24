#!python3

import pylab

tum_raw = [
    (0, 101, 189),  # TUM Blue
    (65, 190, 255),  # TUM Light Blue
    (145, 172, 107),  # TUM Green
    # (181, 202, 130),  # TUM Light Green
    (255, 180, 0),  # TUM Yellow
    (255, 128, 0),  # TUM Orange
    (229, 52, 24),  # TUM Red
    (202, 33, 63)  # TUM Dark Red
]

offsets = [.0, .35, .5, .75, .85, .95, 1.0]

tum_colors = [(offsets[ci], (col[0] / 255.0, col[1] / 255.0, col[2] / 255.0)) for ci, col in enumerate(tum_raw)]

tum_jet = pylab.matplotlib.colors.LinearSegmentedColormap.from_list("tum_jet", tum_colors)


def tum_color(index):
    color = tum_raw[index]
    norm_color = (color[0] / 256, color[1] / 256, color[2] / 256)
    return norm_color
