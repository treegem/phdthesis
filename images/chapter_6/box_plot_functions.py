import random

import numpy


def load_t2s_in_us(preparation_step):
    t2 = numpy.loadtxt(preparation_step + ".txt")
    t2 /= 1000
    return t2


def remove_average(t2):
    t2[-1] = 0


def create_random_x_offsets(t2s, width):
    xs = []
    for t2 in t2s:
        xs_of_this_step = numpy.zeros_like(t2)
        for x_index in range(len(xs_of_this_step)):
            xs_of_this_step[x_index] = random.uniform(-width/2, width/2)
        xs.append(xs_of_this_step)
    return xs
