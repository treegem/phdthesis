# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:21:30 2016

@author: operator
"""

import proem
from proem_constants import *
import numpy
import matplotlib.pylab as plt
import threading
import scipy.signal


def cross_image(image1, image2):
    im1 = image1.copy()
    im2 = image2.copy()
    # get rid of the averages, otherwise the results are not good
    im1 -= numpy.mean(im1)
    im2 -= numpy.mean(im2)

    # calculate the correlation image; note the flipping of onw of the images
    return scipy.signal.fftconvolve(im1, im2[::-1, ::-1], mode='same')


pro = proem.ProEm()

# Save old data
x, y, z = scan.stg.position()

# Set Camera
nexp = 1  # number of exposures per frame
nframe = 1  # number of frames
pro.exposure_time = 5000
pro.set_region(h1=200, h2=400, v1=100, v2=100 + 255)
pro.set_em_gain(300)

###############################################################################
###############################################################################
# Z FOCUS Z FOCUS Z FOCUS Z FOCUS Z FOCUS Z FOCUS Z FOCUS Z FOCUS Z FOCUS #####
###############################################################################

# BLOCKING
# Scan through a row of z-values
for dz in numpy.arange(1.5, 2.6, 0.1):
    scan.stg.set_position(x, y, z + dz)
    pro.acq_single(nexp=nexp, nframes=nframe, exposure_mode='TIMED_MODE')
    pro.save_flag.set()
    pro.plot_frame(frame=pro.frame, filename="proem_%.03f" % (z + dz))


# Same as above, thread
def rough_focus():
    for dz in numpy.arange(-1.0, 1.1, 0.1):
        scan.stg.set_position(x, y, z + dz)
        pro.acq_single(nexp=nexp, nframes=nframe, exposure_mode='TIMED_MODE')
        pro.save_flag.set()
        pro.plot_frame(frame=pro.frame.copy(), filename="proem_%.03f" % (z + dz))
        numpy.savetxt("A:/data/zzz_incoming/rough_dz%03d.txt" % (dz), pro.frame.copy())


thread = threading.Thread(target=rough_focus)
thread.start()

########################
# Same as rough_focus, but with an autoselection of the best z value

import scipy.ndimage
import threading

# Save old data
x, y, z = scan.stg.position()

nexp = 1  # number of exposures per frame
nframe = 1  # number of frames

pro.exposure_time = 2000
pro.set_region()
pro.set_region(h1=50, h2=250, v1=255, v2=255 + 255)
pro.set_em_gain(1000)


def z_focus():
    print('Beginning z-focus')
    maxi = -numpy.infty
    zs = numpy.arange(-0.2, 0.21, 0.1)
    for dz in zs:
        print('dz = %.03f' % (dz))
        scan.stg.set_position(x, y, z + dz)
        pro.acq_single(nexp=nexp, nframes=nframe, exposure_mode='TIMED_MODE')
        fft = numpy.abs(numpy.fft.fft2(pro.frame.copy()))
        sharpness = numpy.linalg.norm(numpy.array(scipy.ndimage.center_of_mass(fft)))
        if sharpness > maxi:
            maxi = sharpness
            maxi_z = dz
    scan.stg.set_position(x, y, z + maxi_z)
    confocal.update_plt.emit()
    pro.acq_single(nexp=nexp, nframes=nframe, exposure_mode='TIMED_MODE')
    pro.save_flag.set()
    pro.plot_frame(frame=pro.frame.copy(), filename="proem_%.03f" % (z + maxi_z))
    print('z-focus done')
    print('sharpest image for z = %0.3f' % (z + maxi_z))
    return z + maxi_z


thread = threading.Thread(target=z_focus)
thread.start()

###############################################################################
###############################################################################
# Load in RABI frames, evaulate only ROI and plot

import os
import numpy
import matplotlib.pyplot as plt
import scipy.io as sio

frames = []
roi = []

n = 11
taus = sio.loadmat("A:/data/zzz_incoming/rabi_%03d/parameters.mat" % (n))['taus'][0]
for i in range(len(taus)):
    frames.append(numpy.loadtxt("A:/data/zzz_incoming/rabi_%03d/frame_%03d.txt" % (n, i)))
result = numpy.loadtxt("A:/data/zzz_incoming/rabi_%03d/frames.txt" % (n))
i = 0
while os.path.isfile("A:/data/zzz_incoming/rabi_%03d/corrected_%03d.txt" % (n, i)):
    corrected.append(numpy.loadtxt("A:/data/zzz_incoming/rabi_%03d/corrected_%03d.txt" % (n, i)))
    i += 1

nvs_roi = numpy.loadtxt("A:/data/zzz_incoming/rabi_%03d/nvs_roi.txt" % (n))
reference = numpy.loadtxt("A:/data/zzz_incoming/rabi_%03d/reference.txt" % (n))

x1 = 0
y1 = 0
x2 = frames[0].shape[1]
y2 = frames[0].shape[0]

# y1= 50
# y2 = 100
# x1 = 100
# x2 = 150

# if needed, one frame can be plotted here, to determine x1 - y2
plt.imshow(frames[0][y1:y2, x1:x2], interpolation='none')
plt.colorbar()

plt.imshow(frames[0] * nvs_roi)
plt.colorbar()

plt.imshow(corrected[0], interpolation='none')
plt.colorbar()

plt.plot(taus, result / max(result))

correlations = []
for frame in frames:
    corr = cross_image(frames[0].copy(), frame.copy())
    correlations.append(numpy.unravel_index(numpy.argmax(corr), corr.shape))

# NOT CORRECTED by dividing through the counts in non-resonant areas
counts_roi = []
for i, frame in enumerate(frames):
    nvs_roi_shifted = nvs_roi.copy()
    # nvs_roi_shifted = numpy.roll(nvs_roi_shifted, correlations[0][0] - correlations[i][0], axis = 0)
    # nvs_roi_shifted = numpy.roll(nvs_roi_shifted, correlations[0][1] - correlations[i][1], axis = 1)
    counts_roi.append((nvs_roi * frame[y1:y2, x1:x2]).sum())
plt.plot(taus, counts_roi / max(counts_roi))

for i, corrs in enumerate(corrected):
    plt.imshow(corrs / corrected[0], vmin=0.5, vmax=1.5)
    plt.colorbar()
    plt.savefig("A:/data/zzz_incoming/rabi_%03d/corrected_%03d.jpg" % (n, i))
    plt.show()

for i, corrs in enumerate(corrected):
    plt.imshow(corrs)
    plt.colorbar()
    plt.savefig("A:/data/zzz_incoming/rabi_%03d/corrected_%03d.jpg" % (n, i))
    plt.show()

for i, corrs in enumerate(corrected):
    plt.imshow(corrs / (reference * frames[i]).sum())
    plt.colorbar()
    plt.savefig("A:/data/zzz_incoming/rabi_%03d/corrected_%03d.jpg" % (n, i))
    plt.show()

for i, frame in enumerate(frames):
    plt.imshow(frame / frames[0], vmin=0.5, vmax=1.5)
    plt.colorbar()
    plt.savefig("A:/data/zzz_incoming/rabi_%03d/frame_%03d.jpg" % (n, i))
    plt.show()

# alternatively get the max value in frames[0] and repeat for several radii
# relevant_frame = frames[0][y1:y2,x1:x2]
# max_ind = numpy.unravel_index(relevant_frame.argmax(), relevant_frame.shape)
for i in [3, 4, 5, 6, 7]:
    counts_roi = []
    for j, frame in enumerate(frames):
        relevant_frame = frame[y1:y2, x1:x2]
        max_ind = numpy.unravel_index(relevant_frame.argmax(), relevant_frame.shape)
        counts_roi.append(frame[y1 + max_ind[0] - i:y1 + max_ind[0] + i, x1 + max_ind[1] - i:x1 + max_ind[1] + i].sum())
        if j == 0:
            plt.imshow(frame[y1 + max_ind[0] - i:y1 + max_ind[0] + i, x1 + max_ind[1] - i:x1 + max_ind[1] + i],
                       interpolation='none')
            plt.colorbar()
            plt.show()
    #    plt.imshow(frames[0][y1+max_ind[0]-i:y1+max_ind[0]+i,x1+max_ind[1]-i:x1+max_ind[1]+i],interpolation = 'none')
    #    plt.colorbar()
    plt.show()
    plt.plot(taus, counts_roi / max(counts_roi))
    plt.show()
    print('contrast * sqrt(max_counts): ',
          str(numpy.sqrt(max(counts_roi)) * (max(counts_roi) - min(counts_roi)) / max(counts_roi)))

for i, frame in enumerate(frames):
    plt.imshow(frame / frames[0], vmin=0.5, vmax=1.5)
    plt.savefig("A:/data/zzz_incoming/rabi_%03d/movie_frame_%03d.jpg" % (n, i))

#############################################
#############################################
# Load in ECHO frames, evaulate only ROI and plot
import os
import numpy
import matplotlib.pyplot as plt
import scipy.io as sio

n = 11
taus = sio.loadmat("A:/data/zzz_incoming/echo_%03d/parameters.mat" % (n))['taus'][0]

frames1 = []
frames2 = []

roi1 = []
roi2 = []

for i in range(len(taus)):
    frames1.append(numpy.loadtxt("A:/data/zzz_incoming/echo_%03d/frame_%03d.txt" % (n, i * 2)))
    frames2.append(numpy.loadtxt("A:/data/zzz_incoming/echo_%03d/frame_%03d.txt" % (n, i * 2 + 1)))
result = numpy.loadtxt("A:/data/zzz_incoming/echo_%03d/result.txt" % (n))
result1 = result[::2]
result2 = result[1::2]
i = 0
while os.path.isfile("A:/data/zzz_incoming/echo_%03d/roi_%03d.txt" % (n, i * 2 + 1)):
    roi1.append(numpy.loadtxt("A:/data/zzz_incoming/echo_%03d/roi_%03d.txt" % (n, i * 2)))
    roi2.append(numpy.loadtxt("A:/data/zzz_incoming/echo_%03d/roi_%03d.txt" % (n, i * 2 + 1)))
    i += 1

reference = numpy.loadtxt("A:/data/zzz_incoming/echo_%03d/reference.txt" % (n))

sweeps = len(result) / len(taus) / 2
# for i in range(len(taus)):
#    value1 = 0
#    value2 = 0
#    for j in range(sweeps):
#        value1 += corrected1[i+j*len(taus)]
#        value2 += corrected2[i+j*len(taus)]
#    corrected1[i] = value1
#    corrected2[i] = value2
# corrected1 = corrected1[:len(taus)]
# corrected2 = corrected2[:len(taus)]

x1 = 0
y1 = 0
x2 = frames1[0].shape[1]
y2 = frames1[0].shape[0]

# if needed, one frame can be plotted here, to determine x1 - y2
plt.imshow(frames1[0][y1:y2, x1:x2], interpolation='none')
plt.colorbar()

plt.plot(taus, result1)
plt.plot(taus, result2)
plt.plot(taus, numpy.array(result1) - numpy.array(result2))

x1 = 80 + 20
y1 = 60 + 49
x2 = 140 - 20
y2 = 130 - 9

plt.imshow(roi1[0][y1:y2, x1:x2])
plt.colorbar()

corrSNV1 = []
corrSNV2 = []
for i, corr in enumerate(corrected1):
    corrSNV1.append(corr[y1:y2, x1:x2].sum())
    corrSNV2.append(corrected2[i][y1:y2, x1:x2].sum())
plt.plot(taus, corrSNV1)
plt.plot(taus, corrSNV2)

for i, corr in enumerate(corrected1):
    plt.imshow(corr)
    plt.colorbar()
    plt.savefig("A:/data/zzz_incoming/echo_%03d/corrected1_%03d.jpg" % (n, i))
    plt.show()

correlations = []
for frame in frames1:
    corr = cross_image(frames1[0].copy(), frame.copy())
    correlations.append(numpy.unravel_index(numpy.argmax(corr), corr.shape))

# NOT CORRECTED by dividing through the counts in non-resonant areas
counts_roi = []
counts_roi2 = []
for i, frame in enumerate(frames1):
    nvs_roi_shifted = nvs_roi.copy()
    nvs_roi_shifted = numpy.roll(nvs_roi_shifted, correlations[0][0] - correlations[i][0], axis=0)
    nvs_roi_shifted = numpy.roll(nvs_roi_shifted, correlations[0][1] - correlations[i][1], axis=1)
    counts_roi.append((nvs_roi_shifted * frame[y1:y2, x1:x2]).sum())
    counts_roi2.append((nvs_roi_shifted * frames2[i][y1:y2, x1:x2]).sum())

counts_roi = numpy.array(counts_roi)
counts_roi2 = numpy.array(counts_roi2)

plt.plot(taus, counts_roi / max(counts_roi))
plt.plot(taus, counts_roi2 / max(counts_roi))

plt.plot(taus, counts_roi - counts_roi2 / max(counts_roi))

# get max value in frames[0] and repeat for several radii
# relevant_frame = frames[0][y1:y2,x1:x2]
# max_ind = numpy.unravel_index(relevant_frame.argmax(), relevant_frame.shape)
for i in [3, 4, 5, 6, 7]:
    counts_roi1 = []
    for j, frame in enumerate(frames1):
        relevant_frame = frame[y1:y2, x1:x2]
        max_ind = numpy.unravel_index(relevant_frame.argmax(), relevant_frame.shape)
        counts_roi1.append(
            frame[y1 + max_ind[0] - i:y1 + max_ind[0] + i, x1 + max_ind[1] - i:x1 + max_ind[1] + i].sum())
        if j == 0:
            plt.imshow(frame[y1 + max_ind[0] - i:y1 + max_ind[0] + i, x1 + max_ind[1] - i:x1 + max_ind[1] + i],
                       interpolation='none')
            plt.colorbar()
            plt.show()
    counts_roi2 = []
    for j, frame in enumerate(frames2):
        relevant_frame = frame[y1:y2, x1:x2]
        max_ind = numpy.unravel_index(relevant_frame.argmax(), relevant_frame.shape)
        counts_roi2.append(
            frame[y1 + max_ind[0] - i:y1 + max_ind[0] + i, x1 + max_ind[1] - i:x1 + max_ind[1] + i].sum())
        if j == 0:
            plt.imshow(frame[y1 + max_ind[0] - i:y1 + max_ind[0] + i, x1 + max_ind[1] - i:x1 + max_ind[1] + i],
                       interpolation='none')
            plt.colorbar()
            plt.show()
    plt.plot(taus, counts_roi1)
    plt.plot(taus, counts_roi2)
    plt.show()

#############################################
#############################################
# Load in OPTIMIZE frames, evaulate only ROI and plot
import numpy
import matplotlib.pyplot as plt
import scipy.io as sio

n = 3
data = sio.loadmat("A:/data/zzz_incoming/optimize_%03d/parameters.mat" % (n))

taus = numpy.arange(50, 450, 50)
# taus = [50]
frames1 = []
frames2 = []
for i in range(len(taus)):
    frames1.append(numpy.loadtxt("A:/data/zzz_incoming/optimize_%03d/no_mw/frame_%03d.txt" % (n, i)))
    frames2.append(numpy.loadtxt("A:/data/zzz_incoming/optimize_%03d/mw/frame_%03d.txt" % (n, i)))

x1 = 90
x2 = 115
y1 = 100
y2 = 130

# if needed, one frame can be plotted here, to determine x1 - y2
plt.imshow(frames1[0][y1:y2, x1:x2], interpolation='none')
plt.colorbar()

# get max value in frames[0] and repeat for several radii
for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    counts_roi1 = []
    for j, frame in enumerate(frames1):
        relevant_frame = frame[y1:y2, x1:x2]
        max_ind = numpy.unravel_index(relevant_frame.argmax(), relevant_frame.shape)
        counts_roi1.append(
            frame[y1 + max_ind[0] - i:y1 + max_ind[0] + i, x1 + max_ind[1] - i:x1 + max_ind[1] + i].sum())
        if j == 0:
            plt.imshow(frame[y1 + max_ind[0] - i:y1 + max_ind[0] + i, x1 + max_ind[1] - i:x1 + max_ind[1] + i],
                       interpolation='none')
            plt.colorbar()
            plt.show()
    counts_roi2 = []
    for j, frame in enumerate(frames2):
        relevant_frame = frame[y1:y2, x1:x2]
        max_ind = numpy.unravel_index(relevant_frame.argmax(), relevant_frame.shape)
        counts_roi2.append(
            frame[y1 + max_ind[0] - i:y1 + max_ind[0] + i, x1 + max_ind[1] - i:x1 + max_ind[1] + i].sum())
        if j == 0:
            plt.imshow(frame[y1 + max_ind[0] - i:y1 + max_ind[0] + i, x1 + max_ind[1] - i:x1 + max_ind[1] + i],
                       interpolation='none')
            plt.colorbar()
            plt.show()
    counts_roi1 = numpy.array(counts_roi1)
    counts_roi2 = numpy.array(counts_roi2)
    plt.plot(taus, counts_roi2 / counts_roi1)
    plt.show()

###############################################################################
###############################################################################
###### SAVE ALL WATERSHED ROIS
pylab.ion()
for i, nv in enumerate(mes_pulsed.nvlist):
    pylab.imshow(mes_pulsed.ws == nv)
    pylab.savefig("A:/data/zzz_incoming/ws_nv%03d.jpg" % (nv))

###############################################################################
###############################################################################
##### PROEM LINE SCAN
import proem

if not globals().has_key('pro'):
    counter.stop()
    pro = proem.ProEm()
    counter.start()

import time
import threading


def x_line_scan():
    x, y, z = scan.stg.position()

    dx = 0

    nexp = 1  # number of exposures per frame
    nframe = 5  # number of frames
    pro.exposure_time = 500
    pro.set_region()
    pro.set_region(h1=120, h2=320, v1=200, v2=200 + 255)
    pro.set_em_gain(900)

    while dx < 30:
        scan.stg.set_position(x - dx, y, z)
        time.sleep(1)  # against bleaching layer
        pro.acq_single(nexp=nexp, nframes=nframe, exposure_mode='TIMED_MODE')
        pro.save_flag.set()
        pro.plot_frame(frame=pro.frame)
        dx += 2

    scan.stg.set_position(x, y, z)


thread = threading.Thread(target=x_line_scan)
thread.start()

###############################################################################
###############################################################################
####### SAVE ALL FRAMES FROM ODMR

for i, frame in enumerate(omes.frames):
    pylab.clf()
    pylab.imshow(frame, vmin=450, vmax=1100)
    pylab.colorbar()
    pylab.savefig('A:/data/zzz_incoming/omes_%03d.jpg' % (i))

###############################################################################
###############################################################################
####### AUTO Z-FOCUS
import threading


def auto_z_focus():
    print('Autofocus started.')
    if globals().has_key('pro'):
        cam = globals()['pro']
    else:
        print('No camera instance found.')
        print('Autofocus aborted.')
        return
    z = cam.z_focus()
    new_z = cam.z_focus()
    while not z == new_z:
        z = new_z
        new_z = cam.z_focus()
    cam.z_focus(output=True)
    print('Autofocus completed.')


focus_thread = threading.Thread(target=auto_z_focus)
if not (pro.diagnose()):
    focus_thread.start()

###############################################################################
###############################################################################
####### Get deepest point in ODMR

from scipy.io import loadmat

odmr = loadmat("A:/data/zzz_incoming/odmr_measurement_001.mat")
sums = odmr['sums'][0]
fs = odmr['fs'][0]

print(fs[numpy.argmin(sums)])

###############################################################################
###############################################################################
###### Cut away beginning/end from ODMR

from scipy.io import loadmat
import matplotlib.pylab as pylab

odmr = loadmat("A:/data/zzz_incoming/odmr_measurement_007.mat")
sums = odmr['sums'][0]
fs = odmr['fs'][0]

skip = 4

pylab.plot(fs[skip:], sums[skip:])
pylab.savefig("A:/data/zzz_incoming/test.jpg")

###############################################################################
###############################################################################
####### Make 2d Plot from B-Scan of ODMR

from scipy.io import loadmat
import os
import numpy
import matplotlib.pylab as pylab

path = "A:/data/zzz_incoming/odmr_scan_004/"
files = os.listdir(path)
data = []
for f in files:
    if f.endswith(".mat"):
        print(f)
        odmr = loadmat(path + f)
        sums = odmr['sums'][0][3:]
        sums /= numpy.mean(sums)
        fs = odmr['fs'][0][3:]
        data.append(sums)
data = numpy.array(data)
pylab.imshow(data)
pylab.savefig("A:/data/zzz_incoming/odmr_2d_plot.jpg")

###############################################################################
###############################################################################
#######
