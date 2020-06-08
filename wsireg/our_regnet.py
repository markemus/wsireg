import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import imagetools.plotter as plo

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from scipy.interpolate import interp2d


def gen_dvf(x, y, dx=2, dy=2):
    """Generates a displacement vector field with shape (x,y).
    This can be used to distort any image of the same size.
    dx, dy: the maximum shift per pixel in each dimension."""
    x_peaks, y_peaks = 5, 5
    # We need to ensure we cover (at least) the whole image, or interpolation will extrapolate a constant.
    spline_length = 2*x/x_peaks

    grid_x_id, grid_y_id = np.mgrid[0:x_peaks - 1:complex(x_peaks), 0:y_peaks - 1:complex(y_peaks)] * spline_length
    # grid_x_large_id, grid_y_large_id = np.mgrid[0:(x_peaks * spline_length)-1:complex(x_peaks * spline_length), 0:(y_peaks*spline_length)-1:complex(y_peaks * spline_length)]
    # perturb
    grid_x_peaks = grid_x_id + np.random.randint(-dx, dx + 1, size=(x_peaks, y_peaks))
    grid_y_peaks = grid_y_id + np.random.randint(-dy, dy + 1, size=(x_peaks, y_peaks))
    # interpolate
    dvf_x = interp2d(grid_x_id, grid_y_id, grid_x_peaks)(np.linspace(0, x-1, x), np.linspace(0, y-1, y)).astype(np.float32)
    dvf_y = interp2d(grid_x_id, grid_y_id, grid_y_peaks)(np.linspace(0, x-1, x), np.linspace(0, y-1, y)).astype(np.float32)

    return dvf_x, dvf_y

def dvfer(arr):
    """Yields a dvf for the next image."""
    for x in arr:
        yield gen_dvf(x.shape[0], x.shape[1])

def morpher(arr):
    """Yields a morphed image for the next image."""
    for x, (dvf_x, dvf_y) in zip(arr, dvfer(arr)):
        yield cv2.remap( x, dvf_x, dvf_y, interpolation=cv2.INTER_LINEAR)

# Main
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
z = zip(iter(x_train), morpher(x_train))
for i in range(5):
    plo.graph_images(next(z))[0].show()


model = Sequential(
    [Conv2D(32, (3,3), activation='relu', input_shape=(32,32,6)),
     Conv2D(32, (3,3), activation='relu'),
     MaxPooling2D(pool_size=(2,2)),

     Conv2D(64, (3,3), activation='relu'),
     Conv2D(64, (3,3), activation='relu'),
     MaxPooling2D(pool_size=(2,2)),

     Flatten(),
     Dense(32*32, activation=None)])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit()