import functools
import random

import cv2
import numpy as np
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.pyplot as plt

import imagetools.convertimage as ci

from itertools import cycle


#TODO vizualization showing original, all stains, and orthogonal values (loss). See Macenko paper fig 5.
# orthogonal values = original - recon?

def checkerboard(shift, shape):
    """Creates a checkerboard of ones and zeros."""
    tile = np.array([[0, 1], [1, 0]]).repeat(shift, axis=0).repeat(shift, axis=1)
    reps = (int(np.ceil(shape[0] / shift)), int(np.ceil(shape[1] / shift)))
    grid = np.tile(tile, reps)[:shape[0], :shape[1]]

    return grid

def compare_checkerboard(im1, im2, display=False):
    """Makes a checkerboard from two large images so the alignment of all regions can be examined."""
    print("Comparing checkerboard alignment: shapes must match: ", im1.shape, im2.shape)
    idx = checkerboard(shift=50, shape=im1.shape)

    board = np.choose(idx, choices=[im1, im2])

    if display:
        plt.imshow(board)
        plt.show()

    return board

# def overlay(im1, im2, mode="BGR"):
#     c_im1 = pseudocolor(ensure_gray(im1), color="yellow", mode=mode)
#     c_im2 = pseudocolor(ensure_gray(im2), color="blue", mode=mode)
#
#     c_im = c_im1 + c_im2
#
#     return c_im

#TODO-DONE add discrete colorbar https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
def overlay(images, colors=None, cmap=None):
    """Merges images together after casting them each to a unique color.
    cmap: Uses plt default if None. Ignored if colors are already manually selected. Qualitative colormaps are recommended."""
    if colors is None:
        cmap = matplotlib.cm.get_cmap(name=cmap)
        colors = cmap(np.linspace(0, 1, len(images)))

    overlays = []
    for im, color in zip(images, colors):
        overlays.append(ci.scale_by_max(pseudocolor(ensure_gray(im), color[:3]), 255).astype(np.uint8))

    # overlay = ci.scale_by_max(np.average(overlays, axis=0), 255).astype(np.uint8)
    overlay = np.sum(overlays, axis=0)
    # overlay = ci.scale_by_max(np.sum(overlays, axis=0), 255).astype(np.uint8)
    return overlay

def pseudocolor(im, color):
    """Turns a grayscale image into a primary color image."""
    c_im = np.full(im.shape + (len(color),), color)
    im = im.reshape(im.shape + (1,))
    c_im = c_im * im

    return c_im

def add_cmap_colorbar(cmap, ax=None):
    """Creates a colorbar for a ListedColormap. Does not accept **kwargs for now cuz yagni."""
    cmap = matplotlib.cm.get_cmap(cmap)

    if ax is None:
        ax = plt.gca()

    c_ax, _ = matplotlib.colorbar.make_axes_gridspec(ax, fraction=0.046, pad=0.04)

    # Offsets + shift
    ticks = (np.linspace(0, 1, len(cmap.colors)+1) - 1/(2*len(cmap.colors)))[1:]
    cbar = matplotlib.colorbar.ColorbarBase(c_ax, cmap=cmap, ticks=ticks)
    cbar.set_label("Signal Channels")
    cbar.set_ticklabels([1, 2, 3, 4])

    return cbar

def compare_alignment(im1, im2, display=True):
    """Compare alignment quality for two images im1 and im2, where im2 has been registered to im1.
    4 square checkerboard allows both vertical and horizontal alignment to be compared."""
    print("Comparing alignment: shapes should be similar: ", im1.shape, im2.shape)
    height, width = im1.shape[:2]
    tl = im1[:int(height / 2), :int(width / 2)]
    tr = im2[:int(height / 2), int(width / 2):]
    bl = im2[int(height / 2):, :int(width / 2)]
    br = im1[int(height / 2):, int(width / 2):]
    top = np.concatenate((tl, tr), axis=1)
    btm = np.concatenate((bl, br), axis=1)
    comparison = np.concatenate((top, btm), axis=0)

    if display:
        plt.imshow(comparison)
        plt.show()

    return comparison

def compare_random_spot(im1, im2, display=True):
    x1 = random.randrange(im1.shape[0])
    y1 = random.randrange(im1.shape[1])
    x2 = x1 + 200
    y2 = y1 + 200

    spot = compare_alignment(im1[x1:x2, y1:y2], im2[x1:x2, y1:y2])

    if display:
        plt.imshow(spot)
        plt.show()

    return spot

def overlay_random_spot(im1, im2, display=True):
    x1 = random.randrange(im1.shape[0])
    y1 = random.randrange(im1.shape[1])
    x2 = x1 + 200
    y2 = y1 + 200

    spot = overlay([im1[x1:x2, y1:y2], im2[x1:x2, y1:y2]])

    if display:
        plt.imshow(spot)
        plt.show()

    return spot

def sum(images):
    return functools.reduce(lambda x, y: cv2.add(x, y), images)

def mean(images):
    """A representative image for the set.
    Intended for use with a list of registered images."""
    return np.mean(np.stack(images), axis=0).astype(np.uint8)

def diff(images):
    """The difference between all images in the stack and the first image in the stack.
    Intended to compare single channel registered images against the fixed image."""
    return [cv2.absdiff(images[0], img) for img in images]

def print_spiral(shape):
    """Shape must be square."""
    arr = np.zeros(shape)
    direction = cycle([(0, 1), (1, 0), (0, -1), (-1, 0)])
    c = 0  # counter

    x = int(np.floor(shape[0] / 2) - 1)
    y = int(np.floor(shape[1] / 2) - 1)

    for k in range(shape[0]):
        for j in range(2):
            shift = next(direction)
            for i in range((c // 2) + 1):
                x += shift[0]
                y += shift[1]
                arr[x, y] = (10 * c) + i
            c += 1

    return arr

def ensure_gray(img):
    """img must be in RGB/A or GRAY format, not BGR/A format."""
    if len(img.shape) == 4:
        return cv2.cvtColor(img, code=cv2.COLOR_RGBA2GRAY)
    elif len(img.shape) == 3:
        return cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 2:
        return img
    else:
        raise ValueError("viz.ensure_gray could not recognize the image format.")


# Colormaps
red = [1,0,0,1]
yellow = [1,1,0,1]
pink = [1,0,1,1]
cyan = [0,1,1,1]
blue = [0,0,1,1]

jhu_vibrant = matplotlib.colors.ListedColormap([red, yellow, pink, cyan], 'jhu_vibrant')
jhu_test = matplotlib.colors.ListedColormap([yellow, red, blue], 'jhu_test')

matplotlib.cm.register_cmap(cmap=jhu_vibrant)
matplotlib.cm.register_cmap(cmap=jhu_test)