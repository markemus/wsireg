"""Harmonize patch level registrations of a large image using bilinear interpolation.

This module turns separate patch registrations into a single global Displacement Vector Field
https://en.wikipedia.org/wiki/Displacement_field_(mechanics).

Use cv2.remap() to apply the DVF- for very large images you may need to tile the image and DVF (untested).
The patch-level registrations can be obtained however you like, but this module should
interface smoothly with patchreg.py.

Note that this module requires fully overlapping patches in order to perform alignments:
Image:
a b c d e
f g h i j
k l m n o
p q r s t

Patches:
[ab fg] [bc gh] [cd hi] [de ij]
[fg kl] [gh lm] ...
[kl pq] ...
"""
import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import sawtooth
from skimage.util import view_as_windows


#TODO with three channel image, why is there a cross through the center? See 3_channel_artifact.png.
def quilter(patches):
    """Turns a set of FULLY OVERLAPPING patches into the set of 4 non-overlapping quilts.
    Fully overlapping means that each patch covers 50% of the region of each neighboring patch.

    Parameters
    ----------
    patches: (rows, cols, stack, prow, pcol, channels)."""
    x_idx_even = np.linspace(0, patches.shape[0] - 1, patches.shape[0]) % 2 == 0
    x_idx_odd = np.linspace(0, patches.shape[0] - 1, patches.shape[0]) % 2 == 1
    y_idx_even = np.linspace(0, patches.shape[1] - 1, patches.shape[1]) % 2 == 0
    y_idx_odd = np.linspace(0, patches.shape[1] - 1, patches.shape[1]) % 2 == 1

    quilts = []
    for x in [x_idx_even, x_idx_odd]:
        for y in [y_idx_even, y_idx_odd]:
            # build quilt
            cols = np.concatenate(patches[x][:, y], axis=-3)
            plane = np.concatenate(cols, axis=-2)
            # padded = utils.buffer_arr()
            quilts.append(plane)

    # Buffer quilts so they're properly aligned with the source image.
    quilts = buffer_quilts(quilts, patches.shape[-3:-1])
    return quilts

def buffer_quilts(quilts, patch_shape):
    """Adds a buffer (shift) to the edge of each quilt to align each pixel across the quilts."""
    bsize = (int(patch_shape[0] / 2), int(patch_shape[1] / 2))
    # Initial buffers- these are the static buffers.
    buffers = [
        [[0, 0], [0, 0], [0, 0], [0, 0]],               #A
        [[0, 0], [0, 0], [bsize[1], 0], [0, 0]],        #B
        [[0, 0], [bsize[0], 0], [0, 0], [0, 0]],        #C
        [[0, 0], [bsize[0], 0], [bsize[1], 0], [0, 0]], #D
    ]

    # Buffer sets
    q_hor = [[0,1],[2,3]]
    q_ver = [[0,2],[1,3]]

    # Add dynamic quilt buffers
    for dim, q_sets in enumerate((q_ver, q_hor), start=1):
        for quilt1, quilt2 in q_sets:
            # There are only 2 possible cases, and the rules are the same in all dimensions.
            if quilts[quilt1].shape[dim] == quilts[quilt2].shape[dim]:
                buffers[quilt1][dim][1] = bsize[dim-1]
            else:
                buffers[quilt2][dim][1] = bsize[dim-1]

    # Apply quilt buffers.
    quilts = list(map(lambda q, pad: np.pad(q, pad_width=pad, mode='constant'), quilts, buffers))

    return quilts

def bilinear_wquilts(patches):
    """Generates a bilinear weight map for a set of patches.

    Parameters
    ----------
    patches: (rows, cols, stack, prow, pcol, channels)."""
    if patches.shape[-3] == patches.shape[-2]:
        side = patches.shape[-3]
    else:
        logging.error("Patches must be square for bilinear interpolation.")
        return False

    tile = bilinear_tile(side)
    weights = np.expand_dims(np.tile(tile, reps=(*patches.shape[:2],1,1,1)), axis=-1)
    wquilts = quilter(weights)

    return wquilts

def bilinear_tile(side):
    t = np.linspace(0, 1, side)
    triangle = sawtooth(2 * np.pi * t, width=0.5)
    triangle = (triangle + 1) * .5
    # weight = np.tile(triangle, (shape[1], int(shape[0] / patch_size)))

    # Single patch weight
    weight = np.tile(triangle, (side, 1))
    weight = weight * weight.T

    return weight

def test():
    # Seems to be a problem with the buffering process. Might only occur with perfectly round numbers, or something... very odd.
    # t_image = utils.make_test_img((5500,5500))
    t_image = np.full((5001,5001,3), 255)
    # plt.imshow(t_image)
    # plt.show()
    w_shape = (1000, 1000, 3)
    w_step = (500, 500, 3)

    # stack1 = np.concatenate((reg1, reg2_aligned), axis=-1)
    patches = view_as_windows(t_image, window_shape=w_shape, step=w_step)

    # patches = test_patches(plates=3, channels=3)[2]
    # plt.imshow(patches[0,0,0,:,:,0].reshape(patches.shape[-3:-1]))
    # plt.show()

    quilts = quilter(patches)
    # Reshape since we don't have multiple plates or channels.
    # quilts = [quilt.reshape(quilt.shape[1:3]) for quilt in quilts]
    for q in quilts:
        plt.imshow(q[0])
        plt.show()

    # Weights for each quilt
    # # wmaps = quilt_weights(quilts[0].shape[1:3], patches.shape[-2])
    # wmaps = quilt_weights(quilts[0].shape, patches.shape[-3:-1])
    # # wmaps = [wmap.reshape(wmap.shape[1:3]) for wmap in wmaps]
    # for w in wmaps:
    #     plt.imshow(w.reshape(w.shape[-3:-1]))
    #     plt.show()
    #
    # wquilts = []
    #
    # # Apply weights to each quilt
    # for (quilt, weight) in zip(quilts, wmaps):
    #     print(quilt.shape, weight.shape)
    #     wquilts.append(quilt * weight)
    wquilts = bilinear_wquilts(patches)

    # Grab first plate from each stack.
    wquilts = [wquilt.reshape(wquilt.shape[1:3]) for wquilt in wquilts]
    for wq in wquilts:
        plt.imshow(wq)
        plt.show()

    # Sum weighted quilts
    summed = wquilts[0] + wquilts[1] + wquilts[2] + wquilts[3]

    # plt.autoscale(False)
    plt.imshow(summed)
    plt.show()
    print("summed.min(): ", summed.min())
    print("summed.max(): ", summed.max())
    print("summed.mean(): ", summed.mean())

if __name__ == "__main__":
    test()
