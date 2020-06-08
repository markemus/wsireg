import cv2
import functools
import glob
# Temp fix to test openslide patch
import os
import sys
sys.path.insert(0, "/home/chaim/dev/my_openslide")
import openslide
import matplotlib.cm
import matplotlib.colors
import matplotlib.colorbar
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import time

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from imagetools.datamanager import ImageWithSample
import imagetools.convertimage as ci
import imagetools.datamanager as dm


import deconvolution as dec
import demo
import registration as reg
# import patchreg
import utils
import viz


im1 = cv2.imread("../data/in/test1.png")
im2 = cv2.imread("../data/in/test2.png")

wsi = openslide.OpenSlide("../data/in/halo/1-cd3.tif")


demo.overlay_test()
# def overlay(images, colors=None, cmap=None):
#     """Merges images together after casting them each to a unique color.
#     cmap: Uses plt default if None. Ignored if colors are already manually selected. Qualitative colormaps are recommended."""
#     if colors is None:
#         cmap = matplotlib.cm.get_cmap(name=cmap)
#         colors = cmap(np.linspace(0, 1, len(images)))
#
#     overlays = []
#     for im, color in zip(images, colors):
#         overlays.append(ci.scale_by_max(pseudocolor(ensure_gray(im), color[:3]), 255).astype(np.uint8))
#
#     overlay = ci.scale_by_max(np.average(overlays, axis=0), 255).astype(np.uint8)
#     return overlay



# def overlay_test():
#     """Demonstrates fading in current overlay technique.
#
#     Suggestion: use addition instead of averaging colors."""
#     infolder = "../data/in/Training/"
#     outfolder = f"../data/out/{time.time()}"
#     print(f"Creating outfolder: {outfolder}")
#     os.makedirs(outfolder)
#     paths = glob.glob(os.path.join(infolder, "Training_*.jpg"))
#
#     images = [dm.Image(path) for path in paths]
#     od_stack = [ci.scale_by_max(img.od.reshape(img.shape)[:1000,:1000], 255).astype(np.uint8) for img in images]
#
#     ol_stack = []
#     for i in range(1, len(od_stack)-1):
#         ol_stack.append(viz.overlay(od_stack[:i]))
#
#     for i, im in enumerate(ol_stack):
#         plt.imshow(im)
#         plt.suptitle(f"ol_{i}")
#         plt.show()
#         cv2.imwrite(os.path.join(outfolder, str(i) + ".jpg"), im)
#
#     print("Done")



#
# ol = viz.overlay([im1, im2])
# # Correct colors
# axes = plt.subplot(111)
# axes.imshow(ol)
# c_ax, _ = matplotlib.colorbar.make_axes_gridspec(axes)
# cmap = matplotlib.cm.get_cmap('jhu_vibrant')
# cbar = matplotlib.colorbar.ColorbarBase(c_ax, cmap=cmap, ticks=[.25-.125,.5-.125,.75-.125,1-.125])
# cbar.set_label("Signal Channels")
# cbar.set_ticklabels([1,2,3,4])
# plt.show()
#
# def add_cmap_colorbar(cmap):
#     """Creates a colorbar for a ListedColormap. Does not accept **kwargs for now cuz yagni."""
#     ax = plt.gca()
#     c_ax, _ = matplotlib.colorbar.make_axes_gridspec(ax, fraction=0.046, pad=0.04)
#     # Offsets + shift
#     ticks = (np.linspace(0, 1, len(cmap.colors)+1) - 1/(2*len(cmap.colors)))[1:]
#     cbar = matplotlib.colorbar.ColorbarBase(c_ax, cmap=cmap, ticks=ticks)
#     cbar.set_label("Signal Channels")
#     cbar.set_ticklabels([1, 2, 3, 4])
#
#     return cbar
#
# axes = plt.subplot(141)
# axes.imshow(ol)
# cbar = add_cmap_colorbar(cmap)
#
# plt.show()
# # # Correct formatting
# # iax = plt.imshow(ol)
# # cmap = matplotlib.cm.get_cmap('jhu_vibrant')
# cbar = plt.colorbar(iax)
# cbar.cmap = cmap
# plt.show()







# import matplotlib.cm
# matplotlib.colors.to_rgb("c")
# matplotlib.colors.to_rgba_array("viridis")
# matplotlib.colors.ListedColormap
#
# mpatch.Rectangle
# plt.imshow
#
# matplotlib.cm.cmap_d
# viridis = matplotlib.cm.get_cmap('viridis', 12)







#
# w_latent.T
# min(w_latent.T, key=lambda x: (x-np.array([0,.3,1.3])).sum())
# recon[:,[0,2,1]]
#
#







# # average
# red = np.full([50,50,3], [255,0,0])
# blue = np.full([50,50,3], [0,0,255])
# green = np.full([50,50,3], [0,255,0])
# white = np.full([50,50,3], [255,255,255])
#
# plt.imshow(red)
# plt.show()
# plt.imshow(blue)
# plt.show()
#
# both = np.average([red, blue, green, white], axis=0).astype(np.uint8)
# plt.imshow(both)
# plt.show()
#







# # cmap
# red = [1,0,0,1]
# yellow = [1,1,0,1]
# pink = [1,0,1,1]
# cyan = [0,1,1,1]
# colors = [red, yellow, pink, cyan]
#
# cmap = matplotlib.colors.ListedColormap(colors, 'overlay')
#
#
# # ol = viz.overlay([im1, im2],colors=[[0,0,.5],[0,1,0],[1,0,0]], cmap='Set3')
# ol = viz.overlay([im1, im2], cmap=cmap)
# plt.imshow(ol)
# plt.show()

