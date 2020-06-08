import glob
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
sys.path.insert(0, "../../my_openslide/")
import openslide
import imagetools.convertimage as ci

import registration as reg
import viz


from sklearn.decomposition import NMF


# Target syntax:
# stack[0].od.reg.deconv[0]
# stack[1].reg.od.deconv[1]
# stack[2].plot()
# This keeps the chaining of the functional approach but tracks state along the way.
# That will solve the data extraction=pulling teeth problem.
# This feels too clever by half?

# Alternative:
# Each call to a Stack returns a new Stack. You can keep all past Stacks in a list or
# garbage collect them after each transformation.
# A History object to track Stacks in order? History.plot() will plot each Stack[i][j]
# on its own Figure, with stack[i][j].name as ax.title?
# This doesn't feel right either. History is a hack.

# Ugh.

# Composite pattern?

# TODO-DONE stack should be iterable.
# TODO __repr__ should show as list.
class Stack:
    """A stack of images.

    This can be a stack of full WSIs, or just regions within them.
    Both should have the same behavior.
    Performing an operation on a Stack returns a new Stack. This allows
    users to maintain a record of past operations or to easily
    clear them from memory."""
    def __init__(self, images):
        self.images = [ROI(img) for img in images]

    @classmethod
    def from_files(cls, paths):
        images = [cv2.imread(path) for path in paths]
        return cls(images)

    def __getitem__(self, item):
        """Allows indexing and looping."""
        return self.images[item]

    def __iter__(self):
        return iter(self.images)

    def read_region(self, loc, size):
        """Returns a subregion of each image."""
        rois = [img.read_region(loc, size) for img in self.images]
        stack = Stack(images=rois)
        return stack
    
    @property
    def shape(self):
        shape = [img.shape for img in self.images]
        return shape

    def od(self):
        ods = [img.od() for img in self.images]
        stack = Stack(images=ods)

        return stack

    def register(self):
        """Returns a new Stack of registered images"""
        registrations = [img.register(self.images[0].img) for img in self.images]
        # zip(*list) is a transpose operation for multidim lists.
        regs, elastixes = zip(*registrations)
        stack = Stack(images=regs)

        return stack, elastixes

    def deconv(self):
        decs = [img.deconv() for img in self.images]
        # zip(*list) is a transpose operation for multidim lists.
        bg, fg, X = zip(*decs)
        # bg, fg = [w[:,0] for w in W], [w[:,1] for w in W]
        bg_stack = Stack(images=bg)
        fg_stack = Stack(images=fg)

        return bg_stack, fg_stack, X


class Image:
    def __init__(self):
        self.img = None

    @property
    def shape(self):
        return self.img.shape

    def deconv(self):
        """Uses non-negative matrix factorization to decompose a (pixels, RGB)
        matrix into (pixels, stain) and (stain, RGB) matrices."""
        print("Deconvolving...")
        nmf = NMF(n_components=2)

        # Flatten for NMF.
        flat = self.img.reshape(1, -1, self.shape[-1])
        sample = ci.arraySample(flat)

        nmf.fit(sample[0])
        W = nmf.transform(flat[0])
        X = nmf.components_

        # Restore shape.
        bg = W[:,0].reshape(self.shape[:-1])
        fg = W[:,1].reshape(self.shape[:-1])

        return bg, fg, X

    def register(self, fixed):
        """Image registration aligns two images by calculating and applying
        a pixel-to-pixel mapping."""
        print("Registering...")
        #TODO-URGENT demuddle this PLEASE.
        ElastixImageFilter = reg.register(fixed, self.img, parameterMapVector=reg.build_params_large_image())
        im = (ci.scale_by_max(sitk.GetArrayFromImage(ElastixImageFilter.GetResultImage())) * 255).astype(np.uint8)
        return im, ElastixImageFilter

    def od(self):
        """Optical density is a color space that is linear for absorbance
        (ie stain amounts)"""
        lum = np.array([255, 255, 255])
        od = np.log10((lum) / ci.replaceZeros(self.img, stable=False))
        return od

    def od_to_i(self):
        """Inverse of optical density conversion."""
        lum = np.array([255, 255, 255])
        img = lum / (10 ** self.img)
        return img


class ROI(Image):
    """A single numpy image."""
    def __init__(self, img):
        self.img = img

    def read_region(self, loc, size):
        roi = self.img[ loc[0]: loc[0] + size[0],
                        loc[1]: loc[1] + size[1]]
        return roi


class WSI(Image):
    """A single whole slide image.

    WARNING: Attempting operations on a whole slide image
    will usually just give you MemoryErrors."""
    def __init__(self, wsi):
        self.wsi = wsi
        self._img = None

    @property
    def img(self):
        if not self._img:
            self._img = self.read_region(loc=(0, 0), size=self.wsi.dimensions[0])
        return self._img

    def read_region(self, loc, size):
        roi = np.array(self.wsi.read_region(location=loc, level=0, size=size))
        return roi


def plot_stack_hists(stacks):
    image_hists = zip(*stacks)
    for history in image_hists:
        plot_image_hist(history)

def plot_image_hist(image_hist):
    fig, axes = plt.subplots(len(image_hist))
    for image, ax in zip(image_hist, axes):
        ax.imshow(image.img)

    fig.show()
# TODO add support for old overlays.
# TODO move plot functions to viz.

if __name__ == "__main__":
    print("Starting...")
    paths = glob.glob("../data/in/reg?.png")
    stack = Stack.from_files(paths)
    # Subsample for testing
    # roi_stack = stack.read_region(loc=(500,500), size=(2000,2000))
    od_stack = stack.od()
    bg_stack, fg_stack, X = od_stack.deconv()
    reg_stack, elastixes = bg_stack.register()

    plot_stack_hists([stack, bg_stack, fg_stack, reg_stack])
    # this should be easier.
    ol = viz.overlay([img.img for img in reg_stack])
    plt.imshow(ol)
    plt.show()

    print("Done!")
