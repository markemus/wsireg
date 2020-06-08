"""[Whole slide] images are processed using two data formats: Patches, and Quilts.
Patches are (overlapping) regions taken from an image.
Quilts are formed by knitting patches together.

For now, all patches and quilts from a single processing routine should have
the same dimensionality so that patches from different stages can be
compared to one another.

This module should interface smoothly with bilinear.py.

Related papers:
Patch-Based Nonlinear Image Registration for Gigapixel Whole Slide Images: https://ieeexplore.ieee.org/document/7335576
Dynamic registration for gigapixel serial whole slide images: https://ieeexplore.ieee.org/document/7950552"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.util import view_as_windows

#TODO if we got the first half-patches as the first patches, we could construct FULL non-overlapping quilts without needing shifts and dead zone buffers.
#TODO convert detectFeatures, alignFeatures to class?
def detectFeatures(im1, im2, scaleFactor=2, nlevels=10):
    """Find matching features with high probabilities in a pair of images using ORB.
    Returns two arrays containing matching points in the image, index aligned."""
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    # Detect ORB features (can use pyramid)
    orb = cv2.ORB_create(MAX_FEATURES, scaleFactor=scaleFactor, nlevels=nlevels)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    # If no features detected, alignment will throw exception.
    if len(keypoints1) and len(keypoints2):
        # Match features
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, mask=None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance)

        # Remove not-so-good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, outImg=None)
        # cv2.imwrite(outfolder+"matches.jpg", imMatches)

        # Extract locations of good matches
        points1 = np.zeros((len(matches), 2), np.float32)
        points2 = np.zeros((len(matches), 2), np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
    else:
        points1, points2 = [], []

    return points1, points2

def alignFeatures(im1, im2, scaleFactor=2, nlevels=10):
    """Feature based alignment.
    https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/"""
    points1, points2 = detectFeatures(im1, im2, scaleFactor=scaleFactor, nlevels=nlevels)

    if len(points1) and len(points2):
        # Find homography
        h, mask = cv2.findHomography(points2, points1, method=cv2.RANSAC)
    else:
        h, mask = None, None

    return h, mask

def calcPlateMorphs(patches):
    """Find homographies to register each patch in a set of images. Call after initial high level registration.
    patches: (row, col, stack, l_row, l_col, l_channel)"""
    morphs = np.zeros(shape=(patches.shape[:3]) + (3,3), dtype=np.float64)
    morphs[:,:,:] = np.identity(3)
    failed = 0
    worked = 0

    for i, row in enumerate(patches):
        for j, stack in enumerate(row):
            # plates are the individual images to be aligned to the single fixed image (row[0])
            fixed_plate = stack[0]
            # reg[i, j, 0] = fixed_plate
            for k, plate in enumerate(stack[1:], start=1):
                # calculate morph from fixed_plate
                H, _ = alignFeatures(fixed_plate, plate)
                if H is None:
                    # H = np.identity(3, np.uint8)
                    failed += 1
                else:
                    morphs[i, j, k] = H
                    worked += 1
                    # Store warp coindexed to patches
    print("Calculated plate-wise morphs for patches. Worked: %s Failed: %s" % (worked, failed))
    return morphs

def applyMorphs(patches, morphs):
    """Applies a set of morphs to a set of patches. Patches and morphs must be coindexed!
    patches: (row, col, stack, l_row, l_col, l_channel)
    morphs: (row, col, stack, x, y)"""
    #TODO add ignore=1 param that sets how many initial plates to ignore.
    warped = np.zeros(shape=patches.shape, dtype=patches.dtype)

    for i, row in enumerate(patches):
        for j, stack in enumerate(row):
            # plates are the individual images to be aligned to the single fixed image (row[0])
            fixed_plate = stack[0]
            warped[i, j, 0] = fixed_plate
            for k, plate in enumerate(stack[1:], start=1):
                # apply warp to plate
                H = morphs[i,j,k]
                warped[i, j, k] = ensure3d(cv2.warpPerspective(plate, H, dsize=(plate.shape[1], plate.shape[0])))
    print("Morphs applied to patches.")
    return warped

def ensure3d(arr):
    """Turns 2 arrays into 3d arrays with a len(1) 3rd dimension. Allows functions to be
    written for both grayscale and color images."""
    if len(arr.shape) == 3:
        return arr
    elif len(arr.shape) == 2:
        return arr.reshape((arr.shape[0], arr.shape[1], 1))

def buffered_id_map(shape, buffer):
    """Creates an identity deformation-result field for an image, mapping each point to itself.
    Map will be extended by buffer in all 4 directions."""
    x_vector = np.linspace(-buffer, shape[1]-1+buffer, shape[1]+(2*buffer), dtype=np.float32)
    y_vector = np.linspace(-buffer, shape[0]-1+buffer, shape[0]+(2*buffer), dtype=np.float32)
    x_map, y_map = np.meshgrid(x_vector, y_vector)

    return x_map, y_map

def calc_id_patches(img_shape, patch_size):
    """Creates a patches object that is an id_map for a given image and patches.
    patch_shape: (height, width)."""
    step_size = int(patch_size/2)
    step = [step_size, step_size, 1]
    buffer = step_size
    window_shape = (patch_size + 2*buffer, patch_size + 2*buffer, 1)

    # We'll need a buffer for each patch, so we need a corresponding buffer for the whole image.
    xv, yv = buffered_id_map(img_shape, buffer=buffer)
    # First plate is a placeholder- it will not be operated on by applyMorphs since it ignores the first layer in the stack.
    id_stack = np.concatenate((xv[:, :, None], xv[:, :, None], yv[:, :, None]), axis=-1)
    # Patches are buffered so we can map to local points beyond the edge of the patch. Otherwise, cv2 will map them to 0- which is NOT the identity in this format!
    id_patches = view_as_windows(id_stack, window_shape=window_shape, step=step)

    return id_patches

def test():
    debug = True
    # debug = False
    root = "../data/in/"

    # WSI
    # wsi1 = openslide.OpenSlide(root + "BUI_52727_020_stain1_0000_00_00_MYC_0000_00_00_JH8718.ndpi")
    # wsi2 = openslide.OpenSlide(root + "BUI_52727_020_stain2_0000_00_00_Ki-67_2017_02_17_OE4.ndpi")
    #
    # reg1 = np.array(wsi1.read_region(location=(45000, 40000), level=0, size=(20000, 10000)))
    # reg2 = np.array(wsi2.read_region(location=(40000, 40000), level=0, size=(20000, 10000)))
    reg1 = cv2.imread(root + "reg1.png")
    reg2 = cv2.imread(root + "reg2.png")

    if debug:
        plt.imshow(reg1)
        plt.show()
        plt.imshow(reg2)
        plt.show()

    # Stage One: Low-precision feature alignment
    h, _ = alignFeatures(reg1, reg2)
    height, width = reg1.shape[:2]
    reg2_aligned = cv2.warpPerspective(reg2, h, (width, height))

    # Stage Two: Calculate patch-level registrations
    w_shape = (1000, 1000, 4)
    w_step = (500, 500, 4)

    stack1 = np.concatenate((reg1, reg2_aligned), axis=-1)
    patches = view_as_windows(stack1, window_shape=w_shape, step=w_step)
    morphs = calcPlateMorphs(patches)

    # Stage Three: Compute patch-level DVFs
    # Ultimately we want to stitch using bilinear interpolation of local maps.
    # We'll need a buffer for each patch, so we need a corresponding buffer for the whole image.
    # xv, yv = buffered_id_map(reg2_aligned.shape, buffer=500)
    # id_w_shape = (2000, 2000, 1)
    # id_w_step = (500, 500, 1)
    #
    # # First xv is a placeholder- it will not be operated on by applyMorphs since it ignores the first layer in the stack.
    # id_stack = np.concatenate((xv[:, :, None], xv[:, :, None], yv[:, :, None]), axis=-1)
    # # shapes are buffered so we can map to local points beyond the edge of the patch. Otherwise, cv2 will map them to 0- which is NOT the identity in this format!
    # id_patches = view_as_windows(id_stack, window_shape=id_w_shape, step=id_w_step)
    id_patches = calc_id_patches(img_shape=reg2_aligned.shape, patch_size=1000)

    # We also copy the morph so it will be applied to both xv and yv- again, first layer is ignored by applyMorphs.
    map_morphs = np.append(morphs, morphs[:, :, 1, None], axis=2)
    # Apply transformation to identity deformation-result fields.
    reg_patches = applyMorphs(id_patches, map_morphs)

    # # Calculate deformation field: result - id: [d = r - i]
    # map_patches = reg_patches - id_patches
    # # Restrict to actual patch regions (remove buffers).
    # fitted_map_patches = map_patches[:, :, 1:, 500:1500, 500:1500, :]
    # fitted_reg_patches = reg_patches[:, :, 1:, 500:1500, 500:1500, :]

    return patches, reg_patches, id_patches
