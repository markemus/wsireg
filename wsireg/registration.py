import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
sys.path.insert(0, "../../my_openslide/")
import openslide

import imagetools.convertimage as ci

import deconvolution as dec


def build_params_test():
    """Cheap parameter set for test runs."""
    params = sitk.GetDefaultParameterMap('translation')
    return params

# TODO store params in text files.
def build_params_translation():
    params = sitk.GetDefaultParameterMap('translation')
    params['FixedImagePyramid'] = ('FixedRecursiveImagePyramid',)
    params['MovingImagePyramid'] = ('MovingRecursiveImagePyramid',)
    params['NumberOfResolutions'] = ('16',)
    params['NumberOfSpatialSamples'] = ('5000',)
    params['MaximumNumberOfIterations'] = ('512',)

    return params

def build_params_large_image():
    """For large test images 10k x 20k."""
    parameterMapVector = sitk.VectorOfParameterMap()

    params1 = build_params_translation()

    params2 = sitk.GetDefaultParameterMap('bspline')
    params2['BSplineTransformSplineOrder'] = ('1',)
    params2['FixedImagePyramid'] = ('FixedRecursiveImagePyramid',)
    params2['MovingImagePyramid'] = ('MovingRecursiveImagePyramid',)
    del params2['GridSpacingSchedule']
    params2['Metric'] = ('AdvancedMattesMutualInformation',)
    del params2['Metric0Weight']
    del params2['Metric1Weight']
    params2['NumberOfResolutions'] = ('16',)
    params2['NumberOfSpatialSamples'] = ('5000',)
    params2['MaximumNumberOfIterations'] = ('512',)
    params2['NumberOfJacobianMeasurements'] = ('10000',)

    parameterMapVector.append(params1)
    parameterMapVector.append(params2)

    return parameterMapVector

def build_params_whole_slide_image():
    """For large test images 10k x 20k."""
    parameterMapVector = sitk.VectorOfParameterMap()

    params1 = sitk.GetDefaultParameterMap('translation')
    params1['FixedImagePyramid'] = ('FixedRecursiveImagePyramid',)
    params1['MovingImagePyramid'] = ('MovingRecursiveImagePyramid',)
    params1['NumberOfResolutions'] = ('16',)
    params1['NumberOfSpatialSamples'] = ('5000',)
    params1['MaximumNumberOfIterations'] = ('512',)

    parameterMapVector.append(params1)

    return parameterMapVector

def register(fixed, moving, parameterMapVector):
    """Uses SimpleElastix to find a translation transform of a very large image.
    For smaller images, just use the default params."""
    elastixImageFilter = sitk.ElastixImageFilter()

    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(fixed))
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(moving))
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()

    return elastixImageFilter

def imgreg(paths, params, pathmode="file"):
    """Deconvolve, then register all signal channels to a (arbitrary) background channel."""
    # params = build_params_large_image()
    # TODO train a new net (old one is from an old version of TF?) and reset new_net to False.
    sig, bg, _ = dec.deconv_all(paths, pathmode=pathmode, epochs=100, verbose=False, new_net=True)
    elastix = [register(bg[0], image, params) for image in bg]
    # TODO-DONE scale_by_max or macenko
    bg_out = [(ci.scale_by_max(sitk.GetArrayFromImage(elx.GetResultImage())) * 255).astype(np.uint8) for elx in elastix]
    # sig_out = [sitk.GetArrayFromImage(sitk.Transformix(sitk.GetImageFromArray(image), elx.GetTransformParameterMap())) for image, elx in zip(sig, elastix)]
    sig_out = [(ci.scale_by_max(sitk.GetArrayFromImage(sitk.Transformix(sitk.GetImageFromArray(image), elx.GetTransformParameterMap()))) * 255).astype(np.uint8) for image, elx in zip(sig, elastix)]

    return elastix, bg_out, sig_out

def wsireg(paths):
    """WARNING: buggy, crashy, and memory hoggy."""
    print("Beginning registration...")
    params = build_params_whole_slide_image()
    sig = []
    for path in paths:
        wsi = openslide.OpenSlide(path)
        image = np.array(wsi.read_region(location=(0, 0), level=0, size=wsi.dimensions))
        print("Image loaded.")
        signal = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        print("Image converted to grayscale.")
        del image
        sig.append(signal)

    # SIGSEGV after 4gb of RAM usage (computingJacobians late in iteration cycle).
    print("Registering images...")
    registered = register(sig[0], sig[1], params).getResultImage()

    print("Registration complete!")
    return registered

# def wsireg_with_deconv(paths):
#     print("Beginning registration...")
#     params = build_params_whole_slide_image()
#     images = []
#     for path in paths:
#         wsi = openslide.OpenSlide(path)
#         image = np.array(wsi.read_region(location=(0, 0), level=0, size=wsi.dimensions))
#         print("Image loaded.")
#         images.append(image)
#
#     print("Deconvolving images...")
#     bg, sig = dec.deconv_all(images, pathmode="memory", epochs=100, verbose=False)
#
#     print("Deconv complete!")
