"""Process fails with large WSIs due to an integer overflow error somewhere in ITK."""
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
sys.path.insert(0, "/home/chaim/dev/my_openslide/")
import openslide

import viz

# Note also groupwise registration exists!
"""To prepare WSIs either: 
1. use openslide.
2. use eg: 'tiffcp -c lzw BI14N7189_081320181246.tif,12 moving.tif'. 
WSIs need to be stripped, not tiled. Associated images should be dropped by the loader, but we should isolate the 
image we want just in case."""


# # Load
root = "../data/in/"
# Test data
# fixed = "./data/test.png"
# moving = "./data/test2.png"
# Large data
fixed = "../data/in/reg1.png"
moving = "../data/in/reg2.png"

reg1 = sitk.ReadImage(fixed)
reg2 = sitk.ReadImage(moving)

# WSI
# wsi1 = openslide.OpenSlide(root + "BUI_52727_020_stain1_0000_00_00_MYC_0000_00_00_JH8718.ndpi")
# wsi2 = openslide.OpenSlide(root + "BUI_52727_020_stain2_0000_00_00_Ki-67_2017_02_17_OE4.ndpi")
#
# reg1 = np.array(wsi1.read_region(location=(0, 0), level=0, size=wsi1.dimensions))
# reg2 = np.array(wsi2.read_region(location=(0, 0), level=0, size=wsi2.dimensions))
#
# reg1 = sitk.GetImageFromArray(reg1, isVector=True)
# reg2 = sitk.GetImageFromArray(reg2, isVector=True)

# Convert our data to some operable format.
reg1 = sitk.VectorIndexSelectionCast(reg1, 0, sitk.sitkFloat32)
reg2 = sitk.VectorIndexSelectionCast(reg2, 0, sitk.sitkFloat32)

########### Functional api ###############
# result = sitk.Elastix(reg1,
#                       reg2,
#                       "affine")

######### Object Oriented api ############
# Initialize transform calculator
elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(reg1)
elastixImageFilter.SetMovingImage(reg2)

# Construct transformation queue
parameterMapVector = sitk.VectorOfParameterMap()
params = sitk.GetDefaultParameterMap("translation")
params['NumberOfResolutions'] = ['8',]
params['NumberOfSpatialSamples'] = ['16384',]
params['NumberOfSamplesForExactGradient'] = ['16384',]
params['MaximumNumberOfIterations'] = ['1000',]
print(dict(params))
parameterMapVector.append(params)
# parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
# parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.PrintParameterMap()

# Run
elastixImageFilter.Execute()
transform = elastixImageFilter.GetTransformParameterMap()
result = elastixImageFilter.GetResultImage()

# Save results
reg1 = sitk.Cast(reg1, sitk.sitkUInt8)
reg2 = sitk.Cast(reg2, sitk.sitkUInt8)
result = sitk.Cast(result, sitk.sitkUInt8)
overlay = viz.overlay([reg1, reg2])

groupfolder = "../data/out/big/"
version = str(2)
outfolder = os.path.join(groupfolder, version)
os.makedirs(outfolder, exist_ok=False)

sitk.WriteImage(reg1, os.path.join(outfolder, "reg1test.jpg"))
sitk.WriteImage(reg2, os.path.join(outfolder, "reg2test.jpg"))
sitk.WriteImage(result, os.path.join(outfolder, "result.jpg"))
cv2.imwrite(os.path.join(outfolder, "overlay.png"), overlay)

# Document experiment
with open(os.path.join(groupfolder, "doc.txt"), "a") as doc:
    doctext = "\n%s: increased MaximumNumberOfIterations to hopefully increase convergence." % outfolder
    # doctext = "\n%s: sitk, default translation params, with the following increased: NumberOfResolutions, NumberOfSpatialSamples, NumberOfSamplesForExactGradient." % outfolder
    doc.write(doctext)



# basic_transform = sitk.Euler2DTransform()
# basic_transform.SetTranslation((2,3))
#
# sitk.WriteTransform(basic_transform, 'euler2D.tfm')
# read_result = sitk.ReadTransform('euler2D.tfm')
#
# assert(str(type(read_result) != type(basic_transform)))
