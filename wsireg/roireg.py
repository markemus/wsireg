"""Registers a region of interest for a set of whole slide images.

All images are stored in a dataframe to make keeping track of them easier. This should really be done with a class
instead, but the dataframe structure works very well for this application and is well supported. The functions in
this module are therefore similar to object methods for the dataframe, rather than proper functions.

roi_reg() will get you started."""
import glob
import os
import sys

import cv2
import numpy as np
sys.path.insert(0, "/home/chaim/dev/my_openslide")
import openslide
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
import imagetools.convertimage as ci

import registration as reg
import viz

from libtiff import TIFF
from sklearn.metrics import mean_absolute_error


def roi_reg(paths, topleft_x, topleft_y, shift_x, shift_y, level, test=False):
    """Registers a region of interest in a series of images. The region is defined on the first image path,
    which will also be the fixed image for registration.

    Image requirements:
    The images should represent the same piece of tissue.
    The images should use the same pair of stains for background and signal channels.

    Returns:
    rois["bg"], rois["fg"] are the final registered images."""
    rois = pd.DataFrame(data={"path": paths}, columns=[
        'path', 'wsi', 'downsample', 'region',
        'topleft_x', 'topleft_y', 'shift_x', 'shift_y',
        'elastix_trans', 'trans_shift_x', 'trans_shift_y', 'final_topleft_x', 'final_topleft_y',
        'elastix_final', 'bg', 'sig'])

    rois["wsi"] = rois["path"].apply(openslide.OpenSlide)
    rois["topleft_x"] = topleft_x
    rois["topleft_y"] = topleft_y
    rois["shift_x"] = shift_x
    rois["shift_y"] = shift_y

    rois["downsample"] = rois.apply(lambda x: np.array(x["wsi"].read_region(location=(0,0), level=level, size=x["wsi"].level_dimensions[level]))[:,:,:3], axis=1)
    rois["elastix_trans"], _, _ = reg.imgreg(paths=rois["downsample"], params=reg.build_params_translation(), pathmode="memory")

    # Get the translation to roughly align the ROIs in each image.
    rois["trans_shift_x"] = pd.Series([x.GetTransformParameterMap()[0]["TransformParameters"][0] for x in rois["elastix_trans"]], dtype=float)
    rois["trans_shift_y"] = pd.Series([x.GetTransformParameterMap()[0]["TransformParameters"][1] for x in rois["elastix_trans"]], dtype=float)

    rois["final_topleft_x"] = (rois["topleft_x"] + rois["trans_shift_x"] * rois.loc[0, "wsi"].level_downsamples[level]).astype(int)
    rois["final_topleft_y"] = (rois["topleft_y"] + rois["trans_shift_y"] * rois.loc[0, "wsi"].level_downsamples[level]).astype(int)

    # Extract the aligned ROIs and do the full registration.
    rois["region"] = rois.apply(lambda x: np.array(x["wsi"].read_region(location=(x["final_topleft_x"], x["final_topleft_y"]), level=0, size=(x["shift_x"], x["shift_y"])))[:,:,:3], axis=1)

    if test:
        rois["elastix_final"], rois["bg"], rois["sig"] = reg.imgreg(paths=rois["region"], params=reg.build_params_test(), pathmode="memory")
    else:
        rois["elastix_final"], rois["bg"], rois["sig"] = reg.imgreg(paths=rois["region"], params=reg.build_params_large_image(), pathmode="memory")

    return rois

def plot(df):
    """One click plotting function to dump all registration data into plots."""
    for elx in df["elastix_trans"]:
        plt.suptitle("Downsample")
        plt.imshow(sitk.GetArrayFromImage(elx.GetResultImage()))
        plt.show()

    for img in df["region"]:
        plt.suptitle("Region")
        plt.imshow(img)
        plt.show()

    for bg, sig in df[["bg", "sig"]].values:
        plt.suptitle("BG")
        plt.imshow(bg)
        plt.show()
        plt.suptitle("SIG")
        plt.imshow(sig)
        plt.show()

    plt.imshow(viz.overlay(list(df["sig"]), cmap="jhu_vibrant"))
    viz.add_cmap_colorbar(cmap="jhu_vibrant")
    plt.show()

    print("Registration complete.\nImage values:\n",
          df[["topleft_x", "topleft_y", "trans_shift_x", "trans_shift_y",
              "final_topleft_x", "final_topleft_y"]].to_string())

def save(df, outroot):
    tiff_bg = TIFF.open(os.path.join(outroot, "bg.tif"), mode="w")
    tiff_sig = TIFF.open(os.path.join(outroot, "sig.tif"), mode="w")

    # Data
    for i, img in enumerate(df["region"]):
        cv2.imwrite(os.path.join(outroot, f"{i}_roi.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    for img in df["bg"]:
        print("Saving layer...")
        tiff_bg.SetField(tag="SubFileType", _value=0)
        # tiff_bg.write_image(img)
        tiff_bg.write_tiles(img, tile_width=512, tile_height=512)
        print("Background channels saved.")

    for img in df["sig"]:
        print("Saving layer...")
        tiff_sig.SetField(tag="SubFileType", _value=0)
        # tiff_sig.write_image(img)
        tiff_sig.write_tiles(img, tile_width=512, tile_height=512)
        print("Signal channels saved.")
        
    tiff_bg.close()
    tiff_sig.close()

    # Vizualizations
    cv2.imwrite(os.path.join(outroot, "registered-0_to_1.png"), viz.overlay([df["bg"][0], df["bg"][1]]))
    cv2.imwrite(os.path.join(outroot, "registered-1_to_2.png"), viz.overlay([df["bg"][1], df["bg"][2]]))
    cv2.imwrite(os.path.join(outroot, "registered-0_to_2.png"), viz.overlay([df["bg"][0], df["bg"][2]]))

    cv2.imwrite(os.path.join(outroot, "mean_bg.png"), viz.mean(df["bg"]))
    cv2.imwrite(os.path.join(outroot, "colorized.png"), viz.overlay(list(df["sig"]), cmap="jhu_vibrant"))

    # Metrics
    with open(os.path.join(outroot, "metrics.txt"),mode="w") as metrics:
        metrics.write("Mean absolute error:\n")
        for image in rois.loc[1:, "bg"]:
            metrics.write(str(mean_absolute_error(rois.loc[0, "bg"].flatten(), image.flatten())) + "\n")

def invert(df):
    df = df.copy()
    df["sig"] = df["sig"].apply(lambda x: ci.scale_by_max(1/(ci.scale_by_max(x, 254)+1), 255).astype(np.uint8))
    df["bg"] = df["bg"].apply(lambda x: ci.scale_by_max(1/(ci.scale_by_max(x, 254)+1), 255).astype(np.uint8))

    return df

# Main
# paths = glob.glob("../data/in/BUI*.ndpi")
paths = glob.glob("../data/in/halo/*.tif")

timestamp = ci.timestamp()
rois = roi_reg(paths=paths, topleft_x=25000, topleft_y=26000, shift_x=5000, shift_y=6000, level=6,
               test=False)

plot(rois)

# Save
outroot = f"../data/out/roireg/{timestamp}/"
outroot_inv = os.path.join(outroot, "invert")
os.mkdir(outroot)
save(rois, outroot)

os.mkdir(outroot_inv)
save(invert(rois), outroot_inv)

# TODO test using macenko_norm instead of scale_by_max

print(f"Process completed at: {ci.timestamp()}.")
