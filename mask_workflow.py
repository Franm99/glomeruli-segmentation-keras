from unet_model import unet_model
import cv2
import numpy as np
from tensorflow.keras.utils import normalize
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from skimage import io
from PIL import Image
from typing import List
import time
import pandas as pd

PATCH_SIZE = 256


def get_model():
    return unet_model(PATCH_SIZE, PATCH_SIZE, 1)


def get_ims_path(ims_dir : str) -> List[str]:
    return [ims_dir + i for i in os.listdir(ims_dir) if i.endswith('.png')]


def get_mask(im : np.array, patch_org_size : int, format="gray") -> List[np.array]:
    [h, w, _] = im.shape
    mask = [np.zeros((h, w), dtype=bool)]
    if format != "gray":
        mask = mask * 3
    for x in range(0, w, patch_org_size):
        if x + patch_org_size >= w:
            x = w - patch_org_size
        for y in range(0, h, patch_org_size):
            if y + patch_org_size >= h:
                y = h - patch_org_size

            # 2.2. Get sub-patch in original size
            patch = im[y:y + patch_org_size, x:x + patch_org_size, :]

            # 2.3. U-Net works with (256,256) grayscale images:
            # TODO: instead of converting RGB to grayscale, use different channels (RGB or HSV) and check results
            patch_channels = get_channels(patch, format=format)  # Change to desired format
            # patch = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY),
            #                    (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)

            for ch, patch_ch in enumerate(patch_channels):
            # 3. Use median filter on histogram to find non-tissue patches
                counts, bins = np.histogram(patch_ch.flatten(), list(range(256 + 1)))
                counts.sort()
                median = np.median(counts)
                if median <= 3.0:
                    # If sub-patch does not contain tissue, prediction mask is filled with 0's
                    prediction_rs = np.zeros((patch_org_size, patch_org_size), dtype=np.uint8)
                else:
                    # 4. Use U-net to predict masks for sub-patches containing tissue.
                    # 4.1. Make prediction (firstly, normalize dimensions)
                    patch_input = np.expand_dims(normalize(np.array([patch_ch]), axis=1), 3)
                    # TODO: Fix problem here: always getting same channel
                    prediction = (model.predict(patch_input)[:, :, :, 0] > 0.5).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (patch_org_size, patch_org_size),
                                                    interpolation=cv2.INTER_AREA)

                # 4.2. Upload patch mask using bool to easier merge redundant patches
                mask[ch][y:y + patch_org_size, x:x + patch_org_size] = np.logical_or(
                    mask[ch][y:y + patch_org_size, x:x + patch_org_size],
                    prediction_rs.astype(np.bool))

        for i in range(len(mask)):
            plt.figure()
            plt.imshow(mask[i], cmap="gray")
        plt.show()
        plt.close()

    return [mask[ch].astype(np.uint8) for ch in range(len(mask))]


def get_channels(patch : np.array, format : str ="gray"):
    if format == "gray":
        return [cv2.resize(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY),
                           (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)]
    elif format == "hsv" or format == "HSV":
        patch = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_RGB2HSV),
                           (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
        return [patch[:, :, i] for i in range(patch.shape[2])]
    elif format == "rgb" or format == "RGB":
        patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
        return [patch[:, :, i] for i in range(patch.shape[2])]


if __name__ == '__main__':
    # 1. Prepare workflow
    # 1.1. Get U-Net model
    model = get_model()

    # 1.2. Load pre-trained weights
    weights_file = 'mitochondria_test.hdf5'
    model.load_weights(weights_file)

    # 1.3. Find tissue patches to work with and save their filenames
    ims_path = 'myImages/'
    ims_names = get_ims_path(ims_path)

    # 2. Get resized sub-patches for each tissue patch (loop)
    reduction_ratio = 5 # TODO: modify
    patch_org_size = PATCH_SIZE * reduction_ratio
    masks = []
    for im_name in ims_names:
        # 2.1. Using OpenCV to read images in array format
        im = cv2.cvtColor(cv2.imread(im_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = get_mask(im, patch_org_size, format="hsv")
        masks.append(mask)

        # 6. Show results (end loop)
        plt.figure(figsize=(8, 8))
        plt.subplot(121)
        plt.title("Patch")
        plt.imshow(im)
        plt.subplot(122)
        plt.title("Mask")
        plt.imshow(mask, cmap="gray")
        plt.show()
        plt.close()

