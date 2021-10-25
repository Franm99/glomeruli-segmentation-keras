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
from utils import show_ims

PATCH_SIZE = 256


def get_model():
    """ returns the U-net model architecture for default input size."""
    return unet_model(PATCH_SIZE, PATCH_SIZE, 1)


def get_ims_path(ims_dir: str) -> List[str]:
    """
    Returns full paths to images (*.png format) in specified folder
    :param ims_dir: folder to search in
    :return: list of filenames
    """
    return [ims_dir + i for i in os.listdir(ims_dir) if i.endswith('.png')]


def get_mask(model, img: np.array, dim: int, im_format: str = "gray") -> List[np.array]:
    """
    Returns the predicted mask for a given renal tissue patch.
    The input image can not be fed directly to the U-net model, as it has bigger dimensions than expected.
    It is neccesary to divide the image into patches with (256, 256) size.
    The number of masks returned by this functions depends on the number of channels (1 or 3) of the desired format.
    :param img: original image
    :param dim: desired size for (squared) patches in original format. Later, it will be resized to default model dims.
    :param im_format: "gray", "RGB" or "HSV". Both RGB and HSV will return a list of 3 masks, one for each channel.
    :return:list of masks by channel.
    """
    [h, w, _] = img.shape
    # Initializing list of masks
    mask = [np.zeros((h, w), dtype=bool)]
    if im_format != "gray":
        mask = [np.zeros((h, w), dtype=bool) for _ in range(3)]

    # Loop through the whole in both dimensions
    for x in range(0, w, dim):
        if x + dim >= w:
            x = w - dim
        for y in range(0, h, dim):
            if y + dim >= h:
                y = h - dim

            # Get sub-patch in original size
            patch = img[y:y + dim, x:x + dim, :]
            patch_channels = get_channels(patch, im_format=im_format)  # Change to desired format

            for idx, patch_channel in enumerate(patch_channels):

                # Median filter applied on image histogram to discard non-tissue sub-patches
                counts, bins = np.histogram(patch_channel.flatten(), list(range(256+1)))
                counts.sort()
                median = np.median(counts)
                if median <= 3.:
                    # Non-tissue sub-patches automatically get a null mask
                    prediction_rs = np.zeros((dim, dim), dtype=np.uint8)
                else:
                    # Tissue sub-patches are fed to the U-net model for mask prediction
                    patch_input = np.expand_dims(normalize(np.array([patch_channel]), axis=1), 3)
                    prediction = (model.predict(patch_input)[:, :, :, 0] > 0.5).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (dim, dim),
                                               interpolation=cv2.INTER_AREA)

                # Final mask is composed by the sub-patches masks (boolean format)
                mask[idx][y:y+dim, x:x+dim] = \
                    np.logical_or(mask[idx][y:y+dim, x:x+dim], prediction_rs.astype(np.bool))
    return [mask[ch].astype(np.uint8) for ch in range(len(mask))] # Change datatype from np.bool to np.uint8


def get_channels(patch: np.array, im_format: str ="gray"):
    """
    Get channels from input image (patch) in the specified format
    :param patch: patch to get channels from
    :param im_format: "gray", "RGB", "HSV"
    :return: list of channels resized to default dimensions for the U-Net model
    """
    if im_format == "gray":
        return [cv2.resize(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY),
                           (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)]
    elif im_format == "hsv" or im_format == "HSV":
        patch = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_RGB2HSV),
                           (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
        return [patch[:, :, i] for i in range(patch.shape[2])]
    elif im_format == "rgb" or im_format == "RGB":
        patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
        return [patch[:, :, i] for i in range(patch.shape[2])]


def workflow_test(wfilename, ims_path, reduction_ratio=5):
    model = get_model()
    model.load_weights(wfilename)
    ims_names = get_ims_path(ims_path)
    patch_org_size = int(PATCH_SIZE * reduction_ratio)
    for im_name in ims_names:
        # 2.1. Using OpenCV to read images in array format
        im = cv2.cvtColor(cv2.imread(im_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = get_mask(model, im, patch_org_size, im_format="rgb")
        # TODO: save images to disk
        # 6. Show results (end loop)
        ims = [im] + mask
        fig = show_ims(ims, 1, len(ims))


# if __name__ == '__main__':
#     model = get_model()
#     weights_file = 'mitochondria_test.hdf5'
#     ims_path = 'myImages/'
#     workflow_test(model, weights_file, ims_path)


