""" Script to get filled masks from manually segmented glomeruli using contours."""
import glob
import os
import numpy as np
import parameters as params
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm


def contours2mask(im: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Get contour-labelled image and generate binary mask filling holes.
    Source: https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    :param im: 3-channel (RGB) image with easy-to-find contours.
    :param color: color used for contours (RGB).
    :return: resulting binary mask (2D array)
    """
    chR = im[..., 0] == color[0]
    chG = im[..., 1] == color[1]
    chB = im[..., 2] == color[2]
    im_th = np.logical_and(chR, chG, chB).astype(np.uint8)

    # Flood-filled image
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Inverted flood-filled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine to get the foreground
    return im_th | im_floodfill_inv


def get_masks():
    contour_color = (0, 255, 0)
    ims_path = params.DATASET_PATH + '/marked_ims'
    output_path = params.DATASET_PATH + '/gt/masks'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    ims_names = glob.glob(ims_path + '/*')
    for im_name in tqdm(ims_names):
        im = cv2.cvtColor(cv2.imread(im_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = contours2mask(im, color=contour_color)
        output_name =  os.path.join(output_path, os.path.basename(im_name))
        cv2.imwrite(output_name, mask)


if __name__ == '__main__':
    get_masks()
