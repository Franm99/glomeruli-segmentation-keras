"""
Author: Fran Moreno
Contact: fran.moreno.se@gmail.com
Date: 15/03/2022
"""
import cv2.cv2 as cv2
import glob
import numpy as np
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import Tuple, List


def contours2mask(im: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Get contour-labelled image and generate binary mask filling holes.

    **Source:** `Filling holes in an image using opencv-python
    <https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/>`_

    :param im: 3-channel (RGB) image with easy-to-find contours.
    :param color: color used for contours (RGB).
    :return: resulting binary mask (2D array)
    """
    ch_r = im[..., 0] == color[0]
    ch_g = im[..., 1] == color[1]
    ch_b = im[..., 2] == color[2]
    im_th = np.logical_and(ch_r, ch_g, ch_b).astype(np.uint8)

    # Flood-filled image
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Inverted flood-filled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine to get the foreground
    return im_th | im_floodfill_inv


def get_masks(target_path: str) -> None:
    """
    Read images with remarked contours and generate binary masks with filled holes.

    Images are saved to the output path.

    :return: None
    """
    contour_color = (0, 255, 0)
    ims_path = target_path
    output_path = os.path.join(target_path, 'masks')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    ims_names = glob.glob(ims_path + '/*')
    for im_name in tqdm(ims_names):
        im = cv2.cvtColor(cv2.imread(im_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = contours2mask(im, color=contour_color)
        output_name = os.path.join(output_path, os.path.basename(im_name))
        cv2.imwrite(output_name, mask)


def coordinates_over_images(target: str) -> None:
    """
    Draw centroids of know blobs over the masks to ease finding them.

    :return: None
    """
    ims_path = os.path.join(target, 'ims')
    xml_path = os.path.join(target, 'xml')
    output_path = os.path.join(target, 'marked_ims')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    ims_names = glob.glob(ims_path + '/*')
    xml_names = [os.path.join(xml_path, os.path.basename(i)[:-4] + '.xml') for i in ims_names]
    for im_name, xml_name in tqdm(zip(ims_names, xml_names), total=len(ims_names)):
        im = cv2.imread(im_name, cv2.IMREAD_COLOR)
        points = get_coords(xml_name)
        for point in points:
            im = cv2.circle(im, point, radius=10, color=(0, 0, 255), thickness=20)
        cv2.imwrite(os.path.join(output_path, os.path.basename(im_name)), im)


def get_coords(xml_name: str) -> List[Tuple[int, int]]:
    with open(xml_name, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")
    counts = bs_data.find("Counts").find_all("Count")
    p = []
    for count in counts:
        points = count.find_all('point')
        for point in points:
            p.append((int(point.get('X')), int(point.get('Y'))))
    return p


# if __name__ == '__main__':
    # get_masks()
    # coordinates_over_images()
