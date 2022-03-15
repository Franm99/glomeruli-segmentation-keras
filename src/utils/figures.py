"""
Author: Fran Moreno
Contact: fran.moreno.se@gmail.com
Date: 15/03/2022
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
from skimage.measure import regionprops, label
import os


def find_blobs_centroids(img: np.ndarray) -> List[Tuple[float, float]]:
    """
    This function implements region labelling and region properties extraction to find, label and compute centroids of
    each blob in binary images.

    **NOTE:** In this context, blob = glomerulus.
    """
    img_th = img.astype(bool)
    img_labels = label(img_th)
    img_regions = regionprops(img_labels)
    centroids = []
    for props in img_regions:
        centroids.append((int(props.centroid[0]), int(props.centroid[1])))  # (y, x)
    return centroids


def t_find_blob_centroids():
    """
    *Test function*

    Testing unit for find_blob_centroids() function.
    """
    import cv2.cv2 as cv2

    data_dir = "../../data"
    mask_sample = os.path.join(data_dir, "segmenter", "HE", "gt", "masks", "20B0004711 HE_x4800y14400s3200.png")

    im = cv2.imread(mask_sample, cv2.IMREAD_GRAYSCALE)
    centroids = find_blobs_centroids(im)
    print(centroids)
    plt.imshow(im, cmap="gray")
    for (cy, cx) in centroids:
        plt.plot(cx, cy, ".r")
    plt.show()


def compute_rows_cols(num: int, m: int, n: int) -> Tuple[int, int]:
    """
    Compute number of rows and columns for subplot.

    :param num: number of subfigures
    :param m: rows preference
    :param n: columns preference
    :return: resulting number of rows and colums
    """
    if m is None:
        m = math.sqrt(num)
        if n is None:
            n = math.ceil(num / m)
        else:
            m = math.ceil(num / n)
    else:
        if n is None:
            n = math.ceil(num / m)
        else:
            m = math.ceil(num / n)
    m, n = max(1, m), max(1, n)
    m, n = math.ceil(m), math.ceil(n)
    return m, n


def show_ims(imgs: List[np.ndarray], m: Optional[int] = None, n: Optional[int] = None,
             title: str = "", subtitles: Optional[List[str]] = None):
    """
    Show a group of images in subplots of the same figure.

    :param imgs: images to show
    :param m: number of rows preference
    :param n: number of columns preference
    :param title: global title
    :param subtitles: list of subtitles for subplots
    :return: None
    """
    N = len(imgs)

    m, n = compute_rows_cols(N, m, n)
    fig = plt.figure(figsize=(12, 6))
    for i in range(1, N + 1):
        plt.subplot(m, n, i)
        if len(imgs[i - 1].shape) == 3:
            plt.imshow(imgs[i - 1])
        else:
            plt.imshow(imgs[i - 1], cmap="gray")
        if subtitles is not None:
            plt.title(subtitles[i - 1])
    fig.suptitle(title)
    plt.show()

    return fig


def show_masked_ims(imgs: List[np.ndarray], masks: List[np.ndarray],
                    m: Optional[int] = None, n: Optional[int] = None,
                    title: str = "", subtitles: Optional[List[str]] = None):
    """
    Show a group of images in subplots of the same figure.

    :param imgs: images to show
    :param m: number of rows preference
    :param n: number of columns preference
    :param title: global title
    :param subtitles: list of subtitles for subplots
    :return: None
    """
    N = len(imgs)

    m, n = compute_rows_cols(N, m, n)
    fig = plt.figure(figsize=(12, 6))
    for i in range(1, N + 1):
        plt.subplot(m, n, i)
        if len(imgs[i - 1].shape) == 3:
            plt.imshow(imgs[i - 1][:, :, 0], cmap="gray")
            plt.imshow(masks[i - 1][:, :, 0], cmap="jet", alpha=0.3)
        else:
            plt.imshow(imgs[i - 1], cmap="gray")
            plt.imshow(masks[i - 1], cmap="jet", alpha=0.3)
        if subtitles is not None:
            plt.title(subtitles[i - 1])
    fig.suptitle(title)
    plt.show()

    return fig


# if __name__ == '__main__':
#     t_find_blob_centroids()
