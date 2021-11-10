from unet_model import unet_model
import os, glob
import cv2.cv2 as cv2
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import normalize
import random
from utils import computeRowsCols
import matplotlib.pyplot as plt
from typing import List, Optional

PATCH_SIZE = 256
IMGS_PATH = "/home/francisco/Escritorio/DataGlomeruli/patches_ims"
MASKS_PATH = "/home/francisco/Escritorio/DataGlomeruli/patches_masks"


def show_ims(imgs: List[np.ndarray],
                    m: Optional[int] = None, n: Optional[int] = None,
                    title: str = "", subtitles: Optional[List[str]] = None):
    """
    Show a group of images in subplots of the same figure
    :param imgs: images to show
    :param m: number of rows preference
    :param n: number of columns preference
    :param title: global title
    :param subtitles: list of subtitles for subplots
    :return: None
    """
    N = len(imgs)
    m, n = computeRowsCols(N, m, n)
    fig = plt.figure(figsize=(12,6))
    for i in range(1, N + 1):
        plt.subplot(m, n, i)
        if len(imgs[i-1].shape) == 3:
            plt.imshow(imgs[i-1][:, :, 0], cmap="gray")
        else:
            plt.imshow(imgs[i-1], cmap="gray")
        if subtitles is not None:
            plt.title(subtitles[i - 1])
    fig.suptitle(title)
    plt.show()

    return fig


def load_data(fpath: str):
    return [cv2.imread(im, cv2.IMREAD_GRAYSCALE) for im in glob.glob(fpath + "/*")]


def get_model():
    return unet_model(PATCH_SIZE, PATCH_SIZE, 1)


if __name__ == '__main__':
    model = get_model()
    weights_file = "last.hdf5"
    model.load_weights(weights_file)

    ims = load_data(IMGS_PATH)
    masks = load_data(MASKS_PATH)

    ims = np.expand_dims(normalize(np.array(ims), axis=1), 3)
    masks = np.expand_dims(normalize(np.array(masks), axis=1), 3)

    ypred = model.predict(ims)

    while True:
        indexes = random.sample(range(len(ypred)), 4)
        ims_sample = [ims[index] for index in indexes]
        masks_sample = [masks[index] for index in indexes]
        pred_sample = [ypred[index] for index in indexes]
        show_ims(ims_sample+masks_sample+pred_sample, 3, 4)
        plt.close()


    ypred_th = ypred > 0.5

    intersection = np.logical_and(masks, ypred_th)
    union = np.logical_or(masks, ypred_th)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU score is ", iou_score)






