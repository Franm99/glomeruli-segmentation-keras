""" Load weights and test performance and improvements on the way predictions are processed. """
from utils import browse_path, show_ims
from unet_model import unet_model
import keras
import parameters as params
import os
import glob
import numpy as np
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import normalize
from typing import List

output_folder = browse_path()
output_dir = os.path.join('output', output_folder)


def get_model() -> keras.Model:
    """ return: U-Net Keras model (TF2 version)"""
    return unet_model(params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE, 1)


def load_model() -> keras.Model:
    model = get_model()
    model_weights = os.path.join(output_dir, "weights/model.hdf5")
    # model = keras.models.load_model(model_weights)
    model.load_weights(model_weights)
    return model


def load_test_set():
    test_folder = os.path.join(output_dir, "test_pred")
    test_names = os.listdir(test_folder)
    preds = [cv2.imread(os.path.join(test_folder, name), cv2.IMREAD_GRAYSCALE) for name in test_names]
    return [load_im(i) for i in test_names], test_names, preds


def load_gt_set(fnames: List[str]):
    gt_folder = os.path.join(params.DATASET_PATH, "gt/masks")
    return [cv2.imread(os.path.join(gt_folder, name), cv2.IMREAD_GRAYSCALE) for name in fnames]


def load_im(fname: str):
    fpath = os.path.join(params.DATASET_PATH, "ims", fname)
    return cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)


def _get_pred_mask(im, dim, model, th: float):
    """ """
    [h, w] = im.shape
    # Initializing list of masks
    mask = np.zeros((h, w), dtype=bool)
    # pred_masks = []
    # Loop through the whole in both dimensions
    for x in range(0, w, dim):
        if x + dim >= w:
            x = w - dim
        for y in range(0, h, dim):
            if y + dim >= h:
                y = h - dim
            # Get sub-patch in original size
            patch = im[y:y + dim, x:x + dim]

            # Median filter applied on image histogram to discard non-tissue sub-patches
            counts, bins = np.histogram(patch.flatten(), list(range(256 + 1)))
            counts.sort()
            median = np.median(counts)
            if median <= 3.:
                # Non-tissue sub-patches automatically get a null mask
                prediction_rs = np.zeros((dim, dim), dtype=np.uint8)
                # prediction = np.zeros((dim, dim), dtype=np.uint8)
            else:
                # Tissue sub-patches are fed to the U-net model for mask prediction
                patch = cv2.resize(patch, (params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE), interpolation=cv2.INTER_AREA)
                patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                prediction = model.predict(patch_input)[:, :, :, 0]
                # prediction = prediction[0, :, :]
                # pass
                prediction = (model.predict(patch_input)[:, :, :, 0] >= th).astype(np.uint8)
                prediction_rs = cv2.resize(prediction[0], (dim, dim), interpolation=cv2.INTER_AREA)

            # pred_masks.append(prediction)
    # return pred_masks
                # Final mask is composed by the sub-patches masks (boolean array)
            mask[y:y + dim, x:x + dim] = np.logical_or(mask[y:y + dim, x:x + dim], prediction_rs.astype(bool))
    return mask.astype(np.uint8)  # Change datatype from np.bool to np.uint8



if __name__ == '__main__':
    model = load_model()
    test_ims, test_names, preds = load_test_set()
    gt_ims = load_gt_set(test_names)
    # plt.imshow(test_ims[0])
    th = 0.9
    predictions = []
    org_size = int(params.UNET_INPUT_SIZE * params.RESIZE_RATIOS[0])
    for im, name, gt, pred in zip(test_ims, test_names, gt_ims, preds):
        new_pred = _get_pred_mask(im, org_size, model, th)
        # show_ims(preds, title=name)
        plt.figure()
        plt.subplot(131)
        plt.imshow(gt, cmap="gray")
        plt.title("Ground-truth")
        plt.subplot(132)
        plt.imshow(pred, cmap="gray")
        plt.title("Previous prediction (th={0.5})")
        plt.subplot(133)
        plt.imshow(new_pred*255, cmap="gray")
        plt.title("New prediction (th={})".format(th))
        plt.show()
        plt.close()
