""" Load weights and test performance and improvements on the way predictions are processed. """
import openslide  # TODO include OS lecture to avoid import errors
import keras
import os
import glob
import numpy as np
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import normalize
from typing import List
import re
from tqdm import tqdm
from PIL import Image

from src.utils.utils import browse_path, show_ims
from src.model.keras_models import simple_unet
import src.parameters as params

output_folder = browse_path()
output_dir = os.path.join('output', output_folder)


def get_model() -> keras.Model:
    """ return: U-Net Keras model (TF2 version)"""
    return simple_unet(params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE, 1)


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


def predict(model, ims, org_size, th):
    return [_get_pred_mask(im, org_size, model, th) for im in tqdm(ims)]


def validate_model():
    """ Load model weights and make predictions, optionally varying the output threshold. """
    model = load_model()
    test_ims, test_names, preds = load_test_set()
    gt_ims = load_gt_set(test_names)
    # plt.imshow(test_ims[0])
    th = 0.7
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


def get_coords(name: str):
    x = int(re.search(r'x([0-9]*)', name).group(1))
    y = int(re.search(r'y([0-9]*)', name).group(1))
    return x, y


def mask_assembly():
    model = load_model()
    par_dir = "/home/francisco/Escritorio/Biopsies"
    filepath = os.path.join(par_dir, os.listdir(par_dir)[0])
    wsi_name = os.path.basename(filepath).split('.')[0]  # Name without extension

    def load_wsi_patches(name):
        ims_folder = os.path.join(params.DATASET_PATH, 'ims')
        ims_names = [i for i in glob.glob(ims_folder + '/*') if name in i]
        ims = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in ims_names]
        return ims, ims_names

    ims, ims_names = load_wsi_patches(wsi_name)

    org_size = int(params.UNET_INPUT_SIZE * 3)
    th = 0.7
    predictions = predict(model, ims, org_size, th)

    def assembly(filepath, predictions, ims_names):
        with openslide.OpenSlide(filepath) as wsiObject:
            w_wsi, h_wsi = wsiObject.level_dimensions[0]
            w_th, h_th = wsiObject.level_dimensions[3]
            lvl = wsiObject.level_downsamples[3]
        mask = np.zeros((w_wsi, h_wsi), dtype=np.uint8)
        for pred, name in zip(predictions, ims_names):
            x, y = get_coords(name)
            mask[y:y+3200, x:x+3200] = np.logical_or(mask[y:y+3200, x:x+3200], pred)
        mask_pil = Image.fromarray(mask*255)
        mask_pil = mask_pil.resize((w_th, h_th))
        return mask_pil

    mask_pil = assembly(filepath, predictions, ims_names)
    mask_pil.save(os.path.join(par_dir, f'{wsi_name}_mask.png'))


class WSI(openslide.OpenSlide):
    def __init__(self, filename):
        super().__init__(filename)
        print(self.__class__)


if __name__ == '__main__':
    # validate_model()
    # mask_assembly()
    par_dir = "/home/francisco/Escritorio/Biopsies"
    filepath = os.path.join(par_dir, os.listdir(par_dir)[0])
    wsi = WSI(filename=filepath)