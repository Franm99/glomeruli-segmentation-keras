from unet_model import unet_model
import keras
import parameters as params
import os
import glob
import cv2.cv2 as cv2
import numpy as np
from tensorflow.keras.utils import normalize
from utils import show_ims
import openslide


# output_folder = browse_path()
output_folder = "2022-01-20_16-40-12"
output_dir = os.path.join('output', output_folder)
par_dir = "/home/francisco/Escritorio/Biopsies"
ims_path = os.path.join(par_dir, "20B0011364 A 1 HE")
WSI = os.path.join(par_dir, "20B0011364 A 1 HE.tif")


def get_model() -> keras.Model:
    """ return: U-Net Keras model (TF2 Version)"""
    return unet_model(params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE, 1)


def load_model() -> keras.Model:
    model = get_model()
    model_weights = os.path.join(output_dir, "weights/model.hdf5")
    model.load_weights(model_weights)
    return model


def get_ims_list(ims_path: str):
    return glob.glob(ims_path + '/*')


def get_pred_mask(im, dim, model, th: float):
    """ """
    [h, w] = im.shape
    # Initializing list of masks
    mask = np.zeros((h, w), dtype=bool)

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
            else:
                # Tissue sub-patches are fed to the U-net model for mask prediction
                patch = cv2.resize(patch, (params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE), interpolation=cv2.INTER_AREA)
                patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                prediction = (model.predict(patch_input)[:, :, :, 0] > th).astype(np.uint8)
                prediction_rs = cv2.resize(prediction[0], (dim, dim), interpolation=cv2.INTER_AREA)

                # Final mask is composed by the sub-patches masks (boolean array)
            mask[y:y + dim, x:x + dim] = np.logical_or(mask[y:y + dim, x:x + dim], prediction_rs.astype(bool))
    return mask.astype(np.uint8)  # Change datatype from np.bool to np.uint8


def main():
    # 1. Load Pre-trained model and images to take predictions from
    model = load_model()

    ims_list = get_ims_list(ims_path)
    ims_names = [os.path.basename(i) for i in ims_list]
    ims = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in ims_list]

    # 2. Predict glomeruli location
    th = 0.8
    predictions = []
    patch_dim = int(params.UNET_INPUT_SIZE * params.RESIZE_RATIOS[0])

    for im, name in zip(ims, ims_names):
        pred = get_pred_mask(im, patch_dim, model, th)
        predictions.append(pred)
    show_ims(predictions)

    # 3.1 Extract info from WSI (dimensions)
    with openslide.OpenSlide(WSI) as slidePtr:
        WSI_dims = slidePtr.dimensions

    print(WSI_dims)

    # 3.2. Extract info from image names (coordinates)
    # TODO complete this part


if __name__ == '__main__':
    main()
