""" Pipeline class """
"""
1. Read WSI
2. Split into smaller images
3. Load pre-trained model
4. Collect predictions
5. Assembly WSI mask
"""
import os
from sys import platform
import math
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time
from dataclasses import dataclass
import cv2.cv2 as cv2
from tensorflow.keras.utils import normalize
import re

import src.parameters as params
import src.constants as const
from src.model.model_utils import get_model, load_model_weights
from src.model.keras_models import simple_unet

if platform == "win32":
    _dll_path = os.getenv('OPENSLIDE_PATH')  # TODO add README.md guide for Openslide import when using Windows
    if _dll_path is not None:
        # Python >= 3.8
        with os.add_dll_directory(_dll_path):
            import openslide
else:
    import openslide


@dataclass
class ModelData:
    """
    Model info structure obtained from model weights filename.
    Required filename format: <model_name>-<staining>-<resize_ratio>-<date>.hdf5
    """
    def __init__(self, model_weights: str):
        self._fields = os.path.basename(model_weights).split('-')
        self._weights = model_weights

    @property
    def weights(self):
        return self._weights

    @property
    def name(self):
        return self._fields[0]

    @property
    def staining(self):
        return self._fields[1]

    @property
    def resize_ratio(self):
        return int(self._fields[2])

    @property
    def date(self):
        return self._fields[3].split('.')[0]


class WSI(openslide.OpenSlide):
    """ Specific Openslide class for Renal biopsy Whole-Slide Images. """
    def __init__(self, filename):
        super().__init__(filename)
        self.dimensions_thumbnail, self.ss_factor = self._find_lower_thumbnail_dimensions()
        self.window_reduced_dim = self.find_window_reduced_dim()

    def get_thumbnail(self, size: Optional[Tuple[int, int]] = None):
        """
        Overrides get_thumbnail method from openslide.Openslide class.
        :param size: if None, the lower allowed thumbnail size is automatically computed. Else, call to the original
         get_thumbnail() method.
        :return: PIL.Image containing RGB thumbnail. """
        if size:
            return super().get_thumbnail(size)
        else:
            return self.get_best_thumbnail()

    def _find_lower_thumbnail_dimensions(self):
        max_level = self.level_count - 1
        ss_factor = None
        for level in range(max_level, 1, -1):
            ss_factor = 2 ** level
            reduced_dim = params.PATCH_SIZE[0] / ss_factor
            stride = reduced_dim * const.STRIDE_PTG
            if math.floor(stride) == stride:
                break

        level = self.get_best_level_for_downsample(ss_factor)
        return self.level_dimensions[level], ss_factor

    def find_window_reduced_dim(self):
        return int(params.PATCH_SIZE[0] / self.ss_factor)

    def get_best_thumbnail(self):
        best_dims, ss_factor = self._find_lower_thumbnail_dimensions()
        window_reduced_dim = self.find_window_reduced_dim()
        return self.get_thumbnail(best_dims)


class SegmentationPipeline:
    """ Pipeline to obtain glomeruli segmentation from a renal biopsy Whole-Slide Image. """
    def __init__(self, model_weights):
        """
        Initializing class.
        :param model: Pre-trained Keras segmentation model to use.
        """
        self.model_info = ModelData(model_weights)
        self.model = self.load_model()
        self.dims = (3200, 3200)
        self.stride_proportion = 1/4
        self.slide = None
        self.prediction = None

    def run(self, slide: str, th: float):
        ims_generator = self.preprocess_slide(slide)
        for im, name in tqdm(ims_generator, desc="Generating predictions"):
            pred = self.predict(im, th)
            self.assembly(pred, name)

    def preprocess_slide(self, slide_file: str) :
        """
        PIPELINE FIRST STAGE: Read and divide WSI in portions or patches for gradually processing.
        :param slide_file: slide filename where the segmentation process will take part.
        :yield: Patch generator object.
        """
        # Initialization
        self.slide_file = slide_file
        self.slide_name = os.path.basename(slide_file).split('.')[0]
        self.patch_dir = slide_file.split('.')[0]
        if not os.path.isdir(self.patch_dir):
            os.mkdir(self.patch_dir)

        self.slide = WSI(slide_file)
        self.init_prediction()

        # Generate thumbnail
        thumbnail = self.slide.get_thumbnail()
        name = self.slide_file.split('.')[0] + '_thumbnail.png'
        thumbnail.save(name)

        # Generate thumbnail patches
        x_list, y_list = self.get_patches_from_thumbnail(thumbnail, self.slide.find_window_reduced_dim())

        # Take images from slide corresponding to thumbnail patches
        for x, y in zip(x_list, y_list):
            ts = time.time()
            ARGB = self.slide.read_region(location=(x, y), size=self.dims, level=0)
            name = self.slide_name.split('.')[0] + f"_x{x}_y{y}_s{self.dims[0]}.png"
            patch_path = os.path.join(self.patch_dir, name)
            yield np.array(ARGB.convert('L')), name  # Numpy grayscale format is required
            # if not os.path.isfile(patch_path):
            #     ARGB.save(patch_path)

    def predict(self, im, th):
        """ SECOND PIPELINE STAGE"""
        dim = const.UNET_INPUT_SIZE * self.model_info.resize_ratio
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
                    # prediction = np.zeros((dim, dim), dtype=np.uint8)
                else:
                    # Tissue sub-patches are fed to the U-net model for mask prediction
                    patch = cv2.resize(patch, (const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE),
                                       interpolation=cv2.INTER_AREA)
                    patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                    prediction = self.model.predict(patch_input)[:, :, :, 0]
                    # prediction = prediction[0, :, :]
                    # pass
                    prediction = (self.model.predict(patch_input)[:, :, :, 0] >= th).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (dim, dim), interpolation=cv2.INTER_AREA)

                # pred_masks.append(prediction)
                # return pred_masks
                # Final mask is composed by the sub-patches masks (boolean array)
                mask[y:y + dim, x:x + dim] = np.logical_or(mask[y:y + dim, x:x + dim], prediction_rs.astype(bool))
        return mask.astype(np.uint8)  # Change datatype from np.bool to np.uint8

    def assembly(self, pred, name):
        """ THIRD PIPELINE STAGE """
        x, y = self.get_coords(name)
        s = params.PATCH_SIZE[0]
        # print(f"x: {x}, y: {y}, s: {s}")
        self.prediction[y:y+s, x:x+s] = np.logical_or(self.prediction[y:y+s, x:x+s], pred)

    def init_prediction(self):
        slide_dims = self.slide.dimensions
        w, h = self.slide.dimensions
        self.prediction = np.zeros((h, w), dtype=np.uint8)

    def load_model(self):
        model = get_model(self.model_info.name, im_h=const.UNET_INPUT_SIZE, im_w=const.UNET_INPUT_SIZE)
        return load_model_weights(model, self.model_info.weights)

    def get_patches_from_thumbnail(self, im_pil: Image, window_reduced_dim):
        im = np.array(im_pil)
        h, w, _ = im.shape
        stride = int(window_reduced_dim * const.STRIDE_PTG)
        ss_factor = self.slide.ss_factor

        X = list()
        Y = list()
        for r in range(0, h, window_reduced_dim - stride):
            if r + window_reduced_dim > h:
                r = h - window_reduced_dim
            for c in range(0, w, window_reduced_dim - stride):
                if c + window_reduced_dim > w:
                    c = w - window_reduced_dim

                patch = im[r:r+window_reduced_dim, c:c + window_reduced_dim, :]

                if self.is_tissue_patch(patch):
                    R = r * ss_factor
                    C = c * ss_factor
                    X.append(C)
                    Y.append(R)
        return X, Y

    def get_scaled_prediction(self, reduction_factor: int):
        w_wsi, h_wsi = self.slide.dimensions
        w_reduced, h_reduced = int(w_wsi / reduction_factor), int(h_wsi / reduction_factor)
        pred_pil = Image.fromarray(self.prediction * 255)  # bool to uint8 casting
        return pred_pil.resize((w_reduced, h_reduced))

    @staticmethod
    def is_tissue_patch(patch):
        counts, bins = np.histogram(patch.flatten(), list(range(256 + 1)))
        counts.sort()
        median = np.median(counts)
        if median >= 1:
            return True
        return False

    @staticmethod
    def get_coords(name):
        x = int(re.search(r'x([0-9]*)', name).group(1))
        y = int(re.search(r'y([0-9]*)', name).group(1))
        return x, y


# def debugger():
#     data_dir = os.path.join(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))), 'data')
#     weights_dir = os.path.join(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))), 'models')
#     weights_file = os.path.join(weights_dir, "simple_unet-HE-4-011422_081632.hdf5")
#     slide_path = os.path.join(data_dir, "raw/21B0000257 A 1 HE.tif")
#
#     segmenter = SegmentationPipeline(weights_file)
#     ims_generator = segmenter.preprocess_slide(slide_path)
#     th = 0.6
#     for im, name in tqdm(ims_generator, desc="Generating predictions"):
#         pred = segmenter.predict(im, th)
#         # plt.figure()
#         # plt.subplot(121)
#         # plt.imshow(im, cmap="gray")
#         # plt.suptitle(name)
#         # plt.subplot(122)
#         # plt.imshow(pred, cmap="gray")
#         # plt.show()
#         segmenter.assembly(pred, name)
#     prediction = segmenter.get_scaled_prediction(reduction_factor=16)
#
#
#
#
#
# if __name__ == '__main__':
#     debugger()
