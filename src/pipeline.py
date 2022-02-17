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

import src.parameters as params
from src.model.model_utils import get_model, load_model_weights
from src.model.keras_models import simple_unet

if platform == "win32":
    _dll_path = os.getenv('OPENSLIDE_PATH')  # TODO add README.md guide for Openslide import while using Windows
    if _dll_path is not None:
        # Python >= 3.8
        with os.add_dll_directory(_dll_path):
            import openslide
else:
    import openslide

from openslide import open_slide


class WSI(openslide.OpenSlide):
    def __init__(self, filename):
        super().__init__(filename)
        self.dimensions_thumbnail, self.ss_factor = self._find_lower_thumbnail_dimensions()
        self.window_reduced_dim = self.find_window_reduced_dim()

    def _find_lower_thumbnail_dimensions(self):
        max_level = self.level_count - 1
        ss_factor = None
        for level in range(max_level, 1, -1):
            ss_factor = 2 ** level
            reduced_dim = params.PATCH_SIZE[0] / ss_factor
            stride = reduced_dim * params.STRIDE_PTG
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

    def get_thumbnail(self, size: Optional[Tuple[int, int]] = None):
        if size:
            return super().get_thumbnail(size)
        else:
            return self.get_best_thumbnail()


class SegmenterPipeline:
    def __init__(self, model):
        self.model = model
        self.dims = (3200, 3200)
        self.stride_proportion = 1/4
        self.slide = None
        self.ss_factor = None
        self.window_reduced_dim = None

    def preprocess_slide(self, slide_file: str) :
        # Initialization
        self.slide_file = slide_file
        self.slide_name = os.path.basename(slide_file).split('.')[0]
        self.patch_dir = slide_file.split('.')[0]
        if not os.path.isdir(self.patch_dir):
            os.mkdir(self.patch_dir)

        self.slide = WSI(slide_file)

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
            # Save to disk
            yield np.array(ARGB)[:, :, :-1]
            # if not os.path.isfile(patch_path):
            #     ARGB.save(patch_path)

    def get_patches_from_thumbnail(self, im_pil: Image, window_reduced_dim):
        im = np.array(im_pil)
        h, w, _ = im.shape
        stride = int(window_reduced_dim * params.STRIDE_PTG)
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

    @staticmethod
    def is_tissue_patch(patch):
        counts, bins = np.histogram(patch.flatten(), list(range(256 + 1)))
        counts.sort()
        median = np.median(counts)
        if median >= 1:
            return True
        return False


def load_model(model_weights):
    model = get_model(simple_unet, im_w=params.UNET_INPUT_SIZE, im_h=params.UNET_INPUT_SIZE)
    return load_model_weights(model, model_weights)


def debugger():
    print(os.path.join(os.path.dirname(os.path.dirname((os.path.abspath("__file__")))), 'output'))
    out_dir = os.path.join(os.path.dirname(os.path.dirname((os.path.abspath("__file__")))), 'output')
    data_dir = os.path.join(os.path.dirname(os.path.dirname((os.path.abspath("__file__")))), 'data')
    weights_file = os.path.join(out_dir, "2022-01-14_08-16-32", "weights/model.hdf5")
    # slide_path = os.path.join(data_dir, "raw/20B0011364 A 1 HE.tif")
    slide_path = os.path.join(data_dir, "raw/21A3 A25 HE.tif")

    model = load_model(weights_file)
    segmenter = SegmenterPipeline(model)
    ims_generator = segmenter.preprocess_slide(slide_path)


if __name__ == '__main__':
    debugger()
