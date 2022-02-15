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
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

import src.parameters as params

if platform == "win32":
    _dll_path = os.getenv('OPENSLIDE_PATH')  # TODO add README.md guide for Openslide import while using Windows
    if _dll_path is not None:
        # Python >= 3.8
        with os.add_dll_directory(_dll_path):
            import openslide
else:
    import openslide

from openslide import open_slide


class SegmenterPipeline:
    def __init__(self, model_weights, slide_file):
        self.model_weights = model_weights
        self.slide_file : str = slide_file
        self.slide_name : str = os.path.basename(self.slide_file).split('.')[0]
        self.patch_dir : str = self.slide_file.split('.')[0]
        os.mkdir(self.patch_dir)
        # self.slide_folder = os.path.dirname(self.slide_file)
        self.dims = (3200, 3200)
        self.stride_proportion = 1/4

        self.slide = None
        self.ss_factor = None
        self.window_reduded_dim = None


    def preprocess_slide(self):
        self.slide = open_slide(self.slide_file)

        # find best reduction level
        max_level = self.slide.level_count - 1
        w_slide, h_slide = self.slide.dimensions
        self.ss_factor, self.window_reduded_dim = self.lower_reduction_level(self.dims, self.stride_proportion, max_level)

        # Generate thumbnail with desired level
        thumbnail = self.get_thumbnail()

        # Generate thumbnail patches
        X, Y = self.get_patches_from_thumbnail(thumbnail)

        # Take images from slide corresponding to thumbnail patches
        for x, y in zip(X, Y):
            ARGB = self.slide.read_region(location=(x, y), size=self.dims, level=0)
            ARGB.show()
            name = self.slide_name.split('.')[0] + f"_x{x}_y{y}_s{self.dims[0]}.png"
            patch_path = os.path.join(self.patch_dir, name)
            # Save to disk
            ARGB.save(patch_path)

    @staticmethod
    def lower_reduction_level(dims, stride_proportion, max_level):
        for level in range(max_level, 1, -1):
            ss_factor = 2 ** level
            window_reduced_dim = dims[0] / ss_factor
            stride = window_reduced_dim * stride_proportion

            if math.floor(stride) == stride:
                return ss_factor, int(window_reduced_dim)

    def get_thumbnail(self):
        level = self.slide.get_best_level_for_downsample(self.ss_factor)

        w_thumb, h_thumb = self.slide.level_dimensions[level]

        argb = self.slide.read_region(location=(0, 0), level=level, size=(w_thumb, h_thumb))
        name = self.slide_file.split('.')[0] + '_thumbnail.png'
        argb.save(name)
        return np.array(argb)[:, :, :-1]

    def get_patches_from_thumbnail(self, im: np.ndarray):
        h, w, _ = im.shape
        stride = int(self.window_reduded_dim * self.stride_proportion)

        X = list()
        Y = list()
        for r in range(0, h, self.window_reduded_dim - stride):
            if r + self.window_reduded_dim > h:
                r = h - self.window_reduded_dim
            for c in range(0, w, self.window_reduded_dim - stride):
                if c + self.window_reduded_dim > w:
                    c = w - self.window_reduded_dim

                patch = im[r:r+self.window_reduded_dim, c:c+self.window_reduded_dim, :]

                if self.is_tissue_patch(patch):
                    R = r * self.ss_factor
                    C = c * self.ss_factor
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



def debugger():
    print(os.path.join(os.path.dirname(os.path.dirname((os.path.abspath("__file__")))), 'output'))
    out_dir = os.path.join(os.path.dirname(os.path.dirname((os.path.abspath("__file__")))), 'output')
    data_dir = os.path.join(os.path.dirname(os.path.dirname((os.path.abspath("__file__")))), 'data')
    weights_file = os.path.join(out_dir, "2022-01-14_08-16-32", "weights/model.hdf5")
    # slide_path = os.path.join(data_dir, "raw/20B0011364 A 1 HE.tif")
    slide_path = os.path.join(data_dir, "raw/21A3 A25 HE.tif")
    pipeline = SegmenterPipeline(weights_file, slide_path)
    pipeline.preprocess_slide()

    thumb = pipeline.get_thumbnail()

if __name__ == '__main__':
    debugger()

