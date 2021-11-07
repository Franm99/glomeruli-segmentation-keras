"""
Dataset class 
"""

import os, glob
import warnings
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2.cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from utils import show_ims, timer
import random

_PATCH_SIZE = 256
_DATASET_PATH = "D:/DataGlomeruli"
_DEFAULT_MASKS_PATH = _DATASET_PATH + '/masks_400'


class Dataset():
    """Dataset class to prepare data for training-test U-Net model for glomeruli segmentation."""
    def __init__(self, masks_path: str = _DEFAULT_MASKS_PATH):
        """ Class constructor
        :param masks_path: path to directory containing binary masks for groundtruth. 
        """
        # Initializing variables
        self._ims_path = _DATASET_PATH + '/ims'  # Path to original images
        self._masks_path = masks_path            
        # Lists to load the set of original patches.
        self.data = []
        self.data_masks = []
        # Lists to save the set of sub-patches taken from original patches
        self.ims = []
        self.masks = []

        # Check if exists a masks folder for the given radius
        if not os.path.exists(self._masks_path):
            warnings.warn(
                "Specified mask folder not found. Getting default source: {}".format(_DEFAULT_MASKS_PATH),
                category=ResourceWarning)
            self._masks_path = _DEFAULT_MASKS_PATH

    @timer
    def load(self, limit_samples: Optional[float] = None):
        """
        Load data to work with.
        :param limit_samples (0.0 - 1.0) Limits the number of samples taken from the specified data folder. If 1.0, the 
        whole set of data is loaded.
        """
        # Get full path to both images and masks directories.
        ims_names = glob.glob(self._ims_path + '/*')
        masks_names = glob.glob(self._masks_path + '/*')
        
        if limit_samples and (0.0 < limit_samples < 1.0):
            last = int(limit_samples * len(ims_names))
            ims_names = ims_names[0:last]
            masks_names = masks_names[0:last]

        for im_name, mask_name in tqdm(zip(ims_names, masks_names), total=len(ims_names),
                                       desc= "Loading images and masks"):
            im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)  # TODO: Also test with RGB or HSV format
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            self.data.append(im)
            self.data_masks.append(mask)

    @timer
    def gen_subpatches(self, rz_ratio: int):
        """
        Generate sub-patches from original patches. Needed to adjust data for U-Net input requirements.
        :param rz_ratio: Resize ratio. Sub-patches can be taken from a resized version of the original patch.
        """
        patch_size_or = _PATCH_SIZE * rz_ratio
        for im, mask in tqdm(zip(self.data, self.data_masks), total=len(self.data),
                             desc = "Generating subpatches"):
            [h, w] = im.shape
            for x in range(0, w, patch_size_or):
                if x+patch_size_or >= w:
                    x = w - patch_size_or
                for y in range(0, h, patch_size_or):
                    if y+patch_size_or >= h:
                        y = h - patch_size_or
                    patch_arr = im[y:y+patch_size_or, x:x+patch_size_or]
                    if self._filter(patch_arr):  # sub-patches not containing any tissue fragment are discarded
                        continue
                    # Converting to PIL Image format for resize operation
                    patch = Image.fromarray(patch_arr)
                    patch = patch.resize((_PATCH_SIZE, _PATCH_SIZE))
                    self.ims.append(np.asarray(patch))
                    patch_mask = Image.fromarray(mask[y:y+patch_size_or, x:x+patch_size_or])
                    patch_mask = patch_mask.resize((_PATCH_SIZE, _PATCH_SIZE))
                    self.masks.append(np.asarray(patch_mask))
        # show_ims(random.sample(self.ims, 6))
        print("Normalizing patches for U-Net input...")
        self._normalize()

    @staticmethod
    def _filter(patch: np.ndarray):
        counts, bins = np.histogram(patch.flatten(), list(range(256+1)))
        counts.sort()
        median = np.median(counts)
        if median <= 3.:
            return True
        return False

    def _normalize(self):
        self.ims = np.expand_dims(normalize(np.array(self.ims), axis=1), 3)
        self.masks = np.expand_dims((np.array(self.masks)), 3) / 255

    def split(self, ratio: float = 0.15) -> Tuple:
        return train_test_split(self.ims, self.masks, test_size=ratio)


# Testing
# if __name__ == '__main__':
#     dataset = Dataset()
#     dataset.load(limit_samples=0.03)
#     dataset.gen_subpatches(rz_ratio=5)
#     xtrain, xtest, ytrain, ytest = dataset.split()
#     print("Training size:", len(xtrain))
#     print("Test size:", len(xtest))

