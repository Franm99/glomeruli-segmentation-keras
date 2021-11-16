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
from utils import show_masked_ims, timer
import random
import matplotlib.pyplot as plt

_PATCH_SIZE = 256
_DATASET_PATH = "D:/DataGlomeruli"
# _DATASET_PATH = "/home/francisco/Escritorio/DataGlomeruli"
# _DATASET_PATH = "images/training"
# NOTE: 250px radius seems to be the better size to fit a whole mean glomeruli
_DEFAULT_MASKS_PATH = _DATASET_PATH + '/masks_250'
# _DEFAULT_MASKS_PATH = "images/training_groundtruth"
_PATCHES_IMS_PATH = "D:/DataGlomeruli/patches_ims"
# _PATCHES_IMS_PATH = "/home/francisco/Escritorio/DataGlomeruli/patches_ims"
_PATCHES_MASKS_PATH = "D:/DataGlomeruli/patches_masks"
# _PATCHES_MASKS_PATH = "/home/francisco/Escritorio/DataGlomeruli/patches_masks"


class Dataset():
    """Dataset class to prepare data for training-test U-Net model for glomeruli segmentation."""
    def __init__(self, masks_path: str = _DEFAULT_MASKS_PATH):
        """ Class constructor
        :param masks_path: path to directory containing binary masks for groundtruth. 
        """
        # Initializing variables
        self._ims_path = _DATASET_PATH + '/ims'  # Path to original images
        # self._ims_path = _DATASET_PATH  # Path to original images
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
    def load(self, limit_samples: Optional[float] = None, staining: Optional[str] = None):
        """
        Load data to work with.
        :param limit_samples (0.0 - 1.0) Limits the number of samples taken from the specified data folder. If 1.0, the 
        whole set of data is loaded.
        :param staining: ["PAS", "PM", "HE"] Parameter to select a specific staining between the 3 available. If set to
        None, all of them are taken for dataset.
        """
        # Get full path to both images and masks directories.
        [ims_names, masks_names] = self._get_data_dirs(staining)

        if limit_samples and (0.0 < limit_samples < 1.0):
            last = int(limit_samples * len(ims_names))
            ims_names = ims_names[0:last]
            masks_names = masks_names[0:last]

        for im_name, mask_name in tqdm(zip(ims_names, masks_names), total=len(ims_names),
                                       desc= "Loading images and masks", ascii=True, ncols=80):
            im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)  # TODO: Also test with RGB or HSV format
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            self.data.append(im)
            self.data_masks.append(mask)

    @timer
    def gen_subpatches(self, rz_ratio: int, save: bool = False, clear: bool = False):
        """
        Generate sub-patches from original patches. Needed to adjust data for U-Net input requirements.
        :param rz_ratio: Resize ratio. Sub-patches can be taken from a resized version of the original patch.
        :param save: If True, saves the set of sub-patches to disk.
        """
        patch_size_or = _PATCH_SIZE * rz_ratio
        for im, mask in tqdm(zip(self.data, self.data_masks), total=len(self.data),
                             desc = "Generating subpatches", ascii=True, ncols=80):
            [h, w] = im.shape
            for x in range(0, w, patch_size_or):
                if x+patch_size_or >= w:
                    x = w - patch_size_or
                for y in range(0, h, patch_size_or):
                    if y+patch_size_or >= h:
                        y = h - patch_size_or
                    patch_arr = im[y:y+patch_size_or, x:x+patch_size_or]
                    mask_arr = mask[y:y+patch_size_or, x:x+patch_size_or]
                    if self._filter(mask_arr):
                        # Convert to PIL for resizing and returning to numpy array format.
                        patch = np.asarray(Image.fromarray(patch_arr).resize((_PATCH_SIZE, _PATCH_SIZE)))
                        patch_mask = np.asarray(Image.fromarray(mask_arr).resize((_PATCH_SIZE, _PATCH_SIZE)))
                        self.ims.append(patch)
                        self.masks.append(patch_mask)

        if save:
            self._save_dataset(clear=clear)

        print("Dataset size: {}".format(len(self.ims)))
        # self.show_random_samples(3)  # DEBUG
        self._normalize()

    @staticmethod
    def _filter(patch: np.ndarray) -> bool:  # Modify: not include sub-patches without glomeruli
        """
        Patch filter based on median value from ordered histogram to find patches containing kidney tissue.
        :param patch: patch to check up.
        :return: True if patch contains tissue, False if not.
        """
        return np.sum(patch) > 0
        # counts, bins = np.histogram(patch.flatten(), list(range(256+1)))
        # counts.sort()
        # median = np.median(counts)
        # if median <= 3.:
        #     return True
        # return False

    def _normalize(self):
        """
        Normalize dataset to U-Net input format:
        np.ndarray with format: (num_of_samples, h, w, num_channels)
        """
        self.ims = np.expand_dims(normalize(np.array(self.ims), axis=1), 3)
        self.masks = np.expand_dims((np.array(self.masks)), 3) / 255

    def _get_data_dirs(self, staining: str):
        ims_names = glob.glob(self._ims_path + '/*')
        masks_names = glob.glob(self._masks_path + '/*')
        if staining:
            ims_names = [i for i in ims_names if staining in i]
            masks_names = [i for i in masks_names if staining in i]
        ims_names.sort()
        masks_names.sort()
        return [ims_names, masks_names]

    def _save_dataset(self, clear: bool):
        if clear:
            ims = glob.glob(_PATCHES_IMS_PATH + '/*')
            masks = glob.glob(_PATCHES_MASKS_PATH + '/*')
            for im, mask in zip(ims, masks):
                try:
                    os.unlink(im)
                    os.unlink(mask)
                except Exception as e:
                    print("Failed to delete files. Reason: {}".format(e))

        num_digits = len(str(len(self.ims))) + 1
        for idx, (im, mask) in enumerate(zip(self.ims, self.masks)):
            bname = str(idx).zfill(num_digits) + ".png"
            cv2.imwrite(os.path.join(_PATCHES_IMS_PATH, bname), im)
            cv2.imwrite(os.path.join(_PATCHES_MASKS_PATH, bname), mask)


    def split(self, ratio: float = 0.15) -> Tuple:
        """
        Split dataset (both images and masks) for training and test
        :param ratio: (0.0 - 1.0) test percentage
        :return: [xtrain, xtest, ytrain, ytest]
        """
        return train_test_split(self.ims, self.masks, test_size=ratio)

    def show_random_samples(self, n: int):
        indexes = random.sample(range(len(self.ims)), n)
        ims_sample = [self.ims[idx] for idx in indexes]
        masks_sample = [self.masks[idx] for idx in indexes]
        show_masked_ims(ims_sample, masks_sample, 1, n)

    def __len__(self):
        return len(self.ims)


def im_debug(dataset: Dataset):
    while True:
        dataset.show_random_samples(4)
        plt.close()


def split_debug(xtrain, xtest, ytrain, ytest):
    while True:
        idx_train = random.randint(0, len(xtrain)-1)
        idx_test = random.randint(0, len(xtest)-1)
        show_masked_ims([xtrain[idx_train][:, :, 0], xtest[idx_test][:, :, 0]],
                        [ytrain[idx_train][:, :, 0], ytest[idx_test][:, :, 0]],
                        1, 2)


# Testing
if __name__ == '__main__':
    dataset = Dataset()
    dataset.load(limit_samples=None, staining="PAS")
    dataset.gen_subpatches(rz_ratio=4, save=True, clear=True)
    im_debug(dataset)  # Debug

    xtrain, xtest, ytrain, ytest = dataset.split()
    split_debug(xtrain, xtest, ytrain, ytest)
    print("Training size:", len(xtrain))
    print("Test size:", len(xtest))

