import os
import glob
from typing import List, Optional
from mask_generator.MaskGenerator import MaskGenerator
from sklearn.model_selection import train_test_split
from utils import print_info, print_error, timer
from tqdm import tqdm
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from tensorflow.keras.utils import normalize

_UNET_INPUT_SIZE = 256
_DEF_MASK_SIZE = 150
_DATASET_PATH = 'D:/DataGlomeruli'
_DEF_TRAIN_SIZE = 0.8
_DEF_STAINING = 'HE'


class Dataset():
    def __init__(self, mask_size: int = _DEF_MASK_SIZE, staining: str = _DEF_STAINING):
        # Paths
        self._ims_path = _DATASET_PATH + '/ims'
        self._xmls_path = _DATASET_PATH + '/xml'
        self._masks_path = _DATASET_PATH + '/gt' + '/circles' + str(mask_size)
        self._train_val_file = _DATASET_PATH + '/train_val.txt'
        self._test_file = _DATASET_PATH + '/test.txt'
        self._train_val_ims_path = _DATASET_PATH + '/train_val/patches'
        self._train_val_masks_path = _DATASET_PATH + '/train_val/patches_masks'

        # Variables
        self._staining = staining
        self._mask_size = mask_size

        # List of data paths (instead of numpy arrays)
        if self._staining == "ALL":
            self.ims_list = glob.glob(self._ims_path + '/*')
            self.masks_list = self._load_masks()
        else:
            self.ims_list = [i for i in glob.glob(self._ims_path + '/*') if self._staining in i]
            self.masks_list = [i for i in self._load_masks() if self._staining in i]

    def split_trainval_test(self, train_size: float, overwrite: bool = True):
        if not overwrite:
            print_info("Split previously done. Reading from file...")
            train_val_list = self.txt2list(self._train_val_file)
            test_list = self.txt2list(self._test_file)
            xtrainval = [os.path.join(self._ims_path, i) for i in train_val_list]
            ytrainval = [os.path.join(self._masks_path, i) for i in train_val_list]
            xtest = [os.path.join(self._ims_path, i) for i in test_list]
            ytest = [os.path.join(self._masks_path, i) for i in test_list]
            return xtrainval, ytrainval, xtest, ytest
        else:
            xtrainval, xtest, ytrainval, ytest = train_test_split(self.ims_list, self.masks_list,
                                                                  train_size=train_size, shuffle=True)
            print_info("Overwriting new split.")
            trainval_names = [os.path.basename(i) for i in xtrainval]
            test_names = [os.path.basename(i) for i in xtest]
            self.list2txt(self._train_val_file, trainval_names)
            self.list2txt(self._test_file, test_names)
            return xtrainval, xtest, ytrainval, ytest

    def split_train_val(self, ims, masks, test_size: float = 0.1):
        xtrain, xval, ytrain, yval = train_test_split(ims, masks, test_size=test_size, shuffle=False)
        print_info("{} sub-patches for training, {} for validation.".format(len(xtrain), len(xval)))
        return xtrain, xval, ytrain, yval

    @timer
    def load_pairs(self, x, y, limit_samples=None):
        ims = []
        masks = []

        if limit_samples and (0.0 < limit_samples < 1.0):
            last = int(limit_samples * len(x))
            x = x[0:last]
            y = y[0:last]

        for im_name, mask_name in tqdm(zip(x, y), total=len(x), desc="Loading images and masks pairs"):
            im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)  # TODO: Also test with RGB or HSV format
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            ims.append(im)
            masks.append(mask)
        return ims, masks

    @timer
    def get_spatches(self, data, data_masks, rz_ratio: int, from_disk: bool = True):
        """
        Generate sub-patches from original patches. Needed to adjust data for U-Net input requirements.
        :param rz_ratio: Resize ratio. Sub-patches can be taken from a resized version of the original patch.
        :param save: If True, saves the set of sub-patches to disk.
        """
        patches = []
        patches_masks = []
        if from_disk:
            print_info("Loading sub-patches pairs from disk...")
            patches_files = glob.glob(self._train_val_ims_path + '/*')
            patches_masks_files = glob.glob(self._train_val_masks_path + '/*')
            patches, patches_masks = self.load_pairs(patches_files, patches_masks_files)
        else:
            print_info("Generating sub-patches and saving to disk...")
            patch_size_or = _UNET_INPUT_SIZE * rz_ratio
            for im, mask in tqdm(zip(data, data_masks), total=len(ims), desc = "Generating subpatches"):
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
                            patch = np.asarray(Image.fromarray(patch_arr).resize((_UNET_INPUT_SIZE, _UNET_INPUT_SIZE)))
                            patch_mask = np.asarray(Image.fromarray(mask_arr).resize((_UNET_INPUT_SIZE, _UNET_INPUT_SIZE)))
                            patches.append(patch)
                            patches_masks.append(patch_mask)
            # Save train dataset to disk for later use
            self._save_train_dataset(patches, patches_masks)

        print_info("Training+Validation set size: {}".format(len(patches)))
        return self._normalize(patches, patches_masks)

    # PRIVATE
    def _load_masks(self):
        # Check for existing masks folder for desired size
        if not os.path.isdir(self._masks_path):
            maskGenerator = MaskGenerator(r=self._mask_size)
            return maskGenerator.get_masks_files()
        return glob.glob(self._masks_path + '/*.png')

    @staticmethod
    def _filter(patch: np.ndarray) -> bool:  # Modify: not include sub-patches without glomeruli
        """
        Patch filter based on median value from ordered histogram to find patches containing kidney tissue.
        :param patch: patch to check up.
        :return: True if patch contains tissue, False if not.
        """
        return np.sum(patch) > 0

    def _save_train_dataset(self, ims, masks):
        # Clear data if existing
        prev_ims = glob.glob(self._train_val_ims_path + '/*')
        prev_masks = glob.glob(self._train_val_masks_path + '/*')
        for im, mask in zip(prev_ims, prev_masks):
            try:
                os.unlink(im)
                os.unlink(mask)
            except Exception as e:
                print_error("Failed to delete files: Reason: {}".format(e))

        num_digits = len(str(len(ims))) + 1
        for idx, (im, mask) in enumerate(zip(ims, masks)):
            bname = str(idx).zfill(num_digits) + ".png"
            cv2.imwrite(os.path.join(self._train_val_ims_path, bname), im)
            cv2.imwrite(os.path.join(self._train_val_masks_path, bname), mask)

    # STATIC
    @staticmethod
    def txt2list(fname):
        with open(fname, 'r') as f:
            return [line for line in f.readlines()]

    @staticmethod
    def list2txt(fname, data):
        with open(fname, 'w') as f:
            for i in data:
                f.write(i + "\n")

    @staticmethod
    def _normalize(ims, masks):
        ims_t = np.expand_dims(normalize(np.array(ims), axis=1), 3)
        masks_t = np.expand_dims((np.array(masks)), 3) / 255
        return ims_t, masks_t


# Testing
# if __name__ == '__main__':
#     print_info("Building dataset...")
#     dataset = Dataset(staining="HE")
#     xtrainval, xtest, ytrainval, ytest = dataset.split_trainval_test(train_size=0.9)
#     print_info("Train/Validation set size: {}".format(len(xtrainval)))
#     print_info("Test set size: {}".format(len(xtest)))
#     ims, masks = dataset.load_pairs(xtrainval, ytrainval, limit_samples=0.1)
#     print_info("Plotting sample:")
#     idx = random.randint(0, len(ims)-1)
#     plt.imshow(ims[idx], cmap="gray")
#     plt.imshow(masks[idx], cmap="jet", alpha=0.3)
#     plt.show()
#     x_t, y_t = dataset.get_spatches(ims, masks, rz_ratio=4, from_disk=True)
#     print(x_t.shape, y_t.shape)
#     xtrain, xval, ytrain, yval = dataset.split_train_val(x_t, y_t)
#     print(xtrain.shape, xval.shape)



