"""
TODO
Write all image filenames to a txt, previous to the train+val - test split. This way, i ensure
the same order when importing to a list. Then, the split is performed, and two new txt will be
generated, containing the names for both the train+val and test sets. SAME PROCESS for sub-patches.

TODO: Refactoring and documentation
"""
import os
import glob
from mask_generator.MaskGenerator import MaskGenerator
from sklearn.model_selection import train_test_split
from utils import print_info, print_warn, print_error, timer
from tqdm import tqdm
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from tensorflow.keras.utils import normalize
import parameters as params
from typing import Optional


class Dataset():
    def __init__(self, staining: str, mask_size: Optional[int], mask_simplex: bool):
        """ Initialize Dataset.
        Paths initialization to find data in disk. Images and ground-truth masks full-paths are loaded.
        """
        # Paths
        self._ims_path = params.DATASET_PATH + '/ims'
        self._xmls_path = params.DATASET_PATH + '/xml'
        self._masks_path = params.DATASET_PATH + '/gt/circles'
        self._train_val_path = params.DATASET_PATH + '/train_val'
        self._train_val_ims_path = self._train_val_path + '/patches'  # For saving train-val sub-patches
        self._train_val_masks_path = self._train_val_path + '/patches_masks'  # For saving train-val sub-masks

        # Files to keep track of the data used
        self._ims_list_path = params.DATASET_PATH + '/ims.txt'  # For saving patches names
        self._train_val_file = params.DATASET_PATH + '/train_val.txt'  # For saving names of train-val images (both)
        self._test_file = params.DATASET_PATH + '/test.txt'  # For saving names of test images
        self._subpatches_file = self._train_val_path + '/subpatches_list.txt'  # For saving names of train-val sub-patches

        # Instance parameters initialization
        self._staining = staining
        self._mask_size = mask_size
        self._mask_simplex = mask_simplex
        self.trainval_list = []
        self.test_list = []

        if self._mask_size:
            self._masks_path = self._masks_path + str(self._mask_size)

        if self._mask_simplex:
            self._masks_path = self._masks_path + "_simplex"

        self.ims_names = [i[:-4] for i in os.listdir(self._ims_path)]
        self.ims_list = [self._ims_path + '/' + i + '.png' for i in self.ims_names]
        self.xml_list = [self._xmls_path + '/' + i + '.xml' for i in self.ims_names]

        # Images names will be writen to a file, and later used to load the remaining data.
        self.list2txt(self._ims_list_path, self.ims_names)

        # By now, everytime the training stage is launched, new masks are computed and saved.
        self.masks_list = self._load_masks(self._mask_size, self._mask_simplex)
        if self._staining != "ALL":  # Load just the samples for the selected staining
            self.ims_list = [i for i in self.ims_list if self._staining in i]
            self.masks_list = [i for i in self.masks_list if self._staining in i]

    def split_trainval_test(self, train_size: float):
        xtrainval, xtest, ytrainval, ytest = train_test_split(self.ims_list, self.masks_list, train_size=train_size,
                                                              shuffle=True, random_state=params.TRAINVAL_TEST_RAND_STATE)
        self.trainval_list = [os.path.basename(i) for i in xtrainval]
        self.test_list = [os.path.basename(i) for i in xtest]
        self.list2txt(self._train_val_file, self.trainval_list)
        self.list2txt(self._test_file, self.test_list)
        print_info("{} images for Train/Validation and {} for testing".format(len(xtrainval), len(xtest)))
        return xtrainval, xtest, ytrainval, ytest

    def split_train_val(self, ims, masks, test_size: float = 0.1):
        xtrain, xval, ytrain, yval = train_test_split(ims, masks, test_size=test_size,
                                                      shuffle=True, random_state=params.TRAIN_VAL_RAND_STATE)
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
            im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            ims.append(im)
            masks.append(mask)
        return ims, masks

    @timer
    def get_spatches(self, data, data_masks, rz_ratio: int):
        patches = []
        patches_masks = []
        print_info("Generating sub-patches for training stage and saving to disk...")
        patch_size_or = params.UNET_INPUT_SIZE * rz_ratio
        for im, mask in tqdm(zip(data, data_masks), total=len(data), desc = "Generating subpatches"):
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
                        patch = np.asarray(Image.fromarray(patch_arr).resize((params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE)))
                        patch_mask = np.asarray(Image.fromarray(mask_arr).resize((params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE)))
                        patches.append(patch)
                        patches_masks.append(patch_mask)
        # Save train dataset to disk for later use
        spatches_names = self._save_train_dataset(patches, patches_masks)

        print_info("{} patches generated from {} images for training and validation.".format(len(patches), len(data)))
        return spatches_names, self._normalize(patches, patches_masks)
        # return self._normalize(patches, patches_masks)

    def get_data_list(self, set: str):
        return eval("self." + set + "_list")

    # PRIVATE
    @timer
    def _load_masks(self, mask_size: Optional[int], mask_simplex: bool):
        """ Function to generate a ground-truth (masks) from xml info. """
        if os.path.isdir(self._masks_path):
            # If masks already exists, delete.
            self.clear_dir(self._masks_path)
        else:
            os.mkdir(self._masks_path)
        ims_names = self.txt2list(self._ims_list_path)
        maskGenerator = MaskGenerator(ims_names=ims_names, mask_size=mask_size, apply_simplex=mask_simplex)
        return maskGenerator.get_masks_files()

    @staticmethod
    def _filter(patch: np.ndarray) -> bool:  # Modify: not include sub-patches without glomeruli
        """
        Patch filter based on median value from ordered histogram to find patches containing kidney tissue.
        :param patch: patch to check up.
        :return: True if patch contains tissue, False if not.
        """
        return np.sum(patch) > 0

    def _save_train_dataset(self, ims, masks):
        # Check if target directories exist
        if not os.path.isdir(self._train_val_ims_path):
            os.mkdir(self._train_val_ims_path)
            os.mkdir(self._train_val_masks_path)

        # Clear data if existing
        self.clear_dir(self._train_val_ims_path)
        self.clear_dir(self._train_val_masks_path)

        num_digits = len(str(len(ims))) + 1
        spatches_names = []
        for idx, (im, mask) in enumerate(zip(ims, masks)):
            bname = str(idx).zfill(num_digits) + ".png"
            spatches_names.append(bname)
            cv2.imwrite(os.path.join(self._train_val_ims_path, bname), im)
            cv2.imwrite(os.path.join(self._train_val_masks_path, bname), mask)
        self.list2txt(self._subpatches_file, spatches_names)
        return spatches_names

    # STATIC
    @staticmethod
    def txt2list(fname):
        with open(fname, 'r') as f:
            return [line.rstrip('\n') for line in f.readlines()]

    @staticmethod
    def list2txt(fname, data):
        with open(fname, 'w') as f:
            for i in data:
                f.write(i + "\n")

    @staticmethod
    def clear_dir(dpath : str):
        files = glob.glob(dpath + '/*')
        for file in files:
            try:
                os.unlink(file)
            except Exception as e:
                print_error("Failed to delete files: Reason: {}".format(e))

    @staticmethod
    def _normalize(ims, masks):
        ims_t = np.expand_dims(normalize(np.array(ims), axis=1), 3)
        masks_t = np.expand_dims((np.array(masks)), 3) / 255
        return ims_t, masks_t


# Testing
if __name__ == '__main__':
    print_info("Building dataset...")
    dataset = Dataset(staining="HE", mask_size=None, mask_simplex=False)
    xtrainval, xtest, ytrainval, ytest = dataset.split_trainval_test(train_size=0.9)
    print_info("Train/Validation set size: {}".format(len(xtrainval)))
    print_info("Test set size: {}".format(len(xtest)))
    ims, masks = dataset.load_pairs(xtrainval, ytrainval, limit_samples=0.1)
    print_info("Plotting sample:")
    idx = random.randint(0, len(ims)-1)
    plt.imshow(ims[idx], cmap="gray")
    plt.imshow(masks[idx], cmap="jet", alpha=0.3)
    plt.show()
    x_t, y_t = dataset.get_spatches(ims, masks, rz_ratio=4, from_disk=True)
    print(x_t.shape, y_t.shape)
    xtrain, xval, ytrain, yval = dataset.split_train_val(x_t, y_t)
    print(xtrain.shape, xval.shape)



