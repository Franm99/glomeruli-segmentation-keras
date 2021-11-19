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
import constants as ct


class Dataset():
    def __init__(self, staining: str = ct.DEF_STAINING):
        """ Initialize Dataset.
        Paths initialization to find data in disk. Images and ground-truth masks full-paths are loaded."""
        # Paths
        self._ims_path = ct.DATASET_PATH + '/ims'
        self._xmls_path = ct.DATASET_PATH + '/xml'
        self._masks_path = ct.DATASET_PATH + '/gt' + '/circles'
        self._train_val_file = ct.DATASET_PATH + '/train_val.txt'
        self._test_file = ct.DATASET_PATH + '/test.txt'
        self._train_val_ims_path = ct.DATASET_PATH + '/train_val/patches'
        self._train_val_masks_path = ct.DATASET_PATH + '/train_val/patches_masks'

        self._staining = staining
        self.trainval_list = []
        self.test_list = []

        # List of data paths (instead of numpy arrays)
        if self._staining == "ALL":
            self.ims_list = glob.glob(self._ims_path + '/*')
            self.ims_list.sort()
            self.masks_list = self._load_masks()
        else:
            tmp = glob.glob(self._ims_path + '/*')
            tmp.sort()
            self.ims_list = [i for i in tmp if self._staining in i]
            self.masks_list = [i for i in self._load_masks() if self._staining in i]

    def split_trainval_test(self, train_size: float, overwrite: bool = True):
        if not overwrite:
            print_info("Split previously done. Reading from file...")
            self.trainval_list = self.txt2list(self._train_val_file)
            self.test_list = self.txt2list(self._test_file)
            xtrainval = [os.path.join(self._ims_path, i) for i in self.trainval_list]
            ytrainval = [os.path.join(self._masks_path, i) for i in self.trainval_list]
            xtest = [os.path.join(self._ims_path, i) for i in self.test_list]
            ytest = [os.path.join(self._masks_path, i) for i in self.test_list]
        else:
            xtrainval, xtest, ytrainval, ytest = train_test_split(self.ims_list, self.masks_list,
                                                                  train_size=train_size, shuffle=True)
            print_info("Overwriting new split.")
            self.trainval_list = [os.path.basename(i) for i in xtrainval]
            self.test_list = [os.path.basename(i) for i in xtest]
            self.list2txt(self._train_val_file, self.trainval_list)
            self.list2txt(self._test_file, self.test_list)
        print_info("{} images for Train/Validation and {} for testing".format(len(xtrainval), len(xtest)))
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
            im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            ims.append(im)
            masks.append(mask)
        return ims, masks

    @timer
    def get_spatches(self, data, data_masks, rz_ratio: int, from_disk: bool = True):
        patches = []
        patches_masks = []
        if from_disk:
            print_info("PATCHES ALREADY EXISTS IN DISK. ")
            print_info("Loading sub-patches pairs for training stage from disk...")
            patches_files = glob.glob(self._train_val_ims_path + '/*')
            patches_files.sort()
            patches_masks_files = glob.glob(self._train_val_masks_path + '/*')
            patches_masks_files.sort()
            patches, patches_masks = self.load_pairs(patches_files, patches_masks_files)
        else:
            print_info("NO PATCHES HAVE BEEN FOUND IN DISK.")
            print_info("Generating sub-patches for training stage and saving to disk...")
            patch_size_or = ct.UNET_INPUT_SIZE * rz_ratio
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
                            patch = np.asarray(Image.fromarray(patch_arr).resize((ct.UNET_INPUT_SIZE, ct.UNET_INPUT_SIZE)))
                            patch_mask = np.asarray(Image.fromarray(mask_arr).resize((ct.UNET_INPUT_SIZE, ct.UNET_INPUT_SIZE)))
                            patches.append(patch)
                            patches_masks.append(patch_mask)
            # Save train dataset to disk for later use
            self._save_train_dataset(patches, patches_masks)

        print_info("{} patches generated from {} images for training and validation.".format(len(patches), len(data)))
        return self._normalize(patches, patches_masks)

    def get_data_list(self, set: str):
        return eval("self." + set + "_list")

    # PRIVATE
    @timer
    def _load_masks(self):
        # Check for existing masks folder for desired size
        print_info("Checking if ground-truth masks exists or either they need to be generated.")
        if not os.path.isdir(self._masks_path):
            print_warn("MASKS DO NOT EXIST YET. GENERATING GROUND-TRUTH.")
            maskGenerator = MaskGenerator()
            return maskGenerator.get_masks_files()
        res = glob.glob(self._masks_path + '/*.png')
        res.sort()
        print_info("TAKING GROUND-TRUTH MASKS FROM DISK.")
        print_info("Masks located in: {}".format(self._masks_path))
        return res

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
        prev_ims = glob.glob(self._train_val_ims_path + '/*')
        prev_ims.sort()
        prev_masks = glob.glob(self._train_val_masks_path + '/*')
        prev_masks.sort()
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
if __name__ == '__main__':
    print_info("Building dataset...")
    dataset = Dataset(staining="HE")
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



