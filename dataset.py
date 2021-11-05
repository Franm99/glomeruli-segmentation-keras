import os, glob
import warnings
from typing import List, Tuple
import numpy as np
import cv2.cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

_DATASET_PATH = "D:/DataGlomeruli"
_DEFAULT_RADIUS = 400

class Dataset():
    def __init__(self, mask_radius: int = 400):
        self._ims_path = _DATASET_PATH + '/ims'
        self._masks_path = _DATASET_PATH + '/masks_' + str(mask_radius)

        # Check if exists a masks folder for the given radius
        if not os.path.exists(self._masks_path):
            warnings.warn(
                "No masks with mask_radius = {}px in dataset. Getting default value: {}px".format(mask_radius, _DEFAULT_RADIUS),
                category=ResourceWarning)
            mask_radius = _DEFAULT_RADIUS

    def load(self):
        ims_names = glob.glob(self._ims_path + '/*')
        masks_names = glob.glob(self._masks_path + '/*')
        self.ims = []
        self.masks = []
        DEBUG_LIMIT = 0
        for im_name, mask_name in tqdm(zip(ims_names, masks_names), total=len(ims_names),
                                       desc= "Loading images and masks"):
            im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)  # TODO: Also test with RGB or HSV format
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            self.ims.append(im)
            self.masks.append(mask)
            DEBUG_LIMIT += 1
            if DEBUG_LIMIT == 10:
                break

    def split(self):
        return train_test_split(self.ims, self.masks, test_size=0.15)


# Testing
if __name__ == '__main__':
    dataset = Dataset()
    dataset.load()
    xtrain, xtest, ytrain, ytest = dataset.split()
    print("Training size:", len(xtrain))
    print("Test size:", len(xtest))

