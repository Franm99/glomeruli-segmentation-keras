""" Data generator class to progressively load data to RAM """
import parameters as params
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from typing import List, Tuple
import numpy as np
import cv2.cv2 as cv2


class DataGeneratorImages(Sequence):
    """ Data generator class to load images and masks pairs in grayscale format. """
    def __init__(self,
                 ims_list: List[str],
                 masks_list: List[str],
                 batch_size: int = 16,
                 shuffle: bool = True,
                 n_channels: int = 1):
        """
        Initializes DataGenerator for images.
        :param ims_list: list of paths to image dataset.
        :param masks_list: list of paths to masks dataset. It must keep correspondence with ims_list.
        :param batch_size: Generator batch size
        :param shuffle: load batches with shuffle (True) or not (False).
        :param n_channels: number of image channels. Default to 1 (Grayscale).
        """
        self.ims_list = ims_list
        self.masks_list = masks_list
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.ims_list) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        ims_list_temp = [self.ims_list[k] for k in indexes]
        masks_list_temp = [self.masks_list[k] for k in indexes]

        return self.__data_generation(ims_list_temp, masks_list_temp)

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.ims_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ims_list_temp, masks_list_temp):
        """ This Data Generator class loads images and masks in a basic format: numpy arrays with shape (w, h). """
        X = list()
        y = list()
        # Generate data
        for i, (im_name, mask_name) in enumerate(zip(ims_list_temp, masks_list_temp)):
            # Load sample
            X.append(cv2.imread(im_name, cv2.IMREAD_GRAYSCALE))
            y.append(cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE))
        return X, y


class DataGeneratorPatches(Sequence):
    """ Data Generator to load patches from memory to a Tensor format. """
    def __init__(self,
                 ims_list: List[str],
                 masks_list: List[str],
                 dims: Tuple[int, int],
                 batch_size: int = params.BATCH_SIZE,
                 shuffle: bool = True,
                 n_channels: int = 1):
        """
        Initializes DataGenerator for patches
        :param ims_list: list of paths to patches images.
        :param masks_list: list of paths to patches masks. It must correspond with ims_list.
        :param dims: patch dimensions. It vary depending on the resize_ratio used to generate patches.
        :param batch_size: generator batch size.
        :param shuffle: load batches with shuffle (True) or not (False).
        :param n_channels: number of image channels. Default to 1 (Grayscale).
        """
        self.ims_list = ims_list
        self.masks_list = masks_list
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.dims = dims
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.ims_list) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        ims_list_temp = [self.ims_list[k] for k in indexes]
        masks_list_temp = [self.masks_list[k] for k in indexes]

        return self.__data_generation(ims_list_temp, masks_list_temp)

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.ims_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ims_list_temp, masks_list_temp):
        X = np.empty((self.batch_size, *self.dims, self.n_channels))
        y = np.empty((self.batch_size, *self.dims, self.n_channels))

        # Generate data
        for i, (im_name, mask_name) in enumerate(zip(ims_list_temp, masks_list_temp)):
            # Load sample
            X[i, :, :, 0] = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
            y[i, :, :, 0] = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        return X, y


def debugger():
    import os
    from glob import glob
    # Prepare lists of images and masks
    im_list = glob(params.DATASET_PATH + '/ims/*')
    masks_list = glob(params.DATASET_PATH + '/gt/masks/*')

    dataGen = DataGeneratorImages(im_list, masks_list)
    print(len(dataGen))
    print(dataGen[0])


# TESTING
if __name__ == '__main__':
    # Prepare lists of images and masks
    debugger()
