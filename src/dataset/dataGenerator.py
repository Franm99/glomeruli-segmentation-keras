""" Data generator class to progressively load data to RAM """
from tensorflow.keras.utils import normalize
from typing import List, Tuple
import numpy as np
import cv2.cv2 as cv2
from PIL import Image

import src.parameters as params
import src.constants as const
from src.utils.utils import DataGenerator


class DataGeneratorImages(DataGenerator):
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

        return self._data_generation(ims_list_temp, masks_list_temp)

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.ims_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, ims_list_temp, masks_list_temp):
        """ This Data Generator class loads images and masks in a basic format: numpy arrays with shape (w, h). """
        X = list()
        y = list()
        # Generate data
        for i, (im_name, mask_name) in enumerate(zip(ims_list_temp, masks_list_temp)):
            # Load sample
            X.append(cv2.imread(im_name, cv2.IMREAD_GRAYSCALE))
            y.append(cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE))
        return X, y


class DataGeneratorPatches(DataGenerator):
    """ Data Generator to load patches from memory to a Tensor format. """
    def __init__(self,
                 ims_list: List[str],
                 masks_list: List[str],
                 dims: Tuple[int, int] = (const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE),
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
        self.class_values = np.array([0, 1])
        self.class_weights = np.array([1, 10])

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.ims_list) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        ims_list_temp = [self.ims_list[k] for k in indexes]
        masks_list_temp = [self.masks_list[k] for k in indexes]

        # Return a three-elements tuple when using sample_weights with data generators.
        # Source: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        return self._data_generation(ims_list_temp, masks_list_temp)

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.ims_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, ims_list_temp, masks_list_temp):
        X = np.empty((self.batch_size, *self.dims, self.n_channels))
        y = np.empty((self.batch_size, *self.dims, self.n_channels))
        w = np.empty((self.batch_size, *self.dims, self.n_channels))

        # Generate data
        for i, (im_name, mask_name) in enumerate(zip(ims_list_temp, masks_list_temp)):
            # Load sample
            im_tensor, mask_tensor = self._normalize_sample(cv2.imread(im_name, cv2.IMREAD_GRAYSCALE),
                                                            cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE))
            X[i] = im_tensor
            y[i] = mask_tensor
            w[i] = self.get_sample_weights(y[i])

        return X, y #, w

    def _normalize_sample(self, im: np.ndarray, mask: np.ndarray) -> Tuple:
        im_tensor = np.expand_dims(normalize(np.array(im), axis=1), 2)
        mask_tensor = np.expand_dims(mask, 2) / 255
        return im_tensor, mask_tensor

    def get_sample_weights(self, y):
        # class_weights = compute_class_weight('balanced', classes=self.class_values, y=y.flatten())
        w = np.zeros(y.shape, dtype=np.uint8)
        w[y == 0] = self.class_weights[0]
        w[y == 1] = self.class_weights[1]
        return w


class PatchGenerator:
    def __init__(self,
                 patch_dim: int,
                 squared_dim: int,
                 filter: bool = True):
        self.patch_dim = patch_dim
        self.filter = filter

        self.batch_counter = 0
        self.global_counter = 0
        self.basename_length = 8

    def generate(self, ims: List[np.ndarray], masks: List[np.ndarray]):
        patches = list()
        patches_masks = list()
        for im, mask in zip(ims, masks):
            h, w = im.shape
            for x in range(0, w, self.patch_dim):
                if x + self.patch_dim >= w:
                    # if w - x <= self.patch_dim // 5:
                    #     continue
                    x = w - self.patch_dim
                for y in range(0, h, self.patch_dim):
                    if y + self.patch_dim >= h:
                        # if h - y <= self.patch_dim // 5:
                        #     continue
                        y = h - self.patch_dim
                    patch_arr = im[y: y + self.patch_dim, x: x + self.patch_dim]
                    mask_arr = mask[y: y + self.patch_dim, x: x + self.patch_dim]
                    # TODO: Add balancing stage when no filtering is used.
                    if self.filter:
                        if not self._include_patch(mask_arr):
                            continue
                    patch = np.asarray(
                        Image.fromarray(patch_arr).resize((const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE)))
                    patch_mask = self.binarize(np.asarray(
                        Image.fromarray(mask_arr).resize((const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE))))
                    patches.append(patch)
                    patches_masks.append(patch_mask)
        self.batch_counter = len(patches)
        self.global_counter += self.batch_counter
        return patches, patches_masks, self._generate_names()

    def _generate_names(self):
        limits = (self.global_counter - self.batch_counter, self.global_counter)
        names = list()
        for i in range(limits[0], limits[1]):
            bname = str(i).zfill(self.basename_length) + ".png"
            names.append(bname)
        return names

    @staticmethod
    def _include_patch(mask) -> bool:
        """
        Patch filter based on median value from ordered histogram to find patches containing kidney tissue.
        :param mask: patch to check up.
        :return: True if patch contains tissue, False if not.
        """
        return np.sum(mask) > 0

    @staticmethod
    def binarize(im):
        th = 200
        return 255 * (im > th)
