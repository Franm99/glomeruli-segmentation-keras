import cv2.cv2 as cv2
import glob
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict
from shutil import rmtree
from random import sample
from abc import ABC, abstractmethod
import random
from tensorflow.keras.utils import normalize, Sequence
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial import distance
from bs4 import BeautifulSoup

from src.utils.misc import print_info, print_error, timer
import src.utils.parameters as params
import src.utils.constants as const
from src.utils.enums import MaskType, Size, GlomeruliClass


class DatasetImages:
    def __init__(self, staining: str, balance: bool):
        self._ims_path = os.path.join(const.SEGMENTER_DATA_PATH, staining, 'ims')
        self._masks_path = os.path.join(const.SEGMENTER_DATA_PATH, staining, 'gt/masks')

        self.ims_names = [i[:-4] for i in os.listdir(self._ims_path)]
        self.ims_list = [os.path.join(self._ims_path, f'{i}.png') for i in self.ims_names]
        lim = self.find_balance_limit()

        # self.ims_list = [i for i in glob.glob(self._ims_path + '/*') if staining in os.path.basename(i)]
        self.ims_list = glob.glob(self._ims_path + '/*')
        if balance:
            self.ims_list = sample(self.ims_list, lim)
        self.masks_list = [os.path.join(self._masks_path, os.path.basename(i)) for i in self.ims_list]

    def split_train_test(self, train_size: float):
        return train_test_split(self.ims_list, self.masks_list, train_size=train_size,
                                shuffle=True, random_state=params.TRAINVAL_TEST_RAND_STATE)

    @staticmethod
    def find_balance_limit():
        counter = {st:len(os.listdir(os.path.join(const.SEGMENTER_DATA_PATH, st, 'ims'))) for st in ['HE', 'PAS', 'PM']}
        return min(counter.values())


class DatasetPatches:
    def __init__(self, dir_path: str):
        self._dir_path = dir_path
        self._dir_patches = os.path.join(self._dir_path, 'patches')
        self._dir_patches_masks = os.path.join(self._dir_path, 'patches_masks')

        self.patches_list = glob.glob(self._dir_patches + '/*')
        self.patches_masks_list = glob.glob(self._dir_patches_masks + '/*')

    def clear(self):
        try:
            rmtree(self._dir_path)
        except:
            print("Failed to delete tmp directory.")


class DatasetTest:
    def __init__(self, ims_files: List[str], masks_files: List[str]):
        self.ims_files = ims_files
        self.masks_files = masks_files
        self.ims_names = [os.path.basename(i) for i in self.ims_files]

    def __getitem__(self, idx):
        return self._load(self.ims_files[idx]), self._load(self.masks_files[idx]), self.ims_names[idx]

    def __len__(self):
        return len(self.ims_files)

    @staticmethod
    def _load(im):
        return cv2.imread(im, cv2.IMREAD_GRAYSCALE)


class Dataset():
    """ Class for data loading and pre-processing before using it as input for the Segmentation keras. """
    def __init__(self, staining: str, mask_type: MaskType, mask_size: Optional[int], mask_simplex: bool):
        """
        Initialize Dataset.
        Paths initialization to find data in disk. Images and ground-truth masks full-paths are loaded.
        """
        # Paths
        self.data_path = os.path.join(const.SEGMENTER_DATA_PATH, staining)
        self._ims_path = self.data_path + '/ims'
        self._xmls_path = self.data_path + '/xml'
        if mask_type == MaskType.HANDCRAFTED:
            self._masks_path = self.data_path + '/gt/masks'
        else:
            if mask_type == MaskType.CIRCULAR:
                self._masks_path = self.data_path + '/gt/circles'
                if mask_size:
                    self._masks_path = self._masks_path + str(mask_size)
                if mask_simplex:
                    self._masks_path = self._masks_path + "_simplex"
            else:
                self._masks_path = self.data_path + '/gt/bboxes'

        # output directories
        # Files to keep track of the data used
        self._ims_list_path = self.data_path + '/ims.txt'  # patches names

        # Instance parameters initialization
        self._staining = staining
        self._mask_type = mask_type
        self._mask_size = mask_size
        self._mask_simplex = mask_simplex
        self.trainval_list = []
        self.test_list = []

        self.ims_names = [i[:-4] for i in os.listdir(self._ims_path)]
        self.ims_list = [self._ims_path + '/' + i + '.png' for i in self.ims_names]
        self.xml_list = [self._xmls_path + '/' + i + '.xml' for i in self.ims_names]

        # Images names will be writen to a file, and later used to load the remaining data.
        self.list2txt(self._ims_list_path, self.ims_names)

        # By now, everytime the training stage is launched, new masks are computed and saved.
        self.masks_list = self._load_masks(self._mask_type, self._mask_size, self._mask_simplex)
        if self._staining != "ALL":  # Load just the samples for the selected staining
            self.ims_list = [i for i in self.ims_list if self._staining in i]
            self.masks_list = [i for i in self.masks_list if self._staining in i]

    def split_trainval_test(self, train_size: float):
        """
        Implements train_test_split Keras method to split data into train+validation and test.
        self.ims_list and self.masks_list are lists with the same length containing path strings for every sample used.
        :param train_size: train+val proportion (0.0 to 1.0)
        :return: Tuple containing data and labels for both sets (train+val and test).
        """
        xtrainval, xtest, ytrainval, ytest = train_test_split(self.ims_list, self.masks_list, train_size=train_size,
                                                              shuffle=True, random_state=params.TRAINVAL_TEST_RAND_STATE)
        self.trainval_list = [os.path.basename(i) for i in xtrainval]
        self.test_list = [os.path.basename(i) for i in xtest]
        # self.list2txt(self._train_val_file, self.trainval_list)
        # self.list2txt(self._test_file, self.test_list)
        print_info("{} images for Train/Validation and {} for testing".format(len(xtrainval), len(xtest)))
        return xtrainval, xtest, ytrainval, ytest

    def split_train_val(self, ims, masks, test_size: float = 0.1):
        """
        Implements train_test_split Keras method to split data into train and validation sets.
        :param ims: set of sub-patches images
        :param masks: set of masks images (same order as ims)
        :param test_size: proportion for the validation set
        :return: tuple containing data and labels for both sets (train and validation)
        """
        xtrain, xval, ytrain, yval = train_test_split(ims, masks, test_size=test_size,
                                                      shuffle=True, random_state=params.TRAIN_VAL_RAND_STATE)
        print_info("{} sub-patches for training, {} for validation.".format(len(xtrain), len(xval)))
        return xtrain, xval, ytrain, yval

    @timer
    def load_pairs(self, x: List[str], y: List[str], limit_samples: bool = None):
        """
        Method to load pairs of images and masks from disk.
        :param x: Set of full-path strings for images
        :param y: Set of full-path strings for labels (masks).
        :param limit_samples: [DEBUG] Select a reduced portion of data to load from disk. This parameter is useful for
        faster debugging, as loading images from disk can take a notorious time.
        :return: tuple containing images and masks in Numpy ndarray format.
        """
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
    def get_spatches(self, ims: List[np.ndarray], masks: List[np.ndarray], rz_ratio: int,
                     filter_spatches: bool = params.FILTER_SUBPATCHES) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Method to generate sub-patches from original patches with compatible dimensions for the keras input.
        :param ims: List of images in Numpy ndarray format.
        :param masks: List of masks in Numpy ndarray format.
        :param rz_ratio: Resize ratio, used to select the relative size of sub-patches on images.
        :param filter_spatches: Choose whether to add patches not containing glomeruli or filter them.
        :return: tuple containing two lists: patches images and masks. Numpy array format, NOT NORMALIZED YET!.
        """
        patches = []
        patches_masks = []
        print_info("Generating sub-patches for training stage and saving to disk...")
        patch_size_or = const.UNET_INPUT_SIZE * rz_ratio
        for im, mask in tqdm(zip(ims, masks), total=len(ims), desc ="Generating subpatches"):
            [h, w] = im.shape
            for x in range(0, w, patch_size_or):
                if x+patch_size_or >= w:
                    x = w - patch_size_or
                for y in range(0, h, patch_size_or):
                    if y+patch_size_or >= h:
                        y = h - patch_size_or
                    patch_arr = im[y:y+patch_size_or, x:x+patch_size_or]
                    mask_arr = mask[y:y+patch_size_or, x:x+patch_size_or]
                    if filter_spatches:
                        if not self._filter(mask_arr):
                            continue
                    patch = np.asarray(
                        Image.fromarray(patch_arr).resize((const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE)))
                    patch_mask = self.binarize(np.asarray(
                        Image.fromarray(mask_arr).resize((const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE))))
                    patches.append(patch)
                    patches_masks.append(patch_mask)

        print_info("{} patches generated from {} images for training and validation.".format(len(patches), len(ims)))
        return patches, patches_masks

    @timer
    def _load_masks(self, mask_type: MaskType, mask_size: Optional[int], mask_simplex: bool) -> List[str]:
        """
        Method to load the groundtruth (masks). Different type of masks can be used.
        :param mask_type: Parameter to specify the mask type to use: HANDCRAFTED masks, loaded from disk, or Synthetic
        masks, that can be either CIRCULAR or BBOX masks. Both of the synthetic masks need coordinates info obtained
        from the xml files associated to each image.
        :param mask_size: In case CIRCULAR masks are used, this parameter sets its radius. If None, radii are computed
        from glomeruli classes.
        :param mask_simplex: In case CIRCULAR masks are used, if True, the Simplex algorithm is applied to avoid masks
        overlap.
        :return: List containing full-path strings with the names of every generated (or existing) mask.
        """
        ims_names = self.txt2list(self._ims_list_path)
        if mask_type != MaskType.HANDCRAFTED:
            if os.path.isdir(self._masks_path):
                # If masks already exists, delete.
                self.clear_dir(self._masks_path)
            else:
                os.mkdir(self._masks_path)
        maskGenerator = MaskGenerator(ims_names=ims_names, mask_type=params.MASK_TYPE,
                                      mask_size=mask_size, apply_simplex=mask_simplex)
        return maskGenerator.get_masks_files()

    # PRIVATE
    @staticmethod
    def _filter(patch: np.ndarray) -> bool:  # Modify: Do not include sub-patches without glomeruli
        """
        Patch filter based on median value from ordered histogram to find patches containing kidney tissue.
        :param patch: patch to check up.
        :return: True if patch contains tissue, False if not.
        """
        return np.sum(patch) > 0

    @staticmethod
    def binarize(im):
        th = 200
        return 255 * (im > th)

    # def _save_train_dataset(self, ims: List[np.ndarray], masks: List[np.ndarray], output_folder: str) -> List[str]:
    #     """
    #     Method to save in disk the set of sub-patches images and masks previously generated.
    #     :param ims: list of sub-patches images in numpy ndarray format.
    #     :param masks: list of sub-patches masks in numpy ndarray format.
    #     :return: list of sub-patches names
    #     """
    #     # Create both train and validation patches folders.
    #     if not os.path.isdir(self._train_val_ims_path):
    #         os.mkdir(self._train_val_ims_path)
    #         os.mkdir(self._train_val_masks_path)
    #
    #     # Clear data if existing
    #     self.clear_dir(self._train_val_ims_path)
    #     self.clear_dir(self._train_val_masks_path)
    #
    #     num_digits = len(str(len(ims))) + 1
    #     spatches_names = []
    #     for idx, (im, mask) in enumerate(zip(ims, masks)):
    #         bname = str(idx).zfill(num_digits) + ".png"
    #         spatches_names.append(bname)
    #         cv2.imwrite(os.path.join(self._train_val_ims_path, bname), im)
    #         cv2.imwrite(os.path.join(self._train_val_masks_path, bname), mask)
    #     self.list2txt(self._subpatches_file, spatches_names)
    #     return spatches_names

    @staticmethod
    def txt2list(fname: str) -> List[str]:
        """
        Method to read file names from a txt file and save to a python list.
        :param fname: txt file full path
        :return: file content in python list format
        """
        with open(fname, 'r') as f:
            return [line.rstrip('\n') for line in f.readlines()]

    # STATIC
    @staticmethod
    def list2txt(fname: str, data: List[str]) -> None:
        """
        Method to save a list of strings to a txt file.
        :param fname: txt file full path
        :param data: list containing the data to save in file
        :return: None
        """
        with open(fname, 'w') as f:
            for i in data:
                f.write(i + "\n")

    @staticmethod
    def clear_dir(dpath : str) -> None:
        """
        Method to clear the content of the specified directory
        :param dpath: folder full path
        :return: None
        """
        files = glob.glob(dpath + '/*')
        for file in files:
            try:
                os.unlink(file)
            except Exception as e:
                print_error("Failed to delete files: Reason: {}".format(e))

    def get_data_list(self, set: str) -> List[str]:
        """
        Method to obtain an specific data list: 'test', 'train', 'val'
        :param set: name of the desired set in string format.
        :return: class attribute containing the name list of the desired set.
        """
        return eval("self." + set + "_list")



class DataGenerator(ABC, Sequence):
    @abstractmethod
    def __init__(self,
                 ims_list: List[str],
                 masks_list: List[str],
                 batch_size: int,
                 shuffle: bool,
                 n_channels: int):
        self.ims_list = ims_list
        self.masks_list = masks_list
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass

    @abstractmethod
    def _data_generation(self, ims_list_temp, masks_list_temp):
        pass


class DataGeneratorImages(DataGenerator):
    """ Data generator class to load images and masks pairs in grayscale format. """
    def __init__(self,
                 ims_list: List[str],
                 masks_list: List[str],
                 batch_size: int = 16,
                 shuffle: bool = True,
                 n_channels: int = 1,
                 augmentation: bool = False):
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
        self.augmentation = augmentation
        if self.augmentation:
            self.ims_list, self.masks_list = self.data_cloning()
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

    def data_cloning(self, target_size: int = 360):
        indexes = list(range(len(self.ims_list)))
        indexes.extend([random.choice(indexes) for _ in range(target_size - len(indexes))])
        ims_aug = [self.ims_list[i] for i in indexes]
        masks_aug = [self.masks_list[i] for i in indexes]
        return ims_aug, masks_aug


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
                    # if w - x <= self.resized_img_dim // 5:
                    #     continue
                    x = w - self.patch_dim
                for y in range(0, h, self.patch_dim):
                    if y + self.patch_dim >= h:
                        # if h - y <= self.resized_img_dim // 5:
                        #     continue
                        y = h - self.patch_dim
                    patch_arr = im[y: y + self.patch_dim, x: x + self.patch_dim]
                    mask_arr = mask[y: y + self.patch_dim, x: x + self.patch_dim]
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



class MaskGenerator:
    """ Class to load or create masks for the groundtruth """
    def __init__(self, ims_names: List[str], mask_type: MaskType = MaskType.CIRCULAR,
                 mask_size: Optional[int] = params.MASK_SIZE, apply_simplex: bool = params.APPLY_SIMPLEX):
        """
        Initialize maskGenerator object.
        :param ims_names: list with full path to every image from which to generate or load masks.
        :param mask_type: HANDCRAFTED, CIRCULAR or BBOX are available mask types.
        :param mask_size: When using CIRCULAR mask type, to specify radius. If None, it depends on the glomeruli class.
        :param apply_simplex: When using CIRCULAR mask type, if True, modifies radii to avoid overlap between masks.
        """
        self._ims_names = ims_names
        self._mask_type = mask_type
        self._mask_size = mask_size
        self._apply_simplex = apply_simplex
        self._ims = params.DATASET_PATH + '/ims'
        self._ims_file_list = [self._ims + '/' + i + '.png' for i in self._ims_names]
        self._glomeruli_coords = params.DATASET_PATH + '/xml'
        self._xml_file_list = [self._glomeruli_coords + '/' + i + '.xml' for i in self._ims_names]

        if mask_type == MaskType.HANDCRAFTED:
            self._masks = params.DATASET_PATH + '/gt/masks'
        else:
            # Generate synthetic masks
            if mask_type == MaskType.CIRCULAR:
                self._masks = params.DATASET_PATH + '/gt/circles'
                if self._mask_size:
                    self._masks = self._masks + str(self._mask_size)
                if self._apply_simplex:
                    self._masks = self._masks + "_simplex"
            elif mask_type == MaskType.BBOX:
                self._masks = params.DATASET_PATH + 'gt/bboxes'
            # Execute the mask generation process to obtain synthetic masks
            self._gen_masks()

    def get_masks_files(self) -> List[str]:
        """
        Method to get full paths for every mask in disk.
        :return: list of masks full paths, ordered as the list of images used.
        """
        return [self._masks + '/' + name + '.png' for name in self._ims_names]

    def _gen_masks(self) -> None:
        """
        Method to generate masks, using xml files to obtain glomeruli coordinates and classes.
        :return: None
        """
        print_info("Generating masks for groundtruth...")
        for xml_file, im_file in tqdm(zip(self._xml_file_list, self._ims_file_list),
                                      total=len(self._ims_file_list), desc="Generating masks"):
            points = get_data_from_xml(xml_file, mask_size=self._mask_size, apply_simplex=self._apply_simplex)
            mask = self._get_mask(points)
            # self._plot_sample(im_file, mask)  # DEBUG
            # Save mask to disk
            self._save_im(mask, im_file)

    def _get_mask(self, data: Dict[int, List[Tuple[int, int]]]) -> np.ndarray:
        """
        Create a binary mask where circles (255) mark the position of glomeruli in image.
        Source:  https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array/8650741
        :param data: dictionary with information about each glomeruli class.
        :return: synthetic circular masks.
        """
        h, w = params.PATCH_SIZE[1], params.PATCH_SIZE[0]
        im_mask = np.zeros((h, w), dtype=np.uint8)
        if self._mask_type == MaskType.CIRCULAR:
            for r in data.keys():
                cs = data[r]
                for c in cs:
                    y, x = np.ogrid[-c[1]:h-c[1], -c[0]:w-c[0]]
                    cmask = x*x + y*y <= r*r
                    im_mask[cmask] = 255
        return im_mask

    def _save_im(self, mask: np.ndarray, im_file: str) -> None:
        """
        Save image (mask) to disk
        :param mask: binary mask to save
        :param im_file: desired name for the mask to save
        :return: None
        """
        mask_name = os.path.basename(im_file)
        cv2.imwrite(os.path.join(self._masks, mask_name), mask)

    @staticmethod
    def _plot_sample(im_path: str, mask: np.ndarray) -> None:
        """
        [DEBUG] Method to show an image overlapped with its own groundtruth mask.
        :param im_path: path to desired image
        :param mask: mask for the specified image
        :return: None
        """
        plt.figure()
        plt.imshow(cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        plt.imshow(mask, cmap="jet", alpha=0.3)
        plt.show()


def get_data_from_xml(xml_file: str, mask_size: Optional[int],
                      apply_simplex: bool) -> Dict[int, List[Tuple[int, int]]]:
    """
    Read glomeruli classes and coordinates from glomeruli xml file.
    :param xml_file: full path to an xml file containing glomeruli information.
    :param mask_size: Desired mask size for synthetic masks. If None, the glomeruli class is used to define radii.
    :param apply_simplex: If True, Simplex algorithm is applied to avoid mask overlap.
    :return: Dictionary with keys referring to glomeruli dimensions, with lists of coordinates as values.
    """
    with open(xml_file, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")
    counts = bs_data.find("Counts").find_all("Count")

    if mask_size:  # Using fixed mask size
        glomeruli = {mask_size: []}
        for count in counts:
            points = count.find_all('point')
            glomeruli[mask_size].extend([(int(point.get('X')), int(point.get('Y'))) for point in points])
    else:  # Using variable mask size based on glomeruli class
        glomeruli = {x.value: [] for x in Size}
        for count in counts:
            name = count.get('name')
            points = count.find_all('point')
            p = []
            for point in points:
                p.append((int(point.get('X')), int(point.get('Y'))))
            glomeruli[GlomeruliClass[name]].extend(p)
    if apply_simplex:
        glomeruli = simplex(glomeruli)
    return glomeruli


def simplex(data: Dict[int, List[Tuple[int, int]]]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Apply Simplex algorithm to avoid masks overlap.
    Simplex algorithm: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-simplex.html
    :param data: dictionary containing glomeruli information.
    :return: dictionary containing glomeruli information, with modified radii just when overlap occurs.
    """

    # Compute D2: size limits for each point (i.e., data key)
    D2 = np.asarray([size for size in data.keys() for _ in range(len(data[size]))])
    # Compute X: Set of points. Format: [xc', yc']
    X = np.asarray([i for size in data.keys() for i in data[size]])  # Set of points
    # data_ = [[sz, p] for sz in data.keys() for p in data[sz]]
    # Compute D1: Distance between each points pair
    D1 = distance.pdist(X, metric='euclidean')
    N = len(X)

    # Search for duplicate labels (very near coordinates)
    c = np.asarray(list(combinations(np.arange(N), 2)))
    targets = c[D1 < 100]  # Threshold set to the minimum radius size allowed
    to_delete = [tg[0] if D2[tg[0]] < D2[tg[1]] else tg[1] for tg in targets]

    # Re-Compute D2,X and N solving duplicates
    D2 = np.delete(D2, to_delete)
    X = np.delete(X, to_delete, axis=0)

    # Re-construct data dict
    data = {i: [] for i in set(D2)}
    for idx, p in enumerate(X):
        data[D2[idx]].append(tuple(p))

    N = len(X)
    # Re-Compute D1
    D1 = distance.pdist(X, metric='euclidean')

    # Lower triangle
    M = np.zeros((N, N), dtype=D1.dtype)
    # Operations to maintain same format as in MATLAB
    (rows_idx, cols_idx) = np.tril_indices(N, k=-1)
    arrlinds = cols_idx.argsort()
    srows_idx, scols_idx = rows_idx[arrlinds], cols_idx[arrlinds]
    M[srows_idx, scols_idx] = D1

    # Prepare V
    fil = len(D1)
    V = np.zeros((fil, N))
    params = 0
    for j in range(N):
        for i in range(N):
            if M[i, j] != 0:
                V[params, i] = 1
                V[params, j] = 1
                params += 1

    # f: function to minimize
    f = np.ones((N, 1)) * 2 * np.pi
    A = np.eye(N)
    A = np.concatenate((A, V), axis=0)
    B = np.concatenate((D2.T, D1.T), axis=0)

    res = linprog(-f, A, B, method="simplex")
    nlims = res.x.astype(np.int)

    # Update data dictionary with new mask sizes
    for idx, (plim, nlim) in enumerate(zip(D2, nlims)):
        if plim != nlim:
            point = data[plim].pop(0)
            if nlim not in data.keys():
                data[nlim] = []
            data[nlim].append(point)
    return data
