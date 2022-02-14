import cv2.cv2 as cv2
import glob
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional, List, Tuple
from shutil import rmtree

from src.utils.utils import print_info, print_warn, print_error, timer
from src.utils.enums import MaskType
from src.dataset.mask_generator.MaskGenerator import MaskGenerator
import src.parameters as params


class DatasetImages:
    def __init__(self, staining: str):
        self._ims_path = os.path.join(params.DATASET_PATH, 'ims')
        self._masks_path = os.path.join(params.DATASET_PATH, 'gt/masks')

        self.ims_names = [i[:-4] for i in os.listdir(self._ims_path)]
        self.ims_list = [os.path.join(self._ims_path, f'{i}.png') for i in self.ims_names]

        self.ims_list = [i for i in glob.glob(self._ims_path + '/*') if staining in os.path.basename(i)]
        self.masks_list = [os.path.join(self._masks_path, os.path.basename(i)) for i in self.ims_list]

    def split_train_test(self, train_size: float):
        return train_test_split(self.ims_list, self.masks_list, train_size=train_size,
                                shuffle=True, random_state=params.TRAINVAL_TEST_RAND_STATE)


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


class TestDataset:
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
    """ Class for data loading and pre-processing before using it as input for the Segmentation model. """
    def __init__(self, staining: str, mask_type: MaskType, mask_size: Optional[int], mask_simplex: bool):
        """
        Initialize Dataset.
        Paths initialization to find data in disk. Images and ground-truth masks full-paths are loaded.
        """
        # Paths
        self._ims_path = params.DATASET_PATH + '/ims'
        self._xmls_path = params.DATASET_PATH + '/xml'
        if mask_type == MaskType.HANDCRAFTED:
            self._masks_path = params.DATASET_PATH + '/gt/masks'
        else:
            if mask_type == MaskType.CIRCULAR:
                self._masks_path = params.DATASET_PATH + '/gt/circles'
                if mask_size:
                    self._masks_path = self._masks_path + str(mask_size)
                if mask_simplex:
                    self._masks_path = self._masks_path + "_simplex"
            else:
                self._masks_path = params.DATASET_PATH + '/gt/bboxes'

        # output directories
        # Files to keep track of the data used
        self._ims_list_path = params.DATASET_PATH + '/ims.txt'  # patches names
        # self._test_file = params.DATASET_PATH + '/test.txt'  # names of test images

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
        Method to generate sub-patches from original patches with compatible dimensions for the model input.
        :param ims: List of images in Numpy ndarray format.
        :param masks: List of masks in Numpy ndarray format.
        :param rz_ratio: Resize ratio, used to select the relative size of sub-patches on images.
        :param filter_spatches: Choose whether to add patches not containing glomeruli or filter them.
        :return: tuple containing two lists: patches images and masks. Numpy array format, NOT NORMALIZED YET!.
        """
        patches = []
        patches_masks = []
        print_info("Generating sub-patches for training stage and saving to disk...")
        patch_size_or = params.UNET_INPUT_SIZE * rz_ratio
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
                    # TODO: Add balancing stage when no filtering is used.
                    if filter_spatches:
                        if not self._filter(mask_arr):
                            continue
                    patch = np.asarray(
                        Image.fromarray(patch_arr).resize((params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE)))
                    patch_mask = self.binarize(np.asarray(
                        Image.fromarray(mask_arr).resize((params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE))))
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


# Testing
# if __name__ == '__main__':
#     print_info("Building dataset...")
#     dataset = Dataset(staining="HE", mask_size=None, mask_simplex=False)
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

def debugging():
    import glob
    im_list = glob.glob(params.DATASET_PATH + '/ims/*')[:10]
    masks_list = glob.glob(params.DATASET_PATH + '/gt/masks/*')[:10]
    testDataset = TestDataset(im_list, masks_list)
    for im, mask, name in testDataset:
        print(im)
        print(mask)
        print(name)


if __name__ == '__main__':
    debugging()



