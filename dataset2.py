import os
import glob
from typing import List, Optional
from mask_generator.MaskGenerator import MaskGenerator
from sklearn.model_selection import train_test_split
from utils import print_info

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

        # Variables
        self._staining = staining
        self._mask_size = mask_size

        # List of data paths (instead of numpy arrays)
        self.ims_list = glob.glob(self._ims_path + '/*')
        self.masks_list = self._load_masks()

    def split(self, train_size: float, overwrite: bool = False):
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


    # PRIVATE
    def _load_masks(self):
        # Check for existing masks folder for desired size
        if not os.path.isdir(self._masks_path):
            maskGenerator = MaskGenerator(r=self._mask_size)
            return maskGenerator.get_masks_files()
        return glob.glob(self._masks_path + '/*.png')

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


# Testing
# if __name__ == '__main__':
#     dataset = Dataset()
#     xtrainval, xtest, ytrainval, ytest = dataset.split(0.9, overwrite=True)
#     print_info("Train/Validation set size: {}".format(len(xtrainval)))
#     print_info("Test set size: {}".format(len(xtest)))

