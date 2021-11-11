import os
import glob
from typing import List, Optional
from mask_generator.MaskGenerator import MaskGenerator


_DEF_MASK_SIZE = 150
_DATASET_PATH = 'D:/DataGlomeruli'
_DEF_TRAIN_SIZE = 0.8
_DEF_STAINING = 'HE'


class Dataset():
    def __init__(self, train_size: float = _DEF_TRAIN_SIZE, mask_size: int = _DEF_MASK_SIZE,
                 staining: str = _DEF_STAINING):
        # Paths
        self._ims_path = _DATASET_PATH + '/ims'
        self._xmls_path = _DATASET_PATH + '/xml'
        self._masks_path = _DATASET_PATH + '/gt' + '/circles' + str(mask_size)
        self._train_file = _DATASET_PATH + '/train.txt'
        self._test_file = _DATASET_PATH + '/test.txt'

        # Variables
        self._staining = staining
        self._mask_size = mask_size
        self._train_size = train_size

        self.masks = self._load_masks()

    def _load_masks(self):

        # Check for existing masks folder for desired size
        if not os.path.isdir(self._masks_path):
            maskGenerator = MaskGenerator(r=self._mask_size)
            return maskGenerator.get_masks_files()
        return glob.glob(self._masks_path + '/*.png')


# Testing
if __name__ == '__main__':
    dataset = Dataset()
    print(dataset.masks)
