import cv2.cv2 as cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import parameters as params
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
from utils import get_data_from_xml, print_info, MaskType


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


# if __name__ == '__main__':
#     maskGenerator = MaskGenerator(mask_type=MaskType.CIRCULAR, mask_size=params.MASK_SIZE, apply_simplex=False)
#     maskGenerator.get_masks_files()
#     # test()

