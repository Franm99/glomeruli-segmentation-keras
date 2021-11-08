import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from typing import Tuple, List
import os, glob
from bs4 import BeautifulSoup
from tqdm import tqdm
from enum import Enum, auto
import warnings

GLOMERULI_SIZE = 400  # Empirically defined glomeruli size limit
DATA_PATH = "D:/DataGlomeruli"


class MaskType(Enum):
    """Type of masks to generate."""
    CIRCULAR = auto()
    BBOX = auto()


class MaskGenerator:
    def __init__(self, mask_type: MaskType = MaskType.CIRCULAR):
        self._mask_type = mask_type
        self._ims = DATA_PATH + '/ims'
        self._glomeruli_coords = DATA_PATH + '/glomeruli_coords'
        self._masks = DATA_PATH + '/masks'

        if mask_type == MaskType.CIRCULAR:
            self._mask_type = mask_type
        elif mask_type == MaskType.BBOX:
            self._mask_type = mask_type
            self._masks = self._masks + '_bbox'
        else:
            warnings.warn("Invalid mask type. Getting default: Circular masks", category=ResourceWarning)
            self._mask_type = MaskType.CIRCULAR

        self._xml_file_list = glob.glob(self._glomeruli_coords + '/*')
        self._ims_file_list = glob.glob(self._ims + '/*')

    def run(self, r: int = GLOMERULI_SIZE):
        for xml_file, im_file in zip(self._xml_file_list, self._ims_file_list):
            points = self._get_points_from_xml(xml_file)
            im = self._get_im(im_file)
            mask = self._get_mask(im, points, r, self._mask_type)

            # self._plot_sample(im, mask)
            self._save_im(mask, im_file, r)

    def _get_points_from_xml(self, xml_file: str) -> List[Tuple[int, int]]:
        with open(xml_file, 'r') as f:
            data = f.read()
        bs_data = BeautifulSoup(data, "xml")
        point_fields = bs_data.find_all('point')
        p = []
        for link in point_fields:
            p.append((int(link.get('X')), int(link.get('Y'))))
        return p

    def _save_im(self, mask, im_file, r: int) -> None:
        mask_name = os.path.basename(im_file).split('.')[0] + '_mask.png'
        mask_dir = self._masks
        if self._mask_type == MaskType.CIRCULAR:
            mask_dir = mask_dir + '_' + str(r)
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
        cv2.imwrite(os.path.join(mask_dir, mask_name), mask)


    @staticmethod
    def _get_im(imfile: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(imfile, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _get_mask(im: np.array, cs: List[Tuple[int, int]], r: int, mask_type: MaskType) -> np.ndarray:
        """
        Create a binary mask where circles (255) mark the position of glomeruli in image.
        Source:  https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array/8650741
        :param im: 3-channel (RGB) image.
        :param c: (cy, cx) Y and X coordinates of the circle (glomeruli) centre.
        :param r: desired radius for each circle mask.
        :return: 4-channel (RGB+mask) image.
        """
        h, w = im.shape[1], im.shape[0]
        im_mask = np.zeros((h, w), dtype=np.uint8)
        if mask_type == MaskType.CIRCULAR:
            for c in cs:
                y, x = np.ogrid[-c[1]:h-c[1], -c[0]:w-c[0]]
                cmask = x*x + y*y <= r*r
                im_mask[cmask] = 255
            return im_mask
        else:
            for c in cs:
                sx, sy = int(c[0] - r/2), int(c[1] - r/2)
                lx = sx + r if sx + r < w else w
                ly = sy + r if sy + r < h else h
                im_mask[max(0, sy):ly, max(0, sx):lx] = 255
            return im_mask

    @staticmethod
    def _plot_sample(im, mask):
        plt.figure()
        plt.imshow(im)
        plt.imshow(mask, cmap="jet", alpha=0.2)
        plt.show()
        # plt.close()


if __name__ == '__main__':
    maskGenerator = MaskGenerator(mask_type=MaskType.CIRCULAR)
    # TODO: relaunch with smaller radii (150, 200, 250, 300)
    radii = np.arange(150, 350, 50)
    for radius in tqdm(radii, desc="Masks folders"):
        maskGenerator.run(radius)
    # test()

