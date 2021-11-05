import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from typing import Tuple, List
import os, glob
from bs4 import BeautifulSoup
from tqdm import tqdm

GLOMERULI_SIZE = 400  # Empirically defined glomeruli size limit
DATA_PATH = "D:/DataGlomeruli"


class MaskCreator:
    def __init__(self):
        self._ims = DATA_PATH + '/ims'
        self._glomeruli_coords = DATA_PATH + '/glomeruli_coords'
        self._masks = DATA_PATH + '/masks'

        self._xml_file_list = glob.glob(self._glomeruli_coords + '/*')
        self._ims_file_list = glob.glob(self._ims + '/*')

    def run(self, r: int = GLOMERULI_SIZE):
        for xml_file, im_file in zip(self._xml_file_list, self._ims_file_list):
            points = self._get_points_from_xml(xml_file)
            im = self._get_im(im_file)
            mask = self._get_rounded_mask(im, points, r)
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
        mask_dir = self._masks + '_' + str(r)
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
        cv2.imwrite(os.path.join(mask_dir, mask_name), mask)


    @staticmethod
    def _get_im(imfile: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(imfile, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _get_rounded_mask(im: np.array, cs: List[Tuple[int, int]], r: int) -> np.ndarray:
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
        for c in cs:
            y, x = np.ogrid[-c[1]:h-c[1], -c[0]:w-c[0]]
            cmask = x*x + y*y <= r*r
            im_mask[cmask] = 255
        return im_mask

    @staticmethod
    def _plot_sample(im, mask):
        plt.figure()
        plt.imshow(im)
        plt.imshow(mask, cmap="jet", alpha=0.2)
        plt.show()
        # plt.close()


if __name__ == '__main__':
    maskCreator = MaskCreator()
    radii = np.arange(250, 600, 50)
    for radius in tqdm(radii, desc="Masks folders"):
        maskCreator.run(radius)
    # test()

