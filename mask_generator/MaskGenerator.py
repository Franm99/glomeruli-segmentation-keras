import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from typing import Tuple, List, Optional
import os
import glob
from bs4 import BeautifulSoup
from tqdm import tqdm
from enum import Enum, auto
from utils import print_info

DATA_PATH = "D:/DataGlomeruli"
IMAGE_SIZE = (3200, 3200)  # Width, Height


class MaskType(Enum):
    """Type of masks to generate."""
    CIRCULAR = auto()
    BBOX = auto()


class MaskGenerator:
    def __init__(self, r: Optional[int], mask_type: MaskType = MaskType.CIRCULAR):
        self._r = r
        self._mask_type = mask_type
        self._ims = DATA_PATH + '/ims'
        self._glomeruli_coords = DATA_PATH + '/xml'

        if mask_type == MaskType.CIRCULAR:
            self._masks = DATA_PATH + '/gt/circles'
            if r:
                self._masks = self._masks + str(r)
        else:
            self._masks = DATA_PATH + 'gt/bboxes'

        self._xml_file_list = glob.glob(self._glomeruli_coords + '/*')
        self._ims_file_list = glob.glob(self._ims + '/*')

        # Execute the mask generation process
        self._masks_file_list = self._run()

    def get_masks_files(self):
        return self._masks_file_list

    def _run(self) -> List[str]:
        print_info("Generating masks for groundtruth...")
        for xml_file, im_file in tqdm(zip(self._xml_file_list, self._ims_file_list),
                                      total=len(self._ims_file_list), desc="Generating masks"):
            # TODO: deal with the case where r is None (more than one radius)
            points = self._get_points_from_xml(xml_file)
            mask = self._get_mask(points)
            # self._plot_sample(im_file, mask)  # DEBUG
            self._save_im(mask, im_file)
        return glob.glob(self._masks + '/*.png')

    @staticmethod
    def _get_points_from_xml(xml_file: str) -> List[Tuple[int, int]]:
        # TODO: modify: use glomeruli type information
        with open(xml_file, 'r') as f:
            data = f.read()
        bs_data = BeautifulSoup(data, "xml")
        point_fields = bs_data.find_all('point')
        p = []
        for link in point_fields:
            p.append((int(link.get('X')), int(link.get('Y'))))
        return p

    def _get_mask(self, cs: List[Tuple[int, int]]) -> np.ndarray:
        # TODO: modify: use glomeruli type information
        """
        Create a binary mask where circles (255) mark the position of glomeruli in image.
        Source:  https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array/8650741
        :param cs: List[(cy, cx)] Y and X coordinates of the circle (glomeruli) centre.
        :return: binary mask with dimensions IMAGE_SIZE.
        """
        h, w = IMAGE_SIZE[1], IMAGE_SIZE[0]
        im_mask = np.zeros((h, w), dtype=np.uint8)
        if self._mask_type == MaskType.CIRCULAR:
            for c in cs:
                y, x = np.ogrid[-c[1]:h-c[1], -c[0]:w-c[0]]
                cmask = x*x + y*y <= self._r * self._r
                im_mask[cmask] = 255
            return im_mask
        else:
            for c in cs:
                sx, sy = int(c[0] - self._r/2), int(c[1] - self._r/2)
                lx = sx + self._r if sx + self._r < w else w
                ly = sy + self._r if sy + self._r < h else h
                im_mask[max(0, sy):ly, max(0, sx):lx] = 255
            return im_mask

    def _save_im(self, mask, im_file) -> None:
        mask_name = os.path.basename(im_file)
        if not os.path.exists(self._masks):
            os.mkdir(self._masks)
        cv2.imwrite(os.path.join(self._masks, mask_name), mask)

    @staticmethod
    def _plot_sample(im_path, mask):
        plt.figure()
        plt.imshow(cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        plt.imshow(mask, cmap="jet", alpha=0.3)
        plt.show()
        # plt.close()


# if __name__ == '__main__':
#     maskGenerator = MaskGenerator(mask_type=MaskType.CIRCULAR)
#     radii = np.arange(150, 350, 50)
#     for radius in tqdm(radii, desc="Masks folders"):
#         maskGenerator.run(radius)
#     # test()

