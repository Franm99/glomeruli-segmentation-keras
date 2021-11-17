import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from typing import Tuple, List, Dict
import os
import glob
from tqdm import tqdm
from utils import get_data_from_xml, print_info, MaskType
import constants as ct


class MaskGenerator:
    def __init__(self, mask_type: MaskType = MaskType.CIRCULAR):
        self._mask_type = mask_type
        self._ims = ct.DATASET_PATH + '/ims'
        self._glomeruli_coords = ct.DATASET_PATH + '/xml'

        if mask_type == MaskType.CIRCULAR:
            self._masks = ct.DATASET_PATH + '/gt/circles'
        else:
            self._masks = ct.DATASET_PATH + 'gt/bboxes'

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
            points = get_data_from_xml(xml_file, apply_simplex=True)
            mask = self._get_mask(points)
            # self._plot_sample(im_file, mask)  # DEBUG
            self._save_im(mask, im_file)
        return glob.glob(self._masks + '/*.png')

    def _get_mask(self, data: Dict[int, List[Tuple[int, int]]]) -> np.ndarray:
        """
        Create a binary mask where circles (255) mark the position of glomeruli in image.
        Source:  https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array/8650741
        """
        h, w = ct._PATCH_SIZE[1], ct._PATCH_SIZE[0]
        im_mask = np.zeros((h, w), dtype=np.uint8)
        if self._mask_type == MaskType.CIRCULAR:
            for r in data.keys():
                cs = data[r]
                for c in cs:
                    y, x = np.ogrid[-c[1]:h-c[1], -c[0]:w-c[0]]
                    cmask = x*x + y*y <= r*r
                    im_mask[cmask] = 255
        return im_mask
        # else:
        #     for c in cs:
        #         sx, sy = int(c[0] - self._r/2), int(c[1] - self._r/2)
        #         lx = sx + self._r if sx + self._r < w else w
        #         ly = sy + self._r if sy + self._r < h else h
        #         im_mask[max(0, sy):ly, max(0, sx):lx] = 255
        #     return im_mask

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


if __name__ == '__main__':
    maskGenerator = MaskGenerator(mask_type=MaskType.CIRCULAR)
    maskGenerator.get_masks_files()
    # test()

