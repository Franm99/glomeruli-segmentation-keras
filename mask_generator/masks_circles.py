import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from typing import Tuple, List

GLOMERULI_SIZE = 600  # Empirically defined glomeruli size limit


class MaskCreator:
    def __init__(self):
        pass

    def get_mask(self, im: np.array):
        # TODO: read centre points from XML
        cs = [(50, 50), (1200, 1450), (300, 3000)]
        mask = self.__create_rounded_masks(im, cs)
        plt.figure()
        plt.subplot(121)
        plt.imshow(im)
        plt.subplot(122)
        plt.imshow(mask, cmap="gray")
        plt.show()

    @staticmethod
    def __create_rounded_masks(im: np.array, cs: List[Tuple[int, int]], r: int = GLOMERULI_SIZE):
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
            y, x = np.ogrid[-c[0]:h-c[0], -c[1]:w-c[1]]
            cmask = x*x + y*y <= r*r
            im_mask[cmask] = 255
        return im_mask


if __name__ == '__main__':
    maskCreator = MaskCreator()

    impath = '21A3 A5 HE_x0y2400s3200.png'
    img = cv2.imread(impath, cv2.IMREAD_COLOR)
    maskCreator.get_mask(img)

