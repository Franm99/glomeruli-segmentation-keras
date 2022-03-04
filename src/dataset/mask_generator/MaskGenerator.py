import cv2.cv2 as cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
from bs4 import BeautifulSoup
from itertools import combinations
from scipy.optimize import linprog
from scipy.spatial import distance

import src.parameters as params
from src.utils.misc import print_info
from src.utils.enums import GlomeruliClass
from src.utils.enums import MaskType, Size


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
