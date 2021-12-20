import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import numpy as np
import math
import time
from bs4 import BeautifulSoup
from enum import Enum, auto
from scipy.spatial import distance
from scipy.optimize import linprog
from collections import OrderedDict
from itertools import combinations


# ---- CLASSES ----

class MaskType(Enum):
    """Type of masks to generate."""
    CIRCULAR = auto()
    BBOX = auto()
    HANDCRAFTED = auto()


class Size(Enum):  # Pixels for radius
    HUGE = 225
    BIG = 175
    MEDIUM = 150
    SMALL = 100


GlomeruliClass = {  # TODO: re-classify sizes
    # Huge
    "MEMBRANOSO": Size.HUGE.value,
    "GNMP": Size.HUGE.value,
    "GSSF": Size.HUGE.value,
    # Big
    "SANO": Size.BIG.value,
    "HIPERCELULAR MES": Size.BIG.value,
    # Medium
    "INCOMPLETO": Size.MEDIUM.value,
    "SEMILUNAS": Size.MEDIUM.value,
    "ISQUEMICO": Size.MEDIUM.value,
    "MIXTO": Size.MEDIUM.value,
    "ENDOCAPILAR": Size.MEDIUM.value,
    # Small
    "ESCLEROSADO": Size.SMALL.value
}


# ---- FUNCTIONS ----
def get_data_from_xml(xml_file: str, mask_size: Optional[int],
                      apply_simplex: bool) -> Dict[int, List[Tuple[int, int]]]:
    """ Read data from glomeruli xml file.
    Data to extract: Glomeruli class and coordinates of each occurrence.
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
    else:  # Using variable mask size based on glmoeruli class
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
    """ Apply simplex algorithm to avoid masks overlap.
     Simplex algorithm: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-simplex.html"""

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

    #Re-construct data dict
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


def timer(f):
    """ Timer decorator to wrap and measure a function time performance."""

    def time_dec(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print_info("{} - {:2.4f} sec".format(f.__name__, te - ts))
        return result

    return time_dec


def computeRowsCols(N: int, m: int, n: int) -> Tuple[int, int]:
    """
    Compute number of rows and columns for subplot
    :param N: number of subfigures
    :param m: rows preference
    :param n: columns preference
    :return: resulting number of rows and colums
    """
    if m is None:
        m = math.sqrt(N)
        if n is None:
            n = math.ceil(N / m)
        else:
            m = math.ceil(N / n)
    else:
        if n is None:
            n = math.ceil(N / m)
        else:
            m = math.ceil(N / n)
    m, n = max(1, m), max(1, n)
    m, n = math.ceil(m), math.ceil(n)
    return m, n


def show_ims(imgs: List[np.ndarray], m: Optional[int] = None, n: Optional[int] = None,
             title: str = "", subtitles: Optional[List[str]] = None):
    """
    Show a group of images in subplots of the same figure
    :param imgs: images to show
    :param m: number of rows preference
    :param n: number of columns preference
    :param title: global title
    :param subtitles: list of subtitles for subplots
    :return: None
    """
    N = len(imgs)

    m, n = computeRowsCols(N, m, n)
    fig = plt.figure(figsize=(12, 6))
    for i in range(1, N + 1):
        plt.subplot(m, n, i)
        if len(imgs[i - 1].shape) == 3:
            plt.imshow(imgs[i - 1])
        else:
            plt.imshow(imgs[i - 1], cmap="gray")
        if subtitles is not None:
            plt.title(subtitles[i - 1])
    fig.suptitle(title)
    plt.show()

    return fig


def show_masked_ims(imgs: List[np.ndarray], masks: List[np.ndarray],
                    m: Optional[int] = None, n: Optional[int] = None,
                    title: str = "", subtitles: Optional[List[str]] = None):
    """
    Show a group of images in subplots of the same figure
    :param imgs: images to show
    :param m: number of rows preference
    :param n: number of columns preference
    :param title: global title
    :param subtitles: list of subtitles for subplots
    :return: None
    """
    N = len(imgs)

    m, n = computeRowsCols(N, m, n)
    fig = plt.figure(figsize=(12, 6))
    for i in range(1, N + 1):
        plt.subplot(m, n, i)
        if len(imgs[i - 1].shape) == 3:
            plt.imshow(imgs[i - 1][:, :, 0], cmap="gray")
            plt.imshow(masks[i - 1][:, :, 0], cmap="jet", alpha=0.3)
        else:
            plt.imshow(imgs[i - 1], cmap="gray")
            plt.imshow(masks[i - 1], cmap="jet", alpha=0.3)
        if subtitles is not None:
            plt.title(subtitles[i - 1])
    fig.suptitle(title)
    plt.show()

    return fig


def print_info(msg):
    info = "--> [I]:  "
    print(info, msg)


def print_warn(msg):
    info = "--> [W]:  "
    print(info, msg)


def print_error(msg):
    info = "--> [E]:  "
    print(info, msg)


# DEBUG
def check_gpu_availability():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


# Testing timer
# if __name__ == '__main__':
#     @timer
#     def test():
#         for i in range(1000000):
#             pass
#
#     test()
#
#     class Test():
#         @timer
#         def testf(self, arg):
#             for i in range(10000000):
#                 pass
#
#     t = Test()
#     t.testf(2)

# Testing xml extractor
if __name__ == '__main__':
    xml_path = "D:\\DataGlomeruli\\xml"
    mask_path = "D:\\DataGlomeruli\\gt\\circles"
    im_path = "D:\\DataGlomeruli\\ims"
    import os, glob, random
    import matplotlib.pyplot as plt
    import cv2.cv2 as cv2

    i = random.randint(0, len(os.listdir(xml_path)) - 1)
    # i = 522
    # i = 10
    print("------", i)
    # xml_f = glob.glob(xml_path + '/*')[i]
    xml_f = xml_path + "\\20B0012178 A 1 PAS_x2400y9600s3200.xml"
    # mask_f = glob.glob(mask_path + '/*')[i]
    mask_f = mask_path + "\\20B0012178 A 1 PAS_x2400y9600s3200.png"
    # im_f = glob.glob(im_path + '/*')[i]
    im_f = im_path + "\\20B0012178 A 1 PAS_x2400y9600s3200.png"

    print(xml_f)
    mask = cv2.cvtColor(cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB)
    im = cv2.cvtColor(cv2.imread(im_f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(121)
    plt.imshow(im)
    plt.subplot(122)
    plt.imshow(mask, cmap="gray")

    data = get_data_from_xml(xml_f, apply_simplex=True)
    print(data)
    plt.show()


