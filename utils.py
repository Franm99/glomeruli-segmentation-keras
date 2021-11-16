import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import numpy as np
import math
import time
from bs4 import BeautifulSoup
from enum import Enum, auto


class MaskType(Enum):
    """Type of masks to generate."""
    CIRCULAR = auto()
    BBOX = auto()


class Size(Enum):  # Pixels for radius
    HUGE = 200
    BIG = 150
    MEDIUM = 125
    SMALL = 100


GlomeruliClass = {
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


def get_points_from_xml(xml_file: str) -> Dict[int, List[Tuple[int, int]]]:
    """ Read data from glomeruli xml file.
    Data to extract: Glomeruli class and coordinates of each occurrence.
    """
    with open(xml_file, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")
    counts = bs_data.find("Counts").find_all("Count")
    glomeruli = {x.value: [] for x in Size}
    print(counts)  # DEBUG
    for count in counts:
        name = count.get('name')
        points = count.find_all('point')
        p = []
        for point in points:
            p.append((int(point.get('X')), int(point.get('Y'))))
        glomeruli[GlomeruliClass[name]].extend(p)
    return glomeruli


def timer(f):
    """ Timer decorator to wrap and measure a function time performance."""

    def time_dec(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print("I: {} - {:2.4f} sec".format(f.__name__, te - ts))
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

# # Testing xml extractor
# if __name__ == '__main__':
#     dpath = "D:\\DataGlomeruli\\xml"
#     import os, glob, random
#
#     i = random.randint(0, len(os.listdir(dpath)) - 1)
#     f = glob.glob(dpath + '/*')[i]
#     print(get_points_from_xml(f))
