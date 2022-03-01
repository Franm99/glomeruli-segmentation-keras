import math
import matplotlib.pyplot as plt
import numpy as np
import time
from bs4 import BeautifulSoup
from itertools import combinations
from scipy.optimize import linprog
from scipy.spatial import distance
from typing import List, Tuple, Optional, Dict
from tkinter import filedialog
import os
from skimage.measure import regionprops, label
from getpass import getpass
from abc import ABC, abstractmethod
from tensorflow.keras.utils import Sequence
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from src.utils.enums import Size


class DataGenerator(ABC, Sequence):
    @abstractmethod
    def __init__(self,
                 ims_list: List[str],
                 masks_list: List[str],
                 batch_size: int,
                 shuffle: bool,
                 n_channels: int):
        self.ims_list = ims_list
        self.masks_list = masks_list
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass

    @abstractmethod
    def _data_generation(self, ims_list_temp, masks_list_temp):
        pass


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


# ---- FUNCTIONS ----
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


class EmailHandler():
    def __init__(self):
        self._sender, self._pass, self._recv = init_email_info()

    def send(self, t: float, fname: str):
        time_mark = time.strftime("%H:%M:%S", time.gmtime(t))
        port = 465  # for SSL
        message = MIMEMultipart("alternative")
        message["Subject"] = "Training finished"
        message["From"] = self._sender
        message["To"] = self._recv

        html = """\
                <html>
                    <body>
                        Training finished. For further info, check log file.<br>
                        Time spent: {} (h:m:s)<br>
                    </body>
                </html>
                """.format(time_mark)

        part1 = MIMEText(html, "html")
        message.attach(part1)

        # Attach log file
        with open(fname, "rb") as att:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(att.read())

        encoders.encode_base64(part)
        part.add_header("Content-disposition",
                        f"attachment; filename= {os.path.basename(fname)}")

        message.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(self._sender, self._pass)
            server.sendmail(self._sender, self._recv, message.as_string())


def init_email_info():
    def_sender_email = "pythonAdvisor22@gmail.com"
    req = input("Use default sender ({}) [Y/n]: ".format(def_sender_email))
    if req.lower() == "y":
        sender_email = def_sender_email
    else:
        sender_email = input("Specify sender email: ")
    password = getpass()  # Just for terminal executions, not IDLE!
    receiver_email = input("Specify receiver email: ")
    return sender_email, password, receiver_email


def browse_path():
    """
    Opens a file browser to select the path from where test prediction images are taken.
    Default initial directory: output/ folder.
    NOTE: To select a certain output folder, you may first enter to that folder!
    """
    full_path = filedialog.askdirectory(initialdir='../output')
    return full_path


def browse_file():
    return filedialog.askopenfilename(initialdir='data/raw')


def find_blobs_centroids(img: np.ndarray) -> List[Tuple[float, float]]:
    """
    This function implements region labelling and region properties extraction to find, label and compute centroids of
    each blob in binary images.
    SOURCE:
    NOTE: In this context, blob = glomerulus.
    """
    img_th = img.astype(bool)
    img_labels = label(img_th)
    img_regions = regionprops(img_labels)
    centroids = []
    for props in img_regions:
        centroids.append((int(props.centroid[0]), int(props.centroid[1])))  # (y, x)
    return centroids


def timer(f):
    """
    Timer decorator to wrap and measure a function time performance.
    :param f: Function to wrap.
    :return: decorated function
    """
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
    """ Log INFO messages """
    info = "--> [I]:  "
    print(info, msg)


def print_warn(msg):
    """ Log WARNING messages """
    info = "--> [W]:  "
    print(info, msg)


def print_error(msg):
    """ Log ERROR messages """
    info = "--> [E]:  "
    print(info, msg)


# DEBUG
def check_gpu_availability():
    """ Source: https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed """
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


def tmp():
    import shutil
    p = "output"
    folders = [os.path.join(p, i) for i in os.listdir(p)]
    folders.remove("output/.gitignore")
    for folder in folders:
        if not os.path.isfile(os.path.join(folder, "test_analysis.txt")):
            shutil.rmtree(folder)


def t_find_blob_centroids():
    data_dir = "../../data"
    mask_sample = os.path.join(data_dir, "segmenter", "HE", "gt", "masks", "20B0004711 HE_x4800y14400s3200.png")
    import cv2.cv2 as cv2
    im = cv2.imread(mask_sample, cv2.IMREAD_GRAYSCALE)
    centroids = find_blobs_centroids(im)
    print(centroids)
    import matplotlib.pyplot as plt
    plt.imshow(im, cmap="gray")
    for (cy, cx) in centroids:
        plt.plot(cx, cy, ".r")
    plt.show()


if __name__ == '__main__':
    # print(Staining.HE)
    # tmp()
    t_find_blob_centroids()