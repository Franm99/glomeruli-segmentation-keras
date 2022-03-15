"""
Author: Fran Moreno
Contact: fran.moreno.se@gmail.com
Date: 15/03/2022
"""
import cv2.cv2 as cv2
from dataclasses import dataclass
import glob
import numpy as np
import os
import tkinter as tk
import re
from PIL import Image as Img, ImageTk
from PIL.ImageTk import PhotoImage
from tkinter import ttk
from typing import List, Optional
from tensorflow.keras.utils import normalize

import src.utils.constants as const
from src.utils.figures import find_blobs_centroids
from src.keras.utils import load_model_weights, get_model


class PredictionViewer(tk.Frame):
    def __init__(self, output_folder: str, th: Optional[float] = None, from_dir: bool = False):
        # Initialize tkinter interface
        super().__init__()

        self._output_folder = output_folder
        self._th = th
        self._from_dir = from_dir

        self.reduction_ratio = self.compute_reduction_ratio()
        self.canvas_w = const.IMG_SIZE[0] // self.reduction_ratio
        self.canvas_h = const.IMG_SIZE[0] // self.reduction_ratio

        self.trainingData = TrainingData(self._output_folder)
        if not self._th:
            self._th = self.trainingData.th

        self._ims_folder = os.path.join(const.SEGMENTER_DATA_PATH, self.trainingData.staining, "ims")
        self._masks_folder = os.path.join(const.SEGMENTER_DATA_PATH, self.trainingData.staining, "gt", "masks")

        if not from_dir:
            print("Loading pre-trained keras...")
            model = get_model("simple_unet", im_h=const.UNET_INPUT_SIZE, im_w=const.UNET_INPUT_SIZE)
            self._model = load_model_weights(model, self.trainingData.model)

        # Load images, ground-truth masks and prediction masks (in numpy array format)
        self.ims_np = self.load_ims()
        self.preds_np = self.load_test_predictions()
        self.masks_np = self.load_gt()

        # Convert to PhotoImage format to display on tkinter canvas
        self.preds = [self.toImageTk(pred) for pred in self.preds_np]
        self.ims = [self.toImageTk(im) for im in self.ims_np]
        self.masks = [self.toImageTk(mask) for mask in self.masks_np]

        # Initialize interface variables
        self.idx = 0
        self.num_ims = len(self.ims_np)
        self.local_true_positives = tk.StringVar()
        self.local_false_negatives = tk.StringVar()
        self.local_false_positives = tk.StringVar()
        self.glomeruli = tk.StringVar()
        self.global_true_positives = tk.StringVar()
        self.global_false_negatives = tk.StringVar()
        self.global_false_positives = tk.StringVar()
        self.accuracy = tk.StringVar()
        self.precision = tk.StringVar()
        self.progress = tk.StringVar()
        self.data = tk.StringVar()

        self.gt_glomeruli_counter = 0
        self.pred_glomeruli_counter = 0
        self.TP = 0
        self.FP = 0
        self.FN = 0

        self.TP_list = []
        self.FN_list = []
        self.FP_list = []

        self.init_counters()
        self.create_widgets()
        self.update_interface()

    def create_widgets(self):
        """
        Create, initialize and place labels, images, etc on frame
        """
        # Guide labels
        tk.Label(self, text="Image", font="Arial 14 bold").grid(row=0, column=0, padx=10, pady=10)
        tk.Label(self, text="Ground-truth", font="Arial 14 bold").grid(row=0, column=2, padx=10, pady=10)
        tk.Label(self, text="Prediction", font="Arial 14 bold").grid(row=0, column=4, padx=10, pady=10)

        # Canvas for images
        self.canvas_im = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h)
        self.canvas_im.grid(row=1, rowspan=13, column=0, padx=10, pady=10)
        self.canvas_gt = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h)
        self.canvas_gt.grid(row=1, rowspan=13, column=1, columnspan=3, padx=10, pady=10)
        self.canvas_pred = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h)
        self.canvas_pred.grid(row=1, rowspan=13, column=4, padx=10, pady=10)

        # Buttons
        self.buttonPrev = tk.Button(self, text=u"\u2190", command=self.cb_prevImage, font="Arial 12",
                                    height=3, width=10, background="#595959", foreground="#F2F2F2")
        self.buttonPrev.grid(row=14, column=1, padx=10, pady=10)
        self.buttonPrev["state"] = tk.DISABLED
        self.buttonNext = tk.Button(self, text=u"\u2192", command=self.cb_nextImage, font="Arial 12",
                                    height=3, width=10, background="#595959", foreground="#F2F2F2")
        self.buttonNext.grid(row=14, column=3, padx=10, pady=10)
        self.buttonSave = tk.Button(self, text="Save", command=self.cb_save_results, font="Arial 12",
                                    height=3, width=10, background="#595959", foreground="#F2F2F2")
        self.buttonSave.grid(row=14, column=4, sticky="e", padx=10, pady=10)

        self.lblProgress = tk.Label(self, textvariable=self.progress, font="Arial 10")
        self.lblProgress.grid(row=14, column=2, padx=10, pady=10)
        self.lblData = tk.Label(self, textvariable=self.data)
        self.lblData.grid(row=14, column=0, pady=10, padx=10)

        # Text panel
        ttk.Separator(self, orient=tk.VERTICAL).grid(column=5, row=0, rowspan=15, sticky='ns')
        tk.Label(self, text="Local values", anchor="w", font="Arial 14 bold").grid(column=6, columnspan=2, row=1, padx=10)

        tk.Label(self, text="True Positives:", anchor="w", font="Arial 10").grid(column=6, columnspan=1, row=2, padx=10, sticky="w")
        tk.Label(self, text="False Negatives:", anchor="w", font="Arial 10").grid(column=6, columnspan=1, row=3, padx=10, sticky="w")
        tk.Label(self, text="False Positives:", anchor="w", font="Arial 10").grid(column=6, columnspan=1, row=4, padx=10, sticky="w")

        ttk.Separator(self, orient=tk.HORIZONTAL).grid(column=5, columnspan=3, row=6, sticky='we')
        tk.Label(self, text="Global values", font="Arial 14 bold").grid(column=6, columnspan=2, row=7, padx=10)

        tk.Label(self, text="Glomeruli:", anchor="w", font="Arial 10").grid(column=6, columnspan=1, row=8, padx=10, sticky="w")
        tk.Label(self, text="True Positives:", anchor="w", font="Arial 10").grid(column=6, columnspan=1, row=9, padx=10, sticky="w")
        tk.Label(self, text="False Negatives:", anchor="w", font="Arial 10").grid(column=6, columnspan=1, row=10, padx=10, sticky="w")
        tk.Label(self, text="False Positives:", anchor="w", font="Arial 10").grid(column=6, columnspan=1, row=11, padx=10, sticky="w")
        tk.Label(self, text="Accuracy (%):", anchor="w", font="Arial 10").grid(column=6, columnspan=1, row=12, padx=10, sticky="w")
        tk.Label(self, text="Precision (%):", anchor="w", font="Arial 10").grid(column=6, columnspan=1, row=13, padx=10, sticky="w")

        # Variable labels
        self.lblLocalTP = tk.Label(self, textvariable=self.local_true_positives)
        self.lblLocalTP.grid(column=7, row=2, padx=10, sticky="w")
        self.lblLocalFN = tk.Label(self, textvariable=self.local_false_negatives)
        self.lblLocalFN.grid(column=7, row=3, padx=10, sticky="w")
        self.lblLocalFP = tk.Label(self, textvariable=self.local_false_positives)
        self.lblLocalFP.grid(column=7, row=4, padx=10, sticky="w")

        self.lblGlomeruli = tk.Label(self, textvariable=self.glomeruli)
        self.lblGlomeruli.grid(column=7, row=8, padx=10, sticky="w")
        self.lblGlobalTP = tk.Label(self, textvariable=self.global_true_positives)
        self.lblGlobalTP.grid(column=7, row=9, padx=10, sticky="w")
        self.lblGlobalFN = tk.Label(self, textvariable=self.global_false_negatives)
        self.lblGlobalFN.grid(column=7, row=10, padx=10, sticky="w")
        self.lblGlobalFP = tk.Label(self, textvariable=self.global_false_positives)
        self.lblGlobalFP.grid(column=7, row=11, padx=10, sticky="w")
        self.lblAccuracy = tk.Label(self, textvariable=self.accuracy)
        self.lblAccuracy.grid(column=7, row=12, padx=10, sticky="w")
        self.lblPrecision = tk.Label(self, textvariable=self.precision)
        self.lblPrecision.grid(column=7, row=13, padx=10, sticky="w")



    def show_images(self):
        """
        Command to display the three images in a row.
        """
        self.canvas_im.configure(width=self.canvas_w, height=self.canvas_h)
        self.canvas_im.create_image(0, 0, image=self.ims[self.idx], anchor=tk.NW)

        self.canvas_gt.configure(width=self.canvas_w, height=self.canvas_h)
        self.canvas_gt.create_image(0, 0, image=self.masks[self.idx], anchor=tk.NW)

        self.canvas_pred.configure(width=self.canvas_w, height=self.canvas_h)
        self.canvas_pred.create_image(0, 0, image=self.preds[self.idx], anchor=tk.NW)
        self.canvas_pred.bind('<Button-1>', self.cb_add_true_positive)
        self.canvas_pred.bind('<Button-3>', self.cb_add_false_positive)

    def update_interface(self):
        """
        Update both canvas and local values
        """
        self.show_images()
        self.progress.set("{}/{}".format(self.idx+1, len(self.ims)))

        # Update text variables
        self.local_true_positives.set(str(self.TP_list[self.idx]))
        self.local_false_negatives.set(str(self.FN_list[self.idx]))
        self.local_false_positives.set(str(self.FP_list[self.idx]))

    def init_counters(self):
        for i, (mask, pred) in enumerate(zip(self.masks_np, self.preds_np)):
            gt_centroids = find_blobs_centroids(mask)
            self.gt_glomeruli_counter += len(gt_centroids)
            self.FN_list.append(len(gt_centroids))
            self.TP_list.append(0)
            self.FP_list.append(0)
        self.local_true_positives.set(str(0))
        self.local_false_negatives.set(str(self.FN_list[self.idx]))
        self.local_false_positives.set(str(0))
        self.glomeruli.set(str(self.gt_glomeruli_counter))
        self.global_true_positives.set(str(0))
        self.global_false_positives.set(str(0))
        self.global_false_negatives.set(str(self.gt_glomeruli_counter))
        self.FN = self.gt_glomeruli_counter
        self.accuracy.set(str(0))
        self.precision.set(str(0))
        self.progress.set("{}/{}".format(self.idx+1, len(self.ims)))
        self.data.set(f"{self.trainingData.staining}, th = {self._th}")

    def cb_add_true_positive(self, event):
        """
        Callback function to add new True Positive case.
        """
        self.TP += 1
        self.TP_list[self.idx] += 1
        self.FN -= 1
        self.FN_list[self.idx] -= 1
        self.local_true_positives.set(str(self.TP_list[self.idx]))
        self.global_true_positives.set(str(self.TP))
        self.local_false_negatives.set(str(self.FN_list[self.idx]))
        self.global_false_negatives.set(str(self.FN))

        # Update accuracy and precision values
        self.accuracy.set("{:.2f}".format(self.compute_accuracy(self.TP, self.FN)))
        self.precision.set("{:.2f}".format(self.compute_precision(self.TP, self.FP)))
        self.plot_mark(event, color="#00ff00")
        # print("Left-click:", event.x, event.y)

    def cb_add_false_positive(self, event):
        """
        Callback function to add new False Positive case.
        """
        self.FP += 1
        self.FP_list[self.idx] += 1
        self.local_false_positives.set(str(self.FP_list[self.idx]))
        self.global_false_positives.set(str(self.FP))

        # Update precision value
        prec = (self.TP / (self.TP + self.FP)) * 100.0
        self.precision.set("{:.2f}".format(self.compute_precision(self.TP, self.FP)))
        self.plot_mark(event, color="#ff0000")

    @staticmethod
    def compute_accuracy(tp, fn):
        return (tp / (tp + fn)) * 100.0

    @staticmethod
    def compute_precision(tp, fp):
        return (tp / (tp + fp)) * 100.0

    def cb_save_results(self):
        """
        Method to save (global) results in a txt file for later study.
        NOTE: output file will be saved in the output directory.
        """
        filename = os.path.join(os.path.dirname(__file__), 'output', self._output_folder, "test_analysis{}.txt".format(self.filename_counter()))
        with open(file=filename, mode="w") as f:
            f.write("-- PARAMETERS --\n")
            f.write("STAINING = {}\n".format(self.trainingData.staining))
            f.write("TH = {}\n".format(str(self._th)))
            f.write("----------------\n")
            f.write("GLOMERULI COUNT: {}\n".format(str(self.gt_glomeruli_counter)))
            f.write("TRUE POSITIVES:  {}\n".format(self.global_true_positives.get()))
            f.write("FALSE NEGATIVES: {}\n".format(self.global_false_negatives.get()))
            f.write("FALSE POSITIVES: {}\n".format(self.global_false_positives.get()))
            f.write("ACCURACY (%):    {}\n".format(self.accuracy.get()))
            f.write("PRECISION (%):   {}\n".format(self.precision.get()))
        print("Results saved to file: {}".format(filename))

    def cb_prevImage(self):
        """
        Display previous image in list, and its corresponding values
        """
        self.idx -= 1
        self.update_interface()
        if self.idx == 0:
            self.buttonPrev["state"] = tk.DISABLED
        else:
            self.buttonPrev["state"] = tk.NORMAL

        if self.buttonNext["state"] == tk.DISABLED:
            self.buttonNext["state"] = tk.NORMAL

    def cb_nextImage(self):
        """
        Display next image in list, and its corresponding values
        """
        self.idx += 1
        self.update_interface()
        if self.idx == self.num_ims - 1:
            self.buttonNext["state"] = tk.DISABLED
        else:
            self.buttonNext["state"] = tk.NORMAL

        if self.buttonPrev["state"] == tk.DISABLED:
            self.buttonPrev["state"] = tk.NORMAL

    def plot_mark(self, event, color: str):
        # cy, cx = event.y, event.x
        # y, x = np.ogrid[-cy:]
        # self.preds_np[self.idx][event.y, event.x] = self.hex2rgb(color)
        # self.preds[self.idx] = self.toImageTk(self.preds_np[self.idx])
        # self.show_images()
        self.create_circle(event.x, event.y, color=color)

    def load_pretrained_model(self):
        return os.path.join(self._output_folder, 'weights/keras.hdf5')

    def load_test_predictions(self):
        """
        Load test prediction masks from the specified folder.
        :return: (filenames list, numpy array images list)
        """
        if self._from_dir:
            print("Loading predictions from previous training...")
            test_pred_list = self.trainingData.predictions
            return [cv2.imread(i, cv2.IMREAD_COLOR) for i in test_pred_list]
        else:
            # Use pre-trained keras to compute predictions
            print("Computing predictions using pre-trained keras using th={} ...".format(self._th))
            return [self._get_pred_mask(i) for i in self.ims_np]

    def load_ims(self) -> List[np.ndarray]:
        """
        Load images from where test prediction images have been obtained.
        :return: numpy array images list
        """
        names = [os.path.basename(i) for i in self.trainingData.predictions]
        ims_names = [os.path.join(self._ims_folder, name) for name in names]
        return [cv2.cvtColor(cv2.imread(i, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for i in ims_names]

    def load_gt(self) -> List[np.ndarray]:
        """
        Load ground-truth masks attached to the images from where test prediction masks have been obtained.
        :return: numpy array images list
        """
        names = [os.path.basename(i) for i in self.trainingData.predictions]
        masks_names = [os.path.join(self._masks_folder, name) for name in names]
        return [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in masks_names]

    # def toImageTk(self, ims: List[np.ndarray]) -> List[PhotoImage]:
    #     """
    #     :param ims: List of images in numpy array format
    #     :return: List of images in PhotoImage format (for tkinter canvas)
    #     """
    #     ims_pil = [Img.fromarray(im) for im in ims]
    #     ims = [im.resize((self.canvas_w, self.canvas_h)) for im in ims_pil]
    #     return [ImageTk.PhotoImage(im) for im in ims]

    def toImageTk(self, im: np.ndarray) -> PhotoImage:
        """
        :param im: image in numpy array format
        :return: image in PhotoImage format (for tkinter canvas)
        """
        im_pil = Img.fromarray(im)
        im = im_pil.resize((self.canvas_w, self.canvas_h))
        return ImageTk.PhotoImage(im)

    def _get_pred_mask(self, im):
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        [h, w] = im_gray.shape
        dim = const.UNET_INPUT_SIZE * self.trainingData.resize_ratio

        # Initializing list of masks
        mask = np.zeros((h, w), dtype=bool)
        # Loop through the whole in both dimensions
        for x in range(0, w, dim):
            if x + dim >= w:
                x = w - dim
            for y in range(0, h, dim):
                if y + dim >= h:
                    y = h - dim
                # Get sub-patch in original size
                patch = im_gray[y:y + dim, x:x + dim]

                # Median filter applied on image histogram to discard non-tissue sub-patches
                counts, bins = np.histogram(patch.flatten(), list(range(256 + 1)))
                counts.sort()
                median = np.median(counts)
                if median <= 3.:
                    # Non-tissue sub-patches automatically get a null mask
                    prediction_rs = np.zeros((dim, dim), dtype=np.uint8)
                else:
                    # Tissue sub-patches are fed to the U-net keras for mask prediction
                    patch = cv2.resize(patch, (const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE),
                                       interpolation=cv2.INTER_AREA)
                    patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                    prediction = (self._model.predict(patch_input)[:, :, :, 0] >= self._th).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (dim, dim), interpolation=cv2.INTER_AREA)

                mask[y:y + dim, x:x + dim] = np.logical_or(mask[y:y + dim, x:x + dim], prediction_rs.astype(bool))
        return mask.astype(np.uint8) * 255  # Change datatype from np.bool to np.uint8

    def create_circle(self, x, y, color, r=5):
        return self.canvas_pred.create_oval(x-r, y-r, x+r, y+r, fill=color, outline="")

    def filename_counter(self):
        test_logs = [i for i in glob.glob(self._output_folder + '/*') if (i.startswith("test") and i.endswith(".txt"))]
        return str(len(test_logs)) if len(test_logs) > 0 else ""

    def compute_reduction_ratio(self):
        sz = const.IMG_SIZE[0]
        for i in range(1, sz):
            if ((sz % i == 0) and ((sz // i) * 3 < self.winfo_screenwidth() * 3/4)):
                return i

    @staticmethod
    def hex2rgb(hex):
        r, g, b = hex[1:3], hex[3:5], hex[5:7]
        return int(r, 16), int(g, 16), int(b, 16)


@dataclass
class TrainingData:
    """
    Training Data
    =============

    Data class containing extracted data from a training output folder.
    """

    def __init__(self, dir_path):
        """ *Class constructor* """
        self._dir_path = dir_path
        self._model = glob.glob(os.path.join(self._dir_path, "weights") + '/*.hdf5')[0]
        self._folder_name = os.path.basename(self._dir_path)
        self._logFile = LogFile(os.path.join(self._dir_path, self._folder_name.replace("-", "") + ".txt"))
        self._predictions = glob.glob(os.path.join(self._dir_path, "test_pred") + '/*')

    @property
    def model(self):
        """ Full path to pre-trained weights file. """
        return self._model

    @property
    def predictions(self):
        """ Name of the set of test predictions for a given training output folder. """
        return self._predictions

    @property
    def staining(self) -> str:
        """ Staining used for the given training process. """
        if hasattr(self._logFile, 'staining'):
            return self._logFile.staining
        else:
            raise AttributeError

    @property
    def th(self) -> float:
        """ Binarization threshold used for the given training process. """
        if hasattr(self._logFile, 'prediction_threshold'):
            return float(self._logFile.prediction_threshold)
        else:
            raise AttributeError

    @property
    def resize_ratio(self) -> int:
        """ Resize ratio used for the given training process. """
        if hasattr(self._logFile, 'resize_ratio'):
            return int(self._logFile.resize_ratio)
        else:
            raise AttributeError


class LogFile:
    """
    Log File
    ========

    Log File class that reads the content of a given report file and automatically create class fields for every
    parameter contained in its lines.
    """
    def __init__(self, filepath):
        """ *Class constructor* """
        self._filepath = filepath
        with open(self._filepath, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "--" not in line:
                fields = re.split(" +", line.strip())
                setattr(self, fields[0].lower(), fields[1])

