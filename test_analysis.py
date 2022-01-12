""" File to analyze the test set prediction results"""
import glob
import cv2.cv2 as cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
import os
from PIL import Image as Img
from PIL.Image import Image
from PIL.ImageTk import PhotoImage
from PIL import ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from typing import List, Tuple
from parameters import DATASET_PATH


class Viewer(tk.Frame):
    def __init__(self, output_folder: str, masks_folder: str):
        # Initialize tkinter interface
        super().__init__()
        self.canvas_w, self.canvas_h = 400, 400

        self._output_folder = output_folder
        self._masks_folder = os.path.join(DATASET_PATH, 'gt', masks_folder)

        # Load images, ground-truth masks and prediction masks
        self.names, self.preds = self.load_test_predictions()
        self.ims = self.load_ims()
        self.masks = self.load_gt()

        # Initialize interface variables
        self.idx = 0
        self.num_ims = len(self.names)
        self.local_true_positives = tk.StringVar()
        self.local_false_negatives = tk.StringVar()
        self.local_false_positives = tk.StringVar()
        self.local_hit_pctg = tk.StringVar()
        self.global_true_positives = tk.StringVar()
        self.global_false_negatives = tk.StringVar()
        self.global_false_positives = tk.StringVar()
        self.global_hit_pctg = tk.StringVar()

        self.counter_localFP = 0
        self.counter_globalFP = 0

        self.create_widgets()
        self.show_images()

    def create_widgets(self):
        """
        Create, initialize and place labels, images, etc on frame
        """
        # Guide labels
        tk.Label(self, text="Image").grid(row=0, column=0, padx=10, pady=10)
        tk.Label(self, text="Ground-truth").grid(row=0, column=2, padx=10, pady=10)
        tk.Label(self, text="Prediction").grid(row=0, column=4, padx=10, pady=10)

        # Canvas for images
        self.canvas_im = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h)
        self.canvas_im.grid(row=1, rowspan=11, column=0, padx=10, pady=10)
        self.canvas_gt = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h)
        self.canvas_gt.grid(row=1, rowspan=11, column=1, columnspan=3, padx=10, pady=10)
        self.canvas_pred = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h)
        self.canvas_pred.grid(row=1, rowspan=11, column=4, padx=10, pady=10)

        # Buttons
        self.buttonPrev = tk.Button(self, text="<", command=self.prevImage)
        self.buttonPrev.grid(row=12, column=1, padx=10, pady=10)
        self.buttonPrev["state"] = tk.DISABLED
        self.buttonNext = tk.Button(self, text=">", command=self.nextImage)
        self.buttonNext.grid(row=12, column=3, padx=10, pady=10)

        # Text panel
        ttk.Separator(self, orient=tk.VERTICAL).grid(column=5, row=0, rowspan=15, sticky='ns')
        tk.Label(self, text="Local values", anchor="w").grid(column=6, columnspan=2, row=1)
        tk.Label(self, text="True positives:", anchor="w").grid(column=6, columnspan=1, row=2)
        tk.Label(self, text="False negatives:", anchor="w").grid(column=6, columnspan=1, row=3)
        tk.Label(self, text="False positives:", anchor="w").grid(column=6, columnspan=1, row=4)
        tk.Label(self, text="Hit ratio:", anchor="w").grid(column=6, columnspan=1, row=5)
        ttk.Separator(self, orient=tk.HORIZONTAL).grid(column=5, columnspan=3, row=6, sticky='we')
        tk.Label(self, text="Global values").grid(column=6, columnspan=2, row=7)
        tk.Label(self, text="True positives:", anchor="w").grid(column=6, columnspan=1, row=8)
        tk.Label(self, text="False negatives:", anchor="w").grid(column=6, columnspan=1, row=9)
        tk.Label(self, text="False positives:", anchor="w").grid(column=6, columnspan=1, row=10)
        tk.Label(self, text="Hit ratio:", anchor="w").grid(column=6, columnspan=1, row=11)

        # Variable labels
        self.lblLocalTP = tk.Label(self, textvariable=self.local_true_positives)
        self.lblLocalTP.grid(column=7, row=2)
        self.lblLocalFN = tk.Label(self, textvariable=self.local_false_negatives)
        self.lblLocalFN.grid(column=7, row=3)
        self.lblLocalFP = tk.Label(self, textvariable=self.local_false_positives)
        self.lblLocalFP.grid(column=7, row=4)
        self.lblLocalPctg = tk.Label(self, textvariable=self.local_hit_pctg)
        self.lblLocalPctg.grid(column=7, row=5)
        self.lblGlobalTP = tk.Label(self, textvariable=self.global_true_positives)
        self.lblGlobalTP.grid(column=7, row=8)
        self.lblGlobalFN = tk.Label(self, textvariable=self.global_false_negatives)
        self.lblGlobalFN.grid(column=7, row=9)
        self.lblGlobalFP = tk.Label(self, textvariable=self.global_false_positives)
        self.lblGlobalFP.grid(column=7, row=10)
        self.lblGlobalPctg = tk.Label(self, textvariable=self.global_hit_pctg)
        self.lblGlobalPctg.grid(column=7, row=11)

    def show_images(self):
        """
        Command to display the three images in a row.
        """
        self.canvas_im.configure(width=self.canvas_w, height=self.canvas_h)
        self.canvas_im.create_image(0, 0, image=self.ims[self.idx], anchor=tk.NW)
        self.canvas_im.bind('<Button-1>', self.add_false_positive)
        self.canvas_im.bind('<Button-3>', self.remove_false_positive)

        self.canvas_gt.configure(width=self.canvas_w, height=self.canvas_h)
        self.canvas_gt.create_image(0, 0, image=self.masks[self.idx], anchor=tk.NW)

        self.canvas_pred.configure(width=self.canvas_w, height=self.canvas_h)
        self.canvas_pred.create_image(0, 0, image=self.preds[self.idx], anchor=tk.NW)

    def prevImage(self):
        self.idx -= 1
        if self.idx == 0:
            self.buttonPrev["state"] = tk.DISABLED
        else:
            self.buttonPrev["state"] = tk.NORMAL

        if self.buttonNext["state"] == tk.DISABLED:
            self.buttonNext["state"] = tk.NORMAL

    def nextImage(self):
        self.idx += 1
        if self.idx >= self.num_ims:
            self.buttonNext["state"] = tk.DISABLED
        else:
            self.buttonNext["state"] = tk.NORMAL

        if self.buttonPrev["state"] == tk.DISABLED:
            self.buttonPrev["state"] = tk.NORMAL

    @staticmethod
    def motion(event):
        x, y = event.x, event.y
        print("{}, {}".format(x, y))

    def add_false_positive(self, event):
        self.counter_localFP += 1
        self.local_false_positives.set(str(self.counter_localFP))
        self.counter_globalFP += 1
        self.global_false_positives.set(str(self.counter_globalFP))

    def remove_false_positive(self, event):
        self.local_false_positives -= 1
        self.global_false_positives -= 1

    # def run(self):
    #     for name, im, gt, pred in zip(self.names, self.ims, self.masks, self.preds):
    #         gt_centroids = find_blobs_centroids(gt)
    #
    #         # 1. Compute True positives (glomeruli correctly predicted).
    #         true_positives = 0
    #         for (cy, cx) in gt_centroids:
    #             true_positives += 1 if pred[cy, cx] == 1 else 0

    def load_test_predictions(self) -> Tuple[List[str], List[PhotoImage]]:
        dir_path = os.path.join('output', self._output_folder, 'test_pred')
        test_pred_list = glob(dir_path + '/*')
        pred_ims = [Img.open(pred_name) for pred_name in test_pred_list]
        test_pred_names = [os.path.basename(i) for i in test_pred_list]
        return test_pred_names, self.toImageTk(pred_ims)

    def load_ims(self) -> List[PhotoImage]:
        dir_path = os.path.join(DATASET_PATH, 'ims')
        ims_list = [Img.open(os.path.join(dir_path, name)) for name in self.names]
        return self.toImageTk(ims_list)

    def load_gt(self) -> List[PhotoImage]:
        dir_path = os.path.join(DATASET_PATH, 'gt', self._masks_folder)
        gt_list = [Img.open(os.path.join(dir_path, name)) for name in self.names]
        return self.toImageTk(gt_list)

    def toImageTk(self, ims: List[Image]) -> List[PhotoImage]:
        ims = [im.resize((self.canvas_w, self.canvas_h)) for im in ims]
        return [ImageTk.PhotoImage(im) for im in ims]


def browse_path():
    full_path = filedialog.askdirectory(initialdir='output')
    return os.path.basename(full_path)


def find_blobs_centroids(img: np.ndarray) -> List[Tuple[float, float]]:
    img_th = img.astype(bool)
    img_labels = label(img_th)
    img_regions = regionprops(img_labels)
    centroids = []
    for props in img_regions:
        centroids.append((props.centroid[0], props.centroid[1]))  # (y, x)
    return centroids

# # TEST FUNCTIONS
# # --------------
#
# def test_loaders():
#     # output_folder = '2022-01-11_08-54-07' # TODO: open selector window
#     output_folder = browse_path()
#     test_pred_paths, preds = load_test_predictions(output_folder)
#     test_pred_names = [os.path.basename(i) for i in test_pred_paths]
#     imgs = load_ims(test_pred_names)
#     gts = load_gt(test_pred_names, 'circles100')
#
#
# def test_centroids():
#     im_path = os.path.join(DATASET_PATH, 'gt', 'masks') + '/04B0006786 A 1 HE_x12000y4800s3200.png'
#     im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
#     plt.imshow(im)
#     plt.show()
#     centroids = find_blobs_centroids(im)
#     print(centroids)
#
#
# def test_TestBench():
#     output_folder = browse_path()
#     testBench = TestBench(output_folder)


if __name__ == '__main__':
    output_path = browse_path()
    viewer = Viewer(output_folder=output_path, masks_folder='circles100')
    viewer.pack(fill="both", expand=True)
    viewer.mainloop()

