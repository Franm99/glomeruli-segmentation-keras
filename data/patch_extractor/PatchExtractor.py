import glob
import os
import tkinter as tk
from PIL import Image as Img
from PIL.ImageTk import PhotoImage
from PIL import ImageTk
import cv2.cv2 as cv2
import numpy as np
import random
from typing import Tuple, Optional
import re

from src.utils.enums import Staining

## PARAMETERS ##
staining = Staining.ALL # Select desired staining: HE, PAS, PM, None to select all of them
DATASET_PATH = '/home/francisco/Escritorio/DataGlomeruli'
RANDOM_ORDER = True

## CONSTANTS ##
IM_SIZE = 3200
PATCH_SIZE = 600

ims_directory = os.path.join(DATASET_PATH, 'ims')
masks_directory = os.path.join(DATASET_PATH, 'gt/masks')
out_folder = 'patches'
if not os.path.isdir(out_folder):
    os.mkdir(out_folder)

checkpoint_file = 'checkpoint.txt'


class Interface(tk.Frame):
    def __init__(self, ims_dir: str, masks_dir: str, staining: Optional[Staining], rand_order: bool = False):
        super().__init__()
        self.ims_dir = ims_dir
        self.masks_dir = masks_dir
        self.rand_order = rand_order
        self.staining = staining

        self.ims_list = [i for i in glob.glob(self.ims_dir + '/*') if self.staining in i]
        self.masks_list = [i for i in glob.glob(self.masks_dir + '/*') if self.staining in i]
        if os.path.isfile(checkpoint_file):
            self.pop_seen_images()

        if self.rand_order:
            index_shuffle = list(range(len(self.ims_list)))
            random.shuffle(index_shuffle)
            self.ims_list = [self.ims_list[i] for i in index_shuffle]
            self.masks_list = [self.masks_list[i] for i in index_shuffle]
        self.ims_iter = iter(self.ims_list)
        self.masks_iter = iter(self.masks_list)

        self.num_ims = len(self.ims_list)
        self.reduction_ratio = self.compute_reduction_ratio()
        self.patch_size = PATCH_SIZE
        self.reduced_patch_size = self.patch_size // self.reduction_ratio
        self.canvas_w = IM_SIZE // self.reduction_ratio
        self.canvas_h = IM_SIZE // self.reduction_ratio

        # Interface variables
        self.viewed_ims = 0
        self.patches_count = self.current_num_patches()
        self.rectangle = None
        self.im = None
        self.name = None
        self.bbox = None
        self.progress = tk.StringVar(value="...")
        self.total_num_patches = tk.StringVar(value=str(self.patches_count))

        # Initialize interface
        self.create_widgets()
        self.update_interface()

    def create_widgets(self):
        """ Create, initialize and place widgets on frame. """
        # Staining note
        stainingNote = "Staining: {}".format(self.staining if (self.staining != '') else 'ALL')
        tk.Label(self, text=stainingNote, font="Arial 10", anchor=tk.S).grid(column=1, row=0, pady=10)
        # Canvas
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h)
        self.canvas.grid(row=1, column=0, columnspan=3, padx=20, pady=10)
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<Button-1>", self.click)

        # Buttons
        self.buttonNext = tk.Button(self, text=u"\u2192", command=self.cb_nextImage, font="Arial 12",
                                    height=3, width=10, background="#595959", foreground="#F2F2F2")
        self.buttonNext.grid(row=2, column=1, rowspan=2, padx=10, pady=10)

        guide_text = "Progress (Random)" if self.rand_order else "Progress (Ordered)"
        tk.Label(self, text=guide_text, font="Arial 10", anchor="center").grid(column=0, row=2, pady=10, padx=10)
        self.lblProgress = tk.Label(self, textvariable=self.progress, font="Arial 10")
        self.lblProgress.grid(row=3, column=0, padx=10, pady=10, sticky=tk.N)
        tk.Label(self, text="Num of images", font="Arial 10", anchor="center").grid(column=2, row=2, pady=10, padx=10)
        self.lblNumber = tk.Label(self, textvariable=self.total_num_patches, font="Arial 10")
        self.lblNumber.grid(row=3, column=2, padx=10, pady=10, sticky=tk.N)

    def load_im(self) -> Tuple[np.ndarray, str, PhotoImage]:
        """ Load image and mask from iterators and send to canvas in an overlayed format. """
        im_path, mask_path = next(self.ims_iter), next(self.masks_iter)
        self.viewed_ims += 1
        name = os.path.basename(im_path)
        self.save_checkpoint(name)

        im = cv2.cvtColor(cv2.imread(next(self.ims_iter), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = np.zeros((3200, 3200, 3), dtype=np.uint8)
        mask[:, :, 2] = cv2.imread(next(self.masks_iter), cv2.IMREAD_GRAYSCALE)
        im_overlayed = cv2.addWeighted(im, 1.0, mask, 0.3, 0.5)
        im_canvas = self.toImageTk(im_overlayed)
        return im, name, im_canvas

    def update_interface(self):
        self.im, self.name, self.im_canvas = self.load_im()
        self.canvas.create_image(0, 0, image=self.im_canvas, anchor=tk.NW)
        self.progress.set("{}/{}".format(self.viewed_ims, self.num_ims))

    def toImageTk(self, im: np.ndarray) -> PhotoImage:
        """
        :param im: image in numpy array format
        :return: image in PhotoImage format (for tkinter canvas)
        """
        im_pil = Img.fromarray(im)
        im = im_pil.resize((self.canvas_w, self.canvas_h))
        return ImageTk.PhotoImage(im)

    def motion(self, event):
        x, y = event.x, event.y
        self.canvas.delete(self.rectangle)  # to refresh the rectangle each motion
        bb = self.get_bbox(x, y, self.reduced_patch_size)
        self.rectangle = self.canvas.create_rectangle(bb[0], bb[1], bb[2], bb[3], outline="red")

    def click(self, event):
        self.mark_patch(event)
        X, Y = event.x * self.reduction_ratio, event.y * self.reduction_ratio
        bb = self.get_bbox(X, Y, self.patch_size)
        self.get_patch(bb)
        self.patches_count += 1
        self.total_num_patches.set(str(self.patches_count))

    # Callback methods
    def cb_nextImage(self):
        self.update_interface()

    def get_patch(self, bb):
        patch = self.im[bb[3]:bb[1], bb[2]:bb[0]]
        im_pil = Img.fromarray(patch)
        name = self.get_name(bb)
        im_pil.save(os.path.join(out_folder, name))

    def mark_patch(self, event):
        bb = self.get_bbox(event.x, event.y, self.reduced_patch_size)
        self.canvas.create_rectangle(bb[0], bb[1], bb[2], bb[3], outline="green", width=4)

    def get_name(self, bb):
        bname = self.name.split("_")[0]
        bx = re.search(r'x([0-9]*)', self.name).group(1)
        by = re.search(r'y([0-9]*)', self.name).group(1)
        gx = int(bx) + bb[2]
        gy = int(by) + bb[3]
        return bname + '_x{}y{}s{}.png'.format(gx, gy, PATCH_SIZE)

    def compute_reduction_ratio(self):
        for i in range(1, IM_SIZE):
            if ((IM_SIZE % i) == 0) and ((IM_SIZE // i) < (self.winfo_screenheight() * 3/4)):
                return i

    def save_checkpoint(self, name):
        with open(checkpoint_file, 'a') as file:
            file.write(f'{name}\n')

    def pop_seen_images(self):
        with open(checkpoint_file, 'r') as file:
            seen_files = [i.strip('\n') for i in file.readlines()]
        self.ims_list = [i for i in self.ims_list if not os.path.basename(i) in seen_files]
        self.masks_list = [i for i in self.masks_list if not os.path.basename(i) in seen_files]

    def current_num_patches(self):
        return len(os.listdir(out_folder))

    @staticmethod
    def get_bbox(x, y, dim):
        """
        Returns bounding box coordinates in [x_max, y_max, x_min, y_min] order.
        :param x: X center coordinate
        :param y: Y center coordinate
        :param dim: square side (in pixels)
        """
        half_dim = dim // 2
        x_max = x + half_dim
        x_min = x - half_dim
        y_max = y + half_dim
        y_min = y - half_dim
        return x_max, y_max, x_min, y_min


if __name__ == '__main__':
    inter = Interface(ims_directory, masks_directory, staining=staining, rand_order=RANDOM_ORDER)
    inter.pack(fill="both", expand=True)
    inter.mainloop()
