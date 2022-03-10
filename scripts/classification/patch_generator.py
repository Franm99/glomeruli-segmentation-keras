import cv2.cv2 as cv2
import glob
import os
import numpy as np
import random
import re
import sys
import tkinter as tk
from os.path import dirname, abspath
from PIL import Image as Img, ImageTk
from PIL.ImageTk import PhotoImage
from typing import Tuple, Optional

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import src.utils.constants as const

# PARAMETERS #
staining = "HE" # Select desired staining: HE, PAS, PM
RANDOM_ORDER = False

# CONSTANTS #
DATASET_PATH = const.SEGMENTER_DATA_PATH
IM_SIZE = 3200
PATCH_SIZE = 600

ims_directory = os.path.join(DATASET_PATH, staining, 'ims')
masks_directory = os.path.join(DATASET_PATH, staining, 'gt', 'masks')
out_folder = os.path.join(dirname(abspath(__file__)), 'patches')
if not os.path.isdir(out_folder):
    os.mkdir(out_folder)

checkpoint_file = 'checkpoint.txt'


class PatchGenerator(tk.Frame):
    def __init__(self, ims_dir: str, masks_dir: str, staining: Optional[str], rand_order: bool = False):
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
        self.canvas = None
        self.im_canvas = None
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
        staining_note = "Staining: {}".format(self.staining if (self.staining != '') else 'ALL')
        tk.Label(self, text=staining_note, font="Arial 10", anchor=tk.S).grid(column=1, row=0, pady=10)
        # Canvas
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h)
        self.canvas.grid(row=1, column=0, columnspan=3, padx=20, pady=10)
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<Button-1>", self.click)

        # Buttons
        tk.Button(self, text=u"\u2192", command=self.cb_next_image, font="Arial 12", height=3, width=10,
                  background="#595959", foreground="#F2F2F2").grid(row=2, column=1, rowspan=2, padx=10, pady=10)

        guide_text = "Progress (Random)" if self.rand_order else "Progress (Ordered)"
        tk.Label(self, text=guide_text, font="Arial 10", anchor="center").grid(column=0, row=2, pady=10, padx=10)
        tk.Label(self, textvariable=self.progress, font="Arial 10").grid(row=3, column=0, padx=10, pady=10, sticky=tk.N)
        tk.Label(self, text="Num of images", font="Arial 10", anchor="center").grid(column=2, row=2, pady=10, padx=10)
        tk.Label(self, textvariable=self.total_num_patches, font="Arial 10").grid(row=3, column=2, padx=10, pady=10,
                                                                                  sticky=tk.N)

    def load_im(self) -> Tuple[np.ndarray, str, PhotoImage]:
        """ Load image and mask from iterators and send to canvas in an overlayed format. """
        im_path, mask_path = next(self.ims_iter), next(self.masks_iter)
        self.viewed_ims += 1
        name = os.path.basename(im_path)
        self.save_checkpoint(name)

        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = np.zeros((3200, 3200, 3), dtype=np.uint8)
        mask_gray = mask_path
        mask[:, :, 1] = cv2.imread(mask_gray, cv2.IMREAD_GRAYSCALE)
        mask[:, :, 2] = cv2.imread(mask_gray, cv2.IMREAD_GRAYSCALE)
        im_overlaid = cv2.addWeighted(im, 1.0, mask, 0.3, 0.5)
        im_canvas = self.to_image_tk(im_overlaid)
        return im, name, im_canvas

    def update_interface(self):
        self.im, self.name, self.im_canvas = self.load_im()
        self.canvas.create_image(0, 0, image=self.im_canvas, anchor=tk.NW)
        self.progress.set("{}/{}".format(self.viewed_ims, self.num_ims))

    def to_image_tk(self, im: np.ndarray) -> PhotoImage:
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
        gx, gy = event.x * self.reduction_ratio, event.y * self.reduction_ratio
        bb = self.get_bbox(gx, gy, self.patch_size)
        self.get_patch(bb)
        self.patches_count += 1
        self.total_num_patches.set(str(self.patches_count))

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

    def pop_seen_images(self):
        with open(checkpoint_file, 'r') as file:
            seen_files = [i.strip('\n') for i in file.readlines()]
        self.ims_list = [i for i in self.ims_list if not os.path.basename(i) in seen_files]
        self.masks_list = [i for i in self.masks_list if not os.path.basename(i) in seen_files]

    """ CALLBACKS """
    def cb_next_image(self):
        self.update_interface()

    """ STATIC """
    @staticmethod
    def save_checkpoint(name):
        with open(checkpoint_file, 'a') as file:
            file.write(f'{name}\n')

    @staticmethod
    def current_num_patches():
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
    inter = PatchGenerator(ims_directory, masks_directory, staining=staining, rand_order=RANDOM_ORDER)
    inter.pack(fill="both", expand=True)
    inter.mainloop()
