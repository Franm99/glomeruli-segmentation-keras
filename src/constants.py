""" CONSTANTS FILE """
import os

# Paths
BASE_DIR = ".."
DATA_PATH = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
SEGMENTER_DATA_PATH = os.path.join(DATA_PATH, "segmenter")

OUTPUT_BASENAME = os.path.join(BASE_DIR, "output")


# miscellaneous
IMG_SIZE = (3200, 3200)
STRIDE_PTG = 1/4
UNET_INPUT_SIZE = 256
