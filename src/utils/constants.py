""" CONSTANTS FILE """
import os

# Paths
PAR_DIR = "../.."

DATA_PATH = os.path.join(PAR_DIR, "data")
MODELS_PATH = os.path.join(PAR_DIR, "models")
SCRIPTS_PATH = os.path.join(PAR_DIR, "scripts")

RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PIPELINE_RESULTS_PATH = os.path.join(DATA_PATH, "results")
SEGMENTER_DATA_PATH = os.path.join(DATA_PATH, "segmenter")

CLASS_SCRIPTS_PATH = os.path.join(SCRIPTS_PATH, "classification")
SEGM_SCRIPTS_PATH = os.path.join(SCRIPTS_PATH, "segmentation")
TRAIN_REPORTS_PATH = os.path.join(SEGM_SCRIPTS_PATH, "reports")

IMG_SIZE = (3200, 3200)
STRIDE_PTG = 1/4
UNET_INPUT_SIZE = 256
