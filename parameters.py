""" Constants to import in other scripts """
from tensorflow.keras.metrics import MeanIoU
# Paths (vary depending on which machine is used)
# Add your own dataset path here:
# DATASET_PATH = '/home/al409458/FISABIO/DataGlomeruli'  # Ubuntu (Guepard) -> MAIN DEVICE
DATASET_PATH = 'D:/DataGlomeruli'  # Windows
# DATASET_PATH = '/home/francisco/Escritorio/DataGlomeruli'  # Ubuntu (Alien5)


OUTPUT_BASENAME = 'output'
TEST_IMS_PATH = DATASET_PATH + '/train_val/patches'
TEST_MASKS_PATH = DATASET_PATH + '/train_val/patches_masks'

# Images and input for model
PATCH_SIZE = (3200, 3200)
UNET_INPUT_SIZE = 256
RESIZE_RATIO = 4
STAINING = 'HE'

# Model parameters
LEARNING_RATE = 0.001
MODEL_METRICS = [  # Metrics: https://keras.io/api/metrics/
    # 'accuracy',
    MeanIoU(num_classes=2)  # Most recommendable metric for Image Segmentation
]

# Training parameters and hyper-parameters
TRAINVAL_TEST_SPLIT_RATE = 0.9
TRAINVAL_TEST_RAND_STATE = 30  # Integer from 0 to 42 (inclusive)
TRAIN_VAL_RAND_STATE = 30  # Integer from 0 to 42 (inclusive)
TRAIN_SIZE = 0.8
MIN_LEARNING_RATE = 0.0001
PREDICTION_THRESHOLD = 0.5
BATCH_SIZE = 16
EPOCHS = 100

# Model callbacks configuration
ES_PATIENCE = 3
REDUCELR_PATIENCE = 2
""" TODO
Think a better way to include callbacks (maybe using args or kwargs)
Something like a dictionary with callback functions as items and 
dictionaries with arguments as values should work:

from tensorflow.keras.callbacks import *
CALLBACKS = {
    EarlyStopping: {"monitor": "val_loss", "patience": 2},
    ...
}

Then:

callbacks = []
for fn, kwargs in CALLBACKS.items():
    callbacks.append(fn(**kwargs))
"""
SAVE_TRAIN_LOGS = False
ACTIVATE_REDUCELR = False
LOAD_SPATCHES = False

DEBUG_LIMIT = None
MASK_SIZE = 100
APPLY_SIMPLEX = False
