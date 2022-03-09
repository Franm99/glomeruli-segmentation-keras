""" Constants to import in other scripts """
from src.utils.enums import MaskType, Staining

# TODO
# Paths (vary depending on which machine is used)
# Add your own dataset path here:
# DATASET_PATH = '/home/al409458/FISABIO/DataGlomeruli'  # Ubuntu (Guepard) -> MAIN DEVICE
# DATASET_PATH = 'D:/DataGlomeruli'  # Windows
DATASET_PATH = '/home/francisco/Escritorio/DataGlomeruli'  # Ubuntu (Alien5)

# Images and input for keras
STAININGS = [Staining.PM, Staining.PAS, Staining.HE]
RESIZE_RATIOS = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
KERAS_MODEL = "simple_unet"  # Select one from src.keras.keras_models.py

# triggers
FILTER_SUBPATCHES = True
SAVE_TRAINVAL = False
SEND_EMAIL = True
CLEAR_DATA = True
SAVE_TRAIN_HISTORY = True
BALANCE_STAINING = False
DATA_CLONING = True

# Model parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# MONITORED_METRIC = "val_mean_io_u"
MONITORED_METRIC = "val_loss"
MODEL_METRICS = [  # Metrics: https://keras.io/api/metrics/
    'accuracy',
    # MeanIoU(num_classes=2)
]

# Training parameters and hyper-parameters
TRAINVAL_TEST_SPLIT_RATE = 0.9
TRAINVAL_TEST_RAND_STATE = None  # Integer from 0 to 42 (inclusive). None for random
TRAIN_VAL_RAND_STATE = None  # Integer from 0 to 42 (inclusive). None for random
TRAIN_SIZE = 0.8
MIN_LEARNING_RATE = 0.0001
PREDICTION_THRESHOLD = 0.5

# Model callbacks configuration
ES_PATIENCE = 4
REDUCELR_PATIENCE = 2
ACTIVATE_REDUCELR = False

# deprecated
MASK_SIZE = 100
APPLY_SIMPLEX = False
MASK_TYPE = MaskType.HANDCRAFTED
