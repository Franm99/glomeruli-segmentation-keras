""" Constants to import in other scripts """
from tensorflow.keras.metrics import MeanIoU
from src.utils.enums import MaskType, Staining

# Paths (vary depending on which machine is used)
# Add your own dataset path here:
# DATASET_PATH = '/home/al409458/FISABIO/DataGlomeruli'  # Ubuntu (Guepard) -> MAIN DEVICE
# DATASET_PATH = 'D:/DataGlomeruli'  # Windows
DATASET_PATH = '/home/francisco/Escritorio/DataGlomeruli'  # Ubuntu (Alien5)

# Images and input for model
STAININGS = [Staining.PM]
RESIZE_RATIOS = [4, 3]

# triggers
FILTER_SUBPATCHES = True
SAVE_TRAINVAL = False
SEND_EMAIL = True
CLEAR_DATA = True
SAVE_TRAIN_HISTORY = True

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
DEBUG_LIMIT = None
MASK_SIZE = 100
APPLY_SIMPLEX = False
MASK_TYPE = MaskType.HANDCRAFTED
# MASK_TYPE = MaskType.CIRCULAR

# constants
OUTPUT_BASENAME = 'output'
PATCH_SIZE = (3200, 3200)  # TODO rename to IMG_SIZE
UNET_INPUT_SIZE = 256
OPENSLIDE_DIR = "C:\\Users\\Usuario\\Documents\\OpenSlide\\openslide-win64-20171122\\bin"
BIOPSY_DIR = "biopsy"
