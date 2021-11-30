""" Constants to import in other scripts """

# Paths (vary depending on which machine is used)
# Add your own dataset path here:
# DATASET_PATH = 'D:/DataGlomeruli'  # Windows
DATASET_PATH = '/home/francisco/Escritorio/DataGlomeruli'  # Ubuntu (Alien5)

OUTPUT_BASENAME = 'output'
TEST_IMS_PATH = DATASET_PATH + '/train_val/patches'
TEST_MASKS_PATH = DATASET_PATH + '/train_val/patches_masks'

# Images and input for model
PATCH_SIZE = (3200, 3200)
UNET_INPUT_SIZE = 256
RESIZE_RATIO = 4
STAINING = 'HE'

# Model parameters
LEARNING_RATE = 0.1
MODEL_METRICS = [  # Metrics: https://keras.io/api/metrics/
    'accuracy',
]

# Training parameters and hyper-parameters
TRAIN_SIZE = 0.8
MIN_LEARNING_RATE = 0.001
PREDICTION_THRESHOLD = 0.5
BATCH_SIZE = 16
EPOCHS = 100

# Model callbacks configuration
ES_PATIENCE = 3
REDUCELR_PATIENCE = 2
# TODO: Think a better way to include callbacks (maybe using args or kwargs)
SAVE_TRAIN_LOGS = False
ACTIVATE_REDUCELR = True

DEBUG_LIMIT = None
MASK_SIZE = 150
APPLY_SIMPLEX = True
