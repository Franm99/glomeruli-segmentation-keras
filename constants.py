""" Constants to import in other scripts """

PATCH_SIZE = (3200, 3200)
UNET_INPUT_SIZE = 256

DEF_TRAIN_SIZE = 0.8
DEF_STAINING = 'HE'
DEF_RZ_RATIO = 4
DEF_PREDICTION_TH = 0.5

DEBUG_LIMIT = 0.1
BATCH_SIZE = 16
EPOCHS = 1
ES_PATIENCE = 2
SAVE_TRAIN_LOGS = False

OUTPUT_BASENAME = 'output'
# DATASET_PATH = 'D:/DataGlomeruli'  # Windows
DATASET_PATH = '/home/francisco/Escritorio/DataGlomeruli'  # Ubuntu (Alien5)
TEST_IMS_PATH = DATASET_PATH + '/train_val/patches'
TEST_MASKS_PATH = DATASET_PATH + '/train_val/patches_masks'

