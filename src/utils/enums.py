""" Enum classes """
from enum import Enum, auto, unique
from strenum import StrEnum

# from src.model import keras_models


# class Wrapper:
#     """ Wrapper class to wrap a specific function. """
#     def __init__(self, f):
#         self.f = f
#
#     def __call__(self, *args, **kwargs):
#         return self.f(*args, **kwargs)
#
# class Models(Enum):
#     """ Enum to select the desired network model to train or work with. """
#     SIMPLE_UNET = Wrapper(keras_models.simple_unet)
#     # ADD YOUR MODELS HERE


@unique
class MetricsEnum(StrEnum):
    LOSS = "loss"
    EPOCHS = "epochs"
    ACCURACY = "accuracy"
    FOLDER = "report_folder"


@unique
class Staining(StrEnum):
    """ Types of staining for Renal biopsy images. """
    HE = 'HE'
    PAS = 'PAS'
    PM = 'PM'
    ALL = ''  # Will take all existing stainings


class MaskType(Enum):
    """ Type of masks that can be used """
    CIRCULAR = auto()
    BBOX = auto()
    HANDCRAFTED = auto()  # These masks are always obtained from disk


class Size(Enum):
    """ Number of pixels for radii in synthetic mask generation. """
    HUGE = 225
    BIG = 175
    MEDIUM = 150
    SMALL = 100

