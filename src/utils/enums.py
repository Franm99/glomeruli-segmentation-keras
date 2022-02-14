""" Enum classes """
from enum import Enum, auto
from strenum import StrEnum

from src.utils.utils import Wrapper
from src.model import keras_models


class Models(Enum):
    """ Enum to select the desired network model to train or work with. """
    SIMPLE_UNET = Wrapper(keras_models.simple_unet)
    # ADD YOUR MODELS HERE


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

