"""
Author: Fran Moreno
Contact: fran.moreno.se@gmail.com
Date: 15/03/2022

Desc: Enum classes
"""
from enum import Enum, IntEnum, auto, unique
from strenum import StrEnum


@unique
class MetricsEnum(StrEnum):
    """ String Enum containing metrics from a training process. """
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
    """ Type of masks that can be used for the ground-truth. """
    CIRCULAR = auto()
    BBOX = auto()
    HANDCRAFTED = auto()  # These masks are always obtained from disk


class Size(IntEnum):
    """ Number of pixels for radii in synthetic mask generation. """
    HUGE = 225
    BIG = 175
    INTERMEDIATE = 150
    SMALL = 100


class GlomeruliClass(IntEnum):
    # Huge
    MEMBRANOSO = Size.HUGE.value
    GNMP = Size.HUGE.value
    GSSP = Size.HUGE.value

    # Big
    SANO = Size.BIG.value

    # Intermediate
    INCOMPLETO = Size.INTERMEDIATE.value
    SEMILUNAS = Size.INTERMEDIATE.value
    ISQUEMICO = Size.INTERMEDIATE.value
    MIXTO = Size.INTERMEDIATE.value
    ENDOCAPILAR = Size.INTERMEDIATE.value

    # Small
    ESCLEROSADO = Size.SMALL.value
