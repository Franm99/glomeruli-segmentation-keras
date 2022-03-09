import cv2.cv2 as cv2
import glob
import math

import keras
import numpy as np
import os
import re
from dataclasses import dataclass
from PIL import Image
from sys import platform
from tensorflow.keras.utils import normalize
from tqdm import tqdm
from typing import Tuple, Optional, Iterator, List

import src.utils.constants as const
import src.utils.parameters as params
from src.keras.utils import get_model, load_model_weights
from src.utils.enums import Staining

if platform == "win32":
    _dll_path = os.getenv('OPENSLIDE_PATH')  # TODO add README.md guide for Openslide import when using Windows
    if _dll_path is not None:
        # Python >= 3.8
        with os.add_dll_directory(_dll_path):
            import openslide
else:
    import openslide


class SegmentationPipeline:
    """
    Segmentation Pipeline
    =====================

    Class to implement the whole segmentation process, starting from reading a Whole-Slide Image (WSI) from renal
    biopsies and finishing with the resulting binary mask containing the segmented set of glomeruli.

    Stages
    ------

    +------------+-----------------+----------+--------------------+
    | 1          | 2               | 3        | 4                  |
    +============+=================+==========+====================+
    | Read WSI   | Extract images  | Predict  | Assembly WSI mask  |
    +------------+-----------------+----------+--------------------+

    Functionalities
    ---------------

    Run
    ~~~

    Run the pipeline process.

    Get scaled prediction
    ~~~~~~~~~~~~~~~~~~~~~

    Get the result of the whole pipeline process: a WSI binary mask with the segmented glomeruli.
    """
    def __init__(self):
        """ *Class constructor* """
        self.dims = const.IMG_SIZE
        self.stride_proportion = const.STRIDE_PTG
        self.model_info = None  # ModelData
        self.model = None       # Keras model
        self.slide = None       # WSI
        self.prediction = None  # WSI mask

    def run(self, slide: str, th: float) -> None:
        """
        Execute the sequence of stages that compose the segmentation pipeline.

        :param slide: WSI object in which the segmentation process will take place.
        :param th: binarization threshold that will be used at the output of the segmentation model.
        :return: None
        """
        self.model_info = ModelData(self._select_model_weights(slide))
        self.model = self._load_model()
        # 1. Pre-process WSI: generate patches to iteratively analyze the whole image.
        ims_generator = self._preprocess_slide(slide)
        for im, name in tqdm(ims_generator, desc="Generating predictions"):
            # 2. Obtain segmentation prediction: patches are one-by-one analyzed
            pred = self._predict(im, th)
            # 3. Assembly the prediction patches
            self._assembly(pred, name)

    def get_scaled_prediction(self, scale: int) -> Image:
        """
        Obtain a scaled version of the resulting WSI prediction from the segmentation pipeline.

        :param scale: scale factor to reduce the original prediction mask.
        :return: scaled version of the prediction mask.
        """
        w_wsi, h_wsi = self.slide.dimensions
        w_scaled, h_scaled = int(w_wsi / scale), int(h_wsi / scale)
        pred_pil = Image.fromarray(self.prediction * 255)  # bool to uint8 casting
        return pred_pil.resize((w_scaled, h_scaled))

    def _load_model(self) -> keras.Model:
        """ *Private*

        Load the proper pre-trained weights for the image that is being processed.
        :return: keras segmentation model with pre-trained weights.
        """
        model = get_model(self.model_info.name, im_h=const.UNET_INPUT_SIZE, im_w=const.UNET_INPUT_SIZE)
        return load_model_weights(model, self.model_info.weights)

    def _preprocess_slide(self, slide_file: str) -> Iterator[Tuple[np.ndarray, str]]:
        """ *Private*

        Pipeline 1st stage: Read and divide WSI in portions (patches) to be gradually processed.

        :param slide_file: full path and name of the WSI to be processed
        :return: [GENERATOR] tuple containing a patch from the original WSI and its name with coordinates.
        """
        # Initialization
        self.slide_file = slide_file
        self.slide_name = os.path.basename(slide_file).split('.')[0]
        self.slide = WSI(slide_file)

        w, h = self.slide.dimensions
        self.prediction = np.zeros((h, w), dtype=np.uint8)

        # A thumbnail is a scaled copy of the WSI.
        thumbnail = self.slide.get_thumbnail()
        thumb_name = self.slide_file.split('.')[0] + '_thumbnail.png'
        thumbnail.save(thumb_name)

        # (x, y) coordinates from the upper-left corner for each patch
        x_list, y_list = self.__get_patches_from_thumbnail(thumbnail)

        # Take images from slide corresponding to thumbnail patches
        for x, y in zip(x_list, y_list):
            wsi_region = self.slide.read_region(location=(x, y), size=self.dims, level=0)
            region_name = self.slide_name.split('.')[0] + f"_x{x}_y{y}_s{self.dims[0]}.png"
            # A generator (yield) is used to avoid excessive use of memory.
            yield np.array(wsi_region.convert('L')), region_name  # Numpy grayscale format is required

    def _predict(self, im: np.ndarray, th: float) -> np.ndarray:
        """ *private*

        Pipeline 2nd stage: generate prediction using a pre-trained model

        :param im: grayscale image where the prediction will take place.
        :param th: binarization threshold to be applied at the output of the segmentation model.
        :return: binary mask with segmented glomeruli.
        """
        dim = const.UNET_INPUT_SIZE * self.model_info.resize_ratio
        [h, w] = im.shape
        # Initializing list of masks
        mask = np.zeros((h, w), dtype=bool)
        # Loop through the whole image in both dimensions avoiding overflow
        for x in range(0, w, dim):
            if x + dim >= w:
                x = w - dim
            for y in range(0, h, dim):
                if y + dim >= h:
                    y = h - dim
                # Get sub-patch in original size
                patch = im[y:y + dim, x:x + dim]

                # Median filter applied on image histogram to discard non-tissue sub-patches
                counts, bins = np.histogram(patch.flatten(), list(range(256 + 1)))
                counts.sort()
                median = np.median(counts)
                if median <= 3.:
                    # Non-tissue sub-patches automatically get a null mask
                    prediction_rs = np.zeros((dim, dim), dtype=np.uint8)
                else:
                    # Tissue sub-patches are fed to the U-net keras for mask prediction
                    patch = cv2.resize(patch, (const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE),
                                       interpolation=cv2.INTER_AREA)
                    patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                    prediction = (self.model.predict(patch_input)[:, :, :, 0] >= th).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (dim, dim), interpolation=cv2.INTER_AREA)
                # Final mask is composed by the sub-patches masks (boolean array)
                mask[y:y + dim, x:x + dim] = np.logical_or(mask[y:y + dim, x:x + dim], prediction_rs.astype(bool))
        return mask.astype(np.uint8)  # Change datatype from np.bool to np.uint8

    def _assembly(self, pred: np.ndarray, name: str) -> None:
        """ *private*

        THIRD PIPELINE STAGE

        :param pred: prediction mask from a patch that will be placed into the WSI prediction mask.
        :param name: name attached to the prediction patch. This name contains information about the coordinates of this
         certain patch.
        :return: None
        """
        x, y = self.__get_coords(name)
        s = params.PATCH_SIZE[0]
        self.prediction[y:y+s, x:x+s] = np.logical_or(self.prediction[y:y+s, x:x+s], pred)

    def __get_patches_from_thumbnail(self, im_pil: Image) -> Tuple[List[int], List[int]]:
        """ *private*

        Uses the thumbnail version of the WSI to get the coordinates of those patches that can potentially contain
        glomeruli.

        :param im_pil: thumbnail image in PIL Image format.
        :return: Both lists of X and Y coordinates (in original scale) from the upper-left corner from each patch.
        """
        im = np.array(im_pil)
        h, w, _ = im.shape
        wrd = self.slide.window_reduced_dim
        stride = int(self.slide.window_reduced_dim * const.STRIDE_PTG)
        ss_factor = self.slide.ss_factor

        gx = list()
        gy = list()
        for r in range(0, h, wrd - stride):
            if r + wrd > h:
                r = h - wrd
            for c in range(0, w, wrd - stride):
                if c + wrd > w:
                    c = w - wrd
                patch = im[r:r+wrd, c:c + wrd, :]
                if self.__is_tissue_patch(patch):
                    gr = r * ss_factor
                    gc = c * ss_factor
                    gx.append(gc)
                    gy.append(gr)
        return gx, gy

    """ STATIC """
    @staticmethod
    def _select_model_weights(wsi_name: str) -> str:
        """ *Private*

        Using the name of the target WSI, select the proper pre-trained weights to use in the segmentation model.

        :param wsi_name: name of the target WSI.
        :return: name of the pre-trained weights to be used for the segmentation model.
        """
        # Using RE rule to deduct the staining used for the target WSI (just one staining can be found in each WSI).
        rex = r'{}'.format('|'.join([i for i in Staining if i != '']))
        slide_st = re.search(rex, wsi_name).group()
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        models_list = [i for i in glob.glob(models_dir + '/*.hdf5') if slide_st in i]
        return models_list[0]

    @staticmethod
    def __is_tissue_patch(patch: np.ndarray) -> bool:
        """ *Private*

        Checks whether a certain patch contains any renal tissue or just background from the WSI capture. As the
        background is nearly white, if the ordered histogram reaches high gray levels, it will be assumed that at least 
        a portion of the image contains renal tissue.

        :param patch: Patch to be analysed.
        :return: True if the patch contains any tissue. False elsewhere.
        """
        counts, bins = np.histogram(patch.flatten(), list(range(256 + 1)))
        counts.sort()
        median = np.median(counts)
        if median >= 1:
            return True
        return False

    @staticmethod
    def __get_coords(name: str) -> Tuple[int, int]:
        """ *Private*
        
        Given a patch name, obtain its coordinates.
        
        :param name: name of a certain patch.
        :return: tuple with the (x, y) coordinates of the patch upper-left corner.
        """
        x = int(re.search(r'x([0-9]*)', name).group(1))
        y = int(re.search(r'y([0-9]*)', name).group(1))
        return x, y


class WSI(openslide.OpenSlide):
    """
    Whole-Slide Image (WSI) class
    =============================

    *Inherits from:* `openslide.Openslide <https://openslide.org/api/python/#openslide.OpenSlide>`_.

    An OpenSlide [#f1]_ Image with added specific functionalities.

    Functionalities
    ---------------

    [Overrides] Get thumbnail
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Overrides the original ``Openslide.get_thumbnail()`` method, allowing to avoid the ``size`` input
    parameter to instead automatically compute the best reduction size.

    .. [#f1] `OpenSlide Python library <https://openslide.org/api/python/>`_
    """
    def __init__(self, filename):
        """ *Class constructor* """
        super().__init__(filename)
        self._best_dims, self._ss_factor = self._find_lower_thumbnail_dimensions()
        self._window_reduced_dim = int(params.PATCH_SIZE[0] / self.ss_factor)

    def get_thumbnail(self, size: Optional[Tuple[int, int]] = None) -> Image:
        """
        Overrides get_thumbnail method from openslide.Openslide class.

        :param size: if None, the lower allowed thumbnail size is automatically computed. Else, call to the original
         get_thumbnail() method.
        :return: PIL.Image containing RGB thumbnail.
        """
        if size:
            return super().get_thumbnail(size)
        else:
            return super().get_thumbnail(self._best_dims)

    def _find_lower_thumbnail_dimensions(self) -> Tuple[int, int]:
        """ *Private*

        Compute the lower scale that can be applied to a certain WSI without getting values with decimal digits
        (integer values)

        :return: Tuple containing the best scale factor and the sub-sampling factor (2^best_level).
        """
        max_level = self.level_count - 1
        ss_factor = -1
        for level in range(max_level, 1, -1):
            ss_factor = 2 ** level
            reduced_dim = params.PATCH_SIZE[0] / ss_factor
            stride = reduced_dim * const.STRIDE_PTG
            if math.floor(stride) == stride:
                break
        level = self.get_best_level_for_downsample(ss_factor)
        return self.level_dimensions[level], ss_factor

    @property
    def window_reduced_dim(self) -> int:
        """ Window reduced dimension. """
        return self._window_reduced_dim

    @property
    def ss_factor(self) -> int:
        """ Sub-sampling factor. """
        return self._ss_factor


@dataclass
class ModelData:
    """
    Model Data
    ==========

    Data class that contains the most relevant information extracted from a given pre-trained model weights file.

    The required naming format for the weights file is as follows:

    ``<model_name>-<staining>-<resize_ratio>-<date>.hdf5``
    """
    def __init__(self, model_weights: str):
        """ *Class constructor* """
        self._fields = os.path.basename(model_weights).split('-')
        self._weights = model_weights

    @property
    def weights(self) -> str:
        """ Full path to the weights file. """
        return self._weights

    @property
    def name(self) -> str:
        """ Model used (e.g., ``simple_unet``)."""
        return self._fields[0]

    @property
    def staining(self) -> str:
        """ Staining (e.g., ``HE``, ``PAS``, ``PM``)."""
        return self._fields[1]

    @property
    def resize_ratio(self) -> int:
        """ Resize ratio that have to be applied at the model input. """
        return int(self._fields[2])

    @property
    def date(self) -> str:
        """ Date when the model weights were obtained. """
        return self._fields[3].split('.')[0]
