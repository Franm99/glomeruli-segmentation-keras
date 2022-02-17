""" File to analyze the test set prediction results"""
import os
import keras

from src.model.keras_models import simple_unet
from src.model.model_utils import *
from src.utils.interface import Viewer
from src.utils.utils import browse_path

# TODO merge pred_analysis and post_process scripts. Compute predictions while loading interface


def launch_interface():
    output_path = browse_path()
    viewer = Viewer(output_folder=output_path, masks_folder='masks')
    viewer.pack(fill="both", expand=True)
    viewer.mainloop()


if __name__ == '__main__':
    launch_interface()


