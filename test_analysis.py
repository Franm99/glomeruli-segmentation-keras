""" File to analyze the test set prediction results"""
import os
from utils import browse_path
from interface import Viewer
import keras
from unet_model import unet_model
import parameters as params


def get_model() -> keras.Model:
    """ return: U-Net Keras model (TF2 version)"""
    return unet_model(params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE, 1)


def prepare_model(dir_path):
    model = get_model()
    weights_filename = os.path.join(dir_path, 'weights/model.hdf5')
    model.load_weights(weights_filename)
    return model


def test_workflow():
    output_path = browse_path()

    # 1. Load model and weights from desired output log folder
    prepare_model(output_path)

    # 2.


def launch_interface():
    output_path = browse_path()
    viewer = Viewer(output_folder=output_path, masks_folder='masks')
    viewer.pack(fill="both", expand=True)
    viewer.mainloop()


if __name__ == '__main__':
    launch_interface()


