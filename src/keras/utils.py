"""
Author: Fran Moreno
Contact: fran.moreno.se@gmail.com
Date: 15/03/2022
"""
import keras
from src.keras.keras_models import simple_unet


def get_model(model: str, **kwargs) -> keras.Model:
    """ Get Keras network keras.
     :param model: defined and compiled Keras keras to use. Choose one from: src.keras.keras_models
     :**kwargs: input arguments and values for the specified keras. Check the keras docstring. """
    return eval(model)(**kwargs)


def load_model_weights(model: keras.Model, weights_filename: str) -> keras.Model:
    """ Load pre-trained weights to the specified keras. It will be assumed that pre-trained weights were obtained with
    the specified keras.
    :param model: Keras keras object.
    :param weights_filename: full path to the weights file (*.hdf5/*.h5 extensions are expected).
    :return: keras keras with loaded pre-trained weights.
    """
    model.load_weights(weights_filename)
    return model


# def debug():
#     keras = get_model(unet, im_h=256, im_w=256, im_ch=1)
#     print(keras)
#
#
# if __name__ == '__main__':
#     debug()