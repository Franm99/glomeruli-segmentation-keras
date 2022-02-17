""" Utility functions to trait keras models. """
import keras
from src.model.keras_models import simple_unet


def get_model(model: str, **kwargs) -> keras.Model:
    """ Get Keras network model.
     :param model: defined and compiled Keras model to use. Choose one from: src.model.keras_models
     :**kwargs: input arguments and values for the specified model. Check the model docstring. """
    return eval(model)(**kwargs)


# TODO use weigths file name to determine the keras model to use. Naming example: simple_unet-HE-3-<date>.hdf5
def load_model_weights(model: keras.Model, weights_filename: str) -> keras.Model:
    """ Load pre-trained weights to the specified model. It will be assumed that pre-trained weights were obtained with
    the specified model.
    :param model: Keras model object.
    :param weights_filename: full path to the weights file (*.hdf5/*.h5 extensions are expected).
    :return: keras model with loaded pre-trained weights.
    """
    model.load_weights(weights_filename)
    return model


# def debug():
#     model = get_model(unet, im_h=256, im_w=256, im_ch=1)
#     print(model)
#
#
# if __name__ == '__main__':
#     debug()