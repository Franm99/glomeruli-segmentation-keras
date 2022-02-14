""" Utility functions to trait keras models. """
import keras
from src.model.keras_models import simple_unet


def get_model(model, **kwargs) -> keras.Model:
    """ Get Keras network model.
     :param model: defined and compiled Keras model to use. Choose one from: src.model.keras_models
     :**kwargs: input arguments and values for the specified model. Check the model docstring. """
    return model(**kwargs)


def load_model_weights(model: keras.Model, weights_filename: str) -> keras.Model:
    model.load_weights(weights_filename)
    return model


# def debug():
#     model = get_model(unet, im_h=256, im_w=256, im_ch=1)
#     print(model)
#
#
# if __name__ == '__main__':
#     debug()