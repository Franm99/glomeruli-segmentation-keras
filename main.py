from unet_model import unet_model
import cv2
import numpy as np
from tensorflow.keras.utils import normalize
import matplotlib.pyplot as plt

patch_size = 256


def get_model():
    return unet_model(patch_size, patch_size, 1)


if __name__ == '__main__':

    model = get_model()

    # Evaluate the model
    weights_fname = 'mitochondria_test.hdf5'  # 'mitochondria_test.hdf5' | 'last.hdf5'
    model.load_weights(weights_fname)

    fname = 'to_fill'
    test_im = cv2.imread(fname, 0)
    test_im_norm = np.expand_dims(normalize(np.array(test_im), axis=1), 3)
    test_im_norm = test_im_norm[:, :, 0][:, :, None]
    test_im_input = np.expand_dims(test_im_norm, 0)

    prediction = (model.predict(test_im_input)[0, :, :, 0] > 0.5).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.title('Testing image')
    plt.imshow(test_im[:, :, 0], cmap="gray")
    plt.subplot(122)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap="gray")
    plt.savefig("my_prediction.png")














