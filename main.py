from unet_model import unet_model
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import normalize
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from skimage import io
import time
from PIL import Image
import random
from sklearn.model_selection import train_test_split

patch_size = 256
image_dataset = []
mask_dataset = []

def get_images_from_tif(data_path: str):
    dataNames = [i for i in os.listdir(data_path) if i.endswith('.tif')]
    for dataName in tqdm(dataNames):
        ims = io.imread(data_path + dataName)
        dirName = data_path + dataName[:-4] + '/'
        if not os.path.isdir(dirName):
            os.mkdir(dirName)

        [num_ims, h, w] = ims.shape
        patch_num = 0
        for i in tqdm(range(num_ims), leave=False):
            im = ims[i]
            stride = 100
            for x in range(0, w, patch_size-stride):
                if x+patch_size >= w:
                    continue
                for y in range(0, h, patch_size-stride):
                    if y+patch_size >= h:
                        continue
                    patch = Image.fromarray(im[y:y+patch_size, x:x+patch_size])
                    patch_name = str(patch_num).rjust(6, '0') + '.png'
                    if not os.path.isfile(dirName + patch_name):
                        patch.save(dirName + patch_name)
                    patch_num += 1


def get_images(imsPath : str):
    image_dataset = []
    images = os.listdir(imsPath)
    for i, imName in enumerate(images):
        image_dataset.append(cv2.imread(imsPath + '/' + imName, 0))
    return image_dataset


def get_model():
    return unet_model(patch_size, patch_size, 1)


# def train_model():
#     train_path = "data/stage1_train/"
#     test_path = "data/stage1_test/"
#
#     train_ids = next(os.walk(train_path))[1]
#     test_ids = next(os.walk(test_path))[1]
#
#     X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
#     Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


def prediction(model, image, patch_size):
    segm_img = np.zeros(image.shape[:2]) # Array with zeros to be filled with segmented values
    patch_num = 1


if __name__ == '__main__':
    im_path = "images/"
    # get_images_from_tif(im_path) # Run just ONCE

    # 1. Get data
    imFolders = [im_path+i for i in os.listdir(im_path) if os.path.isdir(im_path+i)]
    imFolders.sort()
    # testing = get_images(imFolders[0])
    # testing_gt = get_images(imFolders[1])
    image_dataset = get_images(imFolders[2])
    mask_dataset = get_images(imFolders[3])

    image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
    mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

    xtrain, xtest, ytrain, ytest = train_test_split(image_dataset, mask_dataset, test_size=0.10)

    image_number = random.randint(0, len(xtrain))
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(np.reshape(xtrain[image_number], (256, 256)), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(ytrain[image_number], (256, 256)), cmap='gray')
    plt.show()

    model = get_model()
    weights_fname = 'mitochondria_test.hdf5'

    if [fname for fname in os.listdir() if fname.endswith('.hdf5')]:
        history = model.fit(xtrain, ytrain, batch_size=16, verbose=1, epochs=1,
                            validation_data=(xtest, ytest), shuffle=False)
        model.save(weights_fname)

    # Evaluate the model
    model.load_weights(weights_fname)

    _, acc = model.evaluate(xtest, ytest)
    print("Accuracy = ", (acc * 100.0), '%')

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'y', label='Training_loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    #
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    #
    # plt.plot(epochs, acc, 'y', label="Training_acc")
    # plt.plot(epochs, val_acc, 'r', label="Validation_acc")
    # plt.title("Training and validation accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.show()











