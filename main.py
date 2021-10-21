from unet_model import unet_model
import cv2
import numpy as np
from tensorflow.keras.utils import normalize
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from unet_train import get_images
import random
from tqdm import tqdm
from skimage import io
from PIL import Image

patch_size = 256


def get_model():
    return unet_model(patch_size, patch_size, 1)


def test(model):
    im_path = "images/"

    # 1. Get data
    imFolders = [im_path+i for i in os.listdir(im_path) if os.path.isdir(im_path+i)]
    imFolders.sort()
    # testing = get_images(imFolders[0])
    # testing_gt = get_images(imFolders[1])
    image_dataset = get_images(imFolders[2])
    mask_dataset = get_images(imFolders[3])

    image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
    mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

    # 2. Train model
    xtrain, xtest, ytrain, ytest = train_test_split(image_dataset, mask_dataset, test_size=0.10)

    # 4. Compute IoU score
    ypred = model.predict(xtest)
    ypred_th = ypred > 0.5

    intersection = np.logical_and(ytest, ypred_th)
    union = np.logical_or(ytest, ypred_th)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU score is ", iou_score)

    # 4. Show prediction example
    test_img_number = random.randint(0, len(xtest))
    test_img = xtest[test_img_number]
    ground_truth = ytest[test_img_number]
    test_img_norm = test_img[:, :, 0][:, :, None]
    test_img_input = np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(131)
    plt.title('Testing image')
    plt.imshow(test_img[:, :, 0], cmap="gray")
    plt.subplot(132)
    plt.title('Testing label')
    plt.imshow(ground_truth[:, :, 0], cmap="gray")
    plt.subplot(133)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap="gray")
    plt.savefig("prediction.png")


def test2():
    model = get_model()

    # Evaluate the model
    weights_fname = 'mitochondria_test.hdf5'  # 'mitochondria_test.hdf5' | 'last.hdf5'
    model.load_weights(weights_fname)

    im_path = 'myImages/'
    imFolders = [im_path+i for i in os.listdir(im_path) if os.path.isdir(im_path+i)]
    imFolders.sort()
    ims = []
    for imFolder in imFolders:
        ims.extend(get_images(imFolder))
    ims_dataset = np.expand_dims(normalize(np.array(ims), axis=1), 3)
    prediction = (model.predict(ims_dataset)[:, :, :, 0] > 0.5).astype(np.uint8)
    for i in range(len(prediction)):
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.title('test image')
        plt.imshow(ims[i], cmap="gray")
        plt.subplot(122)
        plt.title('prediction')
        plt.imshow(prediction[i], cmap="gray")
        plt.show()


def get_subpatches_glomeruli(data_path: str):
    dataNames = [i for i in os.listdir(data_path) if i.endswith('.png')]
    patch_size_or = patch_size * 5
    for dataName in tqdm(dataNames):
        im = io.imread(data_path + dataName)
        dirName = data_path + dataName[:-4] + '/'
        if not os.path.isdir(dirName):
            os.mkdir(dirName)

        [h, w, _] = im.shape
        patch_num = 0
        for x in range(0, w, patch_size_or):
            if x+patch_size_or >= w:
                x = w - patch_size_or
            for y in range(0, h, patch_size_or):
                if y+patch_size_or >= h:
                    y = h - patch_size_or
                patch = Image.fromarray(im[y:y+patch_size_or, x:x+patch_size_or, :])
                patch = patch.resize((patch_size, patch_size))
                patch_name = str(patch_num).rjust(6, '0') + '.png'
                if not os.path.isfile(dirName + patch_name):
                    patch.save(dirName + patch_name)
                patch_num += 1


if __name__ == '__main__':
    # get_subpatches_glomeruli('myImages/')  # Run just ONCE
    test2()













