from unet_model import unet_model
import cv2.cv2 as cv2
import numpy as np
from tensorflow.keras.utils import normalize
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split

PATCH_SIZE = 256
mask_dataset = []
_DATASET_PATH = "D:/DataFlomeruli"


def get_model():
    return unet_model(PATCH_SIZE, PATCH_SIZE, 1)


def get_images(imsPath: str):
    image_dataset = []
    images = os.listdir(imsPath)
    for i, imName in enumerate(images):
        image_dataset.append(cv2.imread(imsPath + '/' + imName, 0))
    return image_dataset


if __name__ == '__main__':
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

    image_number = random.randint(0, len(xtrain))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.reshape(xtrain[image_number], (256, 256)), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(ytrain[image_number], (256, 256)), cmap='gray')
    plt.savefig("sample.png")
    plt.close()

    model = get_model()
    model.load_weights('mitochondria_test.hdf5')
    # weights_fname = 'mitochondria_test.hdf5'
    # checkpointer = ModelCheckpoint(weights_fname, verbose=1, save_best_only=True)
    # callbacks = [checkpointer]

    # 3. Fit model and save weights
    # history = model.fit(xtrain, ytrain, batch_size=16, verbose=1, epochs=15,
    #                     validation_data=(xtest, ytest), shuffle=False, callbacks=callbacks)
    # model.save('last.hdf5')
    #
    # # 4. Show loss and accuracy results
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'y', label='Training_loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig("loss.png")
    # plt.close()
    #
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    #
    # plt.plot(epochs, acc, 'y', label="Training_acc")
    # plt.plot(epochs, val_acc, 'r', label="Validation_acc")
    # plt.title("Training and validation accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.savefig("accuracy.png")
    # plt.close()

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
