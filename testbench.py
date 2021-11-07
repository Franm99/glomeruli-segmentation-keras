from unet_model import unet_model
from dataset import Dataset
from typing import List, Optional
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import random

PATCH_SIZE = 256
_DATASET_PATH = "D:/DataGlomeruli"  # INFO: To modify in server (Alien5 or Artemisa)
_WEIGHTS_BASENAME = "weights/weights_"
_OUTPUT_BASENAME = "output"

# TRAINING PARAMETERS
_BATCH_SIZE = 16
_EPOCHS = 2


def get_model():
    """ return: U-Net model (TF2 version)"""
    return unet_model(PATCH_SIZE, PATCH_SIZE, 1)


class TestBench:
    def __init__(self, mask_paths: List[str], resize_factors: List[int], limit_samples: Optional[float] = None):
        self._mask_paths = mask_paths
        self._resize_factors = resize_factors
        self._limit_samples = limit_samples
        self.model = None

    def run(self, save: bool = True):
        for mask_path in self._mask_paths:
            for resize_factor in self._resize_factors:
                print("\nMask radius: {}, Resize factor: {}".format(mask_path.split('_')[-1], resize_factor))
                history = self._test(mask_path, resize_factor, self._limit_samples)
                if save:
                    self._save_results(history)

    def _test(self, mask_path, resize_factor, limit_samples):
        # 1. Get dataset and split into train and test
        dataset = Dataset(mask_path)
        dataset.load(limit_samples=limit_samples)
        dataset.gen_subpatches(rz_ratio=5)
        xtrain, xtest, ytrain, ytest = dataset.split(ratio=0.15)

        # 2.  Get model and prepare testbench
        self.model = get_model()
        self._file_bname = mask_path.split('_')[-1] + '_' + str(resize_factor)
        weights_fname = _WEIGHTS_BASENAME + self._file_bname + '.hdf5'
        checkpointer = ModelCheckpoint(weights_fname, verbose=1, save_best_only=True)
        callbacks = [checkpointer]

        # 3. Train the model
        history = self.model.fit(xtrain, ytrain, batch_size=_BATCH_SIZE, verbose=1, epochs=_EPOCHS,
                            validation_data=(xtest, ytest), shuffle=False, callbacks=callbacks)
        weights_fname_final = _WEIGHTS_BASENAME + self._file_bname + 'final.hdf5'
        self.model.save(weights_fname_final)
        self.compute_IoU(xtest, ytest)
        self.save_random_prediction(xtest, ytest)
        return history

    def _save_results(self, history):
        # 4. Show loss and accuracy results
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs, loss, 'y', label="Training_loss")
        plt.plot(epochs, val_loss, 'r', label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(_OUTPUT_BASENAME, 'loss_' + self._file_bname + ".png"))

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.figure()
        plt.plot(epochs, acc, 'y', label="Training_acc")
        plt.plot(epochs, val_acc, 'r', label="Validation_acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(_OUTPUT_BASENAME, 'acc_' + self._file_bname + ".png"))
        # plt.show()

    def compute_IoU(self, xtest, ytest):
        ypred = self.model.predict(xtest)
        ypred_th = ypred > 0.5
        intersection = np.logical_and(ytest, ypred_th)
        union = np.logical_or(ytest, ypred_th)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU score is ", iou_score)

    def save_random_prediction(self, xtest, ytest):
        test_img_number = random.randint(0, len(xtest))
        test_img = xtest[test_img_number]
        ground_truth = ytest[test_img_number]
        test_img_norm = test_img[:, :, 0][:, :, None]
        test_img_input = np.expand_dims(test_img_norm, 0)
        prediction = (self.model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

        plt.figure()
        plt.subplot(131)
        plt.title('Testing image')
        plt.imshow(test_img[:, :, 0], cmap="gray")
        plt.subplot(132)
        plt.title('Testing label')
        plt.imshow(ground_truth[:, :, 0], cmap="gray")
        plt.subplot(133)
        plt.title('Prediction on test image')
        plt.imshow(prediction, cmap="gray")
        plt.savefig(os.path.join(_OUTPUT_BASENAME, 'pred_' + self._file_bname + ".png"))

    def compare(self):
        # TODO: complete
        pass


if __name__ == '__main__':
    # masks_paths = glob.glob(_DATASET_PATH + '/masks_*')
    # resize_factors = list(range(1, 8, 1))
    masks_paths = [_DATASET_PATH + '/masks_250',
                   _DATASET_PATH + '/masks_400']  # Debug
    resize_factors = [5, 7]  # Debug
    limit_samples = 0.05
    testbench = TestBench(masks_paths, resize_factors, limit_samples)
    testbench.run(save=True)