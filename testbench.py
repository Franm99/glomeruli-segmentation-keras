from unet_model import unet_model
from typing import List, Optional
import os
import tensorflow.keras.callbacks as cb
from tensorflow.keras.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
import random
from dataset2 import Dataset
from utils import get_data_from_xml, print_info
import cv2.cv2 as cv2
from tqdm import tqdm

PATCH_SIZE = 256
_DATASET_PATH = "D:/DataGlomeruli"
# _DATASET_PATH = "/home/francisco/Escritorio/DataGlomeruli"  # INFO: To modify in server (Alien5 or Artemisa)
_OUTPUT_BASENAME = "output"

# TRAINING PARAMETERS
_BATCH_SIZE = 16
_EPOCHS = 50
_DEF_RZ_RATIO = 4

DEBUG_LIMIT = None
PREDICTION_TH = 0.5



def get_model():
    """ return: U-Net model (TF2 version)"""
    return unet_model(PATCH_SIZE, PATCH_SIZE, 1)


class TestBench:
    def __init__(self, stainings: List[str], limit_samples: Optional[float] = None):
        """ Initialize class variables and main paths. """
        self._stainings = stainings
        self._limit_samples = limit_samples

        # self._weights_path = _DATASET_PATH + '/weights'
        self._weights_path = 'weights'
        self._weights_bname = self._weights_path + '/weights_'
        self._xml_path = _DATASET_PATH + '/xml'

    def _prepare_data(self, dataset: Dataset):
        print_info("Building dataset...")
        xtrainval, xtest_p, ytrainval, ytest_p = dataset.split_trainval_test(train_size=0.9)
        print_info("Loading Training and Validation images...")
        ims, masks = dataset.load_pairs(xtrainval, ytrainval, limit_samples=self._limit_samples)
        print_info("Loading Testing images...")
        xtest, ytest = dataset.load_pairs(xtest_p, ytest_p, limit_samples=self._limit_samples)
        x_t, y_t = dataset.get_spatches(ims, masks, rz_ratio=_DEF_RZ_RATIO, from_disk=True)
        xtrain, xval, ytrain, yval = dataset.split_train_val(x_t, y_t)
        return xtrain, xval, xtest, ytrain, yval, ytest

    def _prepare_model(self, save_logs=True):
        model = get_model()
        # self._file_bname = mask_path.split('_')[-1] + '_' + str(resize_factor)
        weights_backup = self._weights_path + '/backup.hdf5'
        # weights_fname = _WEIGHTS_BASENAME + self._file_bname + '.hdf5'
        checkpoint_cb = cb.ModelCheckpoint(weights_backup, verbose=1, save_best_only=True)
        earlystopping_cb = cb.EarlyStopping(monitor='loss', patience=2)
        if save_logs:
            tensorboard_cb = cb.TensorBoard(log_dir="./logs")
            csvlogger_cb = cb.CSVLogger('logs/log.csv', separator=',', append=False)
            callbacks = [checkpoint_cb, earlystopping_cb, tensorboard_cb, csvlogger_cb]
        else:
            callbacks = [checkpoint_cb, earlystopping_cb]
        return model, callbacks

    def _prepare_test(self, ims, model):
        predictions = []
        org_size = int(PATCH_SIZE * _DEF_RZ_RATIO)
        for im in tqdm(ims, desc="Computing predictions for test set"):
            predictions.append(self._get_mask(im, org_size, model))
        return predictions

    def _get_mask(self, im, dim, model, th: float = PREDICTION_TH):
        """ """
        [h, w] = im.shape
        # Initializing list of masks
        mask = np.zeros((h, w), dtype=bool)

        # Loop through the whole in both dimensions
        for x in range(0, w, dim):
            if x + dim >= w:
                x = w - dim
            for y in range(0, h, dim):
                if y + dim >= h:
                    y = h - dim
                # Get sub-patch in original size
                patch = im[y:y + dim, x:x + dim]

                # Median filter applied on image histogram to discard non-tissue sub-patches
                counts, bins = np.histogram(patch.flatten(), list(range(256 + 1)))
                counts.sort()
                median = np.median(counts)
                if median <= 3.:
                    # Non-tissue sub-patches automatically get a null mask
                    prediction_rs = np.zeros((dim, dim), dtype=np.uint8)
                else:
                    # Tissue sub-patches are fed to the U-net model for mask prediction
                    patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
                    patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                    prediction = (model.predict(patch_input)[:, :, :, 0] > th).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (dim, dim), interpolation=cv2.INTER_AREA)

                    # Final mask is composed by the sub-patches masks (boolean array)
                mask[y:y + dim, x:x + dim] = np.logical_or(mask[y:y + dim, x:x + dim], prediction_rs.astype(np.bool))
        return mask.astype(np.uint8)  # Change datatype from np.bool to np.uint8

    def run(self, train: bool, wfile: Optional[str]):
        for staining in self._stainings:
            print_info("Testbench launched for {} staining".format(staining))
            # 1. Prepare Dataset
            dataset = Dataset(staining=staining)
            xtrain, xval, xtest, ytrain, yval, ytest = self._prepare_data(dataset)

            # 2. Prepare model
            model, callbacks = self._prepare_model()
            if wfile:
                weights_fname = os.path.join(self._weights_path, wfile)
            else:
                weights_fname = self._weights_bname + staining + '.hdf5'

            if train:
                # 3. Training stage
                history = model.fit(xtrain, ytrain, batch_size=_BATCH_SIZE, verbose=1, epochs=_EPOCHS,
                                         validation_data=(xval, yval), shuffle=False, callbacks=callbacks)
                model.save(weights_fname)
                self._save_results(history, staining)
                self.compute_IoU(xval, yval, model, th=PREDICTION_TH)
                self.save_random_prediction(xval, yval, model, staining)
            else:
                # Load pre-trained weights
                model.load_weights(weights_fname)

            # 4. Test stage
            ptest = self._prepare_test(xtest, model)  # Test predictions
            test_list = dataset.get_data_list(set="test")
            acc_proportion = self.count_segmented_glomeruli(ptest, test_list)
            print_info("Segmented / Total = {}".format(acc_proportion))
            # When a test ends, clear the model to avoid influence in next ones.
            del model

    def count_segmented_glomeruli(self, preds, test_list):
        xml_list = [os.path.join(self._xml_path, i.split('.')[0] + ".xml") for i in test_list]
        counter_total = 0
        counter = 0
        for pred, xml in zip(preds, xml_list):
            data = get_data_from_xml(xml)
            for r in data.keys():
                points = data[r]
                for (cx, cy) in points:
                    counter_total += 1
                    counter += 1 if pred[cy, cx] == 1 else 0
        return counter / counter_total

    def _save_results(self, history, bname):
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
        plt.savefig(os.path.join(_OUTPUT_BASENAME, 'loss_' + bname + ".png"))

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.figure()
        plt.plot(epochs, acc, 'y', label="Training_acc")
        plt.plot(epochs, val_acc, 'r', label="Validation_acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(_OUTPUT_BASENAME, 'acc_' + bname + ".png"))
        # plt.show()

    def compute_IoU(self, xtest, ytest, model, th: float = PREDICTION_TH):
        ypred = model.predict(xtest)
        ypred_th = ypred > th
        intersection = np.logical_and(ytest, ypred_th)
        union = np.logical_or(ytest, ypred_th)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU score is ", iou_score)

    def save_random_prediction(self, xtest, ytest, model, bname):
        test_img_number = random.randint(0, len(xtest)-1)
        test_img = xtest[test_img_number]
        ground_truth = ytest[test_img_number]
        test_img_norm = test_img[:, :, 0][:, :, None]
        test_img_input = np.expand_dims(test_img_norm, 0)
        prediction = (model.predict(test_img_input)[0, :, :, 0] > PREDICTION_TH).astype(np.uint8)

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
        plt.savefig(os.path.join(_OUTPUT_BASENAME, 'pred_' + bname + ".png"))

    def compare(self):
        # TODO: complete
        pass


def Train():
    stainings = ["HE", "PAS", "PM", "ALL"]
    limit_samples = None
    testbench = TestBench(stainings=stainings)
    testbench.run(train=True)


if __name__ == '__main__':
    stainings = ["HE", "PAS", "PM", "ALL"]
    limit_samples = None
    testbench = TestBench(stainings=stainings)
    testbench.run(train=False, wfile='weights_200_4.hdf5')