"""
TODO
Redirect prints to a log file filtering just those with the "info", "warn" and "err" blueprints"

TODO
Write parameters from the last training performed to a txt. For next training, this file will
be read to check if there are changes that force to generate new data.

TODO: Refactoring and documentation
"""

import cv2.cv2 as cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import parameters as params
import random
import tensorflow.keras.callbacks as cb
import time
from dataset import Dataset
from skimage.measure import label, regionprops
from tensorflow.keras.utils import normalize
from tqdm import tqdm
from typing import List, Optional, Tuple
from unet_model import unet_model
from utils import get_data_from_xml, print_info, check_gpu_availability, MaskType

check_gpu_availability()


def get_model() -> keras.Model:
    """ return: U-Net Keras model (TF2 version)"""
    return unet_model(params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE, 1)


class WorkFlow:
    """ Test bench is designed to study how a specific segmentation network works depending on
     a set of parameters, involving the data preprocessing, model definition, training, validation and testing."""

    def __init__(self, stainings: List[str], mask_type: MaskType, mask_size: Optional[int], mask_simplex: bool,
                 limit_samples: Optional[float]):
        """
        Initialize class variables and main paths.
        Every parameter taking part in the test bench configuration must be initialized here.
        :param stainings: List of stainings (HE, PAS, PM or ALL) with which to launch the test bench.
        :param mask_type: Type of mask to use. By now, just HANDCRAFTED and CIRCULAR can be used. The path to the
        directory containing the groundtruth masks varies depending on this parameter.
        :param mask_size: When using MaskType.CIRCULAR, this parameter sets the circles radius. If None, Radii are
        computed depending on the glomeruli class (ESCLEROSADO, SANO, HIPERCELULAR MES, etc).
        :param mask_simplex: When using MaskType.CIRCULAR, this parameter specifies if the Simplex Algorithm will be
        used to avoid circular masks overlap.
        :param limit_samples: Percentage of the whole data in disk to use. JUST FOR DEBUG!
        """
        self._stainings = stainings
        self._limit_samples = limit_samples
        self._mask_size = mask_size
        self._mask_simplex = mask_simplex

        self._ims_path = params.DATASET_PATH + '/ims'
        self._xml_path = params.DATASET_PATH + '/xml'
        if mask_type == MaskType.HANDCRAFTED:
            self._masks_path = params.DATASET_PATH + '/gt/masks'
        else:
            if mask_type == MaskType.CIRCULAR:
                self._masks_path = params.DATASET_PATH + '/gt/circles'
                if mask_size:
                    self._masks_path = self._masks_path + str(self._mask_size)
                if mask_simplex:
                    self._masks_path = self._masks_path + "_simplex"
            else:
                self._masks_path = params.DATASET_PATH + '/gt/bboxes'

    def run(self):
        """ Method to execute the test bench. Sequentially, the following steps will be executed:
        1. Initialize path where output files will be saved.
        2. Data pre-processing using Dataset class.
        3. Model configuration.
        4. Training
        5. Validation
        6. Testing
        This sequence will be executed for each set of configuration parameters used as input for the test bench class.
        """
        for staining in self._stainings:
            print_info("Testbench launched for {} staining".format(staining))
            self._prepare_output()

            print_info("########## PREPARE DATASET ##########")
            dataset = Dataset(staining=staining, mask_type=params.MASK_TYPE,
                              mask_size=self._mask_size, mask_simplex=self._mask_simplex)
            xtrain, xval, xtest, ytrain, yval, ytest = self._prepare_data(dataset)

            print_info("########## PREPARE MODEL: {} ##########".format("U-Net"))  # TODO: Select from set of models
            model, callbacks = self._prepare_model()

            # 3. TRAINING AND VALIDATION STAGE
            print_info("########## TRAINING AND VALIDATION STAGE ##########")
            print_info("Num epochs: {}".format(params.EPOCHS))
            print_info("Batch size: {}".format(params.BATCH_SIZE))
            print_info("Patience for Early Stopping: {}".format(params.ES_PATIENCE))
            print_info("LAUNCHING TRAINING PROCESS:")
            history = model.fit(xtrain, ytrain, batch_size=params.BATCH_SIZE, verbose=1, epochs=params.EPOCHS,
                                validation_data=(xval, yval), shuffle=False, callbacks=callbacks)
            print_info("TRAINING PROCESS FINISHED.")
            wfile = self.weights_path + '/model.hdf5'
            print_info("Saving weights to: {}".format(wfile))
            model.save(wfile)
            print_info("Saving loss and accuracy results collected over epochs.")
            self._save_results(history)
            iou_score = self.compute_IoU(xval, yval, model, th=params.PREDICTION_THRESHOLD)
            print_info("IoU from validation (threshold for binarization={}): {}".format(params.PREDICTION_THRESHOLD,
                                                                                        iou_score))
            print_info("Saving validation predictions (patches) to disk.")
            self._save_val_predictions(xval, yval, model)

            # 4. TESTING STAGE
            print_info("########## TESTING STAGE ##########")
            print_info("Computing predictions for testing set:")
            test_predictions = self._prepare_test(xtest, dataset.test_list, model)  # Test predictions
            test_names = dataset.get_data_list(set="test")
            count_ptg = self.count_segmented_glomeruli(test_predictions, test_names)
            print_info("Segmented glomeruli percentage: counted glomeruli / total = {}".format(count_ptg))
            self.save_train_log(history, iou_score, count_ptg)
            # When a test ends, clear the model to avoid influence in next ones.
            del model

    def _prepare_output(self):
        """
        Initialize directory where output files will be saved for an specific test bench execution.
        :return: None
        """
        self.log_name = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_folder_path = os.path.join(params.OUTPUT_BASENAME, self.log_name)
        os.mkdir(self.output_folder_path)
        self.weights_path = os.path.join(self.output_folder_path, 'weights')
        os.mkdir(self.weights_path)
        self.val_pred_path = os.path.join(self.output_folder_path, 'val_pred')
        os.mkdir(self.val_pred_path)
        self.test_pred_path = os.path.join(self.output_folder_path, 'test_pred')
        os.mkdir(self.test_pred_path)
        self.logs_path = os.path.join(self.output_folder_path, 'logs')
        os.mkdir(self.logs_path)

    def _prepare_data(self, dataset: Dataset) -> Tuple:
        print_info("First split: Training+Validation & Testing split:")
        xtrainval, xtest_p, ytrainval, ytest_p = dataset.split_trainval_test(train_size=params.TRAINVAL_TEST_SPLIT_RATE)

        print_info("LOADING DATA FROM DISK FOR PROCEEDING TO TRAINING AND TEST:")
        print_info("Loading images from: {}".format(self._ims_path))
        print_info("Loading masks from: {}".format(self._masks_path))
        print_info("Training and Validation:")
        ims, masks = dataset.load_pairs(xtrainval, ytrainval, limit_samples=self._limit_samples)

        print_info("Testing:")
        xtest_list, ytest_list = dataset.load_pairs(xtest_p, ytest_p, limit_samples=self._limit_samples)

        print_info("DATA PREPROCESSING FOR TRAINING.")
        patches_names, (patches_tensors, patches_labels_tensors) = dataset.get_spatches(ims, masks, rz_ratio=params.RESIZE_RATIO)
        print_info("Images and labels (masks) prepared for training. Tensor format: (N, W, H, CH)")

        print_info("Second split: Training & Validation split:")
        xtrain, xval, ytrain, yval = dataset.split_train_val(patches_tensors, patches_labels_tensors)
        # train & val sets are returned as ndarray tensors, ready to be used as input for the U-Net, while test set is a
        # list. It will be processed in the TEST stage.
        return xtrain, xval, xtest_list, ytrain, yval, ytest_list

    def _prepare_model(self):
        model = get_model()
        weights_backup = self.weights_path + '/backup.hdf5'
        checkpoint_cb = cb.ModelCheckpoint(filepath=weights_backup,  # TODO: change monitored metric to IoU
                                           verbose=1, save_best_only=True)
        # earlystopping_cb = cb.EarlyStopping(monitor='val_loss', patience=params.ES_PATIENCE)
        earlystopping_cb = cb.EarlyStopping(monitor=params.MONITORED_METRIC, patience=params.ES_PATIENCE)
        callbacks = [checkpoint_cb, earlystopping_cb]  # These callbacks are always used

        if params.SAVE_TRAIN_LOGS:
            tensorboard_cb = cb.TensorBoard(log_dir=self.logs_path)
            callbacks.append(tensorboard_cb)
            csvlogger_cb = cb.CSVLogger(self.logs_path + 'log.csv', separator=',', append=False)
            callbacks.append(csvlogger_cb)

        if params.ACTIVATE_REDUCELR:
            reducelr_cb = cb.ReduceLROnPlateau(monitor='val_loss', patience=params.REDUCELR_PATIENCE)
            callbacks.append(reducelr_cb)

        print_info("Model callback functions for training:")
        print_info("Checkpointer:       {}".format("Y"))
        print_info("EarlyStopper:       {}".format("Y"))
        print_info("TensorBoard logger: {}".format("Y" if params.SAVE_TRAIN_LOGS else "N"))
        print_info("CSV file logger:    {}".format("Y" if params.SAVE_TRAIN_LOGS else "N"))
        return model, callbacks

    def _prepare_test(self, ims, ims_names, model):
        predictions = []
        org_size = int(params.UNET_INPUT_SIZE * params.RESIZE_RATIO)
        for im, im_name in tqdm(zip(ims, ims_names), total=len(ims), desc="Test predictions"):
            pred = self._get_pred_mask(im, org_size, model, th=params.PREDICTION_THRESHOLD)
            predictions.append(pred)
            im_path = os.path.join(self.test_pred_path, im_name)
            # OpenCV works with [0..255] range. pred is in [0..1] format. It might be changed before save.
            cv2.imwrite(im_path, pred*255)
        return predictions

    @staticmethod
    def _get_pred_mask(im, dim, model, th: float):
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
                    patch = cv2.resize(patch, (params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE), interpolation=cv2.INTER_AREA)
                    patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                    prediction = (model.predict(patch_input)[:, :, :, 0] > th).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (dim, dim), interpolation=cv2.INTER_AREA)

                    # Final mask is composed by the sub-patches masks (boolean array)
                mask[y:y + dim, x:x + dim] = np.logical_or(mask[y:y + dim, x:x + dim], prediction_rs.astype(bool))
        return mask.astype(np.uint8)  # Change datatype from np.bool to np.uint8

    def count_segmented_glomeruli(self, preds, test_list):
        xml_list = [os.path.join(self._xml_path, i.split('.')[0] + ".xml") for i in test_list]
        counter_total = 0
        counter = 0
        for pred, xml in zip(preds, xml_list):
            data = get_data_from_xml(xml, mask_size=self._mask_size, apply_simplex=self._mask_simplex)
            for r in data.keys():
                points = data[r]
                for (cx, cy) in points:
                    counter_total += 1
                    counter += 1 if pred[cy, cx] == 1 else 0
        return counter / counter_total

    def count_segmented_glomeruli_adv(self, test_images: List[np.ndarray], test_gt: List[np.ndarray],
                                      test_predictions: List[np.ndarray], test_names: List[str]) -> None:
        """
        Glomeruli hit-miss counter using Region Labelling.
        - Input: Image, ground-truth mask and prediction mask.
        - To Do:
            - Use Region labelling to both count the number of glomeruli contained in the grount-truth mask and get
            their Mass centers.
            - Check if the coordinates where a glomeruli might be found indeed contains a glomeruli.
            - Study false positives (i.e., additional blobs contained in the prediction mask).
            - Optionally show (or save) subfigures to visually check the hit-miss percentage in every test case.
        """
        total_counter = 0
        num_ims = len(test_images)
        for im, gt, pred, name in zip(test_images, test_gt, test_predictions, test_names):
            # 1. Count blobs and find centroids in ground-truth mask.
            gt_th = gt.astype(bool)
            gt_labels = label(gt_th)
            gt_regions = regionprops(gt_labels)

            # 2. Count blobs and find centroids in prediction mask.
            pred_th = gt.astype(bool)
            pred_labels = label(pred_th)
            pred_regions = regionprops(pred_labels)



            # 2. Check if these centroids in the prediction mask correspond to glomeruli.
            counter = 0
            for props in regions:
                cy, cx, _ = props.centroid
                counter += 1 if pred[cy, cx] == 1 else 0








    def _save_results(self, history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs, loss, 'y', label="Training_loss")
        plt.plot(epochs, val_loss, 'r', label="Validation loss")
        plt.title("[{}] Training and validation loss".format(self.log_name))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.output_folder_path, "loss.png"))

        meanIoU = history.history['val_mean_io_u']
        plt.figure()
        plt.plot(epochs, meanIoU, 'y', label="Training_acc")
        plt.title("[{}] Mean Intersection over Union".format(self.log_name))
        plt.xlabel("Epochs")
        plt.ylabel("MeanIoU")
        plt.legend()
        plt.savefig(os.path.join(self.output_folder_path, "mean_iou.png"))
        print_info("You can check the training and validation results during epochs in:")
        print_info("- {}".format(os.path.join(self.output_folder_path, "loss.png")))
        print_info("- {}".format(os.path.join(self.output_folder_path, "acc.png")))

    def compute_IoU(self, xtest, ytest, model, th: float = params.PREDICTION_THRESHOLD):
        ypred = model.predict(xtest)
        ypred_th = ypred > th
        intersection = np.logical_and(ytest, ypred_th)
        union = np.logical_or(ytest, ypred_th)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def _save_val_predictions(self, xval, yval, model):
        for i, (im, mask) in enumerate(zip(xval, yval)):
            val_img_norm = im[:, :, 0][:, :, None]
            val_img_input = np.expand_dims(val_img_norm, 0)
            pred = (model.predict(val_img_input)[0, :, :, 0] > params.PREDICTION_THRESHOLD).astype(np.uint8)
            plt.figure()
            plt.subplot(131)
            plt.imshow(im, cmap="gray")
            plt.title('image')
            plt.subplot(132)
            plt.imshow(mask, cmap="gray")
            plt.title('gt')
            plt.subplot(133)
            plt.imshow(pred, cmap="gray")
            plt.title('pred')
            val_pred_path = os.path.join(self.val_pred_path, str(i) + '.png')
            plt.savefig(val_pred_path)
            plt.close()

    def save_train_log(self, history, iou_score, count_ptg):
        log_fname = os.path.join(self.output_folder_path, time.strftime("%Y%m%d-%H%M%S") + '.txt')
        with open(log_fname, 'w') as f:
            # Write parameters used
            f.write("TRAINING PARAMETERS:\n")
            f.write('TRAIN_SIZE={}\n'.format(params.TRAIN_SIZE))
            f.write('STAINING={}\n'.format(params.STAINING))
            f.write('RESIZE_RATIO={}\n'.format(params.TRAIN_SIZE))
            f.write('PREDICTION_THRESHOLD={}\n'.format(params.RESIZE_RATIO))
            f.write('TRAIN_SIZE={}\n'.format(params.TRAIN_SIZE))
            f.write('BATCH_SIZE={}\n'.format(params.BATCH_SIZE))
            f.write('EPOCHS={}\n'.format(params.EPOCHS))
            f.write('LEARNING_RATE={}\n'.format(params.LEARNING_RATE))
            f.write('REDUCE_LR={}\n'.format('Y' if params.ACTIVATE_REDUCELR else 'N'))
            f.write('MIN_LEARNING_RATE={}\n'.format(params.MIN_LEARNING_RATE))
            f.write('REDUCELR_PATIENCE={}\n'.format(params.REDUCELR_PATIENCE))
            f.write('ES_PATIENCE={}\n'.format(params.ES_PATIENCE))
            f.write("--------------------------------------\n")
            # Write training results
            f.write('TRAINING RESULTS\n')
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            f.write('TRAINING_LOSS={}\n'.format(str(loss[-1])))
            f.write('VALIDATION_LOSS={}\n'.format(str(val_loss[-1])))
            f.write('TRAINING_ACC={}\n'.format(str(acc[-1])))
            f.write('VALIDATION_ACC={}\n'.format(str(val_acc[-1])))
            f.write('IoU_VAL_SCORE={}\n'.format(iou_score))
            f.write("--------------------------------------\n")
            # write testing results
            f.write('TESTING RESULTS\n')
            f.write('GLOMERULI_HIT_PERCENTAGE={}\n'.format(count_ptg))


def Train():
    stainings = ["HE"]
    testbench = WorkFlow(stainings=stainings,
                         limit_samples=params.DEBUG_LIMIT,
                         mask_type=params.MASK_TYPE,
                         mask_size=params.MASK_SIZE,
                         mask_simplex=params.APPLY_SIMPLEX)
    testbench.run()


def test():
    import glob
    ims = glob.glob("D:/DataGlomeruli/gt/Circles/*")
    ims.sort()
    idx = random.randint(0, len(ims)-1)
    print(idx)
    im = cv2.imread(ims[idx], cv2.IMREAD_GRAYSCALE)

    # Apply erosion
    kernel = np.ones((5, 5), np.uint8)
    im = cv2.erode(im, kernel, iterations=1)

    # Count connected components
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(im)

    def imshow_components(labels, im):
        label_hue = np.uint8(179*labels/np.max(labels))
        blanck_ch = 255*np.ones_like(label_hue)
        labeled_im = cv2.merge([label_hue, blanck_ch, blanck_ch])

        labeled_im = cv2.cvtColor(labeled_im, cv2.COLOR_HSV2RGB)
        labeled_im[label_hue==0] = 0
        plt.figure()
        plt.subplot(121)
        plt.imshow(im, cmap="gray")
        plt.subplot(122)
        plt.imshow(labeled_im)
        plt.show()

    imshow_components(labels, im)
    a=1


def testRegionprops():
    im_path = '/home/francisco/Escritorio/DataGlomeruli/gt/masks/04B0006786 A 1 HE_x9600y4800s3200.png'
    im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB)
    im_th = im.astype(bool)
    from skimage.measure import label, regionprops, regionprops_table
    label_im = label(im_th)
    props = regionprops(label_im)
    props_table = regionprops_table(label_im, im_th, properties=['label', 'centroid'])

    import pandas as pd
    data = pd.DataFrame(props_table)
    print(data)

    plt.figure()
    plt.imshow(im, cmap="gray")
    for prop in props:
        y, x, _= prop.centroid
        plt.plot(x, y, '.g', markersize=15)
    plt.show()



if __name__ == '__main__':
    Train()
    # test()
    # testRegionprops()