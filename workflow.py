"""
TODO
Redirect prints to a log file filtering just those with the "info", "warn" and "err" blueprints"
"""

import cv2.cv2 as cv2
import matplotlib
matplotlib.use('Agg')  # Uncomment when working with SSH and background processes!
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import parameters as params
import random
import tensorflow.keras.callbacks as cb
from time import time
from dataset import Dataset
from tensorflow.keras.utils import normalize
from tqdm import tqdm
from typing import List, Optional, Tuple
from unet_model import unet_model
from utils import get_data_from_xml, print_info, MaskType, init_email_info
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import time

if params.SEND_EMAIL:
    sender_email, password, receiver_email = init_email_info()


def get_model() -> keras.Model:
    """ return: U-Net Keras model (TF2 version)"""
    return unet_model(params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE, 1)


class WorkFlow:
    """ Test bench is designed to study how a specific segmentation network works depending on
     a set of parameters, involving the data preprocessing, model definition, training, validation and testing."""

    def __init__(self, mask_type: MaskType, mask_size: Optional[int], mask_simplex: bool,
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

    def run(self, resize_ratios: List[int], stainings: List[str]):
        """ Method to execute the test bench. Sequentially, the following steps will be executed:
        1. Initialize path where output files will be saved.
        2. Data pre-processing using Dataset class.
        3. Model configuration.
        4. Training
        5. Validation
        6. Testing
        This sequence will be executed for each set of configuration parameters used as input for the test bench class.
        """
        for staining in stainings:
            for resize_ratio in resize_ratios:
                print_info("########## CONFIGURATION ##########")
                print_info("Staining:       {}".format(staining))
                print_info("Resize ratio:   {}".format(resize_ratio))
                self._prepare_output()

                ts = time.time()
                print_info("########## PREPARE DATASET ##########")
                dataset = Dataset(staining=staining, mask_type=params.MASK_TYPE,
                                  mask_size=self._mask_size, mask_simplex=self._mask_simplex)
                xtrain, xval, xtest, ytrain, yval, ytest = self._prepare_data(dataset, resize_ratio)


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
                test_predictions = self._prepare_test(xtest, dataset.test_list, model, resize_ratio)  # Test predictions
                test_names = dataset.get_data_list(set="test")
                count_ptg = self.count_segmented_glomeruli(test_predictions, test_names)
                print_info("Segmented glomeruli percentage: counted glomeruli / total = {}".format(count_ptg))
                log_file_name = self.save_train_log(history, iou_score, count_ptg, staining, resize_ratio)
                # When a test ends, clear the model to avoid influence in next ones.
                del model
                exec_time = time.time() - ts
                self.send_log_email(exec_time, log_file_name)

    def _prepare_output(self):
        """
        Initialize directory where output files will be saved for an specific test bench execution.
        :return: None
        """
        # Output log folder
        self.log_name = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_folder_path = os.path.join(params.OUTPUT_BASENAME, self.log_name)
        os.mkdir(self.output_folder_path)
        # Weights saved for later usage
        self.weights_path = os.path.join(self.output_folder_path, 'weights')
        os.mkdir(self.weights_path)
        # Validation and test predictions might be subsequently analyzed, so they are saved into disk.
        self.val_pred_path = os.path.join(self.output_folder_path, 'val_pred')
        os.mkdir(self.val_pred_path)
        self.test_pred_path = os.path.join(self.output_folder_path, 'test_pred')
        os.mkdir(self.test_pred_path)
        # If a log system (Tensorboard) is used, its output can be helpful for later analysis
        self.logs_path = os.path.join(self.output_folder_path, 'logs')
        os.mkdir(self.logs_path)
        # With reproducibility purposes, training, validation and test sets will be saved
        self.patches_train_path = os.path.join(self.output_folder_path, 'patches_train')
        os.mkdir(self.patches_train_path)
        self.patches_val_path = os.path.join(self.output_folder_path, 'patches_val')
        os.mkdir(self.patches_val_path)

    def _prepare_data(self, dataset: Dataset, resize_ratio: int) -> Tuple:
        print_info("First split: Training+Validation & Testing split:")
        xtrainval, xtest_p, ytrainval, ytest_p = dataset.split_trainval_test(train_size=params.TRAINVAL_TEST_SPLIT_RATE)
        self.list2txt(os.path.join(self.output_folder_path, 'test_list.txt'), [os.path.basename(i) for i in xtest_p])

        print_info("LOADING DATA FROM DISK FOR PROCEEDING TO TRAINING AND TEST:")
        print_info("Loading images from: {}".format(self._ims_path))
        print_info("Loading masks from: {}".format(self._masks_path))
        print_info("Training and Validation:")
        ims, masks = dataset.load_pairs(xtrainval, ytrainval, limit_samples=self._limit_samples)

        print_info("Testing:")
        xtest_list, ytest_list = dataset.load_pairs(xtest_p, ytest_p, limit_samples=self._limit_samples)

        print_info("DATA PREPROCESSING FOR TRAINING.")
        patches_ims, patches_masks = dataset.get_spatches(ims, masks, rz_ratio=resize_ratio,
                                                          filter_spatches=params.FILTER_SUBPATCHES)

        # print_info("Images and labels (masks) prepared for training. Tensor format: (N, W, H, CH)")

        print_info("Second split: Training & Validation split:")
        xtrain, xval, ytrain, yval = dataset.split_train_val(patches_ims, patches_masks)

        if params.SAVE_TRAINVAL:
            self._save_spatches(xtrain, ytrain, self.patches_train_path)
            self._save_spatches(xval, yval, self.patches_val_path)

        # train & val sets are returned as ndarray tensors, ready to be used as input for the U-Net, while test set is a
        # list. It will be processed in the TEST stage.
        xtrain_tensor, ytrain_tensor = self._normalize(xtrain, ytrain)
        xval_tensor, yval_tensor = self._normalize(xval, yval)
        return xtrain_tensor, xval_tensor, xtest_list, ytrain_tensor, yval_tensor, ytest_list

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

    def _prepare_test(self, ims, ims_names, model, resize_ratio):
        predictions = []
        org_size = int(params.UNET_INPUT_SIZE * resize_ratio)
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

        # meanIoU = history.history['val_mean_io_u']
        # plt.figure()
        # plt.plot(epochs, meanIoU, 'y', label="Training_acc")
        # plt.title("[{}] Mean Intersection over Union".format(self.log_name))
        # plt.xlabel("Epochs")
        # plt.ylabel("MeanIoU")
        # plt.legend()
        # plt.savefig(os.path.join(self.output_folder_path, "mean_iou.png"))
        print_info("You can check the training and validation results during epochs in:")
        print_info("- {}".format(os.path.join(self.output_folder_path, "loss.png")))
        # print_info("- {}".format(os.path.join(self.output_folder_path, "acc.png")))

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

    def save_train_log(self, history, iou_score, count_ptg, staining, resize_ratio) -> str:
        log_fname = os.path.join(self.output_folder_path, time.strftime("%Y%m%d-%H%M%S") + '.txt')
        with open(log_fname, 'w') as f:
            # Write parameters used
            f.write("TRAINING PARAMETERS:\n")
            f.write('TRAIN_SIZE={}\n'.format(params.TRAIN_SIZE))
            f.write('STAINING={}\n'.format(staining))
            f.write('RESIZE_RATIO={}\n'.format(resize_ratio))
            f.write('PREDICTION_THRESHOLD={}\n'.format(params.PREDICTION_THRESHOLD))
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
            # val_loss = history.history['val_loss']
            # acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            f.write('TRAINING_LOSS={}\n'.format(str(loss[-1])))
            # f.write('VALIDATION_LOSS={}\n'.format(str(val_loss[-1])))
            # f.write('TRAINING_ACC={}\n'.format(str(acc[-1])))
            # f.write('VALIDATION_ACC={}\n'.format(str(val_acc[-1])))
            f.write('IoU_VAL_SCORE={}\n'.format(iou_score))
            f.write("--------------------------------------\n")
            # write testing results
            f.write('TESTING RESULTS\n')
            f.write('APROX_GLOMERULI_HIT_PERCENTAGE={}\n'.format(count_ptg))
        return log_fname

    @staticmethod
    def send_log_email(t: float, fname: str):
        """
        Send informative email to know when a training process has finished. Time spent is specified.
        :param t: time spent (in seconds). Preferably, give HH:MM:SS format to improve readability.
        :return: None
        """
        time_mark = time.strftime("%H:%M:%S", time.gmtime(t))
        port = 465  # for SSL
        message = MIMEMultipart("alternative")
        message["Subject"] = "Training finished"
        message["From"] = sender_email
        message["To"] = receiver_email

        html = """\
        <html>
            <body>
                Training finished. For further info, check log file.<br>
                Time spent: {} (h:m:s)<br>
            </body>
        </html>
        """.format(time_mark)

        part1 = MIMEText(html, "html")
        message.attach(part1)

        # Attach log file
        with open(fname, "rb") as att:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(att.read())

        encoders.encode_base64(part)
        part.add_header("Content-disposition",
                        f"attachment; filename= {os.path.basename(fname)}")

        message.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())


    @staticmethod
    def _save_spatches(x: List[np.ndarray], y: List[np.ndarray], dir_path: str):
        ims_path = os.path.join(dir_path, "ims")
        os.mkdir(ims_path)
        masks_path = os.path.join(dir_path, "masks")
        os.mkdir(masks_path)

        max_num = len(str(len(x))) + 1
        names = []
        print_info("Savinng images and masks: {}".format(dir_path))
        for idx, (im, mask) in tqdm(enumerate(zip(x, y)), total=len(x), desc="Saving images"):
            bname = str(idx).zfill(max_num) + ".png"
            names.append(bname)
            cv2.imwrite(os.path.join(ims_path, bname), im)
            cv2.imwrite(os.path.join(masks_path, bname), mask)

    @staticmethod
    def _normalize(ims: List[np.ndarray], masks: List[np.ndarray]):
        """
        Method to convert pairs of images and masks to the expected format as input for the segmentator model.
        :param ims: list of images in numpy ndarray format, range [0..255]
        :param masks: list of masks in numpy ndarray format, range [0..255]
        :return: tuple with normalized sets: (BATCH_SIZE, W, H, CH) and range [0..1]
        """
        ims_t = np.expand_dims(normalize(np.array(ims), axis=1), 3)
        masks_t = np.expand_dims((np.array(masks)), 3) / 255
        return ims_t, masks_t

    @staticmethod
    def list2txt(fname: str, data: List[str]) -> None:
        """
        Method to save a list of strings to a txt file.
        :param fname: txt file full path
        :param data: list containing the data to save in file
        :return: None
        """
        with open(fname, 'w') as f:
            for i in data:
                f.write(i + "\n")


def train():
    workflow = WorkFlow(limit_samples=params.DEBUG_LIMIT,
                        mask_type=params.MASK_TYPE,
                        mask_size=params.MASK_SIZE,
                        mask_simplex=params.APPLY_SIMPLEX)
    workflow.run(resize_ratios=params.RESIZE_RATIOS,
                 stainings=params.STAININGS)


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
    train()
    # test()
    # testRegionprops()