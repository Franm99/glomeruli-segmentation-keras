import cv2.cv2 as cv2
import matplotlib
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import parameters as params
# import random
import tensorflow.keras.callbacks as cb
from time import time
from dataset import Dataset, DatasetImages, DatasetPatches, TestDataset
from dataGenerator import DataGeneratorImages, DataGeneratorPatches, PatchGenerator
from tensorflow.keras.utils import normalize
from tqdm import tqdm
from typing import List, Optional, Tuple
from unet_model import unet_model
from utils import get_data_from_xml, MaskType, init_email_info
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import time
import logging
# from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')  # Uncomment when working with SSH and background processes!
if params.SEND_EMAIL:
    sender_email, password, receiver_email = init_email_info()


def get_model() -> keras.Model:
    """ return: U-Net Keras model (TF2 version)"""
    return unet_model(params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE, 1)


class WorkFlow:
    """ """

    def __init__(self, mask_type: MaskType, mask_size: Optional[int], mask_simplex: bool,
                 limit_samples: Optional[float]):
        """ """
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

        self.staining = None
        self.patch_dim = None
        self.resize_ratio = None
        self.logger = None

    def start(self, resize_ratios: List[int], stainings: List[str]):
        for staining in stainings:
            for resize_ratio in resize_ratios:
                ts = time.time()

                self.staining, self.resize_ratio = staining, resize_ratio
                self.patch_dim = params.UNET_INPUT_SIZE * self.resize_ratio

                # 1. Each training iteration will generate its proper output folder
                self.prepare_output()

                # 2. Preparing data
                xtrainval, xtest, ytrainval, ytest = self.init_data()

                # 3. Training stage
                history = self.train(xtrainval, ytrainval)

                # 4. Test stage
                aprox_hit_pctg = self.test(xtest, ytest)

                # 5. Saving results to output folder and clearning the model variable
                # self._save_results(history)
                log_file = self.save_train_log(history, aprox_hit_pctg)
                del self.model

                # 6. If specified, send output info via e-mail
                exec_time = time.time() - ts
                if params.SEND_EMAIL:
                    self.send_log_email(exec_time, log_file)

    def init_data(self):
        self.logger.info("\n########## DATASET INFO ##########")
        dataset_ims = DatasetImages(self.staining)
        xtrainval, xtest, ytrainval, ytest = dataset_ims.split_train_test(train_size=params.TRAINVAL_TEST_SPLIT_RATE)
        self.logger.info("Train-Validation:          {} images".format(len(xtrainval)))
        self.logger.info("Test:                      {} images".format(len(xtest)))
        return xtrainval, xtest, ytrainval, ytest

    def train(self, xtrainval, ytrainval):
        """ Execute the training stage """
        # Use generator for train+val images to avoid RAM excessive usage
        dataGenImages = DataGeneratorImages(xtrainval, ytrainval, shuffle=False)
        self.logger.info("Batch size:               {}".format(params.BATCH_SIZE))
        self.logger.info("Num of batches:           {}".format(len(dataGenImages)))
        self.logger.info("--------------------------------")

        # PatchGenerator object can be reused for each images batch.
        patchGenerator = PatchGenerator(patch_dim=self.patch_dim,
                                        squared_dim=params.UNET_INPUT_SIZE,
                                        filter=params.FILTER_SUBPATCHES)

        # Patches are saved in a tmp directory, so a new DataGenerator can be set up for the training stage.
        # Once the iteraton finishes, the tmp directory is deleted to avoid unnecessary memory usage.
        for ims_batch, masks_batch in tqdm(dataGenImages, desc="Getting patches from image batches"):
            patches, patches_masks, patches_names = patchGenerator.generate(ims_batch, masks_batch)
            self.save_imgs_list(self.patches_tmp_path, patches, patches_names)
            self.save_imgs_list(self.patches_masks_tmp_path, patches_masks, patches_names)

        datasetPatches = DatasetPatches(self.tmp_folder)
        self.logger.info("Num of generated patches: {}".format(len(datasetPatches.patches_list)))

        # Train - Validation split
        xtrain, xval, ytrain, yval = train_test_split(datasetPatches.patches_list, datasetPatches.patches_masks_list,
                                                      train_size=params.TRAIN_SIZE)
        self.logger.info("Patches for training:     {}".format(len(xtrain)))
        self.logger.info("Patches for validation:   {}".format(len(xval)))

        # Preparing dataset for the training stage. Patches are normalized to a (0,1) tensor format.
        train_dataGen = DataGeneratorPatches(xtrain, ytrain)
        val_dataGen =  DataGeneratorPatches(xval, yval)
        # dataGenPatches = DataGeneratorPatches(datasetPatches.patches_list, datasetPatches.patches_masks_list)

        self.logger.info("\n########## MODEL: {} ##########".format("Classic U-Net"))
        self.model, callbacks = self._prepare_model()

        # 3. TRAINING AND VALIDATION STAGE
        self.logger.info("\n########## TRAINING AND VALIDATION STAGE ##########")
        self.logger.info("Initial num of epochs:    {}".format(params.EPOCHS))
        self.logger.info("Batch size:               {}".format(params.BATCH_SIZE))
        self.logger.info("Early Stopping patience:  {}".format(params.ES_PATIENCE))
        self.logger.info("--------------------------------")

        history = self.model.fit(train_dataGen,
                                 validation_data=val_dataGen,
                                 epochs=params.EPOCHS,
                                 shuffle=False,
                                 verbose=1,
                                 callbacks=callbacks)

        wfile = self.weights_path + '/model.hdf5'
        self.logger.info("Stopped at epoch:         {}".format(len(history.history["loss"])))
        self.logger.info("Final weights saved to {}".format(wfile))
        self.model.save(wfile)

        if params.CLEAR_DATA:
            datasetPatches.clear()

        return history

    def test(self, xtest, ytest):
        self.logger.info("\n########## TESTING STAGE ##########")
        testData = TestDataset(xtest, ytest)
        test_predictions = self.predict(testData)
        aprox_hit_pctg = self.count_segmented_glomeruli(test_predictions, xtest) * 100
        self.logger.info("Aprox. hit percentage:    {}".format(aprox_hit_pctg))
        return aprox_hit_pctg

    def predict(self, test_data: TestDataset, th: float = params.PREDICTION_THRESHOLD):
        predictions = []
        for im, mask, name in test_data:
            prediction = self.get_pred_mask(im, th) * 255  # bool to uint8 casting
            predictions.append(prediction)
            im_path = os.path.join(self.test_pred_path, name)
            cv2.imwrite(im_path, prediction)
        return predictions

    def get_pred_mask(self, im, th):
        [h, w] = im.shape
        # Initializing list of masks
        mask = np.zeros((h, w), dtype=bool)

        # Loop through the whole in both dimensions
        for x in range(0, w, self.patch_dim):
            if x + self.patch_dim >= w:
                x = w - self.patch_dim
            for y in range(0, h, self.patch_dim):
                if y + self.patch_dim >= h:
                    y = h - self.patch_dim
                # Get sub-patch in original size
                patch = im[y:y + self.patch_dim, x:x + self.patch_dim]

                # Median filter applied on image histogram to discard non-tissue sub-patches
                counts, bins = np.histogram(patch.flatten(), list(range(256 + 1)))
                counts.sort()
                median = np.median(counts)
                if median <= 3.:
                    # Non-tissue sub-patches automatically get a null mask
                    prediction_rs = np.zeros((self.patch_dim, self.patch_dim), dtype=np.uint8)
                else:
                    # Tissue sub-patches are fed to the U-net model for mask prediction
                    patch = cv2.resize(patch, (params.UNET_INPUT_SIZE, params.UNET_INPUT_SIZE),
                                       interpolation=cv2.INTER_AREA)
                    patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                    prediction = (self.model.predict(patch_input)[:, :, :, 0] > th).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (self.patch_dim, self.patch_dim),
                                               interpolation=cv2.INTER_AREA)

                    # Final mask is composed by the sub-patches masks (boolean array)
                mask[y:y + self.patch_dim, x:x + self.patch_dim] = \
                    np.logical_or(mask[y:y + self.patch_dim, x:x + self.patch_dim], prediction_rs.astype(bool))
        return mask.astype(np.uint8)  # Change datatype from np.bool to np.uint8

    def prepare_output(self):
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
        self.tmp_folder = os.path.join(self.output_folder_path, 'tmp')
        os.mkdir(self.tmp_folder)
        self.patches_tmp_path = os.path.join(self.tmp_folder, 'patches')
        os.mkdir(self.patches_tmp_path)
        self.patches_masks_tmp_path = os.path.join(self.tmp_folder, 'patches_masks')
        os.mkdir(self.patches_masks_tmp_path)

        # Create logger for saving console info
        logging.basicConfig(filename=os.path.join(self.output_folder_path, "console.log"),
                            format='[%(levelname)s]: %(message)s',
                            level=logging.INFO)
        logger = logging.getLogger(__name__)
        self.logger = logger

        # Displaying iteration info before start the training process
        self.logger.info("########## CONFIGURATION ##########")
        self.logger.info("Staining:       {}".format(self.staining))
        self.logger.info("Resize ratio:   {}".format(self.resize_ratio))

    def _prepare_data(self, dataset: Dataset, resize_ratio: int) -> Tuple:
        self.logger.info("First split: Training+Validation & Testing split:")
        xtrainval, xtest_p, ytrainval, ytest_p = dataset.split_trainval_test(train_size=params.TRAINVAL_TEST_SPLIT_RATE)
        self.list2txt(os.path.join(self.output_folder_path, 'test_list.txt'), [os.path.basename(i) for i in xtest_p])

        self.logger.info("LOADING DATA FROM DISK FOR PROCEEDING TO TRAINING AND TEST:")
        self.logger.info("Loading images from: {}".format(self._ims_path))
        self.logger.info("Loading masks from: {}".format(self._masks_path))
        self.logger.info("Training and Validation:")
        ims, masks = dataset.load_pairs(xtrainval, ytrainval, limit_samples=self._limit_samples)

        self.logger.info("Testing:")
        xtest_list, ytest_list = dataset.load_pairs(xtest_p, ytest_p, limit_samples=self._limit_samples)

        self.logger.info("DATA PREPROCESSING FOR TRAINING.")
        patches_ims, patches_masks = dataset.get_spatches(ims, masks, rz_ratio=resize_ratio,
                                                          filter_spatches=params.FILTER_SUBPATCHES)

        # self.logger.info("Images and labels (masks) prepared for training. Tensor format: (N, W, H, CH)")
        self.logger.info("Second split: Training & Validation split:")
        xtrain, xval, ytrain, yval = dataset.split_train_val(patches_ims, patches_masks)

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
        earlystopping_cb = cb.EarlyStopping(monitor=params.MONITORED_METRIC, patience=params.ES_PATIENCE)
        callbacks = [checkpoint_cb, earlystopping_cb]  # These callbacks are always used

        if params.SAVE_TRAIN_HISTORY:
            csvlogger_cb = cb.CSVLogger(self.logs_path + 'log.csv', separator=',', append=False)
            callbacks.append(csvlogger_cb)

        if params.ACTIVATE_REDUCELR:
            reducelr_cb = cb.ReduceLROnPlateau(monitor='val_loss', patience=params.REDUCELR_PATIENCE)
            callbacks.append(reducelr_cb)

        self.logger.info("Model callback functions for training:")
        self.logger.info("Checkpoint saver:   {}".format("Yes"))
        self.logger.info("EarlyStopping:      {}".format("Yes"))
        self.logger.info("Train history saver:{}".format("Yes" if params.SAVE_TRAIN_HISTORY else "No"))
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
        # TODO use handcrafted masks instead of xml files (more precise count)
        xml_list = [os.path.join(self._xml_path, os.path.basename(i).split('.')[0] + ".xml") for i in test_list]
        counter_total = 0
        counter = 0
        for pred, xml in zip(preds, xml_list):
            data = get_data_from_xml(xml, mask_size=self._mask_size, apply_simplex=self._mask_simplex)
            for r in data.keys():
                points = data[r]
                for (cx, cy) in points:
                    counter_total += 1
                    counter += 1 if pred[cy, cx] else 0
        return counter / counter_total

    @staticmethod
    def save_imgs_list(dir_path: str, imgs_list: List[np.ndarray],
                      names_list: List[str], full_path_names: bool = False):
        filenames_list = list()
        for im, name in zip(imgs_list, names_list):
            if full_path_names:
                filename = name
            else:
                filename = os.path.join(dir_path, name)
                filenames_list.append(filename)
            cv2.imwrite(filename=filename, img=im)

    def _save_results(self, history):
        loss = history.history['loss']
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs, loss, 'y', label="Training_loss")
        plt.title("[{}] Training and validation loss".format(self.log_name))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.output_folder_path, "loss.png"))
        self.logger.info("Training loss")
        self.logger.info("- {}".format(os.path.join(self.output_folder_path, "loss.png")))
        # self.logger.info("- {}".format(os.path.join(self.output_folder_path, "acc.png")))

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

    def save_train_log(self, history, hit_pctg) -> str:
        log_fname = os.path.join(self.output_folder_path, self.log_name.replace("-", "") + '.txt')
        with open(log_fname, 'w') as f:
            # Write parameters used
            f.write("-- PARAMETERS --\n")
            f.write('STAINING               {}\n'.format(self.staining))
            f.write('RESIZE_RATIO           {}\n'.format(self.resize_ratio))
            f.write('PREDICTION_THRESHOLD   {}\n'.format(params.PREDICTION_THRESHOLD))
            f.write('TRAIN_VAL_SPLIT_RATE   {}\n'.format(params.TRAIN_SIZE))
            f.write('BATCH_SIZE             {}\n'.format(params.BATCH_SIZE))
            f.write('EPOCHS                 {}\n'.format(params.EPOCHS))
            f.write('LEARNING_RATE          {}\n'.format(params.LEARNING_RATE))
            f.write('REDUCE_LR              {}\n'.format('Y' if params.ACTIVATE_REDUCELR else 'N'))
            f.write('MIN_LEARNING_RATE      {}\n'.format(params.MIN_LEARNING_RATE))
            f.write('REDUCELR_PATIENCE      {}\n'.format(params.REDUCELR_PATIENCE))
            f.write('ES_PATIENCE            {}\n'.format(params.ES_PATIENCE))
            f.write("--------------------------------------\n")
            # Write training results
            f.write('-- TRAINING RESULTS --\n')
            loss = history.history['loss']
            num_epochs = len(history.history['loss'])
            f.write('TRAINING_LOSS          {}\n'.format(str(loss[-1])))
            f.write('NUM_EPOCHS             {}\n'.format(str(num_epochs)))
            f.write('APROX_HIT_PCTG         {}\n'.format(str(hit_pctg)))

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


    def _save_spatches(self, x: List[np.ndarray], y: List[np.ndarray], dir_path: str):
        ims_path = os.path.join(dir_path, "ims")
        os.mkdir(ims_path)
        masks_path = os.path.join(dir_path, "masks")
        os.mkdir(masks_path)

        max_num = len(str(len(x))) + 1
        names = []
        self.logger.info("Savinng images and masks: {}".format(dir_path))
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


# def train():
#     workflow = WorkFlow(limit_samples=params.DEBUG_LIMIT,
#                         mask_type=params.MASK_TYPE,
#                         mask_size=params.MASK_SIZE,
#                         mask_simplex=params.APPLY_SIMPLEX)
#     workflow.run(resize_ratios=params.RESIZE_RATIOS,
#                  stainings=params.STAININGS)


# def test():
#     import glob
#     ims = glob.glob("D:/DataGlomeruli/gt/Circles/*")
#     ims.sort()
#     idx = random.randint(0, len(ims)-1)
#     print(idx)
#     im = cv2.imread(ims[idx], cv2.IMREAD_GRAYSCALE)
#
#     # Apply erosion
#     kernel = np.ones((5, 5), np.uint8)
#     im = cv2.erode(im, kernel, iterations=1)
#
#     # Count connected components
#     retval, labels, stats, centroids = cv2.connectedComponentsWithStats(im)
#
#     def imshow_components(labels, im):
#         label_hue = np.uint8(179*labels/np.max(labels))
#         blanck_ch = 255*np.ones_like(label_hue)
#         labeled_im = cv2.merge([label_hue, blanck_ch, blanck_ch])
#
#         labeled_im = cv2.cvtColor(labeled_im, cv2.COLOR_HSV2RGB)
#         labeled_im[label_hue==0] = 0
#         plt.figure()
#         plt.subplot(121)
#         plt.imshow(im, cmap="gray")
#         plt.subplot(122)
#         plt.imshow(labeled_im)
#         plt.show()
#
#     imshow_components(labels, im)
#     a=1

#
# def testRegionprops():
#     im_path = '/home/francisco/Escritorio/DataGlomeruli/gt/masks/04B0006786 A 1 HE_x9600y4800s3200.png'
#     im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB)
#     im_th = im.astype(bool)
#     from skimage.measure import label, regionprops, regionprops_table
#     label_im = label(im_th)
#     props = regionprops(label_im)
#     props_table = regionprops_table(label_im, im_th, properties=['label', 'centroid'])
#
#     import pandas as pd
#     data = pd.DataFrame(props_table)
#     print(data)
#
#     plt.figure()
#     plt.imshow(im, cmap="gray")
#     for prop in props:
#         y, x, _= prop.centroid
#         plt.plot(x, y, '.g', markersize=15)
#     plt.show()


def debugger():
    workflow = WorkFlow(limit_samples=params.DEBUG_LIMIT,
                        mask_type=params.MASK_TYPE,
                        mask_size=params.MASK_SIZE,
                        mask_simplex=params.APPLY_SIMPLEX)
    workflow.start(params.RESIZE_RATIOS, params.STAININGS)


if __name__ == '__main__':
    # train()
    # test()
    # testRegionprops()
    debugger()
