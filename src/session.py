import csv
import cv2.cv2 as cv2
import glob
import logging
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import statistics as stats
import time
from tensorflow.keras.utils import normalize
import tensorflow.keras.callbacks as cb
from tqdm import tqdm
from typing import List, Any, Tuple

import src.utils.constants as const
import src.utils.parameters as params
import src.dataset as dataset
from src.utils.enums import Staining, MetricsEnum
from src.utils.figures import find_blobs_centroids
from src.utils.misc import EmailHandler
from src.keras.utils import get_model

matplotlib.use('Agg')  # Enable when working with SSH and background processes (e.g., screen)


class Session:
    """
    Session
    =======

    Python class that builds a **training session**. A training session will **automatically** launch a block of
    training processes for a given Keras segmentation keras, keeping track of the results for each of them. Later, a
    **results report** will be generated so the user can fastly notice what are the best training conditions for the
    given task.

    Functionalities
    ---------------

    run
    ~~~

    Builds a nested for loop to vary the staining and resize ratio parameters

    Register metrics
    ~~~~~~~~~~~~~~~~

    Gets a set of metrics and their values to be registered in the current experiment.    """
    sessions_dir = const.TRAIN_REPORTS_PATH
    if not os.path.isdir(sessions_dir):
        os.mkdir(sessions_dir)

    def __init__(self, staining_list: List[Staining], resize_ratio_list: List[int], send_report: bool):
        self.staining_list = staining_list
        self.resize_ratio_list = resize_ratio_list
        self.send_report = send_report

        # Initialize directory that will contain the session report
        self.sess_name = "session_" + time.strftime("%d-%m-%Y")
        self.sess_folder = self._init_session_folder(os.path.join(self.sessions_dir, self.sess_name))

        self.metrics = Metrics()
        if self.send_report:
            self.emailInfo = EmailHandler()

    def run(self):
        for st in self.staining_list:
            for rs in self.resize_ratio_list:
                workflow = WorkFlow(staining=st, resize_ratio=rs, session_folder=self.sess_folder)
                self.metrics.register_sample(st, rs)
                workflow.launch()
                sample_metrics = workflow.results
                self.metrics.register_metrics(sample_metrics)
                if self.send_report:
                    self.emailInfo.send_sample_info(workflow.exec_time, workflow.log_filename)
        report_file = self.build_report()
        if self.send_report:
            self.emailInfo.send_session_info(report_file)

    def build_report(self) -> str:
        records_dir = os.path.join(self.sess_folder, "records")
        if not os.path.isdir(records_dir):
            os.mkdir(records_dir)
        return self.metrics.build_report(records_dir)

    @staticmethod
    def _init_session_folder(session_folder):
        tmp = session_folder
        idx = 0
        while os.path.isdir(tmp):
            idx += 1
            tmp = session_folder + f"_{idx}"
        os.mkdir(tmp)
        return tmp


class Metrics:
    """
    Metrics
    =======

    Python class that keep track of the metrics in a training session. A training session is organized in individual
    trainings, or **samples**, and sets of samples attached to the same training conditions, or **experiments**.

    Functionalities
    ---------------

    Register sample
    ~~~~~~~~~~~~~~~

    Register a new sample and set it as the current tracked experiment.

    Register metrics
    ~~~~~~~~~~~~~~~~

    Gets a set of metrics and their values to be registered in the current experiment.

    Export metrics
    ~~~~~~~~~~~~~~

    Export metrics for a given experiment. Both the whole set of sample metric values and the
    results computed from the set of samples are generated.

    Build report
    ~~~~~~~~~~~~

    Generate a results file for the session.
    """
    def __init__(self):
        """
        Initialize variables to gather metrics for each sample and experiment.
        """
        self.experiments = dict()
        self.tracked_metrics = set()
        self.curr_exp = dict()

    def register_sample(self, staining: Staining, resize_ratio: int) -> None:
        """
        Register a new sample and set as current experiment to be tracked.

        :param staining: sample staining
        :param resize_ratio: sample resize ratio
        :return: None
        """
        key = f"{staining}_{resize_ratio}"
        if key not in self.experiments.keys():
            self.experiments[key] = dict()
        self.curr_exp = self.experiments[key]

    def register_metrics(self, sample_metrics: dict) -> None:
        """
        Gets a set of metrics and their values to be registered in the current experiment.

        :param sample_metrics: dictionary with string labels and values of the metrics tracked for a certain sample.
        :return: None
        """
        for k in sample_metrics.keys():
            self._add_metric(key=k, val=sample_metrics[k])

    def export_metrics(self, records_dir: str) -> None:
        """
        Export metrics for a given experiment. Both the whole set of sample metric values and the
        results computed from the set of samples are generated.

        :param records_dir: full path to directory where metrics records will be saved.
        :return: None
        """
        for label in self.experiments.keys():
            experiment_dir = os.path.join(records_dir, label)
            if not os.path.isdir(experiment_dir):
                os.mkdir(experiment_dir)
            self._records_to_csv(os.path.join(experiment_dir, "records.csv"), label)
            self._get_measures(os.path.join(experiment_dir, "results.csv"), label)

    def build_report(self, records_dir: str) -> str:
        """
        Generate a results file (.txt) for the current session.

        :param records_dir: full path to directory where metrics records will be saved.
        :return: None
        """
        self.export_metrics(records_dir)
        files = glob.glob(records_dir + '/*/results.csv')
        report = pd.DataFrame()
        for filename in files:
            df = pd.read_csv(filename)
            sample_name = os.path.basename(os.path.dirname(filename))
            new_row = {"sample": sample_name,
                       "max_acc": df[df["metric"] == "accuracy"]["max"].item(),
                       "folder": df[df["metric"] == "accuracy"]["best"].item()}
            report = report.append(new_row, ignore_index=True)

        best = report.iloc[report["max_acc"].idxmax()]
        report_file = os.path.join(records_dir, "results.txt")
        with open(report_file, "w") as f:
            f.write("BEST SAMPLE:   {}\n".format(best["sample"]))
            f.write("FOLDER NAME:   {}\n".format(best["folder"]))
            f.write("ACCURACY:      {}\n".format(best["max_acc"]))
        return report_file

    def _add_metric(self, key: str, val: Any) -> None:
        """
        *Private method*

        Adds a new metric to the current experiment if it has not been tracked before. If it
        already exists, a new value is appended to the list of this metric in the current experiment.

        :param key: name of the metric to be added.
        :param val: value of the metric to be added.
        :return: None
        """
        if key not in self.curr_exp.keys():
            self.curr_exp[key] = []
            if key not in self.tracked_metrics:
                self.tracked_metrics.add(key)
        self.curr_exp[key].append(val)

    def _records_to_csv(self, filename: str, label: str) -> None:
        """
        *Private method*

        Save the metrics records of a given experiment named <*label*> into a csv file named <*filename*>.

        :param filename: Name of the file where experiment records will be saved.
        :param label: Name of the experiment to be saved.
        :return: None
        """
        keys = sorted(self.tracked_metrics)
        with open(filename, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(keys)
            writer.writerows(zip(*[self.experiments[label][key] for key in keys]))

    def _get_measures(self, filename: str, label: str) -> None:
        """
        *Private method*

        For a given experiment named <*label*>, this method computes some common measures (e.g., mean value) for
        the set of metrics tracked for this experiment. Next, measures are exported in csv format to a file named
        <*filename*>.

        :param filename: Name of the file where measures will be saved.
        :param label: Name of the experiment to be saved.
        :return: None
        """
        data = self.experiments[label]
        results = dict()
        for metric in self.tracked_metrics:
            if metric != "report_folder":
                max_val = max(data[metric])
                max_idx = data[metric].index(max_val)
                results[metric] = {'mean': "{:.2f}".format(stats.mean(data[metric])),
                                   'stdev': "{:.2f}".format(stats.mean(data[metric])),
                                   'max': max_val,
                                   'best': data["report_folder"][max_idx]}
        with open(filename, "w") as f:
            writer = csv.writer(f, delimiter=',')
            header = ["metric", "mean", "stdev", "max", "best"]
            writer.writerow(header)
            for metric in results.keys():
                line = [metric, results[metric]['mean'],
                        results[metric]['stdev'],
                        results[metric]['max'],
                        results[metric]['best']]
                writer.writerow(line)



class WorkFlow:
    if not os.path.isdir(const.TRAIN_REPORTS_PATH):
        os.mkdir(const.TRAIN_REPORTS_PATH)

    """ """
    def __init__(self, staining: Staining, resize_ratio: int, session_folder: str):
        """ """
        self.staining = staining
        self.resize_ratio = resize_ratio
        self.session_folder = session_folder

        self.patch_dim = const.UNET_INPUT_SIZE * self.resize_ratio

        self._ims_path = const.SEGMENTER_DATA_PATH + '/ims'
        self._xml_path = const.SEGMENTER_DATA_PATH + '/xml'
        self._masks_path = os.path.join(const.SEGMENTER_DATA_PATH, "gt", "masks")

        self.logger = self.init_logger()
        self.log_handler = logging.NullHandler()

        # Prepare report folder
        self.log_name = None
        self.output_folder_path = None
        self.weights_path = None
        self.val_pred_path = None
        self.test_pred_path = None
        self.logs_path = None
        self.tmp_folder = None
        self.patches_tmp_path = None
        self.patches_masks_tmp_path = None
        self.prepare_report_folder()

        self._results = dict()
        self._log_fname = None
        self._exec_time = None

    def launch(self):
        ts = time.time()
        # 2. Preparing data
        xtrainval, xtest, ytrainval, ytest = self.init_data()
        # 3. Training stage
        history = self.train(xtrainval, ytrainval)
        # 4. Testing stage
        estimated_accuracy = self.test(xtest, ytest)
        # 5. Saving results to output folder and clearing the keras variable
        # self._save_results(history)
        self._log_file = self.save_train_log(history, estimated_accuracy)
        del self.model
        # 6. If specified, send output info via e-mail
        self._exec_time = time.time() - ts
        # Clear log handlers
        self.logger.handlers = [logging.NullHandler()]

    def init_data(self):
        self.logger.info("\n########## DATASET INFO ##########")
        dataset_ims = dataset.DatasetImages(self.staining, balance=params.BALANCE_STAINING)
        xtrainval, xtest, ytrainval, ytest = dataset_ims.split_train_test(train_size=params.TRAINVAL_TEST_SPLIT_RATE)
        self.logger.info("Train-Validation:          {} images".format(len(xtrainval)))
        self.logger.info("Test:                      {} images".format(len(xtest)))
        return xtrainval, xtest, ytrainval, ytest

    def train(self, xtrainval, ytrainval):
        """ Execute the training stage """
        # Use generator for train+val images to avoid RAM excessive usage
        dataGenImages = dataset.DataGeneratorImages(xtrainval, ytrainval, shuffle=False,
                                                    augmentation=params.DATA_CLONING)
        self.logger.info("Batch size:               {}".format(params.BATCH_SIZE))
        self.logger.info("Num of batches:           {}".format(len(dataGenImages)))
        self.logger.info("--------------------------------")

        # PatchGenerator object can be reused for each images batch.
        patchGenerator = dataset.PatchGenerator(patch_dim=self.patch_dim,
                                        squared_dim=const.UNET_INPUT_SIZE,
                                        filter=params.FILTER_SUBPATCHES)

        # Patches are saved in a tmp directory, so a new DataGenerator can be set up for the training stage.
        # Once the iteraton finishes, the tmp directory is deleted to avoid unnecessary memory usage.
        for ims_batch, masks_batch in tqdm(dataGenImages, desc="Getting patches from image batches"):
            patches, patches_masks, patches_names = patchGenerator.generate(ims_batch, masks_batch)
            self.save_imgs_list(self.patches_tmp_path, patches, patches_names)
            self.save_imgs_list(self.patches_masks_tmp_path, patches_masks, patches_names)

        datasetPatches = dataset.DatasetPatches(self.tmp_folder)
        self.logger.info("Num of generated patches: {}".format(len(datasetPatches.patches_list)))

        # Train - Validation split
        xtrain, xval, ytrain, yval = train_test_split(datasetPatches.patches_list, datasetPatches.patches_masks_list,
                                                      train_size=params.TRAIN_SIZE)
        self.logger.info("Patches for training:     {}".format(len(xtrain)))
        self.logger.info("Patches for validation:   {}".format(len(xval)))

        # Preparing dataset for the training stage. Patches are normalized to a (0,1) tensor format.
        train_dataGen = dataset.DataGeneratorPatches(xtrain, ytrain)
        val_dataGen = dataset.DataGeneratorPatches(xval, yval)
        # dataGenPatches = DataGeneratorPatches(datasetPatches.patches_list, datasetPatches.patches_masks_list)

        self.logger.info("\n########## MODEL: {} ##########".format(params.KERAS_MODEL))
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
        self.logger.info("Stopped at epoch:         {}".format(len(history.history["loss"])))
        self.save_model()

        if params.CLEAR_DATA:
            datasetPatches.clear()

        return history

    def test(self, xtest, ytest):
        self.logger.info("\n########## TESTING STAGE ##########")
        testData = dataset.DatasetTest(xtest, ytest)
        test_predictions = self.predict(testData)
        estimated_accuracy = self.compute_metrics(test_predictions, ytest)
        self.logger.info("Accuracy:                 {}".format(estimated_accuracy))
        return estimated_accuracy

    def predict(self, test_data: dataset.DatasetTest, th: float = params.PREDICTION_THRESHOLD):
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
                    # Tissue sub-patches are fed to the U-net keras for mask prediction
                    patch = cv2.resize(patch, (const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE),
                                       interpolation=cv2.INTER_AREA)
                    patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                    prediction = (self.model.predict(patch_input)[:, :, :, 0] > th).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (self.patch_dim, self.patch_dim),
                                               interpolation=cv2.INTER_AREA)

                    # Final mask is composed by the sub-patches masks (boolean array)
                mask[y:y + self.patch_dim, x:x + self.patch_dim] = \
                    np.logical_or(mask[y:y + self.patch_dim, x:x + self.patch_dim], prediction_rs.astype(bool))
        return mask.astype(np.uint8)  # Change datatype from np.bool to np.uint8

    def init_logger(self):
        logging.basicConfig(handlers=[logging.NullHandler()], level=logging.INFO)
        return logging.getLogger(__name__)

    def prepare_report_folder(self):
        """
        Initialize directory where output files will be saved for an specific test bench execution.
        :return: None
        """
        # Output log folder
        self.log_name = time.strftime("%d-%m-%Y_%H-%M-%S")
        self.output_folder_path = os.path.join(self.session_folder, self.log_name)
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
        self.log_handler = logging.FileHandler(filename=os.path.join(self.output_folder_path, "console.log"))
        formatter = logging.Formatter('[%(asctime)s] %(funcName)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.log_handler.setFormatter(formatter)
        self.logger.addHandler(self.log_handler)

        # Displaying iteration info before start the training process
        self.logger.info("########## CONFIGURATION ##########")
        self.logger.info("Staining:       {}".format(self.staining))
        self.logger.info("Resize ratio:   {}".format(self.resize_ratio))

    def _prepare_data(self, dataset: dataset.Dataset, resize_ratio: int) -> Tuple:
        self.logger.info("First split: Training+Validation & Testing split:")
        xtrainval, xtest_p, ytrainval, ytest_p = dataset.split_trainval_test(train_size=params.TRAINVAL_TEST_SPLIT_RATE)
        self.list2txt(os.path.join(self.output_folder_path, 'test_list.txt'), [os.path.basename(i) for i in xtest_p])

        self.logger.info("LOADING DATA FROM DISK FOR PROCEEDING TO TRAINING AND TEST:")
        self.logger.info("Loading images from: {}".format(self._ims_path))
        self.logger.info("Loading masks from: {}".format(self._masks_path))
        self.logger.info("Training and Validation:")
        ims, masks = dataset.load_pairs(xtrainval, ytrainval)

        self.logger.info("Testing:")
        xtest_list, ytest_list = dataset.load_pairs(xtest_p, ytest_p)

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
        model = get_model(params.KERAS_MODEL)
        weights_backup = self.weights_path + '/ckpt.hdf5'
        checkpoint_cb = cb.ModelCheckpoint(filepath=weights_backup,  # TODO: change monitored metric to IoU
                                           verbose=1, save_best_only=True)
        earlystopping_cb = cb.EarlyStopping(monitor=params.MONITORED_METRIC, patience=params.ES_PATIENCE)
        callbacks = [checkpoint_cb, earlystopping_cb]  # These callbacks are always used

        if params.SAVE_TRAIN_HISTORY:
            csvlogger_cb = cb.CSVLogger(os.path.join(self.logs_path, 'log.csv'), separator=',', append=False)
            callbacks.append(csvlogger_cb)

        if params.ACTIVATE_REDUCELR:
            reducelr_cb = cb.ReduceLROnPlateau(monitor='val_loss', patience=params.REDUCELR_PATIENCE)
            callbacks.append(reducelr_cb)

        self.logger.info("Model callback functions for training:")
        self.logger.info("Checkpoint saver:   {}".format("Yes"))
        self.logger.info("EarlyStopping:      {}".format("Yes"))
        self.logger.info("Train history saver:{}".format("Yes" if params.SAVE_TRAIN_HISTORY else "No"))
        return model, callbacks

    def save_model(self):
        name = f"{params.KERAS_MODEL}-{self.staining}-{self.resize_ratio}-{self.log_name.replace('-', '')}.hdf5"
        weights_file = os.path.join(self.weights_path, name)
        ckpt_file = os.path.join(self.weights_path, "ckpt.hdf5")
        if os.path.isfile(ckpt_file):
            os.remove(ckpt_file)
        self.logger.info("Final weights saved to {}".format(weights_file))
        self.model.save(weights_file)

    def _prepare_test(self, ims, ims_names, model, resize_ratio):
        predictions = []
        org_size = int(const.UNET_INPUT_SIZE * resize_ratio)
        for im, im_name in tqdm(zip(ims, ims_names), total=len(ims), desc="Test predictions"):
            pred = self._get_pred_mask(im, org_size, model, th=params.PREDICTION_THRESHOLD)
            predictions.append(pred)
            im_path = os.path.join(self.test_pred_path, im_name)
            # OpenCV works with [0..255] range. pred is in [0..1] format. It might be changed before save.
            cv2.imwrite(im_path, pred * 255)
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
                    # Tissue sub-patches are fed to the U-net keras for mask prediction
                    patch = cv2.resize(patch, (const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE),
                                       interpolation=cv2.INTER_AREA)
                    patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                    prediction = (model.predict(patch_input)[:, :, :, 0] > th).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (dim, dim), interpolation=cv2.INTER_AREA)

                    # Final mask is composed by the sub-patches masks (boolean array)
                mask[y:y + dim, x:x + dim] = np.logical_or(mask[y:y + dim, x:x + dim], prediction_rs.astype(bool))
        return mask.astype(np.uint8)  # Change datatype from np.bool to np.uint8

    @staticmethod
    def compute_metrics(preds, masks_names):
        gt_count = 0
        # pred_count = 0  # TODO add pre-processing stage to compute estimated precision
        counter = 0
        for pred, mask_name in zip(preds, masks_names):
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            centroids = find_blobs_centroids(mask)
            for (cy, cx) in centroids:
                gt_count += 1
                counter += 1 if pred[cy, cx] else 0
        return (counter / gt_count) * 100  # Percentage

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

    # @staticmethod
    # def compute_IoU(xtest, ytest, keras, th: float = params.PREDICTION_THRESHOLD):
    #     ypred = keras.predict(xtest)
    #     ypred_th = ypred > th
    #     intersection = np.logical_and(ytest, ypred_th)
    #     union = np.logical_or(ytest, ypred_th)
    #     iou_score = np.sum(intersection) / np.sum(union)
    #     return iou_score

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

    def save_results(self, history, estimated_accuracy):
        """
        If more metrics are computed in future versions, just modify the results' dictionary adding more fields
        """
        loss = history.history['loss']
        num_epochs = len(history.history['loss'])

        self._results[MetricsEnum.LOSS] = loss[-1]
        self._results[MetricsEnum.ACCURACY] = estimated_accuracy
        self._results[MetricsEnum.EPOCHS] = num_epochs
        self._results[MetricsEnum.FOLDER] = self.log_name


    def save_train_log(self, history, estimated_accuracy) -> str:
        log_fname = os.path.join(self.output_folder_path, self.log_name.replace("-", "") + '.txt')
        with open(log_fname, 'w') as f:
            # Write parameters used
            f.write("-- PARAMETERS --\n")
            f.write('STAINING               {}\n'.format(self.staining))
            f.write('RESIZE_RATIO           {}\n'.format(self.resize_ratio))
            f.write('MODEL                  {}\n'.format(params.KERAS_MODEL))
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
            self.save_results(history, estimated_accuracy)
            loss = self.results[MetricsEnum.LOSS]
            num_epochs = self.results[MetricsEnum.EPOCHS]
            f.write('TRAINING_LOSS          {}\n'.format(str(loss)))
            f.write('NUM_EPOCHS             {}\n'.format(str(num_epochs)))
            f.write('APROX_HIT_PCTG         {}\n'.format(str(estimated_accuracy)))

        return log_fname

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
        Method to convert pairs of images and masks to the expected format as input for the segmentator keras.
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

    @property
    def results(self):
        return self._results

    @property
    def report_folder(self):
        return self.log_name

    @property
    def exec_time(self):
        return self._exec_time

    @property
    def log_filename(self):
        return self._log_file


def debugger():
    stainings = [Staining.HE, Staining.PAS]
    rratios = [3, 3, 4]
    session = Session(stainings, rratios, False)


if __name__ == '__main__':
    debugger()
