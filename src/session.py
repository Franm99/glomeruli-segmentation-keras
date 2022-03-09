import csv
import cv2.cv2 as cv2
import glob
import logging
import keras.callbacks
import matplotlib
import numpy as np
import os
import pandas as pd
import statistics as stats
import time
import tensorflow.keras.callbacks as cb
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
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

    Run
    ~~~

    Builds a nested ``for`` loop for training a segmentation Keras model using different values for the most relevant
    parameters: *staining* and *resize_ratio*. Additionally, both Metrics and EmailHandler objects are instantiated to
    keep track of the training results.

    Build Report
    ~~~~~~~~~~~~
    Export the current session training results as a report file.

    Notes
    -----

    * Staining

    This ML model is implemented in medical image segmentation. A staining refer to the different chemical
    products that the medical team has been using to ease the visualization of different elements inside a renal biopsy.

    * Resize ratio

    The U-Net keras model has been defined with an input size of 256x256 pixels. As it is a small portion of our biopsy
    images, a resize ratio is used to resize our images to the allowed input size. Resolution loss has to be considered
    as an issue.
    """

    sessions_dir = const.TRAIN_REPORTS_PATH
    if not os.path.isdir(sessions_dir):
        os.mkdir(sessions_dir)

    def __init__(self, staining_list: List[Staining], resize_ratio_list: List[int], send_report: bool):
        """
        *Class constructor*

        Initializes the session environment: directories, Metrics object and Email handler.

        :param staining_list: list of biopsy stainings to use for the subsequent training processes.
        :param resize_ratio_list: list of resize ratios to use for the subsequent training processes.
        :param send_report: If True, a report email is sent to the specified receiver
        """
        self.staining_list = staining_list
        self.resize_ratio_list = resize_ratio_list
        self.send_report = send_report

        # Initialize directory that will contain the session report
        self.sess_name = "session_" + time.strftime("%d-%m-%Y")
        self.sess_folder = self._init_session_folder(os.path.join(self.sessions_dir, self.sess_name))

        self.metrics = Metrics()
        if self.send_report:
            self.emailInfo = EmailHandler()

    def run(self) -> None:
        """
        Method to launch the current session, keeping track of the training results.

        :return: None
        """
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
        """
        Method to analyze the tracked training results and export the most relevants to a report file.

        :return: Full path and name of the generated report file.
        """
        records_dir = os.path.join(self.sess_folder, "records")
        if not os.path.isdir(records_dir):
            os.mkdir(records_dir)
        return self.metrics.build_report(records_dir)

    @staticmethod
    def _init_session_folder(session_folder):
        """
        *Private*

        Modify session folder name if an existing session exactly has the same name. It will just happen if
        several sessions has been launched at the same date.

        :param session_folder: Folder name to check and modify if needed.
        :return: modified session folder name.
        """
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
        *Class constructor*

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
        *Private*

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
        *Private*

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
        *Private*

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
    """
    Workflow
    ========

    Python class to run one unique training process for a specific training parameters.

    Functionalities
    ---------------

    Launch
    ~~~~~~

    Launch automatically the whole training process with the following stages sequence:
        1. Preprocess training-test data
        2. Training and validation stages
        3. Testing stage
        4. Report training log
    """

    if not os.path.isdir(const.TRAIN_REPORTS_PATH):
        os.mkdir(const.TRAIN_REPORTS_PATH)

    def __init__(self, staining: Staining, resize_ratio: int, session_folder: str):
        """ *Class constructor*

        :param staining: staining value: HE, PAS, PM are currently allowed.
        :param resize_ratio: resize ratio for U-net input.
        :param session_folder: directory where the training report has to be saved.
        """
        self.staining = staining
        self.resize_ratio = resize_ratio
        self.session_folder = session_folder

        # Image patches might be resized to fit the input size of the keras model.
        self.resized_img_dim = const.UNET_INPUT_SIZE * self.resize_ratio

        self._ims_path = const.SEGMENTER_DATA_PATH + '/ims'
        self._xml_path = const.SEGMENTER_DATA_PATH + '/xml'
        self._masks_path = os.path.join(const.SEGMENTER_DATA_PATH, "gt", "masks")

        # Initialize logger
        self.logger: logging.Logger
        self.log_handler = logging.NullHandler()
        self._init_logger()

        # Initialize report folder and path fields
        self.log_name: str
        self.output_folder_path: str
        self.weights_path: str
        self.test_prediction_path: str
        self.logs_path: str
        self.tmp_folder: str
        self.patches_tmp_path: str
        self.patches_masks_tmp_path: str
        self._init_report_folder()

        # Initialize instance fields
        self._results = dict()
        self._log_filename = ""
        self._exec_time = 0.0

    def _init_logger(self) -> None:
        """
        *Private*

        Initialize logger object so the whole process can be tracked and later saved into a log file.

        **Check** the `logging module docs <https://docs.python.org/3/library/logging.html>`_ for further info.

        :return: None
        """
        logging.basicConfig(handlers=[logging.NullHandler()], level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _init_report_folder(self):
        """
        *Private*

        Initialize directory where output files will be saved for an specific test bench execution.

        :return: None
        """
        # Output log folder named with the current date
        self.log_name = time.strftime("%d-%m-%Y_%H-%M-%S")
        self.output_folder_path = os.path.join(self.session_folder, self.log_name)
        os.mkdir(self.output_folder_path)

        # Folder to save training weights
        self.weights_path = os.path.join(self.output_folder_path, 'weights')
        os.mkdir(self.weights_path)

        # Testing predictions should be saved for later analysis.
        self.test_prediction_path = os.path.join(self.output_folder_path, 'test_pred')
        os.mkdir(self.test_prediction_path)

        # Folder to save the metrics values for each epoch (useful to build figures)
        self.logs_path = os.path.join(self.output_folder_path, 'logs')
        os.mkdir(self.logs_path)

        # Folder to temporally save generated patches. Later, this folder will be cleared to avoid memory issues.
        self.tmp_folder = os.path.join(self.output_folder_path, 'tmp')
        os.mkdir(self.tmp_folder)
        self.patches_tmp_path = os.path.join(self.tmp_folder, 'patches')
        os.mkdir(self.patches_tmp_path)
        self.patches_masks_tmp_path = os.path.join(self.tmp_folder, 'patches_masks')
        os.mkdir(self.patches_masks_tmp_path)

        # A logger handler is needed for a new workflow process
        self.log_handler = logging.FileHandler(filename=os.path.join(self.output_folder_path, "console.log"))
        formatter = logging.Formatter('[%(asctime)s] %(funcName)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.log_handler.setFormatter(formatter)
        self.logger.addHandler(self.log_handler)

        # This messages will be redirected to a log file, instead of a OS console or IDE console
        self.logger.info("########## CONFIGURATION ##########")
        self.logger.info("Staining:       {}".format(self.staining))
        self.logger.info("Resize ratio:   {}".format(self.resize_ratio))

    def launch(self) -> None:
        """ Method to launch the whole training process.

        :return: None
        """
        ts = time.time()
        # 1. Preprocess data, spliting into training+validation and test sets.
        x_train_val, x_test, y_train_val, y_test = self._preprocess_data()
        # 2. Training stage: compute model weights and save training history
        history = self._train(x_train_val, y_train_val)
        # 3. Testing stage: do an accuracy estimation from a simple ground-truth and prediction comparison
        estimated_accuracy = self._test(x_test, y_test)
        # 4. Save results
        self._log_filename = self._save_train_log(history, estimated_accuracy)
        self._exec_time = time.time() - ts

        # It is a good practice to clean some variables before finishing
        del self.model
        self.logger.handlers = [logging.NullHandler()]

    def _preprocess_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """ Split the dataset into training and test. The training set is itself composed of the training and
        validation data.

        :return: Lists of the whole path (as string values) of the whole data contained into the training and test sets,
            both including the data (x) and their labels (y).
        """
        self.logger.info("\n########## DATASET INFO ##########")
        dataset_ims = dataset.DatasetImages(self.staining, balance=params.BALANCE_STAINING)
        x_train_val, x_test, y_train_val, y_test = dataset_ims.split_train_test(
                                                            train_size=params.TRAINVAL_TEST_SPLIT_RATE)
        self.logger.info("Train-Validation:          {} images".format(len(x_train_val)))
        self.logger.info("Test:                      {} images".format(len(x_test)))
        return x_train_val, x_test, y_train_val, y_test

    def _train(self, x_train_val: List[str], y_train_val: List[str]) -> keras.callbacks.History:
        """
        *Private*


        Workflow training stage:
            1. Build data generators to avoid loading the whole set of data into memory (not recommended [#f1]_).

            2. Temporally save in disk the generated patches from the original images

            3. Split data into training and validation sets.

            4. Build data generators for each set to feed the model.

            5. Fit the model to the given data and export the results as a Keras history [#f2]_.

        :param x_train_val: list of names referring to the train+validation set of images.
        :param y_train_val: list of names referring to the train+validation set of masks (labels).
        :return: training history (results).

        .. [#f1] `How to use data generators with Keras
         <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_

        .. [#f2] `Keras history object
         <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History>`_
        """
        # Use generator for train+val images to avoid RAM excessive usage
        data_generator_images = dataset.DataGeneratorImages(x_train_val, y_train_val, shuffle=False,
                                                            augmentation=params.DATA_CLONING)
        self.logger.info("Batch size:               {}".format(params.BATCH_SIZE))
        self.logger.info("Num of batches:           {}".format(len(data_generator_images)))
        self.logger.info("--------------------------------")

        # PatchGenerator object can be reused for each batch of images.
        patch_generator = dataset.PatchGenerator(patch_dim=self.resized_img_dim,
                                                 squared_dim=const.UNET_INPUT_SIZE,
                                                 filter=params.FILTER_SUBPATCHES)

        # Patches are saved in a temporal directory, so a new DataGenerator can be set up for the training stage.
        # Once the iteration finishes, the tmp directory is deleted to avoid excessive memory usage.
        for ims_batch, masks_batch in tqdm(data_generator_images, desc="Getting patches from image batches"):
            patches, patches_masks, patches_names = patch_generator.generate(ims_batch, masks_batch)
            self.__save_patches_list(self.patches_tmp_path, patches, patches_names)
            self.__save_patches_list(self.patches_masks_tmp_path, patches_masks, patches_names)

        dataset_patches = dataset.DatasetPatches(self.tmp_folder)
        self.logger.info("Num of generated patches: {}".format(len(dataset_patches.patches_list)))

        # Train - Validation split
        x_train, x_val, y_train, y_val = train_test_split(dataset_patches.patches_list,
                                                          dataset_patches.patches_masks_list,
                                                          train_size=params.TRAIN_SIZE)
        self.logger.info("Patches for training:     {}".format(len(x_train)))
        self.logger.info("Patches for validation:   {}".format(len(x_val)))

        # Preparing dataset for the training stage. Patches are normalized to a (0,1) tensor format.
        train_data_generator = dataset.DataGeneratorPatches(x_train, y_train)
        val_data_generator = dataset.DataGeneratorPatches(x_val, y_val)

        self.logger.info("\n########## MODEL: {} ##########".format(params.KERAS_MODEL))
        self.model, callbacks = self.__prepare_model()

        self.logger.info("\n########## TRAINING AND VALIDATION STAGE ##########")
        self.logger.info("Initial num of epochs:    {}".format(params.EPOCHS))
        self.logger.info("Batch size:               {}".format(params.BATCH_SIZE))
        self.logger.info("Early Stopping patience:  {}".format(params.ES_PATIENCE))
        self.logger.info("--------------------------------")

        # Refer to model.fit docs for further info: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        history = self.model.fit(train_data_generator,
                                 validation_data=val_data_generator,
                                 epochs=params.EPOCHS,
                                 shuffle=False,
                                 verbose=1,
                                 callbacks=callbacks)
        self.logger.info("Stopped at epoch:         {}".format(len(history.history["loss"])))
        self.__save_model()

        if params.CLEAR_DATA:
            dataset_patches.clear()

        return history

    def _test(self, x_test: List[str], y_test: List[str]) -> float:
        """
        *Private*

        Workflow testing stage:
            1. Load test set.

            2. Get test predictions.

            3. Estimate the prediction accuracy over the test set

        :param x_test: list of names referring to the test images.
        :param y_test: list of names referring to the test masks (labels).
        :return: estimated prediction accuracy.
        """
        self.logger.info("\n########## TESTING STAGE ##########")
        test_data = dataset.DatasetTest(x_test, y_test)
        test_predictions = self.__predict(test_data)
        estimated_accuracy = self.__compute_metrics(test_predictions, y_test)
        self.logger.info("Accuracy:                 {}".format(estimated_accuracy))
        return estimated_accuracy

    def _save_train_log(self, history: keras.callbacks.History, estimated_accuracy: float) -> str:
        """
        *Private*

        Write and save the training log report.

        :param history: training keras history object.
        :param estimated_accuracy: estimated prediction accuracy.
        :return: training log report filename.
        """
        log_filename = os.path.join(self.output_folder_path, self.log_name.replace("-", "") + '.txt')
        with open(log_filename, 'w') as f:
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
            self.__save_train_metrics(history, estimated_accuracy)
            loss = self._results[MetricsEnum.LOSS]
            num_epochs = self._results[MetricsEnum.EPOCHS]
            f.write('TRAINING_LOSS          {}\n'.format(str(loss)))
            f.write('NUM_EPOCHS             {}\n'.format(str(num_epochs)))
            f.write('APROX_HIT_PCTG         {}\n'.format(str(estimated_accuracy)))
        return log_filename

    def __prepare_model(self) -> Tuple[keras.Model, List[Any]]:
        """
        *Private*

        Prepare the Keras U-Net model:
            1. Get a pre-defined keras model structure.

            2. Prepare the list of callback functions to use [#f3]_.

        :return: Tuple containing a Keras model (no weights computed yet) and a list of Keras Callback functions.

        .. [#f3] `Available Callbacks for a Keras model <https://keras.io/api/callbacks/>`_
        """
        model = get_model(params.KERAS_MODEL)
        weights_backup = self.weights_path + '/ckpt.hdf5'
        checkpoint_cb = cb.ModelCheckpoint(filepath=weights_backup,  # TODO: change monitored metric to IoU
                                           verbose=1, save_best_only=True)
        early_stopping_cb = cb.EarlyStopping(monitor=params.MONITORED_METRIC, patience=params.ES_PATIENCE)
        callbacks = [checkpoint_cb, early_stopping_cb]  # These callbacks are always used

        if params.SAVE_TRAIN_HISTORY:
            csv_logger_cb = cb.CSVLogger(os.path.join(self.logs_path, 'log.csv'), separator=',', append=False)
            callbacks.append(csv_logger_cb)

        if params.ACTIVATE_REDUCELR:
            reduce_lr_cb = cb.ReduceLROnPlateau(monitor='val_loss', patience=params.REDUCELR_PATIENCE)
            callbacks.append(reduce_lr_cb)

        self.logger.info("Model callback functions for training:")
        self.logger.info("Checkpoint saver:   {}".format("Yes"))
        self.logger.info("EarlyStopping:      {}".format("Yes"))
        self.logger.info("Train history saver:{}".format("Yes" if params.SAVE_TRAIN_HISTORY else "No"))
        return model, callbacks

    def __save_model(self) -> None:
        """
        *Private*

        Save to disk the weights computed for a specific Keras model in ``*.hdf5`` format. The weights filename gets the
        following naming structure:

        ``<keras_model>-<staining>-<resize_ratio>-<date>.hdf5``

        :return: None
        """
        name = f"{params.KERAS_MODEL}-{self.staining}-{self.resize_ratio}-{self.log_name.replace('-', '')}.hdf5"
        weights_file = os.path.join(self.weights_path, name)
        ckpt_file = os.path.join(self.weights_path, "ckpt.hdf5")
        # Chckpoint weights saved during the training process are no longer needed.
        if os.path.isfile(ckpt_file):
            os.remove(ckpt_file)
        self.logger.info("Final weights saved to {}".format(weights_file))
        self.model.save(weights_file)

    def __predict(self, test_data: dataset.DatasetTest, th: float = params.PREDICTION_THRESHOLD) -> List[np.ndarray]:
        """
        *Private*

        Method to make predictions over a test set for a given model and an specific binarization threshold. The
        binarization threshold si applied to the model output.

        :param test_data: test set.
        :param th: binarization threshold.
        :return: List of predictions in numpy array format.
        """
        predictions = []
        for im, mask, name in test_data:
            prediction = self.__get_prediction_mask(im, th) * 255  # bool to uint8 casting
            predictions.append(prediction)
            im_path = os.path.join(self.test_prediction_path, name)
            cv2.imwrite(im_path, prediction)
        return predictions

    def __get_prediction_mask(self, im: np.ndarray, th: float) -> np.ndarray:
        """
        *Private*

        Get the prediction mask for a given **grayscale** image using a pre-trained Keras model.

        :param im: image to take prediction from.
        :param th: binarization threshold to apply to the model output.
        :return: binary prediction mask
        """
        [h, w] = im.shape
        # Initializing list of masks
        mask = np.zeros((h, w), dtype=bool)

        # Loop through the whole image in both dimensions avoiding overflow.
        for x in range(0, w, self.resized_img_dim):
            if x + self.resized_img_dim >= w:
                x = w - self.resized_img_dim
            for y in range(0, h, self.resized_img_dim):
                if y + self.resized_img_dim >= h:
                    y = h - self.resized_img_dim
                # Get sub-patch in original size
                patch = im[y:y + self.resized_img_dim, x:x + self.resized_img_dim]

                # Median filter applied to image histogram to discard non-tissue sub-patches
                counts, bins = np.histogram(patch.flatten(), list(range(256 + 1)))
                counts.sort()
                median = np.median(counts)
                if median <= 3.:
                    # Non-tissue sub-patches automatically get a null mask
                    prediction_rs = np.zeros((self.resized_img_dim, self.resized_img_dim), dtype=np.uint8)
                else:
                    # Tissue sub-patches are fed to the U-net keras for mask prediction
                    patch = cv2.resize(patch, (const.UNET_INPUT_SIZE, const.UNET_INPUT_SIZE),
                                       interpolation=cv2.INTER_AREA)
                    patch_input = np.expand_dims(normalize(np.array([patch]), axis=1), 3)
                    prediction = (self.model.predict(patch_input)[:, :, :, 0] > th).astype(np.uint8)
                    prediction_rs = cv2.resize(prediction[0], (self.resized_img_dim, self.resized_img_dim),
                                               interpolation=cv2.INTER_AREA)

                # The resulting mask is composed by the sub-patches masks (boolean array).
                mask[y:y + self.resized_img_dim, x:x + self.resized_img_dim] = \
                    np.logical_or(mask[y:y + self.resized_img_dim, x:x + self.resized_img_dim], 
                                  prediction_rs.astype(bool))
        return mask.astype(np.uint8)  # bool to np.uint8 casting

    def __save_train_metrics(self, history: keras.callbacks.History, estimated_accuracy: float) -> None:
        """
        *Private*

        Save the most relevant metrics to a results variable to be exported.

        :param history: history of results from the training process.
        :param estimated_accuracy: computed estimated prediction accuracy over a testing set.
        :return: None
        """
        loss = history.history['loss']
        num_epochs = len(history.history['loss'])

        # If new metrics are added in future versions, modify MetricsEnum enumerator.
        self._results[MetricsEnum.LOSS] = loss[-1]
        self._results[MetricsEnum.ACCURACY] = estimated_accuracy
        self._results[MetricsEnum.EPOCHS] = num_epochs
        self._results[MetricsEnum.FOLDER] = self.log_name

    """ STATIC """
    @staticmethod
    def __save_patches_list(dir_path: str, im_list: List[np.ndarray],
                            names_list: List[str], full_path_names: bool = False) -> None:
        """
        *Private*

        Save a list of patches (portions of original images) to disk.

        :param dir_path: directory path where the txt file must be saved.
        :param im_list: list of images that are pretended to be saved.
        :param names_list: list of names that will be attached to the images.
        :param full_path_names: if True, names_list is expected to contain the whole filename path for each image. if
         False, it is expected to contain just their basenames.
        :return: None
        """
        filenames_list = list()
        for im, name in zip(im_list, names_list):
            if full_path_names:
                filename = name
            else:
                filename = os.path.join(dir_path, name)
                filenames_list.append(filename)
            cv2.imwrite(filename=filename, img=im)

    @staticmethod
    def __compute_metrics(predictions: List[np.ndarray], masks_names: List[str]) -> float:
        """
        *Private*
        
        Makes an estimation of the prediction accuracy based on a comparison between the ground-truth masks and the
        predicted ones.

        :param predictions: list of prediction masks obtained from the test set.
        :param masks_names: list of names referring to the ground-truth masks (same order as predictions images is
         expected).
        :return: estimated prediction accuracy.
        """
        gt_count = 0
        # pred_count = 0  # TODO add pre-processing stage to compute estimated precision
        counter = 0
        for prediction, mask_name in zip(predictions, masks_names):
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            centroids = find_blobs_centroids(mask)
            for (cy, cx) in centroids:
                gt_count += 1
                counter += 1 if prediction[cy, cx] else 0
        return (counter / gt_count) * 100  # Percentage

    """ PROPERTY """
    @property
    def results(self):
        """ Training resulting metrics. """
        return self._results

    @property
    def exec_time(self):
        """ Time employed for the training process. """
        return self._exec_time

    @property
    def log_filename(self):
        """ Name of the training log report file. """
        return self._log_filename


# def debugger():
#     staining_list = [Staining.HE, Staining.PAS]
#     resize_ratio_list = [3, 3, 4]
#     Session(staining_list, resize_ratio_list, False)
#
#
# if __name__ == '__main__':
#     debugger()
