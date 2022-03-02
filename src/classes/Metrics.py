import csv
import glob
import os
from os.path import dirname
import pandas as pd
import statistics as stats
import sys
from typing import Any

sys.path.append(dirname(dirname(dirname(os.path.abspath(__file__)))))
from src.utils.enums import Staining


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

    def build_report(self, records_dir: str) -> None:
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


# def debugging():
#     metrics = Metrics()
#
#     # Adding some samples manually for debug
#     debug_add_sample(metrics, Staining.HE, 3, 0.88, 0.72, 31, "a")
#     debug_add_sample(metrics, Staining.HE, 3, 0.89, 0.71, 34, "b")
#     debug_add_sample(metrics, Staining.HE, 3, 0.76, 0.70, 35, "c")
#     debug_add_sample(metrics, Staining.HE, 4, 0.79, 0.68, 31, "d")
#     debug_add_sample(metrics, Staining.HE, 4, 0.82, 0.72, 35, "e")
#     debug_add_sample(metrics, Staining.HE, 4, 0.82, 0.72, 35, "f")
#
#     debug_add_sample(metrics, Staining.PAS, 3, 0.80, 0.73, 30, "g")
#     debug_add_sample(metrics, Staining.PAS, 3, 0.82, 0.70, 27, "h")
#     debug_add_sample(metrics, Staining.PAS, 3, 0.81, 0.701, 32, "i")
#     debug_add_sample(metrics, Staining.PAS, 4, 0.78, 0.78, 35, "j")
#     debug_add_sample(metrics, Staining.PAS, 4, 0.76, 0.80, 34, "k")
#     debug_add_sample(metrics, Staining.PAS, 4, 0.775, 0.67, 28, "l")
#
#     debug_add_sample(metrics, Staining.PM, 3, 0.95, 0.80, 35, "m")
#     debug_add_sample(metrics, Staining.PM, 3, 0.965, 0.85, 34, "n")
#     debug_add_sample(metrics, Staining.PM, 3, 0.94, 0.84, 30, "o")
#     debug_add_sample(metrics, Staining.PM, 4, 0.89, 0.72, 28, "p")
#     debug_add_sample(metrics, Staining.PM, 4, 0.91, 0.75, 29, "q")
#     debug_add_sample(metrics, Staining.PM, 4, 0.92, 0.80, 32, "r")
#
#     # Expot metrics for each experiment
#     if not os.path.isdir("records"):
#         os.mkdir("records")
#     # Build report
#     metrics.build_report("records")
#
#
# def debug_add_sample(metrics, st, rs, acc, pre, ep, rf):
#     metrics.register_sample(staining=st, resize_ratio=rs)
#     sample_results = {"accuracy": acc,
#                       "precision": pre,
#                       "num_epochs": ep,
#                       "report_folder": rf}
#     metrics.register_metrics(sample_results)
#
#
# if __name__ == '__main__':
#     debugging()
