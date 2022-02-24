from typing import Any
import sys
import os
from os.path import dirname
import csv
import statistics as stats

sys.path.append(dirname(dirname(dirname(os.path.abspath(__file__)))))
from src.utils.enums import Staining


class Metrics:
    def __init__(self):
        self.experiments = dict()
        self.tracked_metrics = set()
        self.curr_exp = dict()

    def add_sample(self, staining: Staining, resize_ratio: int):
        key = f"{staining}_{resize_ratio}"
        if key not in self.experiments.keys():
            self.experiments[key] = dict()
        self.curr_exp = self.experiments[key]

    def register_metrics(self, sample_metrics: dict):
        for k in sample_metrics.keys():
            self.add_metric(key=k, val=sample_metrics[k])

    def add_metric(self, key: str, val: Any):
        if key not in self.curr_exp.keys():
            self.curr_exp[key] = []
            if key not in self.tracked_metrics:
                self.tracked_metrics.add(key)
        self.curr_exp[key].append(val)

    def export_metrics(self, records_dir: str):
        for label in self.experiments.keys():
            experiment_dir = os.path.join(records_dir, label)
            if not os.path.isdir(experiment_dir):
                os.mkdir(experiment_dir)
            self._records_to_csv(os.path.join(experiment_dir, "records.csv"), label)
            self._get_measures(os.path.join(experiment_dir, "results.csv"), label)

    def _records_to_csv(self, filename, label):
        keys = sorted(self.tracked_metrics)
        with open(filename, "w") as f:
            writer = csv.writer(f, delimiter = ",")
            writer.writerow(keys)
            writer.writerows(zip(*[self.experiments[label][key] for key in keys]))

    def _get_measures(self, filename, label):
        data = self.experiments[label]
        results = dict()
        for metric in self.tracked_metrics:
            results[metric] = {'mean': stats.mean(data[metric]),
                               'stdev': stats.mean(data[metric]),
                               'max': max(data[metric])}
        with open(filename, "w") as f:
            writer = csv.writer(f, delimiter = ',')
            header = ["metric", "mean", "stdev", "max"]
            writer.writerow(header)
            for metric in results.keys():
                line = [metric, results[metric]['mean'], results[metric]['stdev'], results[metric]['max']]
                writer.writerow(line)


def debugging():
    metrics = Metrics()

    # First sample
    metrics.add_sample(staining=Staining.HE, resize_ratio=3)

    sample_results = {"accuracy": 0.1,
                      "precision": 0.2,
                      "num_epochs": 10}
    # Add some metrics and values
    metrics.register_metrics(sample_results)

    # Second sample
    metrics.add_sample(staining=Staining.PAS, resize_ratio=2)

    sample_results = {"accuracy": 0.1,
                      "precision": 0.2,
                      "num_epochs": 20}
    # Add some metrics and values
    metrics.register_metrics(sample_results)

    # Third sample (same conditions as the first one)
    metrics.add_sample(staining=Staining.HE, resize_ratio=3)

    sample_results = {"accuracy": 0.1,
                      "precision": 0.2,
                      "num_epochs": 30}
    # Add some metrics and values
    metrics.register_metrics(sample_results)

    if not os.path.isdir("my_dir"):
        os.mkdir("my_dir")
    metrics.export_metrics("my_dir")


if __name__ == '__main__':
    debugging()