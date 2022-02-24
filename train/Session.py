from typing import List
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.enums import Staining
from src.classes.Metrics import Metrics
import src.constants as const
import src.parameters as params
from workflow import WorkFlow


class Session:
    sessions_dir = os.path.join(os.path.dirname(__file__), "reports")
    if not os.path.isdir(sessions_dir):
        os.mkdir(sessions_dir)

    def __init__(self, staining_list: List[Staining], rratio_list: List[int]):
        self.staining_list = staining_list
        self.rratio_list = rratio_list

        # Path initialization
        self.sess_name = "session_" + time.strftime("%d-%m-%Y")
        self.sess_folder = self._init_session_folder(os.path.join(self.sessions_dir, self.sess_name))

        self.metrics = Metrics()

    def run(self):
        for st in self.staining_list:
            for rs in self.rratio_list:
                self.metrics.add_sample(st, rs)
                workflow = WorkFlow(staining=st, resize_ratio=rs)
                workflow.launch()
                sample_metrics = workflow.results
                self.metrics.register_metrics(sample_metrics)
        self.build_report()

    def build_report(self):
        records_dir = os.path.join(self.sessions_dir, "records")
        if not os.path.isdir(records_dir):
            os.mkdir(records_dir)

        self.metrics.export_metrics(records_dir)

        # TODO write the last results.txt file about the best experiment in session

    # Static methods
    @staticmethod
    def _init_session_folder(session_folder):
        tmp = session_folder
        idx = 0
        while os.path.isdir(tmp):
            idx += 1
            tmp = session_folder + f"_{idx}"
        os.mkdir(tmp)
        return tmp


def debugger():
    stainings = [Staining.HE, Staining.PAS]
    rratios = [3, 3, 4]
    session = Session(stainings, rratios)


if __name__ == '__main__':
    debugger()
