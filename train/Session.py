from typing import List
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.enums import Staining
from src.classes.Metrics import Metrics
from src.utils.utils import EmailHandler
from workflow import WorkFlow


class Session:
    sessions_dir = os.path.join(os.path.dirname(__file__), "reports")
    if not os.path.isdir(sessions_dir):
        os.mkdir(sessions_dir)

    def __init__(self, staining_list: List[Staining], rratio_list: List[int], send_report: bool):
        self.staining_list = staining_list
        self.rratio_list = rratio_list
        self.send_report = send_report

        # Path initialization
        self.sess_name = "session_" + time.strftime("%d-%m-%Y")
        self.sess_folder = self._init_session_folder(os.path.join(self.sessions_dir, self.sess_name))

        self.metrics = Metrics()
        if self.send_report:
            self.emailInfo = EmailHandler()

    def run(self):
        for st in self.staining_list:
            for rs in self.rratio_list:
                workflow = WorkFlow(staining=st, resize_ratio=rs, session_folder = self.sess_folder)
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
    session = Session(stainings, rratios, False)


if __name__ == '__main__':
    debugger()
