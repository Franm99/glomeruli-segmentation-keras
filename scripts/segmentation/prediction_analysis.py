"""
Author: Fran Moreno
Contact: fran.moreno.se@gmail.com
Date: 15/03/2022
"""
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from src.interface import PredictionViewer
from src.utils.misc import browse_path


def launch_interface(from_dir, th):
    output_path = browse_path()
    viewer = PredictionViewer(output_folder=output_path, from_dir=from_dir, th=th)
    viewer.pack(fill="both", expand=True)
    viewer.mainloop()


if __name__ == '__main__':
    from_dir = False
    th = 0.75
    launch_interface(from_dir, th)


