""" File to analyze the test set prediction results"""
from src.utils.interface import Viewer
from src.utils.utils import browse_path

# TODO merge pred_analysis and post_process scripts. Compute predictions while loading interface


def launch_interface(from_dir, th):
    output_path = browse_path()
    viewer = Viewer(output_folder=output_path, from_dir=from_dir, th=th)
    viewer.pack(fill="both", expand=True)
    viewer.mainloop()


if __name__ == '__main__':
    from_dir = False
    th = 0.75
    launch_interface(from_dir, th)


