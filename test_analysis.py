""" File to analyze the test set prediction results"""
from utils import browse_path
from interface import Viewer


def launch_interface(dir_path: str):
    viewer = Viewer(output_folder=dir_path, masks_folder='circles100')
    viewer.pack(fill="both", expand=True)
    viewer.mainloop()


if __name__ == '__main__':
    output_path = browse_path()
    launch_interface(output_path)


