import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SegmenterPipeline
from src.utils.utils import browse_file

def main():
    file_path = browse_file()
    print(file_path)


if __name__ == '__main__':
    main()