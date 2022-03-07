from os.path import dirname, abspath
import sys

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from src.session import Session
import src.utils.parameters as params


def main():
    session = Session(staining_list=params.STAININGS, resize_ratio_list=params.RESIZE_RATIOS,
                      send_report=params.SEND_EMAIL)
    session.run()


if __name__ == '__main__':
    main()
