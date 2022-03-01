import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Session import Session
import src.parameters as params


def main():
    session = Session(staining_list=params.STAININGS, rratio_list=params.RESIZE_RATIOS, send_report=params.SEND_EMAIL)
    session.run()


if __name__ == '__main__':
    main()
