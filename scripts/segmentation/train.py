"""
Author: Fran Moreno
Contact: fran.moreno.se@gmail.com
Date: 15/03/2022
"""
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import src.utils.parameters as params
from src.session import Session


def main():
    session = Session(staining_list=params.STAININGS, resize_ratio_list=params.RESIZE_RATIOS,
                      send_report=params.SEND_EMAIL)
    session.run()


if __name__ == '__main__':
    main()
