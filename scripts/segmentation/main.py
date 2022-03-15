"""
Author: Fran Moreno
Contact: fran.moreno.se@gmail.com
Date: 15/03/2022
"""
import argparse
import glob
import os
import sys
from os.path import dirname, abspath
from pathlib import Path

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import src.utils.constants as const
from src.pipeline import SegmentationPipeline
from src.utils.misc import browse_file, browse_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", "-s", type=int, default=16)
    parser.add_argument("--threshold", "-th", type=float, default=0.7)
    parser.add_argument("--slide", type=str)
    return parser.parse_args()


def run_one_slide():
    args = parse_args()
    if not args.slide:
        print("Select slide to process")
        slide_file = browse_file()
    else:
        slide_file = args.slide

    segmentationPipeline = SegmentationPipeline()
    segmentationPipeline.run(slide_file, th=args.threshold)
    prediction = segmentationPipeline.get_scaled_prediction(scale=args.scale)
    prediction_name = os.path.basename(slide_file).split('.')[0] + f"_mask_s{args.scale}.png"
    output_path = Path(slide_file).parent.joinpath(prediction_name)
    prediction.save(output_path)


def multiple_slide(src_folder: str, scale: int):
    th = 0.75
    slides = glob.glob(src_folder + '/*.tif')

    dest_folder = os.path.join(const.PIPELINE_RESULTS_PATH, os.path.basename(src_folder), f"scale{str(scale)}")
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    segmentationPipeline = SegmentationPipeline()
    for slide in slides:
        slide_basename = os.path.basename(slide)
        prediction_name = os.path.join(dest_folder, slide_basename.split('.')[0] + f"_mask_s{scale}.png")

        if os.path.isfile(prediction_name):
            # Mask already exists. Continue with next WSI.
            continue

        segmentationPipeline.run(slide, th)
        prediction = segmentationPipeline.get_scaled_prediction(scale=scale)
        prediction.save(prediction_name)


def run_multiple_slide():
    print("Source folder: ")
    src_path = browse_path()
    print(src_path)
    multiple_slide(src_path, 16)


if __name__ == '__main__':
    # run_one_slide()
    run_multiple_slide()
