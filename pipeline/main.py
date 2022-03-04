import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SegmentationPipeline
from src.utils.utils import browse_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", "-s", type=int, default=16)
    parser.add_argument("--threshold", "-th", type=float, default=0.7)
    parser.add_argument("--slide", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.slide:
        print("Select slide to process")
        slide_file = browse_file()
    else:
        slide_file = args.slide

    segmentationPipeline = SegmentationPipeline()
    segmentationPipeline.run(slide_file, th=args.threshold)
    prediction = segmentationPipeline.get_scaled_prediction(reduction_factor=args.scale)
    prediction_name = os.path.basename(slide_file).split('.')[0] + f"_mask_s{args.scale}.png"
    output_path = Path(slide_file).parent.joinpath(prediction_name)
    prediction.save(output_path)


if __name__ == '__main__':
    main()
