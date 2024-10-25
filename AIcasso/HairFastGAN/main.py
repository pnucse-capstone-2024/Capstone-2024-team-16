import argparse
import os
from pathlib import Path

import sys

from torchvision.utils import save_image
from tqdm.auto import tqdm

from hair_swap import HairFast, get_parser

def process_image(face_path: Path, shape_path: Path, color_path: Path, output_path: Path, model_args, benchmark=False):

    print(f"Processing image with face_path: {face_path}, shape_path: {shape_path}, color_path: {color_path}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        hair_fast = HairFast(model_args)
        result = hair_fast.swap(face_path, shape_path, color_path, benchmark=benchmark, align=True)

        if isinstance(result, tuple):
            final_image = result[0]
        else:
            final_image = result

        save_image(final_image, output_path)
        return output_path

    except Exception as e:
        print(f"Error during image processing: {e}")
        raise


if __name__ == "__main__":

    print("sys.path:", sys.path)

    model_parser = get_parser()
    parser = argparse.ArgumentParser(description='HairFast evaluate')

    parser.add_argument('--input_dir', type=Path, default='', help='The directory of the images to be inverted')
    parser.add_argument('--benchmark', action='store_true', help='Calculates the speed of the method during the session')

    # Arguments for a set of experiments
    parser.add_argument('--file_path', type=Path, default=None,
                        help='File with experiments with the format "face_path.png shape_path.png color_path.png"')
    parser.add_argument('--output_dir', type=Path, default=Path('output'), help='The directory for final results')

    # Arguments for single experiment
    parser.add_argument('--face_path', type=Path, default=None, help='Path to the face image')
    parser.add_argument('--shape_path', type=Path, default=None, help='Path to the shape image')
    parser.add_argument('--color_path', type=Path, default=None, help='Path to the color image')
    parser.add_argument('--result_path', type=Path, default=None, help='Path to save the result')

    args, unknown1 = parser.parse_known_args()
    model_args, unknown2 = model_parser.parse_known_args()

    if all(path is not None for path in (args.face_path, args.shape_path, args.color_path)):
        output_path = args.result_path or args.output_dir / 'user_result.png'
        process_image(args.face_path, args.shape_path, args.color_path, output_path, model_args, args.benchmark)
