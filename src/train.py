"""
Intel Image Classification - Main Training Script
Usage:
    python train.py --model pytorch --data_dir ./data
    python train.py --model tensorflow --data_dir ./data
"""

import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Intel Image Classification Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["pytorch", "tensorflow"],
        help="Framework to use for training: 'pytorch' or 'tensorflow'"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to the Intel Image Classification dataset root directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=150,
        help="Image size (height and width) after resizing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes (PyTorch only)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save trained models and plots"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("   Intel Image Classification Pipeline")
    print("=" * 60)
    print(f"  Framework  : {args.model.upper()}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Image size : {args.img_size}x{args.img_size}")
    print(f"  Output dir : {args.output_dir}")
    print("=" * 60)

    if args.model == "pytorch":
        from model.pytorch_model import run_pytorch_pipeline
        run_pytorch_pipeline(args)
    elif args.model == "tensorflow":
        from model.tensorflow_model import run_tensorflow_pipeline
        run_tensorflow_pipeline(args)


if __name__ == "__main__":
    main()
