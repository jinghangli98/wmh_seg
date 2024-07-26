# wmh_seg/cli.py
import argparse
import os
import subprocess
from .wmh_pytorch import wmh_seg

def parse_args():
    parser = argparse.ArgumentParser(description='WMH Segmentation CLI')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-b', '--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--fast', action='store_true', help='Fast mode')
    return parser.parse_args()

def main():
    args = parse_args()
    verbose = 'True' if args.verbose else 'False'
    fast_mode = 'True' if args.fast else 'False'

    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Verbose mode: {verbose}")
    print(f"Inference batch size: {args.batch}")
    print(f"Fast mode: {fast_mode}")

    wmh_seg(args.input, args.output, verbose=verbose, fast=fast_mode, batch=args.batch)

if __name__ == "__main__":
    main()
