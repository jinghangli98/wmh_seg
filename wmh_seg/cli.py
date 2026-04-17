import argparse
import sys
import nibabel as nib
import numpy as np
from .wmh_pytorch import wmh_seg


def main():
    parser = argparse.ArgumentParser(
        prog='wmh_seg',
        description='White matter hyperintensity segmentation for FLAIR images',
    )
    parser.add_argument('input', help='Path to input FLAIR NIfTI (.nii/.nii.gz)')
    parser.add_argument('output', help='Path for output segmentation NIfTI')
    parser.add_argument(
        '--mode', choices=['wmh', 'pmb'], default='wmh',
        help='Segmentation model: wmh (default) or pmb (post-mortem brain)',
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Axial-only inference (faster, slightly less accurate)',
    )
    parser.add_argument(
        '--batch', type=int, default=4, metavar='N',
        help='Batch size for inference (default: 4)',
    )
    parser.add_argument(
        '--no-progress', action='store_true',
        help='Suppress progress bar',
    )

    args = parser.parse_args()

    img = nib.load(args.input)
    seg = wmh_seg(
        img,
        verbose=not args.no_progress,
        fast=args.fast,
        batch=args.batch,
        mode=args.mode,
    )

    nib.save(nib.Nifti1Image(seg.astype(np.uint8), affine=img.affine), args.output)
    print(f'Saved segmentation to {args.output}')


if __name__ == '__main__':
    main()
