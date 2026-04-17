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
    parser.add_argument('-i', '--input', required=True, help='Path to input FLAIR NIfTI (.nii/.nii.gz)')
    parser.add_argument('-o', '--output', required=True, help='Path for output segmentation NIfTI')
    parser.add_argument('-p', '--pmb', action='store_true', help='Use post-mortem brain model')
    parser.add_argument('--fast', action='store_true', help='Axial-only inference (faster, slightly less accurate)')
    parser.add_argument('-b', '--batch', type=int, default=4, metavar='N', help='Batch size for inference (default: 4)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show progress bar')

    args = parser.parse_args()

    img = nib.load(args.input)
    seg = wmh_seg(
        img,
        verbose=args.verbose,
        fast=args.fast,
        batch=args.batch,
        mode='pmb' if args.pmb else 'wmh',
    )

    nib.save(nib.Nifti1Image(seg.astype(np.uint8), affine=img.affine), args.output)
    print(f'Saved segmentation to {args.output}')


if __name__ == '__main__':
    main()
