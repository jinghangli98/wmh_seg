import argparse
import nibabel as nib
import numpy as np
from .wmh_pytorch import wmh_seg


def main():
    parser = argparse.ArgumentParser(
        prog='wmh_seg',
        description='White matter hyperintensity / lesion segmentation. Pick a model '
                    'with --invivo (default), --exvivo, or --unified; each takes a '
                    'single image of any one contrast.',
    )
    parser.add_argument('-i', '--input', required=True, help='Path to input NIfTI (.nii/.nii.gz)')
    parser.add_argument('-o', '--output', required=True, help='Path for output segmentation NIfTI')

    # Model selection (mutually exclusive; default is in-vivo).
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--invivo', action='store_true',
                             help='Use the dedicated in-vivo nnU-Net model (default)')
    model_group.add_argument('--exvivo', action='store_true',
                             help='Use the dedicated ex-vivo nnU-Net model')
    model_group.add_argument('--unified', action='store_true',
                             help='Use the unified contrast-agnostic nnU-Net model (in-vivo + ex-vivo)')

    parser.add_argument('--legacy', action='store_true',
                        help='Use the legacy in-vivo transformer model instead of the in-vivo '
                             'nnU-Net model (only valid for the in-vivo model).')
    parser.add_argument('--fast', action='store_true',
                        help='[legacy] Axial-only inference for the transformer model '
                             '(faster, slightly less accurate)')
    parser.add_argument('-b', '--batch', type=int, default=4, metavar='N',
                        help='[legacy] Batch size for the transformer model (default: 4)')

    parser.add_argument('-f', '--folds', type=int, nargs='+', metavar='N', default=None,
                        help='nnU-Net fold(s) to ensemble (default: all 5). Fewer folds = '
                             'faster, slightly less accurate.')
    parser.add_argument('--compile', action='store_true',
                        help='torch.compile the model before inference '
                             '(slower first call, faster afterwards)')
    parser.add_argument('--no-reorient', dest='reorient', action='store_false',
                        help='Do not reorient the input to RAS before inference. '
                             'By default the input is reoriented to canonical RAS and '
                             'the mask is mapped back to the original orientation.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show progress bar')

    args = parser.parse_args()

    # Resolve the model domain (mutual exclusivity is enforced by argparse).
    if args.exvivo:
        domain = 'exvivo'
    elif args.unified:
        domain = 'unified'
    else:
        domain = 'invivo'  # default, and when --invivo is given explicitly

    # --legacy only exists for the in-vivo model.
    if args.legacy and domain != 'invivo':
        parser.error(f'--legacy is only available for the in-vivo model; '
                     f'it cannot be combined with --{domain}.')

    # --- Legacy in-vivo transformer model ------------------------------------
    if args.legacy:
        orig_img = nib.load(args.input)
        src_img = orig_img
        if args.reorient:
            from .orientation import to_ras
            src_img = to_ras(orig_img)

        seg = wmh_seg(
            src_img,
            verbose=args.verbose,
            fast=args.fast,
            batch=args.batch,
            mode='wmh',
            compile=args.compile,
        )

        if args.reorient:
            from .orientation import restore_orientation
            restore_orientation(seg, src_img, orig_img).to_filename(args.output)
        else:
            nib.save(nib.Nifti1Image(seg.astype(np.uint8), affine=orig_img.affine), args.output)
        print(f'Saved segmentation to {args.output}')
        return

    # --- nnU-Net models: invivo (default) / exvivo / unified -----------------
    from .nnunet_infer import nnunet_segment
    nnunet_segment(args.input, args.output, model=domain, folds=args.folds,
                   verbose=args.verbose, compile=args.compile, reorient=args.reorient)
    print(f'Saved segmentation to {args.output}')


if __name__ == '__main__':
    main()
