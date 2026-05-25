"""Reorient images to canonical RAS for inference, then map masks back.

nnU-Net (and the legacy transformer models) operate on the voxel array in
whatever axis order the input file happens to be stored in -- they don't apply
the affine's direction cosines to canonicalize orientation. Test-time mirroring
covers axis *flips* but not axis *swaps*, so a brain stored LAS vs. RAS can give
different predictions. To make results independent of how an input was saved, we
reorient every input to closest-canonical RAS before inference, then transform
the predicted mask back to the input's original orientation so it overlays on
the source scan.
"""
import numpy as np
import nibabel as nib


def to_ras(img):
    """Return the closest-canonical (RAS+) version of a nibabel image."""
    return nib.as_closest_canonical(img)


def restore_orientation(mask_data, ras_img, orig_img):
    """Map a mask array from RAS orientation back to the original image.

    Parameters
    ----------
    mask_data : array-like
        Mask in the orientation of ``ras_img`` (canonical RAS).
    ras_img : nibabel image
        The RAS image that was fed to inference (its affine defines the
        orientation the mask currently lives in).
    orig_img : nibabel image
        The original, un-reoriented input image.

    Returns
    -------
    nib.Nifti1Image
        uint8 mask carrying ``orig_img``'s affine, so it overlays on the source.
    """
    ras_ornt = nib.io_orientation(ras_img.affine)
    orig_ornt = nib.io_orientation(orig_img.affine)
    transform = nib.orientations.ornt_transform(ras_ornt, orig_ornt)
    restored = nib.orientations.apply_orientation(np.asanyarray(mask_data), transform)
    out = nib.Nifti1Image(restored.astype(np.uint8), orig_img.affine)
    out.set_data_dtype(np.uint8)
    return out
