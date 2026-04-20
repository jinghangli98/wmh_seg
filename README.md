# wmh_seg
An automatic white matter lesion segmentation tool for T2-weighted Fluid Attenuated Inverse Recovery (FLAIR) images. The model was trained using more than 300 FLAIR scans at 1.5T, 3T and 7T, including images from University of Pittsburgh, UMC Utrecht, NUHS Singapore, and VU Amsterdam. Data augmentation was implemented using [torchio](https://torchio.readthedocs.io/transforms/transforms.html). `wmh_seg` shows reliable results on par with FreeSurfer white matter lesion segmentations on T1-weighted images. No additional preprocessing is needed.

<p align="center">
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/dataAugmentation.png" width=80% height=80%>
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/comparision2.png" width=80% height=80%>
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/comparision.png">
</p>

## Installation
```bash
pip install wmh_seg
```
GPU (CUDA/MPS) is used automatically if available, otherwise falls back to CPU.

---

## CLI Usage

### Basic WMH segmentation
```bash
wmh_seg -i FLAIR.nii.gz -o FLAIR_wmh.nii.gz
```

### Post-mortem brain segmentation
```bash
wmh_seg -i T1.nii.gz -o T1_wmh.nii.gz -p -v -b 4 --fast
```

### All options
| Flag | Description |
|------|-------------|
| `-i`, `--input` | Input FLAIR NIfTI path (`.nii` / `.nii.gz`) |
| `-o`, `--output` | Output segmentation NIfTI path |
| `-p`, `--pmb` | Use post-mortem brain model |
| `-v`, `--verbose` | Show progress bar |
| `-b`, `--batch N` | Batch size for inference (default: 4) |
| `--fast` | Axial-only inference (faster, slightly less accurate) |

### Batch processing with GNU parallel
```bash
ls *.nii.gz | parallel --jobs 4 wmh_seg -i {} -o {.}_wmh.nii.gz -v
```

---

## Python API

```python
import nibabel as nib
from wmh_seg import wmh_seg

img = nib.load('FLAIR.nii.gz')

# Standard WMH segmentation
seg = wmh_seg(img)

# Post-mortem brain segmentation
seg = wmh_seg(img, mode='pmb', verbose=True, fast=True, batch=4)

# Save result
nib.save(nib.Nifti1Image(seg, affine=img.affine), 'output_wmh.nii.gz')
```

### `wmh_seg()` parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img_orig` | `Nifti1Image` or `np.ndarray` | — | Input FLAIR image |
| `mode` | `'wmh'` \| `'pmb'` | `'wmh'` | Model to use |
| `verbose` | `bool` | `True` | Show progress bar |
| `fast` | `bool` | `True` | Axial-only inference |
| `batch` | `int` | `4` | Inference batch size |

---

### Docker usage example
docker build -t wmh_seg .
docker run -v $PWD:/data -w /data wmh_seg wmh_seg -i FLAIR.nii -o FLAIR_wmh.nii


## Citation
If you find this useful for your research, please cite:
```bibtex
@article{li2024wmh_seg,
  title={wmh\_seg: Transformer based U-Net for Robust and Automatic White Matter Hyperintensity Segmentation across 1.5 T, 3T and 7T},
  author={Li, Jinghang and Santini, Tales and Huang, Yuanzhe and Mettenburg, Joseph M and Ibrahim, Tamer S and Aizenstein, Howard J and Wu, Minjie},
  journal={arXiv preprint arXiv:2402.12701},
  year={2024}
}
```
