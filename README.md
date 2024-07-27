# wmh_seg
An automatic white matter lesion segmentaion tool on T2 weighted Fluid Attenuated Inverse Recovery (FLAIR) images. The model was trained using more than 300 FLAIR scans at 1.5T, 3T and 7T, including images from University of Pittsburgh, UMC Utrecht, NUHS Singapore, and VU Amsterdam. Additionaly, data augmentation was implemented using [torchio](https://torchio.readthedocs.io/transforms/transforms.html). ```wmh_seg``` shows reliable results that are on par with freesurfer white matter lesion segmentations on T1 weighted images. No additional preprocessing is needed. 

<p align="center">
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/dataAugmentation.png" width=80% height=80%>
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/comparision2.png" width=80% height=80%>
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/comparision.png">
</p>

## PyPI Installation
```
pip install wmh_seg
```
### Python Example usage
```python
from wmh_seg import wmh_seg
import nibael as nib
nii = nib.load('/Users/jinghangli/Developer/wmh_seg/FLAIR.nii').get_fdata()
wmh = wmh_seg(nii)
slice = nii.get_fdata()[:,:,50]
wmh_slice = wmh_seg(slice)
```
## CLI Installation

### Cloning repository and trained model
```bash
cd $HOME
git clone https://github.com/jinghangli98/wmh_seg.git
cd wmh_seg
wget https://huggingface.co/jil202/wmh_seg/resolve/main/ChallengeMatched_Unet_mit_b5.pth

```

### Creating conda environment
```bash
cd $HOME/wmh_seg
conda env create -f wmh.yml -n wmh
```

### Add to path
```bash
export wmh_seg_home=$HOME/wmh_seg
export PATH="$wmh_seg_home:$PATH"
```
You can certainly add these two lines of code in your ~/.zshrc or ~/.bashrc files.

### CLI Example usage
```bash
conda activate wmh
wmh_seg -i PITT_001.nii.gz -o PITT_001_wmh.nii.gz -g
```
```-i``` is the input image path

```-o``` is the output image path

```-g``` (optional) specifies whether the model would be configured on nividia gpu

```-v``` (optional) monitor prediction progress

```-p``` (optional) enable segmentation on T1-weighted post mortem brain (left hemisphere)

```bash
ls *.nii | parallel --jobs 6 wmh_seg -i {} -o {.}_wmh.nii.gz -g
```
This line of bash command would process all the .nii files on gpu in the current directory, 6 files at a time. (You might need to install GNU parallel)

## Citation
If you find this useful for your research, please use this bibtex to cite this repository:
```
@article{li2024wmh_seg,
  title={wmh\_seg: Transformer based U-Net for Robust and Automatic White Matter Hyperintensity Segmentation across 1.5 T, 3T and 7T},
  author={Li, Jinghang and Santini, Tales and Huang, Yuanzhe and Mettenburg, Joseph M and Ibrahim, Tamer S and Aizenstein, Howard J and Wu, Minjie},
  journal={arXiv preprint arXiv:2402.12701},
  year={2024}
}
