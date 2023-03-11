# wmh_seg
An automatic white matter lesion segmentaion tool on T2 weighted Fluid Attenuated Inverse Recovery (FLAIR) images. The model was trained using more than 300 FLAIR scans at 1.5T, 3T and 7T, including images from University of Pittsburgh, University of Nottingham, UT Health San Antonio, UMC Utrecht, NUHS Singapore, and VU Amsterdam. Additionaly, data augmentation was implemented using [torchio](https://torchio.readthedocs.io/transforms/transforms.html). ```wmh_seg``` shows reliable results that are on par with freesurfer white matter lesion segmentations on T1 weighted images. No additional preprocessing is needed. 

<p align="center">
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/dataAugmentation.png" width=50% height=50%>
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/comparision.png">
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/comparision2.png">
</p>

## Installation

### Cloning repository and trained model
```bash
cd $HOME
git clone https://github.com/jinghangli98/wmh_seg.git
cd wmh_seg
wget https://huggingface.co/jil202/wmh/resolve/main/multi_site_2d_transformer_Unet_mit_b5_0.81.pth
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

## Example usage
```bash
wmh_seg -i PITT_001.nii.gz -o PITT_001_wmh.nii.gz -g
```
```-i``` is the input image path

```-o``` is the output image path

```-g``` (optional) specifies whether the model would perform on nividia gpu

```bash
ls *.nii | parallel --jobs 6 wmg_seg -i {} -o {.}_wmh.nii.gz -g
```
This line of bash command would process all the .nii files on gpu, 6 files at a time in the current directory.
