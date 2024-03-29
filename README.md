# wmh_seg
An automatic white matter lesion segmentaion tool on T2 weighted Fluid Attenuated Inverse Recovery (FLAIR) images. The model was trained using more than 300 FLAIR scans at 1.5T, 3T and 7T, including images from University of Pittsburgh, University of Nottingham, UT Health San Antonio, UMC Utrecht, NUHS Singapore, and VU Amsterdam. Additionaly, data augmentation was implemented using [torchio](https://torchio.readthedocs.io/transforms/transforms.html). ```wmh_seg``` shows reliable results that are on par with freesurfer white matter lesion segmentations on T1 weighted images. No additional preprocessing is needed. 

<p align="center">
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/dataAugmentation.png" width=80% height=80%>
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/comparision2.png" width=80% height=80%>
<img src="https://github.com/jinghangli98/wmh_seg/blob/main/comparision.png">
</p>

## Installation

### Cloning repository and trained model
```bash
cd $HOME
git clone https://github.com/jinghangli98/wmh_seg.git
cd wmh_seg
wget https://huggingface.co/jil202/wmh/resolve/main/multi_site_2d_transformer_Unet_mit_b5_0.81.pth
wget https://huggingface.co/jil202/wmh/resolve/main/pmb_2d_transformer_Unet_mit_b5.pth

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
