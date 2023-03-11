# wmh_seg

### Cloning repository and trained model
```
cd $HOME
git clone https://github.com/jinghangli98/wmh_seg.git
cd wmh_seg/trained_model
wget https://huggingface.co/jil202/wmh/resolve/main/multi_site_2d_transformer_Unet_mit_b5_0.81.pth
```

### Creating conda environment
```
cd $HOME/wmh_seg
conda env create -f wmh.yml -n wmh
```

### Add to path
```
export wmh_seg_home=$HOME/wmh_seg
export PATH="$wmh_seg_home:$PATH"
```
You can certainly add these two lines of code in your .zshrc or .bashrc files.
