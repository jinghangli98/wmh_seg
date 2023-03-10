#!/usr/bin/env python

import torch
import nibabel as nib
import numpy as np
import pdb
import torchvision
from tqdm import tqdm
from einops import rearrange
import sys

train_transforms = torchvision.transforms.Compose([ 
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.CenterCrop((224, 256,)),
                    ])

in_path=sys.argv[1]
out_path=sys.argv[2]
gpu=sys.argv[3]
wmh_seg_home=sys.argv[4]
verbose=sys.argv[5]

if gpu == 'True':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f"{wmh_seg_home}/multi_site_2d_transformer_Unet_mit_b5_0.81.pth", map_location=torch.device('cuda'))
    model.eval()
    model.to(device)
    print('Configuring model on GPU')
else:
    device = torch.device('cpu')
    model = torch.load(f"{wmh_seg_home}/multi_site_2d_transformer_Unet_mit_b5_0.81.pth", map_location=torch.device('cpu'))
    model.eval()
    model.to(device)
    print('Configuring model on CPU')

def wmh_seg(in_path, out_path, train_transforms, device):
    img = nib.load(in_path)
    input = np.squeeze(img.get_fdata())

    origin_size = img.get_fdata().shape
    affine = img.affine
    prediction = np.zeros((224,256,input.shape[-1]))

    input = train_transforms(input)
    input = input.to(device)
    input = torch.unsqueeze(input, 1)
    prediction_input = input/torch.max(input)
    print(f'Predicting.....')
    if verbose == "True":
        for idx in tqdm(range(input.shape[0])):
            input_image = prediction_input[idx].repeat(3,1,1)
            prediction[:, :, idx] = model(torch.unsqueeze(input_image, 0).float()).squeeze().detach().cpu().numpy()
            
    elif verbose != "True":
        for idx in range(input.shape[0]):
            input_image = prediction_input[idx].repeat(3,1,1)
            prediction[:, :, idx] = model(torch.unsqueeze(input_image, 0).float()).squeeze().detach().cpu().numpy()

    #saving images
    arg = prediction > 0.999
    out = np.zeros(prediction.shape)
    out[arg] = 1
    if gpu == "True":
        img_fit = input.squeeze().cpu().numpy()
    else:
        img_fit = input.squeeze().detach().numpy()
    img_fit = rearrange(img_fit, 'd0 d1 d2 -> d1 d2 d0')
    train_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.CenterCrop((img.get_fdata().shape[0], img.get_fdata().shape[1],)),])
    img_fit = train_transforms(img_fit).detach().cpu().numpy()
    out = train_transforms(out).detach().cpu().numpy()

    img_fit = rearrange(img_fit, 'd0 d1 d2 -> d1 d2 d0')
    out = rearrange(out, 'd0 d1 d2 -> d1 d2 d0')
    
    nii_seg = nib.Nifti1Image(out, affine=affine)

    nib.save(nii_seg, out_path)


filename=in_path.split('/')[-1]
ID=filename.split('.')[0]
wmh_seg(in_path, out_path, train_transforms, device)

