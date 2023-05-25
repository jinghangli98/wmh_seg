#!/usr/bin/env python

import torch
import nibabel as nib
import numpy as np
import torchvision
from tqdm import tqdm
from einops import rearrange
import sys
import pdb
import torchio as tio
import matplotlib.pyplot as plt

in_path=sys.argv[1]
out_path=sys.argv[2]
gpu=sys.argv[3]
wmh_seg_home=sys.argv[4]
verbose=sys.argv[5]
pmb=sys.argv[6]

def reduceSize(prediction):
    arg = prediction > 0.1
    out = np.zeros(prediction.shape)
    out[arg] = 1
    
    return out
    
def wmh_seg(in_path, out_path, train_transforms, device, mode):


    if mode == "True":
        img_orig = nib.load(in_path)
        transform = tio.transforms.Resize((256,256,256))
        img = transform(img_orig)
        input = np.squeeze(img.get_fdata())
        prediction = np.zeros((256,256,256))
        input = torch.tensor(input)
        affine = img.affine
        input = torch.unsqueeze(input, 1)
        
    else:
        img_orig = nib.load(in_path)
        img = train_transforms(img_orig)
        input = np.squeeze(img.get_fdata())
        prediction_axial = np.zeros((256,256,256))
        prediction_cor = np.zeros((256,256,256))
        prediction_sag = np.zeros((256,256,256))
        input = torch.tensor(input)
        affine = img.affine
        input = torch.unsqueeze(input, 1)
        
    input = input.to(device)
    prediction_input = input/torch.max(input)
    print(f'Predicting.....')
    
    if verbose == "True":
        for idx in tqdm(range(input.shape[0])):
            axial_img = rearrange(prediction_input[:,:,:,idx], 'd0 d1 d2 -> d1 d0 d2').repeat(3,1,1)
            cor_img = rearrange(prediction_input[:,:,idx,:], 'd0 d1 d2 -> d1 d0 d2').repeat(3,1,1)
            sag_img = prediction_input[idx,:,:,:].repeat(3,1,1)
            prediction_axial[:, :, idx] = model(torch.unsqueeze(axial_img, 0).float()).squeeze().detach().cpu().numpy()
            prediction_cor[:, idx, :] = model(torch.unsqueeze(cor_img, 0).float()).squeeze().detach().cpu().numpy()
            prediction_sag[idx, :, :] = model(torch.unsqueeze(sag_img, 0).float()).squeeze().detach().cpu().numpy()
        prediction = prediction_axial + prediction_cor + prediction_sag
    elif verbose != "True":
        for idx in range(input.shape[0]):
            input_image = prediction_input[idx].repeat(3,1,1)
            prediction[:, :, idx] = model(torch.unsqueeze(input_image, 0).float()).squeeze().detach().cpu().numpy()

    #saving images
    out = reduceSize(prediction)
    if gpu == "True":
        img_fit = input.squeeze().cpu().numpy()
    else:
        img_fit = input.squeeze().detach().numpy()
    
    if mode == "True":
        out = rearrange(out, 'd0 d1 d2 -> d1 d2 d0')
        out = rearrange(out, 'd0 d1 d2 -> d1 d2 d0')
        transform = tio.transforms.Resize((img_orig.get_fdata().shape[0], img_orig.get_fdata().shape[1], img_orig.get_fdata().shape[2]))
        out = transform(np.expand_dims(out,0))
        out = np.squeeze(out)
        
    else:
        pdb.set_trace()
        transform = tio.transforms.Resize((img_orig.get_fdata().shape[0], img_orig.get_fdata().shape[1], img_orig.get_fdata().shape[2]))
        out = transform(np.expand_dims(out, 0))
        out = np.squeeze(out)
    
    nii_seg = nib.Nifti1Image(out, affine=img_orig.affine)
    nib.save(nii_seg, out_path)

filename=in_path.split('/')[-1]
ID=filename.split('.')[0]

if gpu == 'True':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Configuring model on GPU')
else:
    device = torch.device('cpu')
    
    print('Configuring model on CPU')

if pmb == "False":
    train_transforms = transform = tio.transforms.Resize((256,256,256))
    
    model = torch.load(f"{wmh_seg_home}/ChallengeMatched_Unet_mit_b5.pth", map_location=device)
    model.eval()
    model.to(device)
    wmh_seg(in_path, out_path, train_transforms, device, pmb)
    
elif pmb == "True":
    train_transforms = torchvision.transforms.Compose([ 
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize((256, 256,)), 
                        ])
    model = torch.load(f"{wmh_seg_home}/pmb_2d_transformer_Unet_mit_b5.pth", map_location=device)
    model.eval()
    model.to(device)
    wmh_seg(in_path, out_path, train_transforms, device, pmb)


