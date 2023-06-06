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

in_path=sys.argv[1]
out_path=sys.argv[2]
gpu=sys.argv[3]
wmh_seg_home=sys.argv[4]
verbose=sys.argv[5]
pmb=sys.argv[6]
batch=np.int16(sys.argv[7])
fast=sys.argv[8]
def reduceSize(prediction):
    arg = prediction > 0.5
    out = np.zeros(prediction.shape)
    out[arg] = 1
    
    return out
    
def wmh_seg(in_path, out_path, train_transforms, device, mode):


    if mode == "True":
        img_orig = nib.load(in_path)
        transform = tio.transforms.Resize((256,256,256))
        img = transform(img_orig)
        input = np.squeeze(img.get_fdata())
        input = torch.tensor(input)
        affine = img.affine
        input = torch.unsqueeze(input, 1)
        prediction_axial = np.zeros((256,256,256))
        prediction_cor = np.zeros((256,256,256))
        prediction_sag = np.zeros((256,256,256))
        
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
        if fast == "True":
            for idx in tqdm(range(input.shape[0]//batch)):
                axial_img = rearrange(prediction_input[:,:,:,idx*batch:(idx+1)*batch], 'd0 d1 d2 d3 -> d3 d1 d0 d2').repeat(1,3,1,1)
                prediction_axial[:,:,idx*batch:(idx+1)*batch] = rearrange(model(axial_img.float())[0:batch].detach().cpu().numpy(), 'd0 d1 d2 d3 -> d2 d3 (d0 d1)')
            prediction = prediction_axial
        else:
            for idx in tqdm(range(input.shape[0]//batch)):
                axial_img = rearrange(prediction_input[:,:,:,idx*batch:(idx+1)*batch], 'd0 d1 d2 d3 -> d3 d1 d0 d2').repeat(1,3,1,1)
                cor_img = rearrange(prediction_input[:,:,idx*batch:(idx+1)*batch,:], 'd0 d1 d2 d3 -> d2 d1 d0 d3').repeat(1,3,1,1)
                sag_img = rearrange(prediction_input[idx*batch:(idx+1)*batch,:,:,:], 'd0 d1 d2 d3 -> d0 d1 d2 d3').repeat(1,3,1,1)
                stacked_input = torch.vstack((axial_img, cor_img, sag_img))
                prediction_axial[:,:,idx*batch:(idx+1)*batch] = rearrange(model(stacked_input.float())[0:batch].detach().cpu().numpy(), 'd0 d1 d2 d3 -> d2 d3 (d0 d1)')
                prediction_cor[:,idx*batch:(idx+1)*batch,:] = rearrange(model(stacked_input.float())[batch:2*batch].detach().cpu().numpy(), 'd0 d1 d2 d3 -> d2 (d0 d1) d3')
                prediction_sag[idx*batch:(idx+1)*batch,:,:] = rearrange(model(stacked_input.float())[2*batch::].detach().cpu().numpy(), 'd0 d1 d2 d3 -> (d0 d1) d2 d3')

            prediction = prediction_axial + prediction_cor + prediction_sag
    elif verbose != "True":
        if fast == "True":
            for idx in range(input.shape[0]//batch):
                axial_img = rearrange(prediction_input[:,:,:,idx*batch:(idx+1)*batch], 'd0 d1 d2 d3 -> d3 d1 d0 d2').repeat(1,3,1,1)
                prediction_axial[:,:,idx*batch:(idx+1)*batch] = rearrange(model(axial_img.float())[0:batch].detach().cpu().numpy(), 'd0 d1 d2 d3 -> d2 d3 (d0 d1)')
            prediction = prediction_axial

        else:
            for idx in range(input.shape[0]//batch):
                axial_img = rearrange(prediction_input[:,:,:,idx*batch:(idx+1)*batch], 'd0 d1 d2 d3 -> d3 d1 d0 d2').repeat(1,3,1,1)
                cor_img = rearrange(prediction_input[:,:,idx*batch:(idx+1)*batch,:], 'd0 d1 d2 d3 -> d2 d1 d0 d3').repeat(1,3,1,1)
                sag_img = rearrange(prediction_input[idx*batch:(idx+1)*batch,:,:,:], 'd0 d1 d2 d3 -> d0 d1 d2 d3').repeat(1,3,1,1)
                stacked_input = torch.vstack((axial_img, cor_img, sag_img))
                prediction_axial[:,:,idx*batch:(idx+1)*batch] = rearrange(model(stacked_input.float())[0:batch].detach().cpu().numpy(), 'd0 d1 d2 d3 -> d2 d3 (d0 d1)')
                prediction_cor[:,idx*batch:(idx+1)*batch,:] = rearrange(model(stacked_input.float())[batch:2*batch].detach().cpu().numpy(), 'd0 d1 d2 d3 -> d2 (d0 d1) d3')
                prediction_sag[idx*batch:(idx+1)*batch,:,:] = rearrange(model(stacked_input.float())[2*batch::].detach().cpu().numpy(), 'd0 d1 d2 d3 -> (d0 d1) d2 d3')

            prediction = prediction_axial + prediction_cor + prediction_sag

    #saving images
    out = reduceSize(prediction)
    if gpu == "True":
        img_fit = input.squeeze().cpu().numpy()
    else:
        img_fit = input.squeeze().detach().numpy()
    
    transform = tio.transforms.Resize((img_orig.get_fdata().shape[0], img_orig.get_fdata().shape[1], img_orig.get_fdata().shape[2]))
    out = transform(np.expand_dims(out, 0))
    out = reduceSize(np.squeeze(out))
    
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
    train_transforms = tio.transforms.Resize((256,256,256))
    
    model = torch.load(f"{wmh_seg_home}/ChallengeMatched_Unet_mit_b5.pth", map_location=device)
    model.eval()
    model.to(device)
    wmh_seg(in_path, out_path, train_transforms, device, pmb)
    
elif pmb == "True":
    # train_transforms = torchvision.transforms.Compose([ 
    #                     torchvision.transforms.ToTensor(),
    #                     torchvision.transforms.Resize((256, 256,)), 
    #                     ])
    train_transforms = tio.transforms.Resize((256,256,256))
    model = torch.load(f"{wmh_seg_home}/pmb_2d_transformer_Unet_mit_b5.pth", map_location=device)
    model.eval()
    model.to(device)
    wmh_seg(in_path, out_path, train_transforms, device, pmb)
    


