import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from einops import rearrange
import torchio as tio
import argparse
import pdb
from .model_loader import model

def reduceSize(prediction):
    arg = prediction > 0.5
    out = np.zeros(prediction.shape)
    out[arg] = 1
    return out

def wmh_seg(in_path, out_path, verbose=1, fast=1, batch=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transforms = tio.transforms.Resize((256,256,256))
    img_orig = nib.load(in_path)
    img = train_transforms(img_orig)
    input = np.squeeze(img.get_fdata())
    prediction_axial = np.zeros((256,256,256))
    prediction_cor = np.zeros((256,256,256))
    prediction_sag = np.zeros((256,256,256))
    input = torch.tensor(input)
    input = torch.unsqueeze(input, 1).to(device)
        
    prediction_input = input/torch.max(input)
    print(f'Predicting.....')

    if verbose:
        if fast:
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
    elif not verbose:
        if fast:
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
   
    transform = tio.transforms.Resize((img_orig.get_fdata().shape[0], img_orig.get_fdata().shape[1], img_orig.get_fdata().shape[2]))
    out = transform(np.expand_dims(out, 0))
    out = reduceSize(np.squeeze(out))
    
    nii_seg = nib.Nifti1Image(out, affine=img_orig.affine)
    nib.save(nii_seg, out_path)

def main():
    
    parser = argparse.ArgumentParser(description='WMH Segmentation')
    parser.add_argument('i', type=str, help='Input path for the FLAIR image')
    parser.add_argument('o', type=str, help='Output path for the lesion mask')
    parser.add_argument('v', type=int, default=1, choices=[1, 0], help='Verbose output')
    parser.add_argument('b', type=int, help='Batch size for processing')
    parser.add_argument('fast', type=int, default=1, choices=[1, 0], help='Fast processing mode')
    args = parser.parse_args()
    wmh_seg(args.in_path, args.out_path, args.verbose, args.fast, args.batch, model)
    
if __name__ == "__main__":
    main()
