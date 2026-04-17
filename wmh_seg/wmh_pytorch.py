import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from einops import rearrange
import torchio as tio
from .model_loader import load_model, get_device

def reduceSize(prediction):
    out = np.zeros(prediction.shape)
    out[prediction > 0.5] = 1
    return out

def seg_3d(img_orig, verbose, fast, batch, model):
    device = get_device()
    train_transforms = tio.transforms.Resize((256, 256, 256))

    img_orig_arr = np.expand_dims(img_orig, 0)
    input_arr = np.squeeze(train_transforms(img_orig_arr))
    input_tensor = torch.tensor(input_arr)
    input_tensor = torch.unsqueeze(input_tensor, 1).to(device)

    prediction_input = (input_tensor / torch.max(input_tensor)).float()

    prediction_axial = np.zeros((256, 256, 256))
    prediction_cor = np.zeros((256, 256, 256))
    prediction_sag = np.zeros((256, 256, 256))

    print('Predicting.....')
    n_batches = input_tensor.shape[0] // batch
    iterator = tqdm(range(n_batches)) if verbose else range(n_batches)

    with torch.no_grad():
        if fast:
            for idx in iterator:
                axial_img = rearrange(
                    prediction_input[:, :, :, idx*batch:(idx+1)*batch],
                    'd0 d1 d2 d3 -> d3 d1 d0 d2'
                ).repeat(1, 3, 1, 1)
                out = model(axial_img)[:batch].detach().cpu().numpy()
                prediction_axial[:, :, idx*batch:(idx+1)*batch] = rearrange(
                    out, 'd0 d1 d2 d3 -> d2 d3 (d0 d1)'
                )
        else:
            for idx in iterator:
                axial_img = rearrange(
                    prediction_input[:, :, :, idx*batch:(idx+1)*batch],
                    'd0 d1 d2 d3 -> d3 d1 d0 d2'
                ).repeat(1, 3, 1, 1)
                cor_img = rearrange(
                    prediction_input[:, :, idx*batch:(idx+1)*batch, :],
                    'd0 d1 d2 d3 -> d2 d1 d0 d3'
                ).repeat(1, 3, 1, 1)
                sag_img = rearrange(
                    prediction_input[idx*batch:(idx+1)*batch, :, :, :],
                    'd0 d1 d2 d3 -> d0 d1 d2 d3'
                ).repeat(1, 3, 1, 1)
                stacked_input = torch.vstack((axial_img, cor_img, sag_img))
                # Single forward pass for all three orientations
                out = model(stacked_input).detach().cpu().numpy()
                prediction_axial[:, :, idx*batch:(idx+1)*batch] = rearrange(
                    out[:batch], 'd0 d1 d2 d3 -> d2 d3 (d0 d1)'
                )
                prediction_cor[:, idx*batch:(idx+1)*batch, :] = rearrange(
                    out[batch:2*batch], 'd0 d1 d2 d3 -> d2 (d0 d1) d3'
                )
                prediction_sag[idx*batch:(idx+1)*batch, :, :] = rearrange(
                    out[2*batch:], 'd0 d1 d2 d3 -> (d0 d1) d2 d3'
                )
            prediction_axial = prediction_axial + prediction_cor + prediction_sag

    out = reduceSize(prediction_axial)
    orig_shape = img_orig.shape
    transform = tio.transforms.Resize((orig_shape[0], orig_shape[1], orig_shape[2]))
    out = transform(np.expand_dims(out, 0))
    return reduceSize(np.squeeze(out))

def wmh_seg(img_orig, verbose=True, fast=True, batch=4, mode='wmh'):
    model = load_model(mode)

    if isinstance(img_orig, nib.Nifti1Image):
        img_orig = np.squeeze(img_orig.get_fdata())
    elif isinstance(img_orig, np.ndarray):
        img_orig = np.squeeze(img_orig)

    if len(img_orig.shape) == 3:
        return seg_3d(img_orig, verbose, fast, batch, model)
    elif len(img_orig.shape) == 2:
        pass
