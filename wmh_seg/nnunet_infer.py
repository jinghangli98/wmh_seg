"""nnU-Net (v2) inference backend for the wmh_seg models.

Several nnU-Net models share this backend:

* ``invivo`` (default) -- Dataset002_WMHinvivo, dedicated to in-vivo scans
  (challenge FLAIR/T1 + COV-IBR/IBR-SVD FLAIR).
* ``exvivo`` -- Dataset003_WMHexvivo, dedicated to ex-vivo scans (T1/T2/GRE).
* ``unified`` -- Dataset001_WMH, the contrast-agnostic generalist trained on
  both in-vivo and ex-vivo data.
* ``pmb`` -- legacy ex-vivo nnU-Net (Dataset011_WMH); superseded by ``exvivo``
  and not exposed on the CLI, but still reachable from the Python API.

Each is single-channel, so you feed exactly one image of whatever contrast you
have. All are 3d_fullres 5-fold ensembles. nnU-Net's loader expects a model
*folder* laid out as ``<dir>/{dataset.json,plans.json,fold_<i>/checkpoint_final.pth}``.
The fold checkpoints are distributed as flat files on Hugging Face
(``checkpoint_final_{fold}_{model}.pth`` in the jil202/wmh_seg repo) and are
downloaded + cached on first use; we pair them with the dataset.json/plans.json
bundled in this package (which always match the uploaded weights) to rebuild
that folder under ``~/.cache/wmh_seg``.

Set a model's env var (see ``MODELS`` below) to an existing nnU-Net model folder
to bypass the download -- useful for development, offline use, or custom weights.
"""
import os
import shutil
import tempfile
from importlib import resources

import nibabel as nib
import torch

HF_REPO_ID = 'jil202/wmh_seg'
CHECKPOINT_NAME = 'checkpoint_final.pth'  # name nnU-Net expects inside each fold dir
DEFAULT_MODEL = 'invivo'

# Registry of nnU-Net models. Each entry knows where its bundled configs live,
# how its fold checkpoints are named on Hugging Face, and which env var lets a
# user point at a pre-existing local model folder instead of downloading.
MODELS = {
    'invivo': {
        'assets_subdir': 'invivo',
        'checkpoint_template': 'checkpoint_final_{fold}_invivo.pth',
        'env_var': 'WMH_SEG_INVIVO_DIR',
        'folds': (0, 1, 2, 3, 4),
    },
    'exvivo': {
        'assets_subdir': 'exvivo',
        'checkpoint_template': 'checkpoint_final_{fold}_exvivo.pth',
        'env_var': 'WMH_SEG_EXVIVO_DIR',
        'folds': (0, 1, 2, 3, 4),
    },
    'unified': {
        'assets_subdir': 'unified',
        'checkpoint_template': 'checkpoint_final_{fold}_unified.pth',
        'env_var': 'WMH_SEG_UNIFIED_DIR',
        'folds': (0, 1, 2, 3, 4),
    },
    # Legacy ex-vivo nnU-Net (Dataset011). Superseded by the dedicated 'exvivo'
    # model and no longer exposed on the CLI, but kept for the Python API.
    'pmb': {
        'assets_subdir': 'pmb',
        'checkpoint_template': 'pmb_nnunet_checkpoint_final_{fold}.pth',
        'env_var': 'WMH_SEG_NNUNET_DIR',
        'folds': (0, 1, 2, 3, 4),
    },
}

# Cache the initialized predictor per (model, fold-selection, compile) so
# repeated calls in one process don't reload the networks from disk.
_predictor_cache = {}


def _import_nnunet():
    # nnunetv2 prints three warnings at import when its training/preprocessing
    # folders are not configured. Inference needs none of them, so point them
    # at a scratch folder unless the user has set them.
    scratch = os.path.join(os.path.expanduser('~'), '.cache', 'wmh_seg', 'nnunet_paths')
    for var in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        if not os.environ.get(var):
            os.makedirs(scratch, exist_ok=True)
            os.environ[var] = scratch
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    except ImportError as e:
        raise ImportError(
            'The nnU-Net backend requires nnunetv2. Install it with:\n'
            '    pip install nnunetv2'
        ) from e
    return nnUNetPredictor, SimpleITKIO


def _cache_dir(model):
    d = os.path.join(os.path.expanduser('~'), '.cache', 'wmh_seg', f'nnunet_{model}_model')
    os.makedirs(d, exist_ok=True)
    return d


def _bundled_asset(model, name):
    """The config file bundled in the package for ``model`` (an importlib.resources handle)."""
    return resources.files('wmh_seg').joinpath('nnunet_assets', MODELS[model]['assets_subdir'], name)


def _link_or_copy(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy(src, dst)


def _build_model_folder(model, folds):
    """Download checkpoints from HF and assemble the folder nnU-Net expects."""
    from huggingface_hub import hf_hub_download

    spec = MODELS[model]
    model_dir = _cache_dir(model)
    # Configs travel with the package so they always match this model version.
    for cfg in ('dataset.json', 'plans.json'):
        with open(os.path.join(model_dir, cfg), 'wb') as fh:
            fh.write(_bundled_asset(model, cfg).read_bytes())

    print(f'Fetching nnU-Net {model} checkpoints from {HF_REPO_ID} (cached after first run)...')
    for f in folds:
        ckpt = hf_hub_download(repo_id=HF_REPO_ID,
                               filename=spec['checkpoint_template'].format(fold=f))
        fold_dir = os.path.join(model_dir, f'fold_{f}')
        os.makedirs(fold_dir, exist_ok=True)
        _link_or_copy(ckpt, os.path.join(fold_dir, CHECKPOINT_NAME))
    return model_dir


def _resolve_model_folder(model, folds):
    override = os.environ.get(MODELS[model]['env_var'])
    if override:
        if not os.path.isdir(override):
            raise FileNotFoundError(
                f"{MODELS[model]['env_var']}={override!r} is not a directory.")
        return override
    return _build_model_folder(model, folds)


def _get_predictor(model=DEFAULT_MODEL, folds=None, compile=False, mirroring=True):
    if folds is None:
        folds = MODELS[model]['folds']
    folds = tuple(folds)
    cache_key = (model, folds, compile, mirroring)
    if cache_key in _predictor_cache:
        return _predictor_cache[cache_key]

    nnUNetPredictor, _ = _import_nnunet()
    from .model_loader import get_device

    if compile:
        # nnU-Net compiles self.network internally when this env var is set,
        # and its predict loop already handles the resulting OptimizedModule.
        os.environ['nnUNet_compile'] = 'true'

    device = get_device()
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=mirroring,
        perform_everything_on_device=(device.type == 'cuda'),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        _resolve_model_folder(model, folds),
        use_folds=folds,
        checkpoint_name=CHECKPOINT_NAME,
    )
    if device.type == 'cuda':
        # nnU-Net runs the network under fp16 autocast. On recent torch/cuDNN/GPU
        # combinations an fp16 3D convolution in the default NCDHW layout takes a
        # fallback kernel that unfolds its input: one full-resolution layer of
        # this U-Net then needs several GB and ~20x the time of the cuDNN kernel.
        # Channels-last weights steer every convolution onto the proper kernel
        # (measured: 1.3 s -> 0.2 s and 9.5 GB -> 1.4 GB per patch, same output).
        # nnU-Net's per-fold load_state_dict copies into these tensors, so the
        # layout survives fold switching.
        predictor.network.to(memory_format=torch.channels_last_3d)
    _predictor_cache[cache_key] = predictor
    return predictor


def nnunet_segment(input_path, output_path, model=DEFAULT_MODEL, folds=None,
                   verbose=False, compile=False, reorient=True, mirroring=True):
    """Segment WMH from a single NIfTI using an nnU-Net model.

    nnU-Net reads the image with SimpleITK, runs preprocessing + sliding-window
    inference over the fold ensemble, resamples the prediction back to the input
    geometry, and writes a uint8 label map (0=bg, 1=WMH) to ``output_path``.

    Parameters
    ----------
    input_path, output_path : str
        Single-image input NIfTI and destination for the label map.
    model : {'invivo', 'exvivo', 'unified', 'pmb'}
        Which nnU-Net model to run. ``'invivo'`` (default) and ``'exvivo'`` are
        the dedicated models, ``'unified'`` is the contrast-agnostic generalist,
        and ``'pmb'`` is the legacy ex-vivo model.
    folds : iterable[int] or None
        Fold subset for the ensemble. ``None`` uses all of the model's folds.
    reorient : bool
        Reorient the input to canonical RAS before inference and map the mask
        back to the input's original orientation afterwards (default True).
    mirroring : bool
        Test-time augmentation: average the prediction over the 8 axis flips of
        every patch (default True). Turning it off makes inference about 8x
        faster at a small cost in accuracy.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown nnU-Net model {model!r}; choose from {sorted(MODELS)}.")
    _, SimpleITKIO = _import_nnunet()
    predictor = _get_predictor(model, folds, compile=compile, mirroring=mirroring)
    io = SimpleITKIO()

    def _predict(path, dst):
        img, props = io.read_images([path])
        if verbose:
            print('Predicting.....')
        seg = predictor.predict_single_npy_array(img, props, None, None, False)
        io.write_seg(seg, dst, props)

    if not reorient:
        _predict(input_path, output_path)
        return output_path

    # Reorient to canonical RAS, infer, then map the mask back to the original
    # orientation so it overlays on the source scan. See orientation.py.
    from .orientation import to_ras, restore_orientation
    orig_img = nib.load(input_path)
    ras_img = to_ras(orig_img)
    with tempfile.TemporaryDirectory(prefix='wmh_seg_') as tmp:
        ras_in = os.path.join(tmp, 'input_0000.nii.gz')
        nib.save(ras_img, ras_in)
        ras_seg = os.path.join(tmp, 'seg.nii.gz')
        _predict(ras_in, ras_seg)
        mask = nib.load(ras_seg)
        restore_orientation(mask.dataobj, ras_img, orig_img).to_filename(output_path)
    return output_path
