"""nnU-Net (v2) inference backend for the post-mortem brain (pmb) WMH model.

This wraps the trained nnUNetv2 model (Dataset011_WMH, 3d_fullres, 5-fold
ensemble) behind a simple file-in/file-out call so it can be driven from the
``wmh_seg`` CLI without touching nnU-Net's own command line tools.

The 5 fold checkpoints are distributed as flat files on Hugging Face
(``pmb_nnunet_checkpoint_final_<fold>.pth`` in the jil202/wmh_seg repo) and are
downloaded + cached on first use. nnU-Net's loader expects a model *folder*
laid out as ``<dir>/{dataset.json,plans.json,fold_<i>/checkpoint_final.pth}``,
so we reconstruct that layout in a local cache, pairing the downloaded
checkpoints with the dataset.json/plans.json bundled in this package.

Set ``WMH_SEG_NNUNET_DIR`` to an existing nnU-Net model folder to bypass the
download entirely (useful for development / offline use / custom weights).
"""
import os
import shutil

import pkg_resources

# Hugging Face repo + flat checkpoint naming used at upload time.
HF_REPO_ID = 'jil202/wmh_seg'
CHECKPOINT_TEMPLATE = 'pmb_nnunet_checkpoint_final_{fold}.pth'
ALL_FOLDS = (0, 1, 2, 3, 4)
CHECKPOINT_NAME = 'checkpoint_final.pth'  # name nnU-Net expects inside each fold dir

# Cache the initialized predictor per fold-selection so repeated calls in one
# process don't reload five networks from disk.
_predictor_cache = {}


def _import_nnunet():
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    except ImportError as e:
        raise ImportError(
            'The --nnunet backend requires nnunetv2. Install it with:\n'
            '    pip install "wmh_seg[nnunet]"\n'
            'or\n'
            '    pip install nnunetv2'
        ) from e
    return nnUNetPredictor, SimpleITKIO


def _cache_dir():
    d = os.path.join(os.path.expanduser('~'), '.cache', 'wmh_seg', 'nnunet_pmb_model')
    os.makedirs(d, exist_ok=True)
    return d


def _bundled_asset(name):
    """Absolute path to a config file bundled in the package."""
    return pkg_resources.resource_filename('wmh_seg', os.path.join('nnunet_assets', name))


def _link_or_copy(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy(src, dst)


def _build_model_folder(folds):
    """Download checkpoints from HF and assemble the folder nnU-Net expects."""
    from huggingface_hub import hf_hub_download

    model_dir = _cache_dir()
    # Configs travel with the package so they always match this model version.
    for cfg in ('dataset.json', 'plans.json'):
        shutil.copy(_bundled_asset(cfg), os.path.join(model_dir, cfg))

    print(f'Fetching nnU-Net pmb checkpoints from {HF_REPO_ID} (cached after first run)...')
    for f in folds:
        ckpt = hf_hub_download(repo_id=HF_REPO_ID,
                               filename=CHECKPOINT_TEMPLATE.format(fold=f))
        fold_dir = os.path.join(model_dir, f'fold_{f}')
        os.makedirs(fold_dir, exist_ok=True)
        _link_or_copy(ckpt, os.path.join(fold_dir, CHECKPOINT_NAME))
    return model_dir


def _resolve_model_folder(folds):
    override = os.environ.get('WMH_SEG_NNUNET_DIR')
    if override:
        if not os.path.isdir(override):
            raise FileNotFoundError(f'WMH_SEG_NNUNET_DIR={override!r} is not a directory.')
        return override
    return _build_model_folder(folds)


def _get_predictor(folds=ALL_FOLDS, compile=False):
    folds = tuple(folds)
    cache_key = (folds, compile)
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
        use_mirroring=True,
        perform_everything_on_device=(device.type == 'cuda'),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        _resolve_model_folder(folds),
        use_folds=folds,
        checkpoint_name=CHECKPOINT_NAME,
    )
    _predictor_cache[cache_key] = predictor
    return predictor


def nnunet_segment(input_path, output_path, folds=ALL_FOLDS, verbose=False, compile=False):
    """Segment WMH from a single NIfTI using the nnU-Net pmb model.

    nnU-Net reads the image with SimpleITK (handling spacing/orientation),
    runs preprocessing + sliding-window inference over the 5-fold ensemble,
    resamples the prediction back to the input geometry, and writes a uint8
    label map (0=bg, 1=WMH) to ``output_path``.
    """
    _, SimpleITKIO = _import_nnunet()
    predictor = _get_predictor(folds, compile=compile)

    io = SimpleITKIO()
    img, props = io.read_images([input_path])
    if verbose:
        print('Predicting.....')
    seg = predictor.predict_single_npy_array(img, props, None, None, False)
    io.write_seg(seg, output_path, props)
    return output_path
