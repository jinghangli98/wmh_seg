import os
import pkg_resources
import torch

_model_cache = {}

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def _get_cache_dir():
    cache = os.path.join(os.path.expanduser('~'), '.cache', 'wmh_seg')
    os.makedirs(cache, exist_ok=True)
    return cache

def _download_hf_model(filename, repo_id='jil202/wmh_seg'):
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except ImportError:
        pass

    import requests
    url = f'https://huggingface.co/{repo_id}/resolve/main/{filename}'
    dest = os.path.join(_get_cache_dir(), filename)
    if not os.path.exists(dest):
        print(f'Downloading {filename} from HuggingFace...')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f'\r  {downloaded / total * 100:.1f}%', end='', flush=True)
        print()
    return dest

def load_model(mode='wmh'):
    if mode in _model_cache:
        return _model_cache[mode]

    device = get_device()

    if mode == 'wmh':
        model_path = pkg_resources.resource_filename('wmh_seg', 'ChallengeMatched_Unet_mit_b5.pth')
    elif mode == 'pmb':
        model_path = _download_hf_model('pmb_2d_transformer_Unet_mit_b5.pth')
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'wmh' or 'pmb'.")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    _model_cache[mode] = model
    return model
