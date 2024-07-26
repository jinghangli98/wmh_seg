import pkg_resources
import torch

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = pkg_resources.resource_filename('wmh_seg', 'ChallengeMatched_Unet_mit_b5.pth')
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    
    return model

model = load_model()