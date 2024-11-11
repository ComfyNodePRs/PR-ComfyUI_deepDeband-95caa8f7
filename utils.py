import gc
import torch
from PIL import Image


def imgarg2PIL(image):
    """return RGB Pillow Image object from ComfyUI image node argument"""
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)).convert("RGB")   


def PIL2imgarg(img):
    return torch.clamp(torch.from_numpy(np.array(img)), min=0, max=1.0)