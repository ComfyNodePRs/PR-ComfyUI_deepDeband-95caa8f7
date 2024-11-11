import gc
import torch
import numpy as np
from PIL import Image


def imgbatch2PIL(batch):
    """return RGB Pillow Image object from ComfyUI image node argument"""
    # note: image from VHS Load Video has shape [frames, h, w, bgr] (bgr? or rgb?)
    return [Image.fromarray(np.clip(frame.cpu().numpy(),0, 255).astype(np.uint8)) for frame in batch]

def PIL2imgbatch(batch):
    import pdb; pdb.set_trace()
    return torch.tensor([torch.clamp(torch.from_numpy(np.array(img)), min=0, max=1.0) for img in batch])