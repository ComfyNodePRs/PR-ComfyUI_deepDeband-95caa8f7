import gc
import torch
import numpy as np
from PIL import Image


def imgbatch2PIL(batch):
    """return RGB Pillow Image object from ComfyUI image node argument"""
    # note: image from ComfyUI has shape [frames, h, w, bgr] (rgb)
    # Image.fromarray(frame.cpu.permute((2,0,1)).numpy())
    return [Image.fromarray((frame.cpu().numpy()*255).astype(np.uint8)) for frame in batch]

def PIL2imgbatch(pil_batch,progress=None):
    imgbatch = []
    for img in pil_batch:
        imgbatch.append(np.array(img.convert("RGB")).astype(np.float32) / 255.0)
        if progress: progress.update(1)
    return torch.tensor(np.array(imgbatch))