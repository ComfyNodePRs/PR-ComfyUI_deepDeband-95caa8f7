# Imports:
import os, sys, gc
import logging
import comfy.model_management as mm
from comfy.utils import ProgressBar

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import Pad

from .utils import imgbatch2PIL, PIL2imgbatch
from .wrappers import pad_image, deband_image_full, deband_batch

# Logging configuration:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# model paths
model_D = 'custom_nodes/ComfyUI_deepDeband/deepDeband/pytorch-CycleGAN-and-pix2pix/checkpoints/deepDeband-f/latest_net_D.pth'
model_G = 'custom_nodes/ComfyUI_deepDeband/deepDeband/pytorch-CycleGAN-and-pix2pix/checkpoints/deepDeband-f/latest_net_G.pth'
install_model_msg = """ERROR: The model weights were not correctly installed.
Currently the deepDeband repository is over its data quota.
Please install manually from https://doi.org/10.5281/zenodo.7523437 following the README.md instructions"""

# ComnfyUI: Node definitions
class deepDebandInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "img_batch": ("IMAGE", {"tooltip": "Provide an image to be debanded"}),
            "width": ("INT", {"tooltip": "Video frames width"}),
            #"version": (["full", "weighted"], {"default": "weighted", "tooltip": "Choose the debanding model version. Please refer to the original paper"}),
        },
        "optional": {
            "unload_model": ("BOOLEAN", {"default": True, "tooltip": "Unload the model after use to free up memory"}),
        }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("debanded_image",)
    FUNCTION = "infer_batch"
    CATEGORY = "debanding"

    def infer_batch(self, 
              img_batch,
              width, 
              #version, 
              unload_model=True,
              ):
        # Empty cache
        mm.soft_empty_cache()

        # Check model weights are properly installed
        assert os.path.isfile(model_D), install_model_msg
        assert os.path.isfile(model_G), install_model_msg

        # Get image from ComfyUI
        pil_batch = imgbatch2PIL(img_batch)

        batch_process = True
        if batch_process:
            out = self.deband_batch(pil_batch, width)
            return (out,)
        else: #serial processing is very slow, loads and unloads the model for each img inference
            debanded = []
            pbar = ProgressBar(len(pil_batch))
            for i,image in enumerate(pil_batch):
                debanded.append(self.deband_image(image))
                #TODO: add option to clear cache every n images
                pbar.update(1)
        
        return (PIL2imgbatch(debanded),)
        


    def deband_image(self,image):
        # Pad image
        padded_image, original_size = pad_image(image)
        debanded_image = deband_image_full(padded_image, original_size)

        return debanded_image

    def deband_batch(self,pil_batch, width):
        import pdb; pdb.set_trace()

        ww,hh = pil_batch[0].size
        dim = max([ww,hh])
        if ww>hh: # pad height
            pad_f = Pad([0,0,0,ww-hh], padding_mode='reflect')
        else: # pad width
            pad_f = Pad([0,0,hh-ww,0], padding_mode='reflect')
        
        pbar = ProgressBar(len(pil_batch)*3)
        # for i,image in enumerate(pil_batch):
        #     #turn image tensor to array
        #     padded_image, original_size, new_dim = pad_image(image, return_dim=True)
        #     padded_imgs.append(padded_image)
        #     original_sizes.append(original_size)
        #     new_dims.append(new_dim)
        #     pbar.update(1)

        # pil_batch = deband_batch(padded_imgs,original_sizes,new_dims)

        pil_batch = deband_batch(pil_batch, dim) #,original_sizes,new_dims)
        pbar.update(len(pil_batch))

        import pdb; pdb.set_trace()
        
        return PIL2imgbatch(pil_batch,pbar)

        

    #needs to be adapted to my class
    def clearCache(self):
        mm.soft_empty_cache()
        if self.transformer:
            self.transformer.cpu()
            del self.transformer
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        self.tokenizer = None
        self.transformer = None
        self.model_name = None
        self.precision = None
        self.quantization = None