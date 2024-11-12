# Imports:
import os, sys, gc
import logging
import comfy.model_management as mm
from comfy.utils import ProgressBar

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import Pad

#from .utils import imgbatch2PIL, PIL2imgbatch
#from .wrappers import pad_image, deband_image_full, deband_batch, new_dimentions
from .wrappers import comfy2images, run_inference, load_images

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

        pbar = ProgressBar(len(img_batch)*3)

        # Get image from ComfyUI
        comfy2images(img_batch,pbar)

        # Run batch inference
        run_inference(pbar)
        pbar.update(len(img_batch))
        
        # Load inferred images
        out = load_images(pbar)

        return (out,)


    #     # Get image from ComfyUI
    #     pil_batch = imgbatch2PIL(img_batch)

    #     batch_process = True
    #     if batch_process:
    #         out = self.run_deband_batch(pil_batch)
    #         return (out,)
    #     else: #serial processing is very slow, loads and unloads the model for each img inference
    #         debanded = []
    #         pbar = ProgressBar(len(pil_batch))
    #         for i,image in enumerate(pil_batch):
    #             debanded.append(self.deband_image(image))
    #             #TODO: add option to clear cache every n images
    #             pbar.update(1)
        
    #     return (PIL2imgbatch(debanded),)
        


    # def deband_image(self,image):
    #     # Pad image
    #     padded_image, original_size = pad_image(image)
    #     debanded_image = deband_image_full(padded_image, original_size)

    #     return debanded_image

    # def run_deband_batch(self,pil_batch):
    #     # pad frames to squares
    #     ww,hh = pil_batch[0].size
    #     nw,nh = new_dimentions(ww,hh)
    #     dim = max([ww,hh])
    #     if ww>hh: # pad height
    #         pad_f = Pad([0,0,256,ww-hh+256], padding_mode='reflect')
    #     else: # pad width
    #         pad_f = Pad([0,0,hh-ww+256,256], padding_mode='reflect')
        
    #     pbar = ProgressBar(len(pil_batch)*3)
        
    #     #enlarging not enough, need multiple
    #     padded_batch = [pad_f(img) for img in pil_batch]
        
        

    #     out_pil_batch = deband_batch(padded_batch, dim) #,original_sizes,new_dims)
    #     pbar.update(len(pil_batch))

    #     cropped = [img.crop([0,0,ww,hh]) for img in out_pil_batch]

    #     return PIL2imgbatch(cropped,pbar)

        

    # #needs to be adapted to my class
    # def clearCache(self):
    #     mm.soft_empty_cache()
    #     if self.transformer:
    #         self.transformer.cpu()
    #         del self.transformer
    #     if self.tokenizer:
    #         del self.tokenizer
    #     torch.cuda.empty_cache()
    #     torch.cuda.synchronize()
    #     torch._C._cuda_clearCublasWorkspaces()
    #     gc.collect()
    #     self.tokenizer = None
    #     self.transformer = None
    #     self.model_name = None
    #     self.precision = None
    #     self.quantization = None