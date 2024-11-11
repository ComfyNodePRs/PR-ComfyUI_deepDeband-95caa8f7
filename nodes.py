# Imports:
import os, sys, gc
import logging
import comfy.model_management as mm

import numpy as np
from PIL import Image

import torch

from .utils import imgbatch2PIL, PIL2imgbatch
from .wrappers import pad_image, deband_image_full

# Logging configuration:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# ComnfyUI: Node definitions
class deepDebandInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "img_batch": ("IMAGE", {"tooltip": "Provide an image to be debanded"}),
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
              #version, 
              unload_model=True,
              ):

        # Empty cache
        mm.soft_empty_cache()

        # Get image from ComfyUI
        pil_batch = imgbatch2PIL(img_batch)

        debanded = []
        for i,image in enumerate(pil_batch):
            debanded.append(self.deband_image(image))
            #TODO: add option to clear cache every n images
        
        return (PIL2imgbatch(debanded),)
        


    def deband_image(self,image):
        # Pad image
        padded_image, original_size = pad_image(image)

        if True: #version == "full":
            debanded_image = deband_image_full(padded_image, original_size)
        elif version == "weighted":
            pass
            # implement equivalent of:
            # deband_weighted.deband_images(image_sizes, gpu_ids)

        return debanded_image

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