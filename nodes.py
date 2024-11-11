# Imports:
import os, sys
import logging
#import comfy.model_management as mm

# add ComfyUI_deepDeband to path
#sys.path.insert(0,'custom_nodes/ComfyUI_deepDeband/')

import numpy as np
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .utils import imgarg2PIL, PIL2imgarg
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
            "image": ("IMAGE", {"tooltip": "Provide an image to be debanded"}),
            #"version": (["full", "weighted"], {"default": "weighted", "tooltip": "Choose the debanding model version. Please refer to the original paper"}),
        },
        "optional": {
            "unload_model": ("BOOLEAN", {"default": True, "tooltip": "Unload the model after use to free up memory"}),
        }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("debanded_image",)
    FUNCTION = "infer"
    CATEGORY = "debanding"

    def infer(self, 
              image, 
              #version, 
              unload_model=True,
              ):

        # Empty cache
        mm.soft_empty_cache()

        # Get image from ComfyUI
        image = imgarg2PIL(image)

        # Pad image
        padded_image, original_size = pad_image(img_path, self.image_sizes)

        if True: #version == "full":
            debanded_image = deband_image_full(img, original_size)
        elif version == "weighted":
            pass
            # implement equivalent of:
            # deband_weighted.deband_images(image_sizes, gpu_ids)

        return (PIL2imgarg(debanded_image),)


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