# Imports:
import os
import logging
import comfy.model_management as mm
from comfy.utils import ProgressBar

import numpy as np
from PIL import Image

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