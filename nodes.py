# Imports:
import os, sys
import logging
#import comfy.model_management as mm

# add deepDeband scripts to path
from deepDeband.src.padding import pad_image
import numpy as np
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import clearCache

# Logging configuration:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# ComnfyUI: Node definitions
class deepDebandInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "version": (["full", "weighted"], {"default": "weighted", "tooltip": "Choose the debanding model version. Please refer to the original paper"}),
            "image": ("IMAGE", {"tooltip": "Provide an image to use as input for inferencing. Only supported for glm-4v-9b, glm-4v-9b-gptq-4bit and glm-4v-9b-gptq-3bit models."}),
        },
        "optional": {
            "unload_model": ("BOOLEAN", {"default": True, "tooltip": "Unload the model after use to free up memory"}),
        }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("debanded_image",)
    FUNCTION = "infer"
    CATEGORY = "debanding"

    def infer(self, image=None, unload_model=True):
        # Empty cache
        mm.soft_empty_cache()

        debanded_image = None

        return (debanded_image,)


# ComfyUI: Node class mappings
NODE_CLASS_MAPPINGS = {
  "deepDeband Inference": deepDebandInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "deepDebandInference": "deepDeband Inference",
}