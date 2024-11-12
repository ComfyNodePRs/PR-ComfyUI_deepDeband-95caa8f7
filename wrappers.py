import os
from glob import glob
from PIL import Image

from .utils import imgbatch2PIL, PIL2imgbatch, run_async_callback

import asyncio

root_path=os.getcwd()+"/custom_nodes/ComfyUI_deepDeband/"

tempdir_in="deepDeband/input/"
tempdir_out="deepDeband/output/deepDeband-f/"
modeldir="deepDeband/src/"




# Get image from ComfyUI
def comfy2images(img_batch,pbar):
    os.makedirs(root_path+tempdir_in, exist_ok=True)
    os.makedirs(root_path+tempdir_out, exist_ok=True)

    cleanup()

    for idx,img in enumerate(imgbatch2PIL(img_batch)):
        tmp_path = root_path+tempdir_in+f"{idx:0>8}.png"
        img.save(tmp_path, "PNG") # dump temp image
        pbar.update(1)

# Run batch inference
def run_inference(pbar):
    os.chdir(root_path+modeldir)
    try:
        cmd = ['python','deepDeband_batch.py']
        def callback(line):
            print(line)
            if line.startswith('processing'):
                pbar.update(5)

        asyncio.run(run_async_callback(cmd, callback))

    except:
        print("infer failed!")
    os.chdir(root_path)


# Load inferred images
def load_images(pbar):
    generated = glob(f'{root_path}{tempdir_out}*.png')
    generated.sort() # keep images order
    pil_batch = []
    for img in generated:
        pil_batch.append(Image.open(img))
        pbar.update(1)
    cleanup()
    return PIL2imgbatch(pil_batch)

# Cleanup result
def cleanup():
    in_path = root_path+tempdir_in
    out_path = root_path+tempdir_out
    for name in os.listdir(in_path):
        os.remove(in_path+name)
    for name in os.listdir(out_path):
        os.remove(out_path+name)