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



# async update

import asyncio

# Function to run the command and process its output
async def run_async_callback(cmd, callback):
    """Run an async command of the type:

        cmd = ['ping', '-c', '5', 'google.com']

    And for each output line run a callback of type:

        # Callback function to handle each line of output
        def callback(line):
            print(f"Callback handling line: {line.strip()}")

    Usage:

    asyncio.run(run_async_callback(cmd, callback))
            
    """


    # Run the command asynchronously
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Read output line-by-line
    async for line in process.stdout:
        # Decode the byte output to string and strip any extra newlines
        decoded_line = line.decode('utf-8').strip()
        # Call the callback function for each line
        callback(decoded_line)

    # Wait for the process to finish and get the return code
    await process.wait()

    # Print done when the process is complete
    print("done")