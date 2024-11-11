# ComfyUI_deepDeband
ComyUI wrapper for RaymondLZhou/deepDeband image and video debanding


<hr>
WARNING: thit is an experimental development repo, you have to expect bugs, not to be used in a production environment.
<hr>

## Install
Please run `pip install -r requirements.txt` in the environment where ComfyUI will be running, at the moment some depencies are not automatically installed.

Currently the deepDeband repository is over its data quota. Account responsible for LFS bandwidth should purchase more data packs to restore access. Check it with `git lfs pull`.  If this happens please download the model checkpoints manually as stated in the original [README](https://github.com/RaymondLZhou/deepDeband/blob/master/README.md#model) from this file archive [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7523437.svg)](https://doi.org/10.5281/zenodo.7523437)

After downloading and extracting the zip file, place the checkpoint found in ` deepDeband-f ` into 

```
ComfyUI/custom_nodes/ComfyUP_deepDeband/deepDeband/pytorch-CycleGAN-and-pix2pix/checkpoints/deepDeband-f 
```




## Notes
When the repo is downloaded the model are automatically downloaded via GIT LFS, this should require ~~ 300 MB of storage.

The default implementation of deepDeband uses the following functions:
```
cleanup.cleanup() # delete precedent ./temp/*
cleanup.setup(...) # create empty ./temp/*
padding.pad_images(...) # load images from ./input, pad, save to ./temp
deband.deband_images(...) # load images from ./temp, deband, save to ./output
cleanup.cleanup() # delete precedent ./temp/*
```
this method pipes all images through disk read/write which is good for low ram envs but impacts inference time negatively.

the deband process is calling a bash script, so at the moment such stage still requires to write the image (at least this happens only once per image now insted of 3 times)

Our patch transforms this behavior into a RAM-based pipeline.


# Acknowledgements

All credits for the ComfyUI platform, model development, model framework, go to:
- [deepDeband](https://github.com/RaymondLZhou/deepDeband)
- [CylceGan and Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [ComfyUI](https://github.com/comfyui)
- [pytorch](https://github.com/pytorch/pytorch)

all respective licence terms are located in the relative subfolders.