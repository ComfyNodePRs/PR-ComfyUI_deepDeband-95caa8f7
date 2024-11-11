# ComfyUI_deepDeband
ComyUI wrapper for RaymondLZhou/deepDeband image and video debanding


<hr>
WARNING: thit is an experimental development repo, you have to expect bugs, not to be used in a production environment.
<hr>

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