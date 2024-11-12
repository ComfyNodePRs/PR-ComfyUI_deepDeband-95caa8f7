import shutil, os
from glob import glob
from PIL import Image, ImageOps
from .deepDeband.src.padding import new_dimentions

root_path=os.getcwd()+"/custom_nodes/ComfyUI_deepDeband/"

tempdir_in="temp/in/"
tempdir_out="temp/out/"
modeldir="deepDeband/pytorch-CycleGAN-and-pix2pix/"

def get_output_imgpath(fname):
    return f'{root_path}{tempdir_out}/deepDeband-f/test_latest/images/{fname}_fake.png'

def pad_image(img, return_dim=False):
    width, height = img.size
    new_width, new_height = new_dimentions(width, height)

    padded_img = Image.new("RGB", (max(2*width, new_width), max(2*height, new_height)))
    flipped_img = ImageOps.flip(img)
    mirrored_img = ImageOps.mirror(img)
    flipped_mirrored_img = ImageOps.flip(ImageOps.mirror(img))

    for i in range(0, new_width+1, width*2):
        for j in range(0, new_height+1, height*2):
            padded_img.paste(img, (i, j))
            padded_img.paste(flipped_img, (i, j+height))
            padded_img.paste(mirrored_img, (i+width, j))
            padded_img.paste(flipped_mirrored_img, (i+width, j+height))

    padded_img = padded_img.crop((0, 0, new_width, new_height))

    #used by batch processing
    if return_dim:
        dim = max(padded_img.size)
        return padded_img, img.size, dim

    return padded_img, img.size #return padded image and original size


def deband_image_full(img, original_size, gpu_ids=0):
    os.makedirs(root_path+tempdir_in, exist_ok=True)
    os.makedirs(root_path+tempdir_out, exist_ok=True)
    dim = max(img.size) # get padded image size
    img.save(root_path+tempdir_in+"img.png", "PNG") # dump temp image

    # run inference
    os.chdir(root_path+modeldir)
    command = f'python test.py \
        --name deepDeband-f \
        --model test \
        --netG unet_256 \
        --norm batch \
        --dataroot ../../{tempdir_in} \
        --results_dir ../../{tempdir_out} \
        --dataset_mode single \
        --gpu_ids {gpu_ids} \
        --num_test {100}\
        --preprocess none'  # \
        
        # deepDeband inference uses these two lines even if preprocess none is set. this should not be necessary!
        #--crop_size {dim} \
        #--load_size {dim}'
        # this should not be necessary, read:
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#preprocessing


    os.system(command)
    # go back to the root
    os.chdir(root_path)
    
    #load back image and crop
    img = Image.open(get_output_imgpath("img"))
    img = img.crop((0, 0, original_size[0], original_size[1]))

    #delete tempfiles...
    os.remove(root_path+tempdir_in+"img.png")
    os.remove(get_output_imgpath("img"))

    return img

# batch processing

def clean():
    in_path = root_path+tempdir_in
    out_path = f'{root_path}{tempdir_out}/deepDeband-f/test_latest/images/'
    for name in os.listdir(in_path):
        os.remove(in_path+name)
    for name in os.listdir(out_path):
        os.remove(out_path+name)
    
def dump_tempimg(idx,img):
    os.makedirs(root_path+tempdir_in, exist_ok=True)
    os.makedirs(root_path+tempdir_out, exist_ok=True)
    dim = max(img.size) # get padded image size
    tmp_path = root_path+tempdir_in+f"{idx:0>8}.png"
    img.save(tmp_path, "PNG") # dump temp image
    return tmp_path

def deband_batch(padded_batch, width): #,original_sizes,new_dims):
    #clean()
    # write all temp images
    tmp_in=[dump_tempimg(idx, img) for idx,img in enumerate(padded_batch)]
        
    # run all inferences
    # run inference
    os.chdir(root_path+modeldir)
    command = f'python test.py \
        --name deepDeband-f \
        --model test \
        --netG unet_256 \
        --norm batch \
        --dataroot ../../{tempdir_in} \
        --results_dir ../../{tempdir_out} \
        --dataset_mode single \
        --gpu_ids 0 \
        --num_test {len(tmp_in)} \
        --preprocess scale_width \
        --load_size {width}    '
        # --preprocess none'
        
        # deepDeband inference uses these two lines even if preprocess none is set. this should not be necessary!
        #--crop_size {dim} \
        #--load_size {dim}'
        # this should not be necessary, read:
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#preprocessing
    os.system(command)
    os.chdir(root_path)

    # load all images in order and recrop
    generated = glob(f'{root_path}{tempdir_out}/deepDeband-f/test_latest/images/*_fake.png')
    pil_batch = [Image.open(img) for img in generated]
    #clean()
    return pil_batch