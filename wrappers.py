import shutil
from PIL import Image

def pad_image(img):
    img = Image.open(img_path)
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
    return padded_img, img.size #return padded image and original size


def deband_image_full(img, original_size, gpu_ids=0):
    os.makedirs("temp/in/", exist_ok=True)
    os.makedirs("temp/out/", exist_ok=True)
    dim = max(img.size) # get padded image size
    img.save("temp/in/img.png", "PNG") # dump temp image

    # run inference
    os.chdir('deepDeband/pytorch-CycleGAN-and-pix2pix')
    command = f'python test.py --name deepDeband-f --model test --netG unet_256 --norm batch \
        --dataroot ../../temp/in --results_dir ../../temp/out \
        --dataset_mode single --gpu_ids {gpu_ids} --preprocess none --crop_size {dim} --load_size {dim}'


    os.system(command)
    # go back to the root
    os.chdir('../..')
    
    #load back image and crop
    img = Image.open('temp/out/img.png')
    img = img.crop((0, 0, original_size[0], original_size[1]))

    #delete tempfiles...
    os.remove("temp/in/img.png")
    os.remove("temp/out/img.png")

    return img