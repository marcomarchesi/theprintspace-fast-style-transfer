from argparse import ArgumentParser
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import blending

parser = ArgumentParser()
parser.add_argument("--path", dest="path")

args = parser.parse_args()

def mask_img(a_image, mask, b_image, output):

    _mask = Image.open(mask)
        # open images A,B
    _a_image = Image.open(a_image)
    _b_image = Image.open(b_image)
    # resize images 
    m_w, m_h = _mask.size
    _a_image = _a_image.resize((m_w,m_h), Image.ANTIALIAS)
    a_w, a_h = _a_image.size
    _b_image = _b_image.resize((a_w,a_h), Image.ANTIALIAS)

    mask_array = np.asarray(_mask)
    a_array = np.asarray(_a_image)
    b_array = np.asarray(_b_image)
    idx=(mask_array==0) # mask white pixels

    a_array.setflags(write=1)
    a_array[idx] = b_array[idx]
    # a_array[idx[...,0],1]= b_array[idx[...,0],1]
    # a_array[idx[...,0],2]= b_array[idx[...,0],2]
    im = Image.fromarray(a_array)
    im.save(output)

# images to combine
files = os.listdir(os.path.join(args.path, "A"))
for f in files:
    if f != ".DS_Store":
        a_image_path = os.path.join(os.path.join(args.path, "A"),f)
        if os.path.exists(os.path.join(os.path.join(args.path, "mask"),f)):
            print("masking " + f)
            mask_path = os.path.join(os.path.join(args.path, "mask"), f)
            b_image_path = os.path.join(os.path.join(args.path, "B"), f)
            output_path = os.path.join(os.path.join(args.path, "output"), f)
            mask_img(a_image_path, mask_path, b_image_path, output_path)
