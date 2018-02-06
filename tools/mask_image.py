from argparse import ArgumentParser
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--foreground", dest="foreground")
parser.add_argument("--mask", dest="mask")
parser.add_argument("--background", dest="background")
parser.add_argument("--output", dest="output")

args = parser.parse_args()

def mask_img(foreground, mask, background, output):
    image_array = np.asarray(Image.open(foreground))
    mask_array = np.asarray(Image.open(mask))
    pattern_array = np.asarray(Image.open(background))
    idx=(mask_array==0) # mask white pixels
    image_array.setflags(write=1)
    image_array[idx]=pattern_array[idx]
    im = Image.fromarray(image_array)
    im.save(output)

# images to combine
files = os.listdir(args.foreground)
for f in files:
    foreground_path = os.path.join(args.foreground,f)
    if os.path.exists(os.path.join(args.mask,f)):
        print("masking " + f)
        mask_path = os.path.join(args.mask, f)
        background_path = os.path.join(args.background, f)
        output_path = os.path.join(args.output, f)
        mask_img(foreground_path, mask_path, background_path, output_path)
