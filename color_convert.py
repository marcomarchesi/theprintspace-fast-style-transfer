from argparse import ArgumentParser
import sys
sys.path.insert(0, 'src')
import cv2
import numpy as np
import os
import skimage.io
import tensorflow as tf
from PIL import Image
from utils import list_files
from tqdm import tqdm

VGG_MEAN = [103.939, 116.779, 123.68]

parser = ArgumentParser()
parser.add_argument('--dir', type=str,
                        dest='dir', help='dir',
                        metavar='DIR', required=True)
parser.add_argument('--image', type=str,
                        dest='image', help='image to convert',
                        metavar='IMAGE', required=False)
parser.add_argument('--style', type=str,
                        dest='style', help='style to use',
                        metavar='STYLE', required=True)

parser.add_argument('--path', type=str,
                        dest='path', help='path to save',
                        metavar='PATH', required=False)

args = parser.parse_args()


def convert_image(content_image, style_image, path):
    # convert style to yuv
    style_img = Image.open(style_image)
    content_img = Image.open(content_image)

    style_img = style_img.resize((content_img.size[0], content_img.size[1]), Image.ANTIALIAS)
    style_array = np.asarray(style_img)
    style_yuv = cv2.cvtColor(np.float32(style_array), cv2.COLOR_RGB2YUV)

    img = np.asarray(content_img)
    img = np.squeeze(img)
    img = img[:,:,(2,1,0)]  # bgr to rgb
    img = img + VGG_MEAN
    yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)
    yuv[:,:,1:3] = style_yuv[:,:,1:3]
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    img = np.clip(img, 0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    # skimage.io.imsave(path, img)
    im.save(path)

# save_image(args.image, args.style, args.path) 

assert os.path.exists(args.dir)
files = list_files(args.dir)
for i in tqdm(range(len(files))):
    if files[i] != ".DS_Store":
        src = os.path.join(args.dir, files[i])
        dst = os.path.join(args.dir, "yuv_" + files[i])
        convert_image(src, args.style, dst)
    # print(src)
