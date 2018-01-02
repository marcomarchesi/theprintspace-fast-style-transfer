# preprocess

import numpy as np
import sys, os, pdb
sys.path.insert(0, 'src')
from argparse import ArgumentParser
from closed_form_matting import getLaplacianAsThree
from utils import get_img, list_files
from tqdm import tqdm
from random import sample
from shutil import copyfile
from PIL import Image

# format
import h5py

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str,
                            dest='dir', help='dataset dir',
                            metavar='DIR', required=True)
    parser.add_argument('--dest', type=str,
                        dest='dest')
    parser.add_argument('--size', dest='size', type=int,
                            default=100)

    parser.add_argument('--hdf5', action='store_true', default=False)
    parser.add_argument('--crop', action='store_true', default=False)
    parser.add_argument('--laplacian', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=30)
    return parser

# def random_select(files, size):
#     return sample(range(len(files)), size)

def calculate_laplacian(img):
    X = get_img(img, (256,256,3)).astype(np.float32)
    indices, values, __ = getLaplacianAsThree(X / 255.)
    return values

def grayscale_to_rgb(img, size):
    ret = np.empty((size, size, 3), dtype=np.uint8)
    ret[:, :, 0] = img
    ret[:, :, 1] = img
    ret[:, :, 2] = img
    return ret


def main():
    parser = build_parser()
    args = parser.parse_args()
    assert os.path.exists(args.dir)
    files = list_files(args.dir)
    dictionary = {}
    counter = 0

    if args.crop:
        print("Cropping images...")
        for i in tqdm(range(len(files))):
            src = os.path.join(args.dir, files[i])
            # print(src)
            if src.endswith('jpg'):
                im = Image.open(src)
                width, height = im.size   # Get dimensions
                left = (width - 256)/2
                top = (height - 256)/2
                right = (width + 256)/2
                bottom = (height + 256)/2
                im = im.crop((left, top, right, bottom))
                dst = os.path.join(args.dest, files[i])
                im.save(dst)
        return
    if args.hdf5:
        train_shape = (len(files), 256, 256, 3)
        with h5py.File('./data/data.h5', 'w') as hf:
            hf.create_dataset("train_img", train_shape, np.uint8)
            for i in tqdm(range(len(files))):
                src = os.path.join(args.dir, files[i])
                im = Image.open(src)
                im_arr = np.asarray(im)
                if im.mode == 'L':

                    im_arr = grayscale_to_rgb(im_arr,256)
                    # img = Image.fromarray(im_arr, 'RGB')
                    # img.show()

                # print(im_arr.shape)
                # print(i)
                hf["train_img"][i] = im_arr
                im.close()
        return
    if args.laplacian:
        batch_size = args.batch_size
        num_samples = int(len(files) / batch_size) + 1
        laplacian_shape = (num_samples, 1623076)
        # laplacian_shape = (len(files), 1623076)
        with h5py.File('./data/laplacian.h5', 'w') as hf:
            hf.create_dataset("laplacian_img", laplacian_shape, np.float32)
            current_sample = 0
            for i in tqdm(range(len(files))):
                if i % batch_size == 0:
                    src = os.path.join(args.dir, files[i])
                    value = calculate_laplacian(src)
                    # print(value.shape)
                    # return
                    hf["laplacian_img"][current_sample] = value
                    current_sample += 1
        return

if __name__ == '__main__':
    main()





