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
                        dest='dest', required=True)
    parser.add_argument('--size', dest='size', type=int,
                            default=100)

    parser.add_argument('--hdf5', action='store_true', default=False)
    parser.add_argument('--crop', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False)
    return parser

def random_select(files, size):
    return sample(range(len(files)), size)

def calculate_laplacian(img):
    X = get_img(img, (256,256,3)).astype(np.float32)
    indices, values, __ = getLaplacianAsThree(X / 255.)
    return values


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
        print("HDF5 selected")
        hf = h5py.File('data.h5', 'w')
        return
    if args.random:
        #random select the images for calculate affine loss
        selection = random_select(files, args.size)
        for i in tqdm(range(len(selection))):
            copyfile(os.path.join(args.dir, files[i]), os.path.join(args.dest, files[i]))
            abs_path = os.path.join(args.dest, files[i])
            key = os.path.split(abs_path)[1]
            value = calculate_laplacian(abs_path)
            dictionary[key] = value
        np.save('./laplacian_data/dataset_laplacian.npy', dictionary)
        return

if __name__ == '__main__':
    main()





