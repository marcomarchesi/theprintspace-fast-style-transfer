# preprocess laplacian

import numpy as np
import sys, os, pdb
sys.path.insert(0, 'src')
from argparse import ArgumentParser
from closed_form_matting import getLaplacianAsThree
from utils import get_img, list_files
import progressbar
from random import sample
from shutil import copyfile

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str,
                            dest='dir', help='dataset dir',
                            metavar='DIR', required=True)
    parser.add_argument('--dest', type=str,
                        dest='dest', required=True)
    parser.add_argument('--size', dest='size', type=int,
                            default=100)
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
    # random select the images for calculate affine loss
    selection = random_select(files, args.size)
    bar = progressbar.ProgressBar()
    dictionary = {}
    counter = 0
    for i in bar(range(len(selection))):
        copyfile(os.path.join(args.dir, files[i]), os.path.join(args.dest, files[i]))
        abs_path = os.path.join(args.dest, files[i])
        key = os.path.split(abs_path)[1]
        value = calculate_laplacian(abs_path)
        dictionary[key] = value
    np.save('./laplacian_data/dataset_laplacian.npy', dictionary)


    # abs_paths = [os.path.join(args.dir,x) for x in files]
    # # got the files, calculate the laplacian for each of them

    
    # bar = progressbar.ProgressBar()
    # for i in bar(range(len(abs_paths))):
    #     key = os.path.split(abs_paths[i])[1]
    #     value = calculate_laplacian(abs_paths[i])
    #     dictionary[key] = value
    #     # print("%f" % float((counter / len(abs_paths)) * 100))
    #     # counter +=1/
    # np.save('./laplacian_data/dataset_laplacian.npy', dictionary)


if __name__ == '__main__':
    main()





