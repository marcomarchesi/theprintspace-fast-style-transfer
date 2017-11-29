import sys, os
sys.path.insert(0, 'src')
import numpy as np, scipy.misc
from argparse import ArgumentParser
from utils import get_img, save_img

parser =  ArgumentParser()
parser.add_argument('--content-image', type=str,
                    dest='content_image', metavar='CONTENT_IMAGE')
parser.add_argument('--stylized-image', type=str,
                    dest='stylized_image', metavar='STYLIZED_IMAGE')
parser.add_argument('--output-dir', type=str,
                    dest='output_dir', metavar='OUTPUT_DIR')
parser.add_argument('--weight', type=float, nargs='+',
                    dest='weight', metavar='WEIGHT')

args = parser.parse_args()

def grad(img):
    gx, gy = np.gradient(img)
    return gx, gy

# TODO
def rgb2lab(img):
    return img
# TODO
def lab2rgb(img):
    return img

def get_grad_mat(w, h, mask):
    img_size = w * h
    

def poisson_combination(channel, dx, dy, grad_weight):
    # compute mean of the image channel
    mean = np.mean(channel)
    h, w = channel.shape

    # init
    weight_grad = grad_weight * np.ones((w,h), dtype=np.float)
    weight_color = np.ones((w,h), dtype=np.float)

    # sparse rappresentation
    gx, gy = get_grad_mat(w, h, weight_grad)



    return channel - mean


def solve_poisson(content_img, stylized_img, weight):
    lab_content_img = rgb2lab(content_img)
    lab_stylized_img = rgb2lab(stylized_img)
    # initialize
    lab_output_img = lab_stylized_img
    # todo make better
    for c in range(3):
        dx, dy = grad(lab_content_img[:,:,c])
        lab_output_img[:,:,c] = poisson_combination(lab_stylized_img[:,:,c], dx, dy, weight[c])
    return lab2rgb(lab_output_img)

    
content_image = get_img(args.content_image)
stylized_image = get_img(args.stylized_image)
output_image = solve_poisson(content_image, stylized_image, args.weight)
save_img(os.path.join(args.output_dir, "test.png"), output_image)


