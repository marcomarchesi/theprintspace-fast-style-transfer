# color matching
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from color_transfer import color_transfer
import cv2

parser = ArgumentParser()
parser.add_argument("--source", dest="source")
parser.add_argument("--target", dest="target")
parser.add_argument("--output", dest="output")

args = parser.parse_args()

def color_matching(source_image, target_image, output_image):
    source = cv2.imread(source_image)
    target = cv2.imread(target_image)
    transfer = color_transfer(source, target)
    cv2.imwrite(output_image, transfer)

color_matching(args.source, args.target, args.output)
