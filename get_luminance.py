from PIL import Image
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument("--content", dest="content")
parser.add_argument("--stylized", dest="stylized")
parser.add_argument("--output", dest="output")

args = parser.parse_args()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

# get the images
content = np.asarray(Image.open(args.content))
stylized = np.asarray(Image.open(args.stylized))


# convert
grayscale_input = rgb2gray(content)
rgb_input = gray2rgb(grayscale_input)
yuv_input = np.array(Image.fromarray(rgb_input.astype(np.uint8)).convert('YCbCr'))

yuv_content = np.array(Image.fromarray(stylized.astype(np.uint8)).convert('YCbCr'))

# combine
w, h, _ = content.shape
combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
combined_yuv[..., 0] = yuv_input[..., 0] * 0.5 + yuv_content[..., 0] * 0.5
combined_yuv[..., 1] = yuv_content[..., 1]
combined_yuv[..., 2] = yuv_content[..., 2]

# save combined image
img_out = Image.fromarray(combined_yuv, 'YCbCr').convert('RGB')
img_out.save(args.output)




