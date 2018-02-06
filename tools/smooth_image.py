from smooth_local_affine import smooth_local_affine
from argparse import ArgumentParser
import numpy as np
from PIL import Image

parser = ArgumentParser()
parser.add_argument("--content", dest="content")
parser.add_argument("--stylized", dest="result")
parser.add_argument("--output", dest="output")

args = parser.parse_args()

def smooth(content, stylized, output):
    content_input = np.array(Image.open(content).convert("RGB"), dtype=np.float32)
    # RGB to BGR
    content_input = content_input[:, :, ::-1]
    # H * W * C to C * H * W
    content_input = content_input.transpose((2, 0, 1))
    input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.

    _, H, W = np.shape(input_)

    output_ = np.ascontiguousarray(stylized.transpose((2, 0, 1)), dtype=np.float32) / 255.
    best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, 15, 1e-1).transpose(1, 2, 0)
    result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
    result.save(output)

smooth(args.content, args.stylized, args.output)