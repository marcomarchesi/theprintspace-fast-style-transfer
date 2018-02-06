import numpy as np
from PIL import Image
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--image", dest="image")
parser.add_argument("--source_path", dest="source_path")
parser.add_argument("--template", dest="template")
parser.add_argument("--output_path", dest="output_path")

args = parser.parse_args()

def hist_match(source_image, template_image, output, grayscale=False):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    # load images
    source = np.asarray(Image.open(source_image))
    template = np.asarray(Image.open(template_image))

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    output_array = interp_t_values[bin_idx].reshape(oldshape)
    output_image = Image.fromarray(np.uint8(output_array))
    if grayscale:
        output_image = output_image.convert('L')
    output_image.save(output)

if args.image != None:
    hist_match(args.image, args.template, "output.png")
else:
    files = os.listdir(args.source_path)
    for f in files:
        src = os.path.join(args.source_path, f)
        dst = os.path.join(args.output_path, f)
        # perform histogram matching
        hist_match(src, args.template, dst)

