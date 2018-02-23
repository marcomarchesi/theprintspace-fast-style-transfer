from PIL import Image
from argparse import ArgumentParser
import numpy as np
import os
from tqdm import tqdm

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

parser = ArgumentParser()
parser.add_argument("--content", dest="content")
parser.add_argument("--stylized", dest="stylized")
parser.add_argument("--output", dest="output")
parser.add_argument("--ratio", dest="ratio", type=float, choices=[Range(0.0, 1.0)], default=0.5)

args = parser.parse_args()

#input is a RGB numpy array with shape (height,width,3), can be uint,int, float or double, values expected in the range 0..255
#output is a double YUV numpy array with shape (height,width,3), values in the range 0..255
def RGB2YUV( rgb ):
      
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
      
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv
 
#input is an YUV numpy array with shape (height,width,3) can be uint,int, float or double,  values expected in the range 0..255
#output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
def YUV2RGB( yuv ):
       
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
     
    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304
    return rgb


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

def convert(filename, ratio):
    content_img = Image.open(os.path.join(args.content, filename))
    c_w, c_h = content_img.size
    content_array = np.asarray(content_img)
    stylized_img = Image.open(os.path.join(args.stylized, filename))
    stylized_img = stylized_img.resize((c_w, c_h), Image.ANTIALIAS)
    stylized_array = np.asarray(stylized_img)

    # convert
    yuv_input = np.array(Image.fromarray(content_array.astype(np.uint8)).convert('YCbCr'))
    yuv_content = np.array(Image.fromarray(stylized_array.astype(np.uint8)).convert('YCbCr'))


    # combine
    w, h, _ = content_array.shape
    combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
    combined_yuv[..., 0] = yuv_input[..., 0] * ratio + yuv_content[..., 0] * (1 - ratio)
    combined_yuv[..., 1] = yuv_content[..., 1]
    combined_yuv[..., 2] = yuv_content[..., 2]

    # save combined image
    img_out = Image.fromarray(combined_yuv, 'YCbCr').convert('RGB')
    img_out.save(os.path.join(args.output, filename))

def main():
    files = os.listdir(args.content)
    for file in tqdm(files):
        if file != ".DS_Store":
            if os.path.exists(os.path.join(args.stylized, file)):
                convert(file, args.ratio)

if __name__ == '__main__':
    main()
