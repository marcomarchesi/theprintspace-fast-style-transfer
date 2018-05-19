from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, get_bw_img, exists, list_files, resize_img
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
from luminance_utils import convert
from deeplab import run_segmentation
from knn_matting import image_matte


import cv2
from PIL import Image, ImageFilter


from datetime import datetime 
#startTime= datetime.now() 

BATCH_SIZE = 4
DEVICE = '/cpu:0'
soft_config = tf.ConfigProto(allow_soft_placement=True)
soft_config.gpu_options.allow_growth = True

def mask_img(a_image, b_image, mask, output):

    _mask = Image.open(mask)
        # open images A,B
    _a_image = Image.open(a_image)
    _b_image = Image.open(b_image)
    # resize images 
    m_w, m_h = _mask.size
    _a_image = _a_image.resize((m_w,m_h), Image.ANTIALIAS)
    a_w, a_h = _a_image.size
    _b_image = _b_image.resize((a_w,a_h), Image.ANTIALIAS)

    mask_array = np.asarray(_mask)
    a_array = np.asarray(_a_image)
    b_array = np.asarray(_b_image)
    
    # write permissions
    a_array.setflags(write=1)

    # weighting
    mask_array = mask_array / 255
    for i in range(3):
        if len(mask_array.shape) == 3:
            a_array[:,:,i] = b_array[:,:,i] * (1 - mask_array[:,:,0]) + a_array[:,:,i] * mask_array[:,:,0]
        else:
            a_array[:,:,i] = b_array[:,:,i] * (1 - mask_array[:,:]) + a_array[:,:,i] * mask_array[:,:]


    # mask with idx
    # idx=(mask_array==0) # mask white pixels
    # a_array[idx] = b_array[idx]

    im = Image.fromarray(a_array)
    im.save(output)


# get img_shape
def ffwd(data_in, paths_out, checkpoint_dir, device_t='/cpu:0', bw=False):

    img_shape = get_img(data_in).shape

    g = tf.Graph()

    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
        batch_shape = (1,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)

        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        X = np.zeros(batch_shape, dtype=np.float32)
        if bw:
            img = get_bw_img(data_in)

        else:
            img = get_img(data_in)
        X[0] = img

        _preds = sess.run(preds, feed_dict={img_placeholder:X})
        save_img(paths_out, _preds[0])


def ffwd_combine(in_path, out_path, foreground_ckpt, background_ckpt, 
    device_t=DEVICE, auto_seg=False, bw=False):
    
    

    background_out_name = "back_" + os.path.basename(out_path)
    background_out_path = os.path.join(os.path.dirname(out_path), background_out_name)

    print("FOREGROUND STYLE TRANSFER")
    ffwd(in_path, out_path, foreground_ckpt, device_t, bw)
    print("BACKGROUND STYLE TRANSFER")
    ffwd(in_path, background_out_path, background_ckpt, device_t, bw)

    print("Post_processing...")
    content_image_name = os.path.basename(in_path)
    segmented_image_name = "seg_" + content_image_name
    mask_image_name = "mask_" + content_image_name
    output_image_name = "out_" + content_image_name
    stylized_image_dir = os.path.dirname(out_path)
    stylized_image_path = os.path.join(stylized_image_dir, content_image_name)
    segmented_image_path = os.path.join(stylized_image_dir, segmented_image_name)
    mask_image_path = os.path.join(stylized_image_dir, mask_image_name)
    output_image_path = os.path.join(stylized_image_dir, output_image_name)

    print("Segmenting image...")
    if auto_seg:
        run_segmentation(in_path, segmented_image_path)
        # refining segmentation
        image_matte(in_path, segmented_image_path, mask_image_path)


    # C = A * mask + B * (1 - mask)
    # convert(path_in, stylized_image_path, out_path[0])
    convert(in_path, out_path, out_path)
    convert(in_path, background_out_path, background_out_path)

    # masking
    print("Masking...")
    mask_img(out_path, background_out_path, mask_image_path, output_image_path)


def stylize(in_path, out_path, foreground_ckpt, background_ckpt=None, 
            device_t=DEVICE, auto_seg=False, bw=False):


    images_in_path = []
    images_out_path = []
    for i in range(len(in_path)):
        if in_path[i].lower().endswith(('.png', '.jpg', '.jpeg')): 
            images_in_path.append(in_path[i])
            images_out_path.append(out_path[i])

    # write log file
    log_file_path = os.path.join(os.path.dirname(out_path[0]), 'style_transfer.log')
    
    for i in range(len(images_in_path)):
        f = open(log_file_path, 'a')
        
        startTime= datetime.now()
        # add shape for image
        img_shape = get_img(images_in_path[i]).shape
        out_string = 'Image of size %sx%s\n' % (img_shape[0], img_shape[1])
        out_string += '-- with path: %s\n' % images_in_path[i]
        f.write(out_string)

        if background_ckpt != None:
            ffwd_combine(images_in_path[i], images_out_path[i], 
                foreground_ckpt, background_ckpt, device_t, auto_seg, bw)
        else:
            ffwd(images_in_path[i], images_out_path[i], 
                foreground_ckpt, device_t, auto_seg, bw)

        timeElapsed=datetime.now()-startTime 
        print('Time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))
        f.write('-- written in (hh:mm:ss.ms) %s\n\n' % timeElapsed)
        f.close()

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--background-checkpoint', type=str,
                        dest='background_ckpt_dir',
                        help="dir or .ckpt for background style",
                        metavar='BACKGROUND_CHECKPOINT', required=False)

    parser.add_argument('--in-path', type=str, default='./input',
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=True)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str, default='./output',
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--automatic-segmentation', action='store_true',
                        dest='automatic_segmentation',
                        help='run automatic segmentation', default=False)

    parser.add_argument('--black-white', action='store_true',
                    dest='black_white',
                    help='black & white style transfer', default=False)


    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    files = list_files(opts.in_path)
    full_in = [os.path.join(opts.in_path,x) for x in files]
    full_out = [os.path.join(opts.out_path,x) for x in files]
    # multiple dimensions allowed
    stylize(full_in, full_out, opts.checkpoint_dir, opts.background_ckpt_dir, 
        device_t=opts.device, 
        auto_seg=opts.automatic_segmentation, bw=opts.black_white)

if __name__ == '__main__':
    main()
