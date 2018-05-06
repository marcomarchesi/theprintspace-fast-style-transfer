from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files, resize_img
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
from luminance_utils import convert
from deeplab import run_segmentation


import cv2
from PIL import Image


from datetime import datetime 
#startTime= datetime.now() 

BATCH_SIZE = 4
DEVICE = '/cpu:0'
soft_config = tf.ConfigProto(allow_soft_placement=True)
soft_config.gpu_options.allow_growth = True

def mask_img(a_image, mask, b_image, output):

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
    idx=(mask_array==0) # mask white pixels

    a_array.setflags(write=1)
    a_array[idx] = b_array[idx]
    # a_array[idx[...,0],1]= b_array[idx[...,0],1]
    # a_array[idx[...,0],2]= b_array[idx[...,0],2]
    im = Image.fromarray(a_array)
    im.save(output)


# get img_shape
def ffwd(data_in, paths_out, checkpoint_dir, device_t='/cpu:0', batch_size=4):

    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str

    if is_paths:
        assert len(data_in) == len(paths_out)
        img_shape = get_img(data_in[0]).shape
    else:
        assert data_in.size[0] == len(paths_out)
        img_shape = X[0].shape


    g = tf.Graph()
    batch_size = min(len(paths_out), batch_size)
    curr_num = 0

    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
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

        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos+batch_size]
            if is_paths:
                curr_batch_in = data_in[pos:pos+batch_size]
                X = np.zeros(batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = get_img(path_in)
                    assert img.shape == img_shape, \
                        'Images have different dimensions. ' +  \
                        'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos+batch_size]

            _preds = sess.run(preds, feed_dict={img_placeholder:X})

            for j, path_out in enumerate(curr_batch_out):

                save_img(path_out, _preds[j])

        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]
    if len(remaining_in) > 0:
        ffwd(remaining_in, remaining_out, checkpoint_dir, 
            device_t=device_t, batch_size=1)

def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/cpu:0'):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, checkpoint_dir, batch_size=1, device_t=device)

def ffwd_combine(in_path, out_path, foreground_ckpt, background_ckpt, 
    device_t=DEVICE, batch_size=4, train=False):

    background_out_path = [os.path.join(os.path.dirname(i), "back_" + os.path.basename(i)) for i in out_path]

    ffwd(in_path, out_path, foreground_ckpt, device_t, batch_size)
    ffwd(in_path, background_out_path, background_ckpt, device_t, batch_size)

    print("Post_processing...")
    content_image_name = os.path.basename(in_path[0])
    segmented_image_name = "seg_" + content_image_name
    output_image_name = "out_" + content_image_name
    stylized_image_dir = os.path.dirname(out_path[0])
    stylized_image_path = os.path.join(stylized_image_dir, content_image_name)
    segmented_image_path = os.path.join(stylized_image_dir, segmented_image_name)
    output_image_path = os.path.join(stylized_image_dir, output_image_name)

    print("Segmenting image...")
    run_segmentation(in_path[0], segmented_image_path)

    # C = A * mask + B * (1 - mask)
    # convert(path_in, stylized_image_path, out_path[0])
    convert(in_path[0], out_path[0], out_path[0])
    convert(in_path[0], background_out_path[0], background_out_path[0])

    # masking
    print("Masking...")
    mask_img(out_path[0], segmented_image_path, background_out_path[0], output_image_path)


def ffwd_different_dimensions(in_path, out_path, foreground_ckpt, background_ckpt=None, 
            device_t=DEVICE, batch_size=4, train=False):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    for i in range(len(in_path)):
        if in_path[i].lower().endswith(('.png', '.jpg', '.jpeg')): 
            in_image = in_path[i]
            out_image = out_path[i]
            shape = "%dx%dx%d" % get_img(in_image).shape
            in_path_of_shape[shape].append(in_image)
            out_path_of_shape[shape].append(out_image)
    for shape in in_path_of_shape:
        startTime= datetime.now()
        print('Processing images of shape %s' % shape)
        print('with path: %s' % in_path_of_shape[shape][0])
        if background_ckpt != None:
            ffwd_combine(in_path_of_shape[shape], out_path_of_shape[shape], 
                foreground_ckpt, background_ckpt, device_t, batch_size, train)
        else:
            ffwd(in_path_of_shape[shape], out_path_of_shape[shape], 
                foreground_ckpt, device_t, batch_size, train)

        timeElapsed=datetime.now()-startTime 
        print('Time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--background_checkpoint', type=str,
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

    parser.add_argument('--train', action='store_true',
                        dest='train', 
                        help='evaluate during training process', default=False)

    parser.add_argument('--smooth-affine', action='store_true',
                        dest='smooth_affine', 
                        help='smooth affine')


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

    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = \
                    os.path.join(opts.out_path,os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path

        ffwd_to_img(opts.in_path, out_path, opts.checkpoint_dir,
                    device=opts.device)
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path,x) for x in files]
        full_out = [os.path.join(opts.out_path,x) for x in files]
        # multiple dimensions allowed
        ffwd_different_dimensions(full_in, full_out, opts.checkpoint_dir, opts.background_ckpt_dir, 
            device_t=opts.device, batch_size=opts.batch_size, train=opts.train)

if __name__ == '__main__':
    main()
    # timeElapsed=datetime.now()-startTime 
    # print('Time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))
