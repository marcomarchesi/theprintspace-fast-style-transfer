from __future__ import print_function
import sys, os, pdb
sys.path.insert(0, 'src')
import numpy as np, scipy.misc 
from optimize import optimize
from argparse import ArgumentParser
from utils import save_img, get_img, exists, list_files
import evaluate
import time
import json

#for copying
from shutil import copyfile

CONTENT_WEIGHT = 1.5e1
CONTRAST_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
AFFINE_WEIGHT = 5e2
LUMA_WEIGHT = 1e1

LEARNING_RATE = 1e-3
NUM_EPOCHS = 3
NUM_EXAMPLES = 1000
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 1000
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014_256x256'
#TRAIN_PATH = 'data/celebs_80k'
BATCH_SIZE = 30
FRAC_GPU = 1

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--style-image', type=str,
                        dest='style_image', help='style image path',
                        metavar='STYLE_IMAGE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--no-gpu', dest='no_gpu', action='store_true', help='not using GPU', default=False)
    parser.add_argument('--logs', dest='logs', action='store_true', help='whether using Tensorboard', default=False)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--luma-weight', type=float,
                        dest='luma_weight',
                        help='luma weight (default %(default)s)',
                        metavar='LUMA_WEIGHT', default=LUMA_WEIGHT)

    parser.add_argument('--luma', dest='luma', action='store_true',
                        help='luma loss enabled', default=True)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--contrast-weight', type=float,
                        dest='contrast_weight',
                        help='contrast weight (default %(default)s)',
                        metavar='CONTRAST_WEIGHT', default=CONTRAST_WEIGHT)

    parser.add_argument('--contrast', dest='contrast', action='store_true',
                        help='contrast loss enabled', default=False)
    
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--affine', dest='affine', action='store_true',
                        help='affine loss enabled', default=False)
    
    parser.add_argument('--affine-weight', type=float,
                        dest='affine_weight',
                        help='affine regularization weight (default %(default)s)',
                        metavar='AFFINE_WEIGHT', default=AFFINE_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--num-examples', type=float,
                    dest='num_examples',
                    help='number of examples (default %(default)s)',
                    metavar='NUM_EXAMPLES', default=NUM_EXAMPLES)



    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style_image, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        exists(opts.test_dir, "test directory not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.contrast_weight >= 0
    assert opts.luma_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0
    assert opts.num_examples >= 0

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

    
def main():
    start_time = time.time()
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    style_target = options.style_image
    if 1:
        content_targets = _get_files(options.train_path)
    elif options.test:
        content_targets = [options.test]
    print(len(content_targets))

    kwargs = {
        "epochs":options.epochs,
        "print_iterations":options.checkpoint_iterations,
        "batch_size":options.batch_size,
        "save_path":os.path.join(options.checkpoint_dir,'fns.ckpt'),
        "learning_rate":options.learning_rate,
        "num_examples": options.num_examples,
        "no_gpu":options.no_gpu,
        "logs":options.logs,
        "affine":options.affine,
        "luma":options.luma,
        "contrast":options.contrast
    }

    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.contrast_weight,
        options.tv_weight,
        options.affine_weight,
        options.luma_weight,
        options.vgg_path
    ]



    # save options as json file
    with open(os.path.join(options.test_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(options), sort_keys=True, indent=4))


    for preds, losses, i, epoch in optimize(*args, **kwargs):

        if options.luma:
            style_loss, content_loss, tv_loss, contrast_loss, affine_loss, luma_loss, loss = losses
            to_print = (style_loss, content_loss, tv_loss, contrast_loss, affine_loss, luma_loss)
        elif options.affine:
            style_loss, content_loss, tv_loss, contrast_loss, affine_loss, loss = losses
            to_print = (style_loss, content_loss, tv_loss, contrast_loss, affine_loss)
        elif options.contrast:
            style_loss, content_loss, tv_loss, contrast_loss, loss = losses
            to_print = (style_loss, content_loss, tv_loss, contrast_loss) 
        else:
            style_loss, content_loss, tv_loss, loss = losses
            to_print = (style_loss, content_loss, tv_loss) 

        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        # print('style: %s, content:%s, tv: %s, contrast: %s, gradient: %s' % to_print)

        if options.test:
            assert options.test_dir != False
            preds_path = '%s/%s_%s.png' % (options.test_dir,epoch,i)
            
                # copy ckpt
            src = os.path.join(options.checkpoint_dir, "fns.ckpt.data-00000-of-00001")
            dst = os.path.join(options.checkpoint_dir, "fns.ckpt")
            copyfile(src, dst)

            ckpt_dir = os.path.dirname(options.checkpoint_dir)
            evaluate.ffwd_to_img(options.test,preds_path,
                                 options.checkpoint_dir)

    ckpt_dir = options.checkpoint_dir
    end_time  = time.time()
    elapsed_time = end_time - start_time
    print("Training complete in %s seconds." % (elapsed_time))

    # final cleanup
    os.remove(os.path.join(options.checkpoint_dir, "fns.ckpt.meta"))


if __name__ == '__main__':
    main()
