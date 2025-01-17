from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img, list_abs_files, get_img_from_hdf5, get_laplacian_from_hdf5, num_files
import random
from scipy import ndimage
from random import randint

import h5py
from PIL import Image

# add laplacian
from closed_form_matting import getLaplacian, getLaplacianAsThree

#STYLE_LAYERS = ('relu1_2', 'relu2_2', 'relu3_3')
STYLE_LAYERS = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3') # blurred bw
# STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = '/gpu:0'

laplacian_shape = (65536, 65536)
laplacian_indices = np.load('./laplacian_data/indices.npy')

# hfd5
hf = h5py.File('./data/data.h5', 'r')
# laplacian_hf_size = 82783
laplacian_hf_size = num_files('./data/laplacian/')


def sobel(img_array):
    '''
    image: image to convert with Sobel filter
    '''
    col = np.zeros((img_array.shape))

    for i in range(img_array.shape[0]):
        # print(i)
        img = img_array[i]
        dx = ndimage.sobel(img, 0)  # horizontal derivative
        dy = ndimage.sobel(img, 1)  # vertical derivative
        mag = np.hypot(dx, dy)  # magnitude, equivalent to sqrt(dx**2 + dy**2)
        mag *= 255.0 / np.max(mag)  # normalize (Q&D)
        col[i] = mag

    return col

def get_affine_loss_plus(output, MM, weight):
    loss_affine = 0.0
    _M = tf.SparseTensor(laplacian_indices, MM[0], laplacian_shape)
    output_t = output[0] / 255.
    for Vc in tf.unstack(output_t, axis=-1):
        Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
        loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0), tf.sparse_tensor_dense_matmul(_M, tf.expand_dims(Vc_ravel, -1)))

    return loss_affine * weight

def get_luma_loss(content, preds, weight):
    rgb2yuv_m = tf.constant([[ 0.29900, -0.16874,  0.50000],
         [0.58700, -0.33126, -0.41869],
         [ 0.11400, 0.50000, -0.08131]])

    _loss = 0.0

    for i in range(content.shape[0]):
        yuv_content_mat = tf.reshape(content[i], shape=(256*256, 3))
        yuv_content = tf.matmul(yuv_content_mat, rgb2yuv_m)
        yuv_preds_mat = tf.reshape(preds[i], shape=(256*256, 3))
        yuv_preds = tf.matmul(yuv_preds_mat, rgb2yuv_m)

        # go back to 3d
        yuv_content = tf.reshape(yuv_content, shape=(256, 256, 3))
        yuv_preds = tf.reshape(yuv_preds, shape=(256, 256, 3))

        # get the Y (luma) channel
        y_content = tf.unstack(yuv_content, axis=2)
        y_preds = tf.unstack(yuv_preds, axis=2)

        y_content = tf.subtract(y_content, tf.constant(179.45477266423404))
        y_preds = tf.subtract(y_preds, tf.constant(179.45477266423404))

        _loss += tf.reduce_mean(tf.squared_difference(y_content, y_preds)) * weight

    return _loss



def show_features(features, image):
    img = np.expand_dims(np.array(image), axis=0)
    feed_dict = {
           style_image:img
    }
    blocks = tf.unstack(features, axis=3)
    filters = []
    for block in blocks:
        filters.append(tf.squeeze(block))
    arr =  sess.run(filters, feed_dict=feed_dict)
    return Image.fromarray(np.uint8(arr[10]))

# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight, contrast_weight,
             tv_weight, affine_weight, luma_weight, vgg_path, epochs=2, print_iterations=1,
             batch_size=4, save_path='saver/fns.ckpt',
             learning_rate=1e-3, debug=False, no_gpu=False, logs=False, 
             affine=False, gradient=False, contrast=False, luma=False,
             multiple_style_images=False, num_examples=1000):


    DEVICES = '/gpu:0'
    config = tf.ConfigProto(allow_soft_placement=True)

    
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod] 

    style_features = {}

    style_images = []
    style_images.append(get_img(style_target))

    batch_shape = (batch_size,256,256,3)
    style_shape = (1,) + style_images[0].shape

    if no_gpu:
        DEVICES = '/cpu:0' 

    # precompute style features
    with tf.Graph().as_default(), tf.device(DEVICES), tf.Session(config=config) as sess:

        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        # style_pre = np.array([style_target])

        filters = []

        for layer in STYLE_LAYERS:
            features = net[layer]
            # to show the cnn filters

            features = tf.squeeze(features)
            features = tf.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
            features_size = tf.to_float(features.shape[0] * features.shape[1])
            gram = tf.matmul(features, features, adjoint_a=True) / features_size
            style_features[layer] = gram

        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        X_contrast = tf.placeholder(tf.float32, shape=batch_shape, name="X_contrast")
        X_pre_contrast = vgg.preprocess(X_contrast)

        # for affine loss
        X_MM = tf.placeholder(tf.float32, name="X_MM")
 
        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        contrast_features = {}
        contrast_net = vgg.net(vgg_path, X_pre_contrast)
        contrast_features[CONTENT_LAYER] = contrast_net[CONTENT_LAYER]

        preds = transform.net(X_content/255.0)
        preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        # affine loss
        #affine_loss = tf.constant(0.0)
        if affine:
            affine_loss = get_affine_loss_plus(preds_pre, X_MM, affine_weight)
        if luma:
            affine_loss = get_affine_loss_plus(preds_pre, X_MM, affine_weight)
            luma_loss = get_luma_loss(X_content, preds, luma_weight)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size)

        contrast_loss = contrast_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - contrast_features[CONTENT_LAYER]) / content_size)


        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]

            # style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/ tf.to_float(style_gram.shape[0] *\
                style_gram.shape[1]))

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

        # affine
        batch_laplacian_shape = (batch_size, 1623076)
        # batch_laplacian_shape = (laplacian_hf_size, 1623076)
        # M = np.zeros(batch_laplacian_shape, dtype=np.float32)
        M = np.zeros(batch_laplacian_shape, dtype=np.float32)



        # DOES THIS POSITION COULD AFFECT THE TRAINING?  
        X_batch = np.zeros(batch_shape, dtype=np.float32)


        # loss contributions
        if luma:
            loss = content_loss + style_loss + tv_loss + contrast_loss + affine_loss + luma_loss
        elif affine:
            loss = content_loss + style_loss + tv_loss + contrast_loss + affine_loss
        elif contrast:
            loss = content_loss + style_loss + tv_loss + contrast_loss
        else:
            loss = content_loss + style_loss + tv_loss

        if logs:
            # summaries for TensorBoard
            with tf.name_scope('loss_summaries'):
                tf.summary.scalar('content_loss', content_loss)
                tf.summary.scalar('style_loss', style_loss)
                tf.summary.scalar('tv_loss', tv_loss)
                #if affine:
                #    tf.summary.scalar('affine_loss', affine_loss)
                if contrast:
                    tf.summary.scalar('contrast_loss', contrast_loss)
                if gradient:
                    tf.summary.scalar('gradient_loss', gradient_loss)
                #tf.summary.scalar('total_loss', loss)

                tf.summary.image('batch', X_content)
                tf.summary.image('predicted', preds_pre)

            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('./logs',
                                          sess.graph)

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # initialize variables
        sess.run(tf.global_variables_initializer())

        global_step = 0

        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            index = 0
            laplacian_index = 0

            # style image to use
            style_pre = np.expand_dims(np.array(style_images[0]), axis=0)
                
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size

                for j, img_p in enumerate(content_targets[curr:step]):
                   # print(img_p)
                   img_filename = os.path.basename(img_p)
                   #print(j)
                   # read images from hdf5
                   #X_batch[j] = get_img_from_hdf5(index, hf)
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)
                   if affine:
                    if j == 0:
                        filepath = './data/laplacian/' + str(laplacian_index) + '.h5'
                        laplacian_hf = h5py.File(filepath, 'r')
                        M[0] = get_laplacian_from_hdf5(0, laplacian_hf)
                        laplacian_hf.close()

                   index += 1
                   
                laplacian_index += 1
                   
                print ("Iteration: %i" % iterations)  
                iterations += 1
                assert X_batch.shape[0] == batch_size

                # TODO add condition for gradient
                if luma:
                    feed_dict = {
                       style_image:style_pre, X_content:X_batch, X_contrast: sobel(X_batch), X_MM: M
                    }
                elif affine:
                    feed_dict = {
                       style_image:style_pre, X_content:X_batch, X_contrast: sobel(X_batch), X_MM: M
                    }
                elif contrast:
                    feed_dict = {
                       style_image:style_pre, X_content:X_batch, X_contrast: sobel(X_batch)
                    }
                else:
                    feed_dict = {
                       style_image:style_pre, X_content:X_batch
                    }

                if luma:
                    to_get = [style_loss, content_loss, tv_loss, contrast_loss, affine_loss, luma_loss, loss, preds]
                elif affine:
                    to_get = [style_loss, content_loss, tv_loss, contrast_loss, affine_loss, loss, preds]
                elif contrast:
                    to_get = [style_loss, content_loss, tv_loss, contrast_loss, loss, preds]
                else:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]


                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time

                global_step = (epoch + 1) * iterations
                # print("Global Step: %i" % global_step)
                if logs:
                    summary, tup = sess.run([merged, to_get], feed_dict = feed_dict)
                    summary_writer.add_summary(summary, global_step)
                else:
                    tup = sess.run(to_get, feed_dict = feed_dict)


                is_print_iter = int(iterations) % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:

                    if luma:
                        _style_loss,_content_loss,_tv_loss, _contrast_loss, _affine_loss, _luma_loss, _loss,_preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _contrast_loss, _affine_loss, _luma_loss, _loss)
                    elif affine:
                        _style_loss,_content_loss,_tv_loss, _contrast_loss, _affine_loss, _loss,_preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _contrast_loss, _affine_loss, _loss)
                    elif contrast:
                        _style_loss,_content_loss,_tv_loss, _contrast_loss, _loss,_preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _contrast_loss, _loss)
                    else:
                        _style_loss,_content_loss,_tv_loss, _loss,_preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _loss)                   

                    saver = tf.train.Saver()
                    res = saver.save(sess, save_path)
                    yield(_preds, losses, iterations, epoch)

            # check GraphDef size
            print("GraphDef size: %i" % sess.graph_def.ByteSize())

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
