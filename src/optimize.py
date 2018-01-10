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

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = '/gpu:0'

laplacian_shape = (65536, 65536)
laplacian_indices = np.load('./laplacian_data/indices.npy')


# hfd5
hf = h5py.File('./data/data.h5', 'r')
# laplacian_hf_size = 82783
laplacian_hf_size = num_files('./data/laplacian/')


# for debug mode
uid = random.randint(1, 100)
# print("UID: %s" % uid)

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


# def get_affine_loss_plus(output, M, weight):
#     loss_affine = 0.0
#     output_t = output / 255.
#     for Vc in tf.unstack(output_t, axis=-1):
#         Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
#         loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0), tf.sparse_tensor_dense_matmul(M, tf.expand_dims(Vc_ravel, -1)))

#     return loss_affine * weight



def get_affine_loss(output, MM, weight):
    loss_affine = 0.0
    # for i in range(batch_size):
    _M = tf.SparseTensor(laplacian_indices, MM[0], laplacian_shape)
    output_t = output[0] / 255.
    for Vc in tf.unstack(output_t, axis=-1):
        Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
        ravel_0 = tf.expand_dims(Vc_ravel, 0)
        ravel_0 = tf.cast(ravel_0, tf.float32)
        ravel_1 = tf.expand_dims(Vc_ravel, -1)
        ravel_1 = tf.cast(ravel_1, tf.float32)
        loss_affine += tf.matmul(ravel_0, tf.sparse_tensor_dense_matmul(_M, ravel_1))

    return tf.reduce_mean(loss_affine * weight)


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
def optimize(content_targets, style_targets, content_weight, style_weight, contrast_weight, gradient_weight,
             tv_weight, affine_weight, vgg_path, epochs=2, print_iterations=1,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False, no_gpu=False, logs=False, 
             affine=False, affine_plus=False, gradient=False, contrast=False, 
             multiple_style_images=False, num_examples=1000):


    DEVICES = '/gpu:0'
    config = tf.ConfigProto(allow_soft_placement=True)

    
    if affine_plus:
        batch_size = 1
        reduced_image = get_img("./content/portrait_03.jpg", (256,256,3)).astype(np.float32)
        content_image = get_img("./content/portrait_03.jpg").astype(np.float32)
        print("Calculating laplacian...")
    
    batch_laplacian_shape = (batch_size, 1623076)
    M = np.zeros(batch_laplacian_shape, dtype=np.float32)

    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod] 

    style_features = {}

    style_images = []
    for style_target in list_abs_files(style_targets):
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

        if slow:
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            preds = transform.net(X_content/255.0)
            preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        # affine loss
        affine_loss = tf.constant(0.0)
        if affine:
            affine_loss = get_affine_loss(preds_pre, X_MM, affine_weight)


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



        # DOES THIS POSITION COULD AFFECT THE TRAINING?  
        X_batch = np.zeros(batch_shape, dtype=np.float32)


        # loss contributions
        if gradient:
            gradient_loss = grad_image_loss(X_content, preds, gradient_weight)
            loss = content_loss + style_loss + tv_loss + contrast_loss + affine_loss + gradient_loss
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
                if affine:
                    tf.summary.scalar('affine_loss', affine_loss)
                if contrast:
                    tf.summary.scalar('contrast_loss', contrast_loss)
                if gradient:
                    tf.summary.scalar('gradient_loss', gradient_loss)
                tf.summary.scalar('total_loss', loss)

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

        if affine_plus:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            print("Checkpoint restored")


            _, M[0], _ = getLaplacianAsThree(content_image / 255.)

            # training
            iterations = 0
            epoch = 0
            style_pre = np.expand_dims(np.array(style_images[0]), axis=0)
            while iterations < 100:
                X_batch[0] = reduced_image


                print ("Iteration: %i" % iterations)  
                iterations += 1


                feed_dict = {
                    style_image:style_pre, 
                    X_content:X_batch, 
                    X_contrast: sobel(X_batch), 
                    X_MM: M
                }

                train_step.run(feed_dict=feed_dict)

                # print results
                to_get = [style_loss, content_loss, tv_loss, contrast_loss, affine_loss, loss, preds]
                test_feed_dict = {
                    style_image:style_pre, 
                    X_content:X_batch, 
                    X_contrast: sobel(X_batch), 
                    X_MM: M
                }

                tup = sess.run(to_get, feed_dict = test_feed_dict)
                _style_loss,_content_loss,_tv_loss, _contrast_loss, _affine_loss, _loss,_preds = tup
                losses = (_style_loss, _content_loss, _tv_loss, _contrast_loss, _affine_loss, _loss)
                saver = tf.train.Saver()
                res = saver.save(sess, save_path)
                yield(_preds, losses, iterations, epoch)


            print("style transfer with affine plus completed")
            return




        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            index = 0
            laplacian_index = 0

            # style image to use
            if multiple_style_images: 
                random_item = randint(0, len(style_images))
                if random_item == len(style_images):
                    random_item -= 1
                print("style image no.%i" % random_item)
                style_pre = np.expand_dims(np.array(style_images[random_item]), axis=0)
            else:
                style_pre = np.expand_dims(np.array(style_images[0]), axis=0)
                
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size

                for j, img_p in enumerate(content_targets[curr:step]):
                   # print(img_p)
                   # read images from hdf5
                   X_batch[j] = get_img_from_hdf5(index, hf)
                   # X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)

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
                if affine:
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

                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:

                    if gradient:
                        to_get = [style_loss, content_loss, tv_loss, contrast_loss, affine_loss, gradient_loss, loss, preds]
                    elif affine:
                        to_get = [style_loss, content_loss, tv_loss, contrast_loss, affine_loss, loss, preds]
                    elif contrast:
                        to_get = [style_loss, content_loss, tv_loss, contrast_loss, loss, preds]
                    else:
                        to_get = [style_loss, content_loss, tv_loss, loss, preds]

                    # TODO add condition for gradient
                    if affine:
                        test_feed_dict = {
                           style_image:style_pre, X_content:X_batch, X_contrast: sobel(X_batch), X_MM: M
                        }
                    elif contrast:
                        test_feed_dict = {
                           style_image:style_pre, X_content:X_batch, X_contrast: sobel(X_batch)
                        }
                    else:
                        test_feed_dict = {
                           style_image:style_pre, X_content:X_batch
                        }

                    global_step = (epoch + 1) * iterations
                    # print("Global Step: %i" % global_step)
                    if logs:
                        summary, tup = sess.run([merged, to_get], feed_dict = test_feed_dict)
                        summary_writer.add_summary(summary, global_step)
                    else:
                        tup = sess.run(to_get, feed_dict = test_feed_dict)

                    if gradient:
                        _style_loss,_content_loss,_tv_loss, _contrast_loss, _affine_loss, _gradient_loss, _loss,_preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _contrast_loss, _affine_loss, _gradient_loss, _loss)
                    elif affine:
                        _style_loss,_content_loss,_tv_loss, _contrast_loss, _affine_loss, _loss,_preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _contrast_loss, _affine_loss, _loss)
                    elif contrast:
                        _style_loss,_content_loss,_tv_loss, _contrast_loss, _loss,_preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _contrast_loss, _loss)
                    else:
                        _style_loss,_content_loss,_tv_loss, _loss,_preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _loss)                   

                    if slow:
                       _preds = vgg.unprocess(_preds)
                    else:
                        saver = tf.train.Saver()
                        res = saver.save(sess, save_path)
                    yield(_preds, losses, iterations, epoch)

            # check GraphDef size
            print("GraphDef size: %i" % sess.graph_def.ByteSize())

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
