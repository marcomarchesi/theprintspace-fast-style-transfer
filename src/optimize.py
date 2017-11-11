from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img
import random

# add laplacian
from closed_form_matting import getLaplacian, getLaplacianAsThree

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = '/gpu:0'

laplacian_shape = (65536, 65536)
laplacian_indices = np.load('./laplacian_data/indices.npy')

# for debug mode
uid = random.randint(1, 100)
# print("UID: %s" % uid)


def get_affine_loss(output, batch_size, MM, weight):
    loss_affine = 0.0
    for i in range(batch_size):
        _M = tf.SparseTensor(laplacian_indices, MM[i], laplacian_shape)
        output_t = output[i] / 255.
        for Vc in tf.unstack(output_t, axis=-1):
            Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
            ravel_0 = tf.expand_dims(Vc_ravel, 0)
            ravel_0 = tf.cast(ravel_0, tf.float32)
            ravel_1 = tf.expand_dims(Vc_ravel, -1)
            ravel_1 = tf.cast(ravel_1, tf.float32)
            loss_affine += tf.matmul(ravel_0, tf.sparse_tensor_dense_matmul(_M, ravel_1))

    return loss_affine * weight

# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False, gpu=True):
    
    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod] 

    style_features = {}

    batch_shape = (batch_size,256,256,3)
    style_shape = (1,) + style_target.shape

    if gpu == True:
        DEVICES = '/gpu:0'
    else:
        DEVICES = '/cpu:0'

    # precompute style features
    with tf.device(DEVICES), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # placeholder for M
        X_MM = tf.placeholder(tf.float32, name="X_MM")
 
        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

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
        affine_loss = get_affine_loss(preds_pre, batch_size, X_MM, 1e3)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

        # scalar_affine_loss = np.ravel(affine_loss)
        # sc_loss = np.asscalar(scalar_affine_loss)
        # print(style_loss)
        # print(sc_loss)

        loss = content_loss + style_loss + tv_loss + affine_loss


        # summaries for TensorBoard
        with tf.name_scope('loss_summaries'):
            tf.summary.scalar('content_loss', content_loss)
            tf.summary.scalar('style_loss', style_loss)
            tf.summary.scalar('tv_loss', tv_loss)
            tf.summary.scalar('total_loss', loss)
            # tf.summary.scalar('affine_loss', sc_loss)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./logs',
                                      sess.graph)

        

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        batch_laplacian_shape = (batch_size, 1623076)
        M = np.zeros(batch_laplacian_shape, dtype=np.float32)
        X_batch = np.zeros(batch_shape, dtype=np.float32)

        print("Number of examples: %i" % len(content_targets))
        print("Batch size: %i" % batch_size)

        global_step = 0

        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size
                
                for j, img_p in enumerate(content_targets[curr:step]):
                   # print(img_p)
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)
                   _, values, __ = getLaplacianAsThree(X_batch[j] / 255.)
                   M[j] = values
                   
                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                   X_content:X_batch,
                   X_MM: M
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
                    to_get = [style_loss, content_loss, tv_loss, affine_loss, loss, preds]
                    test_feed_dict = {
                       X_content:X_batch,
                       X_MM: M
                    }

                    global_step = (epoch + 1) * iterations
                    # print("Global Step: %i" % global_step)
                    summary, tup = sess.run([merged, to_get], feed_dict = test_feed_dict)
                    summary_writer.add_summary(summary, global_step)
                    _style_loss,_content_loss,_tv_loss, _affine_loss, _loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _affine_loss, _loss)
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
