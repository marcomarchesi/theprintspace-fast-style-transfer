# Batch Laplacian test

from closed_form_matting import getlaplacian1
import numpy as np
import tensorflow as tf

BATCH_SIZE = 2

def getBatchLaplacian(img, batch_size):
    laplacian = []
    for i in range(batch_size):
        item = img[i,:,:,:]
        h, w, _ = item.shape
        coo = getlaplacian1(item, np.zeros(shape=(h, w)), 1e-5, 1).tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        data = [indices, coo.data, coo.shape]
        laplacian.append(data)
    return laplacian

def main():
    M = []
    X_batch = np.zeros((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
    X_shape = np.squeeze(X_batch)
    # indices, data, shape = getLaplacianAsThree(X_shape / 255.)
    laplacian = getBatchLaplacian(X_shape / 255., BATCH_SIZE)
    indices, data, shape = laplacian[0]
    # print(indices.shape)
    M_X = tf.SparseTensor(indices, data, shape)

if __name__ == '__main__':
    main()