import numpy as np

def inverse_mat_4x4(m_in):
  m = np.zeros([16])
  inv = np.zeros([16])

  inv_out = [4][4]

  for i in range(4):
    for j in range(4):
      m[i * 4 + j] = m_in[i][j]

  inv[0] = m[5]  * m[10] * m[15] -  \
           m[5]  * m[11] * m[14] - \
            m[9]  * m[6]  * m[15] + \
             m[9]  * m[7]  * m[14] + \
             m[13] * m[6]  * m[11] - \
             m[13] * m[7]  * m[10]
  inv[4] = -m[4]  * m[10] * m[15] + \
    m[4]  * m[11] * m[14] + \
    m[8]  * m[6]  * m[15] - \
    m[8]  * m[7]  * m[14] - \
    m[12] * m[6]  * m[11] + \
    m[12] * m[7]  * m[10]
  inv[8] = m[4]  * m[9] * m[15] -	\
             m[4]  * m[11] * m[13] -	\
             m[8]  * m[5] * m[15] +	\
             m[8]  * m[7] * m[13] +	\
             m[12] * m[5] * m[11] -	\
             m[12] * m[7] * m[9]
  inv[12] = -m[4]  * m[9] * m[14] +	\
               m[4]  * m[10] * m[13] +	\
               m[8]  * m[5] * m[14] -	\
               m[8]  * m[6] * m[13] -	\
               m[12] * m[5] * m[10] +	\
               m[12] * m[6] * m[9]
  inv[1] = -m[1]  * m[10] * m[15] +	\
              m[1]  * m[11] * m[14] +	\
              m[9]  * m[2] * m[15] -	\
              m[9]  * m[3] * m[14] -	\
              m[13] * m[2] * m[11] +	\
              m[13] * m[3] * m[10]
  inv[5] = m[0]  * m[10] * m[15] -	\
             m[0]  * m[11] * m[14] -	\
             m[8]  * m[2] * m[15] +	\
             m[8]  * m[3] * m[14] +	\
             m[12] * m[2] * m[11] -	\
             m[12] * m[3] * m[10]
  inv[9] = -m[0]  * m[9] * m[15] +	\
              m[0]  * m[11] * m[13] +	\
              m[8]  * m[1] * m[15] -	\
              m[8]  * m[3] * m[13] -	\
              m[12] * m[1] * m[11] +	\
              m[12] * m[3] * m[9]
  inv[13] = m[0]  * m[9] * m[14] -	\
              m[0]  * m[10] * m[13] -	\
              m[8]  * m[1] * m[14] +	\
              m[8]  * m[2] * m[13] +	\
              m[12] * m[1] * m[10] -	\
              m[12] * m[2] * m[9]
  inv[2] = m[1]  * m[6] * m[15] -	\
             m[1]  * m[7] * m[14] -	\
             m[5]  * m[2] * m[15] +	\
             m[5]  * m[3] * m[14] +	\
             m[13] * m[2] * m[7] -	\
             m[13] * m[3] * m[6]
  inv[6] = -m[0]  * m[6] * m[15] +	\
              m[0]  * m[7] * m[14] +	\
              m[4]  * m[2] * m[15] -	\
              m[4]  * m[3] * m[14] -	\
              m[12] * m[2] * m[7] +	\
              m[12] * m[3] * m[6]
  inv[10] = m[0]  * m[5] * m[15] -	\
              m[0]  * m[7] * m[13] -	\
              m[4]  * m[1] * m[15] +	\
              m[4]  * m[3] * m[13] +	\
              m[12] * m[1] * m[7] -	\
              m[12] * m[3] * m[5]
  inv[14] = -m[0]  * m[5] * m[14] +	\
               m[0]  * m[6] * m[13] +	\
               m[4]  * m[1] * m[14] -	\
               m[4]  * m[2] * m[13] -	\
               m[12] * m[1] * m[6] +	\
               m[12] * m[2] * m[5]
  inv[3] = -m[1] * m[6] * m[11] +	\
              m[1] * m[7] * m[10] +	\
              m[5] * m[2] * m[11] -	\
              m[5] * m[3] * m[10] -	\
              m[9] * m[2] * m[7] +	\
              m[9] * m[3] * m[6]
  inv[7] = m[0] * m[6] * m[11] -	\
             m[0] * m[7] * m[10] -	\
             m[4] * m[2] * m[11] +	\
             m[4] * m[3] * m[10] +	\
             m[8] * m[2] * m[7] -	\
             m[8] * m[3] * m[6]
  inv[11] = -m[0] * m[5] * m[11] +	\
               m[0] * m[7] * m[9] +	\
               m[4] * m[1] * m[11] -	\
               m[4] * m[3] * m[9] -	\
               m[8] * m[1] * m[7] +	\
               m[8] * m[3] * m[5]
  inv[15] = m[0] * m[5] * m[10] -	\
              m[0] * m[6] * m[9] -	\
              m[4] * m[1] * m[10] +	\
              m[4] * m[2] * m[9] +	\
              m[8] * m[1] * m[6] -	\
              m[8] * m[2] * m[5]


  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12]

  if abs(det) < 1e-9:
    return False

  det = 1.0 / det

  for i in range(4):
    for j in range(4):
      inv_out[i][j] = inv[i * 4 + j] * det

  return True, inv_out

def best_local_affine(output, input, h, w, epsilon, radius):

	# block, grid, thread

  
  tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1


	size = h * w
	id = block_idx.x * block_dim.x + thread_idx.x

	if id < size:
		x = id % w
		y = id / w
		






