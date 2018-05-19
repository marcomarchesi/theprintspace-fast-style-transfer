import scipy.misc, numpy as np, os, sys

import imageio

from PIL import ImageFile, Image, ImageEnhance

ImageFile.LOAD_TRUNCATED_IMAGES = True


def save_conv(out_path, img, filters):
    img_array = np.split(img, indices_or_sections=filters, axis=2)
    
    for i in range(filters):
      path, basename = os.path.split(out_path)
      out_img = img_array[i] 
      out_img = np.squeeze(out_img)
      # print(out_img.shape)
      # print(out_img)
      basename = str(filters) + "_" + str(i) + "_" + basename
      filename = os.path.join(path,basename)
      out = Image.fromarray(out_img, "L")
      out.save(filename)
      # scipy.misc.imsave(basename, out_img)

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = img[0]
    print("saving %s" % out_path)

    imageio.imwrite(out_path, img)

def resize_img(src, size):
    img = scipy.misc.imresize(src, size)
    return img

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

def get_img_from_hdf5(index, hf):
  return hf["train_img"][index]

def get_laplacian_from_hdf5(index, hf):
  return hf["laplacian_img"][index]

def get_img(src, img_size=False):
   img = np.asarray(Image.open(src))
#    img = imageio.imread(src) # misc.imresize(, (256, 256, 3))
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img

def get_bw_img(src):
  image = Image.open(src)
  desaturated_op = ImageEnhance.Color(image)
  desaturated_image = desaturated_op.enhance(0.0)
  img = np.asarray(desaturated_image)
  if not (len(img.shape) == 3 and img.shape[2] == 3):
    img = np.dstack((img,img,img))
  return img


def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files

def num_files(in_path):
    counter = 0
    files = list_files(in_path)
    for filename in files:
      counter += 1
    return counter

def list_abs_files(in_path):
    files = list_files(in_path)
    abs_files = []
    for filename in files:
      abs_files.append(os.path.join(in_path, filename))
    return abs_files

