import struct
import numpy as np

file_path = './data/minst_images'

def mnist_load():
    fp = open(file_path, 'rb')
    data = fp.read()
    fp.close()
    
    fmt = '>iiii' # big-endian, 4 integers at head

    magic, images_num, rows, cols = struct.unpack_from(fmt, data)
    image_size = rows * cols
    data_size = images_num * image_size

    mat_data = struct.unpack_from('>' + str(data_size) + 'B', data, struct.calcsize(fmt))
    mat_data = np.reshape(mat_data, [images_num, image_size]).astype(np.uint8) # target image is a grat image

    print("Load image from path: %s, images_num: %d, rows: %d, cols: %d, dara_shape: %s" % (file_path, images_num, rows, cols, mat_data.shape))
    return mat_data

def mnist_show_test():
    import cv2

    data = mnist_load()

    cv2.imshow('Image', data[0].reshape(28, 28))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

