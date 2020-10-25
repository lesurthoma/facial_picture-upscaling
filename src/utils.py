import math
from tensorflow.keras import backend as K


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - 
y_true))))

def scaling(input_image):
    input_image = input_image / 255.0
    return input_image
