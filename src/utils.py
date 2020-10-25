import math
from tensorflow.keras import backend as K

#Metrics used to evaluate the upscaling model
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - 
y_true))))

#A function that put all pixel values in an image between 0 and 1
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image
