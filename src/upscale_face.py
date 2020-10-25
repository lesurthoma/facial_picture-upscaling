import cv2
from tensorflow.keras.models import load_model
import sys
import numpy as np

import constants
import utils

def upscale_image(model, img):
    img_scaled = utils.scaling(img)
    input = np.expand_dims(img_scaled, axis=0)
    out = model.predict(input)

    out_img = out[0] * 255
    out_img.clip(0, 255)
    return out_img

def run():
    if (len(sys.argv) != 3):
        print("help : python src/upscale_face.py [src_image] [dest_file]")
    else:
        img = cv2.imread(sys.argv[1])
        if not img is None:
            try:
                upscale_model = load_model(constants.MODEL_NAME, custom_objects={"PSNR" : utils.PSNR})
                upscaled_image = upscale_image(upscale_model,img)
                cv2.imwrite(sys.argv[2], upscaled_image)
            except:
                print("Trained model not found. Please train the model first")
        else:
            print("Source image doesn't not exist")

run()