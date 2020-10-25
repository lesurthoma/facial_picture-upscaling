import cv2
import numpy as np
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory

import constants
import utils

#Set of variables used for training the model
crop_size = 300
input_size = crop_size // constants.UPSCALE_FACTOR
batch_size = 8
epochs = 100

#Function that creates training dataset and validation dataset
def create_datasets(dataset_dir):

    train_ds = image_dataset_from_directory(
        dataset_dir,
        batch_size=batch_size,
        image_size=(crop_size, crop_size),
        validation_split=0.2,
        subset="training",
        seed=1337,
        label_mode=None,
    )

    valid_ds = image_dataset_from_directory(
        dataset_dir,
        batch_size=batch_size,
        image_size=(crop_size, crop_size),
        validation_split=0.2,
        subset="validation",
        seed=1337,
        label_mode=None,
    )
    return train_ds, valid_ds


#Function that apply treatment to create input images and result images
def preprocess_dataset(dataset):
    preprocessed_dataset = dataset.map(utils.scaling)
    preprocessed_dataset = preprocessed_dataset.map(
        lambda x: (tf.image.resize(x, [input_size, input_size], method="area"), x)
    )
    return preprocessed_dataset

#Creates the image upscaling model
def create_model(channels=3):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(256, 5, **conv_args)(inputs)
    x = layers.Conv2D(128, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (constants.UPSCALE_FACTOR ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, constants.UPSCALE_FACTOR)

    return keras.Model(inputs, outputs)

#Trains the model
def train_model(model, train_ds, valid_ds, epochs=100):
    loss = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=loss, metrics=utils.PSNR)

    model.fit(train_ds, epochs=epochs, validation_data=valid_ds)

#Save the model in a file
def save_model(model, model_name):
    model.save(model_name)

#Launch the training process of the face upscaling model
def run_train():
    if (len(sys.argv) != 2):
        print("help : python src/train.py DATASET_DIRECTORY_PATH")
    else:
        train_ds, valid_ds = create_datasets(sys.argv[1])
        train_ds = preprocess_dataset(train_ds)
        valid_ds = preprocess_dataset(valid_ds)
        model = create_model()
        train_model(model, train_ds, valid_ds, epochs=epochs)
        save_model(model, constants.MODEL_NAME)

run_train()