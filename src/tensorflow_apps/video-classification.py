import tqdm
import random
import pathlib

import collections
import numpy as np

import cv2
import keras
import einops

import remotezip as rz
import tensorflow as tf

from keras import layers
from utils import download_ufc_101_subset, FrameGenerator
from utils import *

URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
HEIGHT = 224
WIDTH = 224

download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ufc_101_subset(
    URL,
    num_classes=10,
    splits={"train": 30, "val": 10, "test": 10},
    download_dir=download_dir)

n_frames = 10
batch_size = 8

output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))

train_ds = tf.data.Dataset.from_generator(
    FrameGenerator(subset_paths['train'], n_frames, training=True),
    output_signature=output_signature)

train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(
    FrameGenerator(subset_paths['val'], n_frames),
    output_signature=output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(
    FrameGenerator(subset_paths['test'], n_frames),
    output_signature=output_signature)

test_ds = test_ds.batch(batch_size)


class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()
        self.seq = keras.Sequential([

            layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding),

            layers.Conv3D(filters=filters,
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
        ])

    def call(self, x, **kwargs):
        return self.seq(x)


class ResidualMain(keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x, **kwargs):
        return self.seq(x)


class Project(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x, **kwargs):
        return self.seq(x)


def add_residual_block(_input, filters, kernel_size):
    out = ResidualMain(filters, kernel_size)(_input)
    res = _input
    if out.shape[-1] != _input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video, **kwargs):
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(images, '(b t) h w c -> b t h w c', t=old_shape['t'])
        return videos


input_shape = (None, 10, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

x = add_residual_block(x, 64, (3, 3, 3))
x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

x = add_residual_block(x, 128, (3, 3, 3))

x = layers.GlobalAveragePooling3D()(x)
x = layers.Flatten()(x)
x = layers.Dense(10)(x)

model = keras.Model(input, x)

history = model.fit(x=train_ds, epochs=50, validation_data=val_ds)
result = model.evaluate(test_ds, return_dict=True)
print('video-classification result: ', result)
