from __future__ import print_function, division
import keras
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, GaussianNoise, multiply, Embedding,Dropout,ZeroPadding2D
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import seaborn as sns;
import tensorflow as tf
from keras import backend as k
from custom_keras_layers import Bilinear_kernel, UpSampling_1Dcubical,clip_layer

sns.set()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

n_classes=2
# Channels, time_step= 1, 64
Channels, time_step= 1, 231
weights = Bilinear_kernel(2)

def eeg_generator():

    model = Sequential()

    model.add(Dense(7680, input_dim=120, name='linear'))
    model.add(LeakyReLU())

    model.add(Dense(128 * 60 * 1))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    if K.image_data_format() == 'channels_first':
        model.add(Reshape((128, 1, 60), input_shape=(128 * 60 * 1,)))
        bn_axis = 1
    else:
        model.add(Reshape((1, 60, 128), input_shape=(128 * 60 * 1,)))
        bn_axis = -1

    model.add(UpSampling_1Dcubical(2))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Convolution2D(64, (1, 3), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, (1, 4), strides=(1, 2),
                              kernel_initializer=keras.initializers.Constant(weights), padding='same'))

    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Convolution2D(1, (1, 3), padding='same', activation='tanh'))

    model.add(clip_layer())

    noise = Input(shape=(120,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(n_classes, 120)(label))

    model_input = multiply([noise, label_embedding])
    eeg = model(model_input)

    model.summary()

    return Model([noise, label], eeg)

    
def eeg_discriminator():

    model = Sequential()

    model.add(Reshape((1, 231, 1), input_shape=(231,)))

    model.add(GaussianNoise(0.05, input_shape=(1, 231, 1)))  # Add this layer to prevent D from overfitting!

    if K.image_data_format() == 'channels_first':
        model.add(Convolution2D(64, (1, 3), padding='same', input_shape=(1, 1, 231)))
    else:
        model.add(Convolution2D(64, (1, 3), padding='same', input_shape=(1, 231, 1)))
    model.add(LeakyReLU())
    
    model.add(Convolution2D(128, (1, 3), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())

    model.add(Convolution2D(128, (1, 3), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())
    
    model.add(Flatten())
    
    model.add(Dense(1024, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    
    model.add(Dense(1, kernel_initializer='he_normal'))

    eeg = Input(shape=(Channels, time_step, 1))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(n_classes, 231)(label))
    flat_eeg = Flatten()(eeg)

    model_input = multiply([flat_eeg, label_embedding])
    validity = model(model_input)

    model.summary()

    return Model([eeg, label], validity)


