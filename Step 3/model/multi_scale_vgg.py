#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2020/12/13 15:42
@project: MalariaDetection
@description: 
"""

from keras.layers import *      #For adding convolutional layer
from keras.layers import MaxPooling2D  #For adding max pooling layer
from keras.layers import Flatten, Add       #For flattening max pooled layer values into a single vector
from keras.layers import Dense, BatchNormalization,  Activation        #For adding layers to NN
from keras.optimizers import adam
from keras.models import *                  #for loading the model
from keras import layers
from keras import initializers

from utils import module


def multi_scale_vgg():
    img_input, f = module.vgg_encode()

    reps = [Flatten()(f[0]), Flatten()(f[1]), Flatten()(f[2]), Flatten()(f[3]), Flatten()(f[4])]
    reps = layers.concatenate(reps, axis=-1)
    reps = Dense(512, activation="relu")(reps)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)
    return model


def aggregate_vgg():
    img_input, f = module.vgg_encode()

    reps = [Dense(512, activation="relu")(Flatten()(f[0])),
            Dense(512, activation="relu")(Flatten()(f[1])),
            Dense(512, activation="relu")(Flatten()(f[2])),
            Dense(512, activation="relu")(Flatten()(f[3])),
            Dense(512, activation="relu")(Flatten()(f[4])),]
    reps = Lambda(K.stack, arguments={"axis": 1})(reps)
    reps = Lambda(get_attention_weights, name="representation")(reps)

    print(reps)
    # reps = Dense(512, activation="relu")(reps)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)
    return model


def aggregate_norm_vgg():
    img_input, f = module.vgg_encode()

    reps = [Dense(512, activation="relu")(BatchNormalization()(Flatten()(f[0]))),
            Dense(512, activation="relu")(BatchNormalization()(Flatten()(f[1]))),
            Dense(512, activation="relu")(BatchNormalization()(Flatten()(f[2]))),
            Dense(512, activation="relu")(BatchNormalization()(Flatten()(f[3]))),
            Dense(512, activation="relu")(BatchNormalization()(Flatten()(f[4]))), ]
    reps = Lambda(K.stack, arguments={"axis": 1})(reps)
    reps = Lambda(get_attention_weights, name="representation")(reps)

    print(reps)
    # reps = Dense(512, activation="relu")(reps)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)
    return model


def get_attention_weights(X):
    init = initializers.get('normal')
    input_shape = X.shape
    print(input_shape)
    attention_dim = 512
    W = K.variable(init((512, 512)))
    b = K.variable(init((attention_dim,)))
    u = K.variable(init((attention_dim, 1)))
    uit = K.tanh(K.bias_add(K.dot(X, W), b))
    ait = K.dot(uit, u)
    ait = K.squeeze(ait, -1)
    ait = K.exp(ait)
    ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
    weighted_input = X * K.expand_dims(ait)
    output = K.sum(weighted_input, axis=1)

    output = Dense(1, activation="sigmoid")(output)
    return output
