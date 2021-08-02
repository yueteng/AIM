#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2020/12/10 17:53
@project: MalariaDetection
@description: 
"""

from keras.layers import *      #For adding convolutional layer
from keras.layers import Flatten     #For flattening max pooled layer values into a single vector
from keras.layers import Dense     #For adding layers to NN
from keras.optimizers import adam
from keras.models import Model
from utils import module

pad = 1
kernel = 3
filter_size = 64
pool_size = 2


def decode_unet(f):

    [f1, f2, f3, f4, f5] = f
    model = module.dep_conv2(1024,f5)
    model = module.dep_conv2(512, model)
    model = UpSampling2D((pool_size, pool_size))(model)
    model = Concatenate()([f4, model])

    model = module.dep_conv2(512, model)
    model = module.dep_conv2(256, model)
    model = UpSampling2D((pool_size, pool_size))(model)
    model = Concatenate()([f3, model])

    model = module.dep_conv2(256, model)
    model = module.dep_conv2(128, model)
    model = UpSampling2D((pool_size, pool_size))(model)
    model = Concatenate()([f2, model])

    model = module.dep_conv2(128, model)
    model = module.dep_conv2(64, model)
    model = UpSampling2D((pool_size, pool_size))(model)
    model = Concatenate()([f1, model])

    model = Conv2D(64, (kernel, kernel), padding="valid", activation="relu")(model)
    model = Conv2D(64, (kernel, kernel), padding="valid", activation="relu")(model)

    model = Flatten()(model)
    model = Dense(512, activation="relu")(model)
    print(f"unet: {model}")
    o = Dense(1, activation="sigmoid")(model)
    return o


def unet():
    img_input, f = module.base_encode()
    o = decode_unet(f)
    model = module.model_compile(img_input, o)
    return model


def vgg_unet():
    img_input, f = module.vgg_encode()
    o = decode_unet(f)
    model = module.model_compile(img_input, o)
    return model


def res50_unet():
    img_input, f = module.res50_encode()
    o = decode_unet(f)
    model = module.model_compile(img_input, o)
    return model
