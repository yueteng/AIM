#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2020/12/10 18:21
@project: MalariaDetection
@description: 
"""
from keras.layers import  Flatten, Dense, BatchNormalization, UpSampling2D
from keras.models import Model
from keras.optimizers import adam

from utils import module


pad = 1
kernel = 3
filter_size = 64
pool_size = 2


def decode_segnet(img_input, f5):

    model = module.dep_conv2(512, f5)
    model = UpSampling2D((pool_size, pool_size))(model)

    model = module.dep_conv2(256, model)
    model = UpSampling2D((pool_size, pool_size))(model)

    model = module.dep_conv2(128, model)
    model = UpSampling2D((pool_size, pool_size))(model)

    model = module.dep_conv2(64, model)
    model = UpSampling2D((pool_size, pool_size))(model)

    model = Flatten()(model)
    model = Dense(512)(model)
    print(f"segnet: {model}")

    o = Dense(1, activation="sigmoid")(model)

    model = Model(img_input, o)
    model.compile(optimizer=adam(lr=0.00001), loss='binary_crossentropy',
                  metrics=['accuracy'])  # define optimizer and loss functions as well as required metrics
    return model


def segnet():
    img_input, [f1,f2, f3, f4, f5] = module.base_encode()
    model = decode_segnet(img_input, f5)
    return model


def vgg_segnet():
    img_input, [f1,f2, f3, f4, f5] = module.vgg_encode()
    model = decode_segnet(img_input, f5)
    return model


def res_segnet():
    img_input, [f1,f2, f3, f4, f5] = module.res50_encode()
    model = decode_segnet(img_input, f5)
    return model