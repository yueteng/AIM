#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2020/12/10 18:23
@project: MalariaDetection
@description: 
"""
from keras.layers import *      #For adding convolutional layer
from keras.layers import Dense, ZeroPadding2D, BatchNormalization,  Activation        #For adding layers to NN
from keras.optimizers import adam
from keras.models import *                  #for loading the model
from utils import module as base_model


def pool_block(feats, pool_factor):

    pool_size = strides = [int(64/pool_factor),
                           int(64/pool_factor)]
    x = AveragePooling2D(pool_size, strides=strides, padding='same')(feats)
    x = Conv2D(512, (1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def decode_pspnet(img_input, o):
    pool_factors = [1, 2, 3, 6]
    pool_outs = []

    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    o = Concatenate()(pool_outs)
    o = UpSampling2D((4, 4))(o)
    o = Concatenate()([o, o])
    o = Conv2D(512, (1, 1), name='decode_pspnet_conv1')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Flatten()(o)
    o = Dense(512, activation="relu")(o)
    print(f"pspnet: {o}")

    o = Dense(1, activation="sigmoid")(o)

    model = Model(img_input, o)
    model.compile(optimizer=adam(lr=0.00001), loss='binary_crossentropy',
                  metrics=['accuracy'])  # define optimizer and loss functions as well as required metrics
    return model


def pspnet():
    img_input, [f1, f2, f3, f4, f5] = base_model.base_encode()
    o = f5
    model = decode_pspnet(img_input, o)

    return model


def vgg_pspnet():
    img_input, [f1, f2, f3, f4, f5] = base_model.vgg_encode()
    o = f5
    model = decode_pspnet(img_input, o)

    return model


def res_pspnet():
    img_input, [f1, f2, f3, f4, f5] = base_model.res50_encode()
    o = f5
    model = decode_pspnet(img_input, o)

    return model




