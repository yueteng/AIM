#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2020/12/10 16:02
@project: MalariaDetection
@description: 
"""

from keras.layers import *      #For adding convolutional layer
from keras.layers import Dense, ZeroPadding2D, BatchNormalization,  Activation        #For adding layers to NN
from keras.optimizers import adam
from keras.models import *                  #for loading the model
from keras import initializers
from keras import backend as K

from utils import module


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
    o = Conv2D(512, (1, 1))(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Flatten()(o)
    o = Dense(512, activation="relu")(o)

    return o


def decode_segnet(img_input, f5):
    pad = 1
    kernel = 3
    filter_size = 64
    pool_size = 2

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

    return model


def decode_unet(f):
    pad = 1
    kernel = 3
    filter_size = 64
    pool_size = 2

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


def att_reps():
    img_input, f = module.base_encode()
    # unet = decode_unet(f)
    pspnet = decode_pspnet(img_input, f[-1])
    segnet = decode_segnet(img_input, f[-1])
    representation = Lambda(K.stack, arguments={"axis": 1})([ pspnet, segnet])
    # representation = K.stack([unet, pspnet, segnet], axis=1)
    # new_reps = get_attention_weights(representation)
    new_reps = Lambda(get_attention_weights, name="representation")(representation)
    print(new_reps)

    model = Model(input=img_input, output=new_reps)

    model.compile(optimizer=adam(lr=0.00001), loss='binary_crossentropy',
                  metrics=['accuracy'])  # define optimizer and loss functions as well as required metrics
    # model = module.model_compile(img_input, new_reps)
    return model

