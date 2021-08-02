#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2020/12/10 17:12
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
pad = 1
kernel = 3
filter_size = 64
pool_size = 2


def dep_conv2(filter, f):
    model = ZeroPadding2D((1,1))(f)
    model = Conv2D(filter, (kernel, kernel), padding="valid", activation="relu")(model)
    model = BatchNormalization()(model)
    return model


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def one_side_pad(x):
    x = ZeroPadding2D((1, 1))(x)
    x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2) and the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters


    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def base_encode():
    img_input = Input(shape=(64, 64, 3))
    # Block 1
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same',name='block1_conv1_1')(img_input)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='block1_conv1_2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    f1 = x

    # Block 2
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same',name='block2_conv2_1')(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='block2_conv2_2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    f2 = x

    # Block 3
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same',name='block3_conv3_1')(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='block3_conv3_2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    f3 = x

    # Block 4
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same',name='block4_conv1')(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    f4 = x

    # Block 5
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Conv2D(1024, (kernel, kernel), activation='relu', padding='same',name='block5_conv')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    f5 = x
    level = [f1, f2, f3, f4, f5]
    return img_input, level


def vgg_encode():
    img_input = Input(shape=(64, 64, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]


def vgg():
    img_input, f = vgg_encode()

    model = Flatten()(f[-1])
    model = Dense(512, activation="relu")(model)
    o = Dense(1, activation="sigmoid")(model)
    model = model_compile(img_input, o)
    return model


def res50_encode():
    img_input = Input(shape=(64, 64, 3))
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    f1 = x

    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]


def res50():
    img_input, f = res50_encode()

    model = Flatten()(f[-1])
    model = Dense(512, activation="relu")(model)
    o = Dense(1, activation="sigmoid")(model)
    model = model_compile(img_input, o)
    return model


def model_compile(img_input, o):
    model = Model(img_input, o)
    model.compile(optimizer=adam(lr=0.00001), loss='binary_crossentropy',
                  metrics=['accuracy'])  # define optimizer and loss functions as well as required metrics
    return model
