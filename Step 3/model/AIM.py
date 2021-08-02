#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2020/12/11 16:17
@project: MalariaDetection
@description: 
"""
from keras.models import Sequential    #For initialising the model
from keras.layers import Conv2D, Add, Dropout, DepthwiseConv2D        #For adding convolutional layer
from keras.layers import MaxPooling2D  #For adding max pooling layer
from keras.layers import Flatten       #For flattening max pooled layer values into a single vector
from keras.layers import Dense         #For adding layers to NN
from keras.models import Input, Model
from keras.optimizers import adam

def AIM():
    img_input = Input(shape=(64,64,3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    o = Dense(1000, activation="relu")(x)

    o4 = f4
    o4 = Flatten()(o4)
    o4 = Dense(1000, activation="relu")(o4)
    o = Add()([o, o4])

    o3 = f3
    o3 = Flatten()(o3)
    o3 = Dense(1000, activation="relu")(o3)
    o = Add()([o, o3])
    o = Dense(1000, activation="relu")(o)

    o2 = f2
    o2 = Flatten()(o2)
    o2 = Dense(1000, activation="relu")(o2)
    o = Add()([o, o2])
    o = Dropout(0.5)(o)
    o = Dense(1000, activation="relu")(o)
    o = Dropout(0.5)(o)
    o = Dense(1, activation="sigmoid")(o)


    model = Model(img_input, o)

    model.compile(optimizer=adam(lr=0.00001), loss='binary_crossentropy',
                  metrics=['accuracy'])  # define optimizer and loss functions as well as required metrics


    return model