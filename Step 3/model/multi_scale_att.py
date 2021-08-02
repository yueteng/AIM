#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2021/1/9 15:28
@project: MalariaDetection
@description: 
"""
from keras.layers import *      #For adding convolutional layer
from keras.layers import Dense, ZeroPadding2D, BatchNormalization,  Activation        #For adding layers to NN
from keras.optimizers import adam
from keras.models import *                  #for loading the model
from keras.backend import stack, squeeze, expand_dims
from utils import module
from keras import backend as K
from utils.attention import AttentionLayer
import tensorflow as tf
import keras

def multi_scale_att():
    size = 64
    down_sampling_4 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down4'),
                                  BatchNormalization(), PReLU()])
    down_sampling_3 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down3'),
                                  BatchNormalization(), PReLU()])
    down_sampling_2 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down2'),
                                  BatchNormalization(), PReLU()])
    down_sampling_1 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down1'),
                                  BatchNormalization(), PReLU()])
    # img_input, scale_outputs = module.base_encode()
    img_input, scale_outputs = module.vgg_encode()

    down4 = down_sampling_4(scale_outputs[4])
    down3 = down_sampling_3(scale_outputs[3])
    down2 = down_sampling_2(scale_outputs[2])
    down1 = down_sampling_1(scale_outputs[1])

    upsampling4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(down4)
    upsampling3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(down3)            # bilinear
    upsampling2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(down2)
    upsampling1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(down1)
    Flatten1 = Flatten()(scale_outputs[0])
    Flatten2 = Flatten()(upsampling1)
    Flatten3 = Flatten()(upsampling2)
    Flatten4 = Flatten()(upsampling3)
    Flatten5 = Flatten()(upsampling4)

    hidden_states = [Flatten1, Flatten2, Flatten3, Flatten4, Flatten5]  #

    hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    hidden_size = 32 * 32 * size
    hidden_states = Reshape([5, hidden_size])(hidden_states)
    #hidden_states = keras.layers.Lambda(reshapes)(hidden_states)
    output = AttentionLayer(512, name='attention')(hidden_states)           # 可以调整512为[64, 128, 256, 512, 1024, 2048]

    reps = Dense(512, activation="relu")(output)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # reps = output   #  Dense(512, activation="relu")(output)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)

    return model

def reshapes(hidden_states):
    size = 64
    hidden_size = 32 * 32 * size
    hidden_states = tf.reshape(hidden_states, [-1,5, hidden_size])
    return hidden_states

def multi_scale_han():
    size = 64
    down_sampling_4 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down4'),
                                  BatchNormalization(), PReLU()])
    down_sampling_3 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down3'),
                                  BatchNormalization(), PReLU()])
    down_sampling_2 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down2'),
                                  BatchNormalization(), PReLU()])
    down_sampling_1 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down1'),
                                  BatchNormalization(), PReLU()])
    # img_input, scale_outputs = module.base_encode()
    img_input, scale_outputs = module.vgg_encode()

    down4 = down_sampling_4(scale_outputs[4])
    down3 = down_sampling_3(scale_outputs[3])
    down2 = down_sampling_2(scale_outputs[2])
    down1 = down_sampling_1(scale_outputs[1])

    upsampling4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(down4)
    upsampling3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(down3)            # bilinear
    upsampling2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(down2)
    upsampling1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(down1)

    hidden_states = [Flatten()(scale_outputs[0]), Flatten()(upsampling1), Flatten()(upsampling2),
                     Flatten()(upsampling3), Flatten()(upsampling4)]  #
    hidden_states = concatenate([Flatten()(scale_outputs[0]), Flatten()(upsampling1),
                                 Flatten()(upsampling2), Flatten()(upsampling3), Flatten()(upsampling4)])
    # hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    # hidden_size = 32 * 32 * size
    # hidden_states = Reshape([5, hidden_size])(hidden_states)
    hidden_states = keras.layers.Lambda(reshapes)(hidden_states)
    output = AttentionLayer(512, name='attention')(hidden_states)

    reps = Reshape([32, 32, size])(output)   #  Dense(512, activation="relu")(output)
    filter_dim = 256                   # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    output = Conv2D(filter_dim, kernel_size=(3, 3), activation='relu', padding='same', name='conv_han')(reps)
    reps = Reshape([32*32, filter_dim])(output)
    reps = AttentionLayer(filter_dim, name='attention2')(reps)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)
    return model


def multi_scale_att1():
    hidden_size = 512
    img_input, [f1, f2, f3, f4, f5] = module.base_encode()

    f1_att = Dense(hidden_size, activation=None)(Flatten()(f1))
    f2_att = Dense(hidden_size, activation=None)(Flatten()(f2))
    f3_att = Dense(hidden_size, activation=None)(Flatten()(f3))
    f4_att = Dense(hidden_size, activation=None)(Flatten()(f4))
    f5_att = Dense(hidden_size, activation=None)(Flatten()(f5))

    reps = Lambda(lambda x: stack(x, axis=1))([f1_att, f2_att, f3_att, f4_att, f5_att])
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(reps)

    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(reps)
    score = Lambda(lambda x: dot(x, [2, 1]))([score_first_part, h_t])
    attention_weights = Activation('sigmoid', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    # context_vector = dot([reps, attention_weights], [1, 1], name='context_vector')
    context_vector = Lambda(lambda x: dot(x, [1, 1]))([reps, attention_weights])
    # pre_activation = concatenate([context_vector, h_t], name='attention_output')
    # attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)

    output = Dense(1, activation="sigmoid")(context_vector)
    model = module.model_compile(img_input, output)
    return model


