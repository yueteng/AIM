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
from model.transformer4 import LayerNormalization, TokenAndPositionEmbedding
from keras_transformer.attention import MultiHeadSelfAttention, MultiHeadAttention


def multi_scale_transformer4():
    size = 64
    rate = 0.1
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

    down4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(down_sampling_4(scale_outputs[4]))
    down3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(down_sampling_3(scale_outputs[3]))
    down2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(down_sampling_2(scale_outputs[2]))
    down1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(down_sampling_1(scale_outputs[1]))
    down0 = scale_outputs[0]

    # Transformer
    embed_dim = 64  # Embedding size for each token
    num_heads = 8  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    rs_down4 = Reshape([32*32, embed_dim])(down4)
    rs_down3 = Reshape([32*32, embed_dim])(down3)
    rs_down2 = Reshape([32*32, embed_dim])(down2)
    rs_down1 = Reshape([32*32, embed_dim])(down1)
    rs_down0 = Reshape([32*32, embed_dim])(down0)

    # down3 = TokenAndPositionEmbedding(4*4, embed_dim)(rs_down3)
    ts_att_down3 = MultiHeadAttention(num_heads, use_masking=False)([rs_down3, rs_down4])
    ts_down3 = LayerNormalization()(Add()([rs_down3, Dropout(rate)(ts_att_down3)]))
    ts_down3 = LayerNormalization()(Add()([ts_down3, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down3))]))

    # down2 = TokenAndPositionEmbedding(8*8, embed_dim)(down2)
    ts_att_down2 = MultiHeadAttention(num_heads, use_masking=False)([rs_down2, rs_down3])
    ts_down2 = LayerNormalization()(Add()([rs_down2, Dropout(rate)(ts_att_down2)]))
    ts_down2 = LayerNormalization()(Add()([ts_down2, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down2))]))

    # down1 = TokenAndPositionEmbedding(16*16, embed_dim)(down1)
    ts_att_down1 = MultiHeadAttention(num_heads, use_masking=False)([rs_down1, rs_down2])
    ts_down1 = LayerNormalization()(Add()([rs_down1, Dropout(rate)(ts_att_down1)]))
    ts_down1 = LayerNormalization()(Add()([ts_down1, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down1))]))

    # down0 = TokenAndPositionEmbedding(32*32, embed_dim)(down0)
    ts_att_down0 = MultiHeadAttention(num_heads, use_masking=False)([rs_down0, rs_down1])
    ts_down0 = LayerNormalization()(Add()([rs_down0, Dropout(rate)(ts_att_down0)]))
    ts_down0 = LayerNormalization()(Add()([ts_down0, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down0))]))

    hidden_states = [Flatten()(ts_down0), Flatten()(ts_down1), Flatten()(ts_down2),
                     Flatten()(ts_down3), Flatten()(rs_down4)]  #
    # hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    # hidden_size = 32 * 32 * size
    # hidden_states = Reshape([5, hidden_size])(hidden_states)
    # output = AttentionLayer(512, name='attention')(hidden_states)           # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    output = Concatenate(axis=-1)(hidden_states)

    reps = Dense(512, activation="relu")(output)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # reps = output   #  Dense(512, activation="relu")(output)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)

    return model

def AIM_without_local_context_aligner():
    size = 64
    rate = 0.1
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

    down4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(down_sampling_4(scale_outputs[4]))
    down3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(down_sampling_3(scale_outputs[3]))
    down2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(down_sampling_2(scale_outputs[2]))
    down1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(down_sampling_1(scale_outputs[1]))
    down0 = scale_outputs[0]

    # Transformer
    embed_dim = 64  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer
    print(f"embed_dim: {embed_dim}, num_heads: {num_heads}, ff_dim: {ff_dim}")

    rs_down0 = Reshape([32*32, embed_dim])(down0)
    rs_down4 = Reshape([32*32, embed_dim])(down4)
    rs_down3 = Reshape([32*32, embed_dim])(down3)
    rs_down2 = Reshape([32*32, embed_dim])(down2)
    rs_down1 = Reshape([32*32, embed_dim])(down1)


    if 1:
        print(f"+++++++++++++++++++ Position ++++++++++++++++++++++++++++++++")
        half_embed_dim = int(embed_dim/2)
        rs_down0 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down0)
        rs_down4 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down4)
        rs_down3 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down3)
        rs_down2 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down2)
        rs_down1 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down1)

    hidden_states = [Flatten()(rs_down0), Flatten()(rs_down1), Flatten()(rs_down2),
                     Flatten()(rs_down3), Flatten()(rs_down4)]  #
    hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    hidden_size = 32 * 32 * size
    hidden_states = Reshape([5, hidden_size])(hidden_states)

    output = AttentionLayer(512, name='attention')(hidden_states)           # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    #

    reps = Dense(256, activation="relu")(output)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # reps = output   #  Dense(512, activation="relu")(output)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)

    return model


def AIM_without_multi_scale_attention():
    size = 64
    rate = 0.1
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

    down4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(down_sampling_4(scale_outputs[4]))
    down3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(down_sampling_3(scale_outputs[3]))
    down2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(down_sampling_2(scale_outputs[2]))
    down1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(down_sampling_1(scale_outputs[1]))
    down0 = scale_outputs[0]

    # Transformer
    embed_dim = 64  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer
    print(f"embed_dim: {embed_dim}, num_heads: {num_heads}, ff_dim: {ff_dim}")

    rs_down0 = Reshape([32*32, embed_dim])(down0)
    rs_down4 = Reshape([32*32, embed_dim])(down4)
    rs_down3 = Reshape([32*32, embed_dim])(down3)
    rs_down2 = Reshape([32*32, embed_dim])(down2)
    rs_down1 = Reshape([32*32, embed_dim])(down1)


    if 1:
        print(f"+++++++++++++++++++ Position ++++++++++++++++++++++++++++++++")
        half_embed_dim = int(embed_dim/2)
        rs_down0 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down0)
        rs_down4 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down4)
        rs_down3 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down3)
        rs_down2 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down2)
        rs_down1 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down1)


    ts_att_down3 = MultiHeadAttention(num_heads, use_masking=False)([rs_down3, rs_down4])
    ts_down3 = LayerNormalization()(Add()([rs_down3, Dropout(rate)(ts_att_down3)]))
    ts_down3 = LayerNormalization()(Add()([ts_down3, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down3))]))

    ts_att_down2 = MultiHeadAttention(num_heads, use_masking=False)([rs_down2, rs_down3])
    ts_down2 = LayerNormalization()(Add()([rs_down2, Dropout(rate)(ts_att_down2)]))
    ts_down2 = LayerNormalization()(Add()([ts_down2, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down2))]))

    ts_att_down1 = MultiHeadAttention(num_heads, use_masking=False)([rs_down1, rs_down2])
    ts_down1 = LayerNormalization()(Add()([rs_down1, Dropout(rate)(ts_att_down1)]))
    ts_down1 = LayerNormalization()(Add()([ts_down1, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down1))]))

    ts_att_down0 = MultiHeadAttention(num_heads, use_masking=False)([rs_down0, rs_down1])
    ts_down0 = LayerNormalization()(Add()([rs_down0, Dropout(rate)(ts_att_down0)]))
    ts_down0 = LayerNormalization()(Add()([ts_down0, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down0))]))


    hidden_states = concatenate([Flatten()(rs_down0), Flatten()(rs_down1), Flatten()(rs_down2),
                     Flatten()(rs_down3), Flatten()(rs_down4)])  #
    reps = Dense(256, activation="relu")(hidden_states)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # reps = output   #  Dense(512, activation="relu")(output)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)

    return model


def multi_scale_transformer():
    size = 64
    rate = 0.1
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

    down4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(down_sampling_4(scale_outputs[4]))
    down3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(down_sampling_3(scale_outputs[3]))
    down2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(down_sampling_2(scale_outputs[2]))
    down1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(down_sampling_1(scale_outputs[1]))
    down0 = scale_outputs[0]

    # Transformer
    embed_dim = 64  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer
    print(f"embed_dim: {embed_dim}, num_heads: {num_heads}, ff_dim: {ff_dim}")

    rs_down0 = Reshape([32*32, embed_dim])(down0)
    rs_down4 = Reshape([32*32, embed_dim])(down4)
    rs_down3 = Reshape([32*32, embed_dim])(down3)
    rs_down2 = Reshape([32*32, embed_dim])(down2)
    rs_down1 = Reshape([32*32, embed_dim])(down1)


    if 1:
        print(f"+++++++++++++++++++ Position ++++++++++++++++++++++++++++++++")
        half_embed_dim = int(embed_dim/2)
        rs_down0 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down0)
        rs_down4 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down4)
        rs_down3 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down3)
        rs_down2 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down2)
        rs_down1 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down1)


    ts_att_down3 = MultiHeadAttention(num_heads, use_masking=False)([rs_down3, rs_down4])
    ts_down3 = LayerNormalization()(Add()([rs_down3, Dropout(rate)(ts_att_down3)]))
    ts_down3 = LayerNormalization()(Add()([ts_down3, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down3))]))

    ts_att_down2 = MultiHeadAttention(num_heads, use_masking=False)([rs_down2, rs_down3])
    ts_down2 = LayerNormalization()(Add()([rs_down2, Dropout(rate)(ts_att_down2)]))
    ts_down2 = LayerNormalization()(Add()([ts_down2, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down2))]))

    ts_att_down1 = MultiHeadAttention(num_heads, use_masking=False)([rs_down1, rs_down2])
    ts_down1 = LayerNormalization()(Add()([rs_down1, Dropout(rate)(ts_att_down1)]))
    ts_down1 = LayerNormalization()(Add()([ts_down1, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down1))]))

    ts_att_down0 = MultiHeadAttention(num_heads, use_masking=False)([rs_down0, rs_down1])
    ts_down0 = LayerNormalization()(Add()([rs_down0, Dropout(rate)(ts_att_down0)]))
    ts_down0 = LayerNormalization()(Add()([ts_down0, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down0))]))

    hidden_states = [Flatten()(ts_down0), Flatten()(ts_down1), Flatten()(ts_down2),
                     Flatten()(ts_down3), Flatten()(rs_down4)]  #
    hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    hidden_size = 32 * 32 * size
    hidden_states = Reshape([5, hidden_size])(hidden_states)
    output = AttentionLayer(512, name='attention')(hidden_states)           # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # output = concatenate([Flatten()(ts_down0), Flatten()(ts_down1), Flatten()(ts_down2),
    #                  Flatten()(ts_down3), Flatten()(rs_down4)])  #

    reps = Dense(256, activation="relu")(output)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)

    return model

def multi_scale_transformer2():
    size = 64
    rate = 0.1
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

    down4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(down_sampling_4(scale_outputs[4]))
    down3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(down_sampling_3(scale_outputs[3]))
    down2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(down_sampling_2(scale_outputs[2]))
    down1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(down_sampling_1(scale_outputs[1]))
    down0 = scale_outputs[0]

    # Transformer
    embed_dim = 64  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer
    print(f"embed_dim: {embed_dim}, num_heads: {num_heads}, ff_dim: {ff_dim}")

    rs_down0 = Reshape([32*32, embed_dim])(down0)
    rs_down4 = Reshape([32*32, embed_dim])(down4)
    rs_down3 = Reshape([32*32, embed_dim])(down3)
    rs_down2 = Reshape([32*32, embed_dim])(down2)
    rs_down1 = Reshape([32*32, embed_dim])(down1)


    if 1:
        print(f"+++++++++++++++++++ Position ++++++++++++++++++++++++++++++++")
        half_embed_dim = int(embed_dim/2)
        rs_down0 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down0)
        rs_down4 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down4)
        rs_down3 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down3)
        rs_down2 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down2)
        rs_down1 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down1)


    ts_att_down3 = MultiHeadAttention(num_heads, use_masking=False)([rs_down3, rs_down4])
    ts_down3 = LayerNormalization()(Add()([rs_down3, Dropout(rate)(ts_att_down3)]))
    ts_down3 = LayerNormalization()(Add()([ts_down3, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down3))]))

    ts_att_down2 = MultiHeadAttention(num_heads, use_masking=False)([rs_down2, rs_down3])
    ts_down2 = LayerNormalization()(Add()([rs_down2, Dropout(rate)(ts_att_down2)]))
    ts_down2 = LayerNormalization()(Add()([ts_down2, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down2))]))

    ts_att_down1 = MultiHeadAttention(num_heads, use_masking=False)([rs_down1, rs_down2])
    ts_down1 = LayerNormalization()(Add()([rs_down1, Dropout(rate)(ts_att_down1)]))
    ts_down1 = LayerNormalization()(Add()([ts_down1, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down1))]))

    ts_att_down0 = MultiHeadAttention(num_heads, use_masking=False)([rs_down0, rs_down1])
    ts_down0 = LayerNormalization()(Add()([rs_down0, Dropout(rate)(ts_att_down0)]))
    ts_down0 = LayerNormalization()(Add()([ts_down0, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down0))]))

    hidden_states = [Flatten()(ts_down0), Flatten()(ts_down1), Flatten()(ts_down2),
                     Flatten()(ts_down3), Flatten()(rs_down4)]  #
    hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    hidden_size = 32 * 32 * size
    hidden_states = Reshape([5, hidden_size])(hidden_states)
    output = AttentionLayer(512, name='attention')(hidden_states)           # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # output = Flatten(hidden_states)
    reps = Dense(256, activation="relu")(output)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # reps = output   #  Dense(512, activation="relu")(output)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)

    return model



def multi_scale_transformer_1():
    size = 64
    rate = 0.1
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

    # Transformer
    embed_dim = 64  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    rs_down4 = Reshape([2*2, embed_dim])(down4)
    rs_down3 = Reshape([4*4, embed_dim])(down3)
    rs_down2 = Reshape([8*8, embed_dim])(down2)
    rs_down1 = Reshape([16*16, embed_dim])(down1)
    rs_down0 = Reshape([32*32, embed_dim])(scale_outputs[0])

    # down3 = TokenAndPositionEmbedding(4*4, embed_dim)(rs_down3)
    ts_att_down3 = MultiHeadAttention(num_heads, use_masking=False)([rs_down4, rs_down3])
    ts_down3 = LayerNormalization()(Add()([rs_down3, Dropout(rate)(ts_att_down3)]))
    ts_down3 = LayerNormalization()(Add()([ts_down3, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down3))]))
    ts_down3 = Reshape([4, 4, embed_dim])(ts_down3)

    # down2 = TokenAndPositionEmbedding(8*8, embed_dim)(down2)
    ts_att_down2 = MultiHeadAttention(num_heads, use_masking=False)([rs_down3, rs_down2])
    ts_down2 = LayerNormalization()(Add()([rs_down2, Dropout(rate)(ts_att_down2)]))
    ts_down2 = LayerNormalization()(Add()([ts_down2, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down2))]))
    ts_down2 = Reshape([8, 8, embed_dim])(ts_down2)

    # down1 = TokenAndPositionEmbedding(16*16, embed_dim)(down1)
    ts_att_down1 = MultiHeadAttention(num_heads, use_masking=False)([rs_down2, rs_down1])
    ts_down1 = LayerNormalization()(Add()([rs_down1, Dropout(rate)(ts_att_down1)]))
    ts_down1 = LayerNormalization()(Add()([ts_down1, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down1))]))
    ts_down1 = Reshape([16, 16, embed_dim])(ts_down1)

    # down0 = TokenAndPositionEmbedding(32*32, embed_dim)(down0)
    ts_att_down0 = MultiHeadAttention(num_heads, use_masking=False)([rs_down1, rs_down0])
    ts_down0 = LayerNormalization()(Add()([rs_down0, Dropout(rate)(ts_att_down0)]))
    ts_down0 = LayerNormalization()(Add()([ts_down0, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down0))]))
    ts_down0 = Reshape([32, 32, embed_dim])(ts_down0)

    upsampling4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(down4)
    upsampling3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(ts_down3)            # bilinear
    upsampling2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(ts_down2)
    upsampling1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(ts_down1)

    hidden_states = [Flatten()(ts_down0), Flatten()(upsampling1), Flatten()(upsampling2),
                     Flatten()(upsampling3), Flatten()(upsampling4)]  #
    hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    hidden_size = 32 * 32 * size
    hidden_states = Reshape([5, hidden_size])(hidden_states)
    output = AttentionLayer(512, name='attention')(hidden_states)           # 可以调整512为[64, 128, 256, 512, 1024, 2048]

    reps = Dense(512, activation="relu")(output)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # reps = output   #  Dense(512, activation="relu")(output)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)

    return model


def multi_scale_transformer2():
    size = 64
    rate = 0.1
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

    # Transformer
    embed_dim = 64  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    down4 = Reshape([2*2, embed_dim])(down4)
    # down4 = TokenAndPositionEmbedding(2*2, embed_dim)(down4)
    att_down4 = MultiHeadSelfAttention(num_heads, use_masking=False)(down4)
    down4 = LayerNormalization()(Add()([down4, Dropout(rate)(att_down4)]))
    down4 = LayerNormalization()(Add()([down4, Dropout(rate)(Dense(embed_dim, activation="relu")(down4))]))
    down4 = Reshape([2, 2, embed_dim])(down4)

    down3 = Reshape([4*4, embed_dim])(down3)
    # down3 = TokenAndPositionEmbedding(4*4, embed_dim)(down3)
    att_down3 = MultiHeadSelfAttention(num_heads, use_masking=False)(down3)
    down3 = LayerNormalization()(Add()([down3, Dropout(rate)(att_down3)]))
    down3 = LayerNormalization()(Add()([down3, Dropout(rate)(Dense(embed_dim, activation="relu")(down3))]))
    down3 = Reshape([4, 4, embed_dim])(down3)

    down2 = Reshape([8*8, embed_dim])(down2)
    # down2 = TokenAndPositionEmbedding(8*8, embed_dim)(down2)
    att_down2 = MultiHeadSelfAttention(num_heads, use_masking=False)(down2)
    down2 = LayerNormalization()(Add()([down2, Dropout(rate)(att_down2)]))
    down2 = LayerNormalization()(Add()([down2, Dropout(rate)(Dense(embed_dim, activation="relu")(down2))]))
    down2 = Reshape([8, 8, embed_dim])(down2)

    down1 = Reshape([16*16, embed_dim])(down1)
    # down1 = TokenAndPositionEmbedding(16*16, embed_dim)(down1)
    att_down1 = MultiHeadSelfAttention(num_heads, use_masking=False)(down1)
    down1 = LayerNormalization()(Add()([down1, Dropout(rate)(att_down1)]))
    down1 = LayerNormalization()(Add()([down1, Dropout(rate)(Dense(embed_dim, activation="relu")(down1))]))
    down1 = Reshape([16, 16, embed_dim])(down1)

    down0 = Reshape([32*32, embed_dim])(scale_outputs[0])
    # down0 = TokenAndPositionEmbedding(32*32, embed_dim)(down0)
    att_down0 = MultiHeadSelfAttention(num_heads, use_masking=False)(down0)
    down0 = LayerNormalization()(Add()([down0, Dropout(rate)(att_down0)]))
    down0 = LayerNormalization()(Add()([down0, Dropout(rate)(Dense(embed_dim, activation="relu")(down0))]))
    down0 = Reshape([32, 32, embed_dim])(down0)

    upsampling4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(down4)
    upsampling3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(down3)            # bilinear
    upsampling2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(down2)
    upsampling1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(down1)

    hidden_states = [Flatten()(down0), Flatten()(upsampling1), Flatten()(upsampling2),
                     Flatten()(upsampling3), Flatten()(upsampling4)]  #
    hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    hidden_size = 32 * 32 * size
    hidden_states = Reshape([5, hidden_size])(hidden_states)
    output = AttentionLayer(512, name='attention')(hidden_states)           # 可以调整512为[64, 128, 256, 512, 1024, 2048]

    reps = Dense(512, activation="relu")(output)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # reps = output   #  Dense(512, activation="relu")(output)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)

    return model


def multi_scale_transformer0():
    size = 64
    rate = 0.1
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer
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
    hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    hidden_size = 32 * 32 * size
    hidden_states = Reshape([5, hidden_size])(hidden_states)
    output = AttentionLayer(512, name='attention')(hidden_states)           # 可以调整512为[64, 128, 256, 512, 1024, 2048]

    # Transformer
    embed_dim = 64  # Embedding size for each token

    output = Reshape([32*32, embed_dim])(output)
    # output = TokenAndPositionEmbedding(32*32, embed_dim)(output)
    att_output = MultiHeadSelfAttention(num_heads, use_masking=False)(output)
    output = LayerNormalization()(Add()([output, Dropout(rate)(att_output)]))
    output = LayerNormalization()(Add()([output, Dropout(rate)(Dense(embed_dim, activation="relu")(output))]))
    # output = TransformerBlock()(output)

    output = Reshape([32 * 32 * embed_dim])(output)

    reps = Dense(512, activation="relu")(output)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # reps = output   #  Dense(512, activation="relu")(output)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)

    return model


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

    hidden_states = [Flatten()(scale_outputs[0]), Flatten()(upsampling1), Flatten()(upsampling2),
                     Flatten()(upsampling3), Flatten()(upsampling4)]  #
    hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    hidden_size = 32 * 32 * size
    hidden_states = Reshape([5, hidden_size])(hidden_states)
    output = AttentionLayer(512, name='attention')(hidden_states)           # 可以调整512为[64, 128, 256, 512, 1024, 2048]

    reps = Dense(512, activation="relu")(output)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # reps = output   #  Dense(512, activation="relu")(output)
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)

    return model


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
    hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    hidden_size = 32 * 32 * size
    hidden_states = Reshape([5, hidden_size])(hidden_states)
    output = AttentionLayer(512, name='attention')(hidden_states)

    reps = Reshape([32, 32, size])(output)   #  Dense(512, activation="relu")(output)
    filter_dim = 64                    # 可以调整512为[64, 128, 256, 512, 1024, 2048]
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
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    # context_vector = dot([reps, attention_weights], [1, 1], name='context_vector')
    context_vector = Lambda(lambda x: dot(x, [1, 1]))([reps, attention_weights])
    # pre_activation = concatenate([context_vector, h_t], name='attention_output')
    # attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)

    output = Dense(1, activation="sigmoid")(context_vector)
    model = module.model_compile(img_input, output)
    return model


