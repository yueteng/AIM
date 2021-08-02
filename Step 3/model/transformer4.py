#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2021/1/29 13:59
@project: MalariaDetection
@description: 
"""

import math
from typing import Union, Callable, Optional

import numpy as np
import keras
from keras.layers import Layer, Add, Activation, Dropout
from keras import initializers
# noinspection PyPep8Naming
from keras import layers
from keras import backend as K
from keras.utils import get_custom_objects
from keras.initializers import Ones, Zeros

from keras_transformer.attention import MultiHeadSelfAttention


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,  **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.init = initializers.get('normal')
        super(TransformerBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = K.variable(self.init((input_shape[-1], self.ff_dim)))
        self.b = K.variable(self.init((self.ff_dim, )))
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)
        self.trainable_weights = [self.W, self.b]
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs):
        attn_output = self.dropout1(inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = K.relu(K.bias_add(K.dot(out1, self.W), self.b))
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = {
            'embed_dim': self.embed_dim,
            'ff_dim': self.ff_dim,
            'num_heads': self.num_heads,
            'rate': self.rate,
        }
        base_config = super(TransformerBlock, self).get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNormalization(layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
        }
        base_config = super(LayerNormalization, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, **kwargs):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_rows, max_cols, embed_dim,  **kwargs):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.embed_dim = embed_dim
        super(TokenAndPositionEmbedding, self).__init__()

    def build(self, input_shape):
        row_pos_emb = np.zeros((self.max_rows, self.embed_dim), dtype=np.float)
        col_pos_emb = np.zeros((self.max_cols, self.embed_dim), dtype=np.float)
        for i in range(self.max_rows):
            tmp = np.arange(self.embed_dim)
            row_pos_emb[i, ::2] = np.sin(i/10000.0**(tmp[::2]/float(self.embed_dim)))
            row_pos_emb[i, 1::2] = np.cos(i/10000.0**(tmp[1::2]/float(self.embed_dim)))

        for i in range(self.max_cols):
            tmp = np.arange(self.embed_dim)
            col_pos_emb[i, ::2] = np.sin(i/10000.0**(tmp[::2]/float(self.embed_dim)))
            col_pos_emb[i, 1::2] = np.cos(i/10000.0**(tmp[1::2]/float(self.embed_dim)))

        row_pos_emb = np.tile(np.expand_dims(row_pos_emb, 1), (1, self.max_cols, 1))
        col_pos_emb = np.tile(np.expand_dims(col_pos_emb, 0), (self.max_rows, 1, 1))
        pos_emb = np.concatenate([row_pos_emb, col_pos_emb], axis=-1)
        self.pos_emb = K.constant(np.reshape(pos_emb, (self.max_rows*self.max_cols, self.embed_dim*2)))

        super(TokenAndPositionEmbedding, self).build(input_shape)

    def call(self, x):
        # maxlen = K.shape(x)[1]
        # positions = K.arange(0, maxlen)
        # positions = self.pos_emb(positions)
        # return x + positions
        return x + self.pos_emb

    def get_config(self):
        config = {
            'max_rows': self.max_rows,
            'max_cols': self.max_cols,
            'embed_dim': self.embed_dim,
        }
        base_config = super(TokenAndPositionEmbedding, self).get_config()
        return {**base_config, **config}

class TokenAndPositionEmbedding0(layers.Layer):
    def __init__(self, max_rows, max_cols, embed_dim,  **kwargs):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.embed_dim = embed_dim
        super(TokenAndPositionEmbedding, self).__init__()

    def build(self, input_shape):
        row_pos_emb = np.zeros((self.max_rows, self.embed_dim), dtype=np.float)
        col_pos_emb = np.zeros((self.max_cols, self.embed_dim), dtype=np.float)
        for i in range(self.max_rows):
            tmp = np.arange(self.embed_dim)
            row_pos_emb[i, ::2] = np.sin(i/10000.0**(tmp[::2]/float(self.embed_dim)))
            row_pos_emb[i, 1::2] = np.cos(i/10000.0**(tmp[1::2]/float(self.embed_dim)))

        row_pos_emb = np.tile(np.expand_dims(row_pos_emb, 1), (1, self.max_cols, 1))
        pos_emb = np.concatenate([row_pos_emb, row_pos_emb], axis=-1)
        self.pos_emb = K.constant(np.reshape(pos_emb, (self.max_rows*self.max_cols, self.embed_dim*2)))

        super(TokenAndPositionEmbedding, self).build(input_shape)

    def call(self, x):
        # maxlen = K.shape(x)[1]
        # positions = K.arange(0, maxlen)
        # positions = self.pos_emb(positions)
        # return x + positions
        return x + self.pos_emb

    def get_config(self):
        config = {
            'max_rows': self.max_rows,
            'max_cols': self.max_cols,
            'embed_dim': self.embed_dim,
        }
        base_config = super(TokenAndPositionEmbedding, self).get_config()
        return {**base_config, **config}


get_custom_objects().update({
    'LayerNormalization': LayerNormalization,
    'TransformerBlock': TransformerBlock,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
})
