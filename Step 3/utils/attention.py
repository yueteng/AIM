#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2021/1/17 1:34
@project: MalariaDetection
@description: https://github.com/Wanke15/HAN-keras/blob/master/HAN-keras.ipynb
"""
import keras
from keras import backend as K
from keras import initializers
import numpy as np
import tensorflow as tf

class AttentionLayer(keras.layers.Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def _get_attention_weights(self, X):
        print(np.shape(X))
        X = tf.convert_to_tensor(X)
        uit = K.tanh(K.bias_add(K.dot(X, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        return ait

    def get_config(self):
        config = {
            'attention_dim': self.attention_dim
        }
        base_config = super(AttentionLayer, self).get_config()
        return {**base_config, **config}

    #
    # def __init__(self, context_vector_length=100, **kwargs):
    #     """
    #     An implementation of a attention layer. This layer
    #     accepts a 3d Tensor (batch_size, time_steps, input_dim) and
    #     applies a single layer attention mechanism in the time
    #     direction (the second axis).
    #     :param context_vector_lenght: (int) The size of the hidden context vector.
    #         If set to 1 this layer reduces to a standard attention layer.
    #     :param kwargs: Any argument that the baseclass Layer accepts.
    #     """
    #     self.context_vector_length = context_vector_length
    #     super(AttentionLayer, self).__init__(**kwargs)
    #
    # def build(self, input_shape):
    #     dim = input_shape[2]
    #
    #     # Add a weights layer for the
    #     self.W = self.add_weight(
    #         name='W', shape=(dim, self.context_vector_length),
    #         initializer=keras.initializers.get('uniform'),
    #         trainable=True
    #     )
    #
    #     self.u = self.add_weight(
    #         name='context_vector', shape=(self.context_vector_length, 1),
    #         initializer=keras.initializers.get('uniform'),
    #         trainable=True
    #     )
    #
    #     super(AttentionLayer, self).build(input_shape)
    #
    # def _get_attention_weights(self, X):
    #     """
    #     Computes the attention weights for each timestep in X
    #     :param X: 3d-tensor (batch_size, time_steps, input_dim)
    #     :return: 2d-tensor (batch_size, time_steps) of attention weights
    #     """
    #     # Compute a time-wise stimulus, i.e. a stimulus for each
    #     # time step. For this first compute a hidden layer of
    #     # dimension self.context_vector_length and take the
    #     # similarity of this layer with self.u as the stimulus
    #     u_tw = K.tanh(K.dot(X, self.W))
    #     tw_stimulus = K.dot(u_tw, self.u)
    #
    #     # Remove the last axis an apply softmax to the stimulus to
    #     # get a probability.
    #     tw_stimulus = K.reshape(tw_stimulus, (-1, tw_stimulus.shape[1]))
    #     att_weights = K.softmax(tw_stimulus)
    #
    #     return att_weights
    #
    # def call(self, X):
    #     att_weights = self._get_attention_weights(X)
    #
    #     # Reshape the attention weights to match the dimensions of X
    #     att_weights = K.reshape(att_weights, (-1, att_weights.shape[1], 1))
    #     att_weights = K.repeat_elements(att_weights, X.shape[-1], -1)
    #
    #     # Multiply each input by its attention weights
    #     weighted_input = keras.layers.Multiply()([X, att_weights])
    #
    #     # Sum in the direction of the time-axis.
    #     return K.sum(weighted_input, axis=1)
    #
    # def compute_output_shape(self, input_shape):
    #     return input_shape[0], input_shape[2]
    #
    # def get_config(self):
    #     config = {
    #         'context_vector_length': self.context_vector_length
    #     }
    #     base_config = super(AttentionLayer, self).get_config()
    #     return {**base_config, **config}
