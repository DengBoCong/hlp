# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:46:38 2020
formatted
@author: 九童
"""
# !/usr/bin/env python

import tensorflow as tf


class pBLSTM(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(pBLSTM, self).__init__()
        self.dim = dim
        self.LSTM = tf.keras.layers.LSTM(self.dim, return_sequences=True)
        self.bidi_LSTM = tf.keras.layers.Bidirectional(self.LSTM)

    @tf.function
    def call(self, inputs):
        y = self.bidi_LSTM(inputs)

        if (tf.shape(inputs)[1] % 2 == 1):
            y = tf.keras.layers.ZeroPadding1D(padding=(0, 1))(y)

        y = tf.keras.layers.Reshape(target_shape=(-1, int(self.dim * 4)))(y)
        return y


class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.dim = embedding_dim
        # Listen; Lower resoultion by 8x
        self.plstm1 = pBLSTM(self.dim // 2)
        self.plstm2 = pBLSTM(self.dim // 2)
        self.plstm3 = pBLSTM(self.dim // 2)
        
    def call(self, x, hidden):
        x = self.plstm1(x)
        x = self.plstm2(x)
        output = self.plstm3(x)
        return output, hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
