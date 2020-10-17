# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:46:38 2020

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
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    #self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    
    
    '''
    modify
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    '''

  def call(self, x, hidden):
    input_1 = tf.keras.Input(shape=(None, x))
    #x = self.embedding(x)
    #modify
    #output, state = self.gru(x, initial_state = hidden)  # 编码器需要显示初始状态吗？
    # Listen; Lower resoultion by 8x
    x = pBLSTM(self.embedding_dim // 2)(input_1)
    x = pBLSTM(self.embedding_dim // 2)(x)
    output, state = pBLSTM(self.embedding_dim // 2)(x)

    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))