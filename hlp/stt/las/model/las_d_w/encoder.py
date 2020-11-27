# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:46:01 2020
encoder部分采用2层cnn+max_pooling(stride=2)+d层的Bi-LSTM(cell=w)
a 2-layer (CNN) with max-pooling and stride of 2. The output of the CNN is passes through
 an encoder consisting of d stacked bi-directional LSTMs with cell
size w to yield a series of attention vectors. 
@author: 九童
"""
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, d, w, batch_sz):
        super(Encoder, self).__init__()
        self.d = d
        self.w = w
        self.batch_sz = batch_sz
        self.cnn1 = tf.keras.layers.Conv1D(filters=2, kernel_size=2, activation='relu')
        self.cnn2 = tf.keras.layers.Conv1D(filters=2, kernel_size=2, activation='relu')
        self.max_pool = tf.keras.layers.MaxPooling1D(strides=2)
        self.LSTM = tf.keras.layers.LSTM(w, return_sequences=True)
        self.bi_lstm = []
        for i in range(d):
            self.bi_lstm.append(tf.keras.layers.Bidirectional(self.LSTM))

    def call(self, x, hidden):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.max_pool(x)
        for i in range(self.d):
            x = self.bi_lstm[i](x)
        return x, hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.w))
