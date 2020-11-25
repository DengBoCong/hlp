# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:46:01 2020
encoder部分采用2层cnn+max_pooling(stride=2)+d层的Bi-LSTM(cell=w)
@author: 九童
"""
import tensorflow as tf



class Encoder(tf.keras.Model):
    def __init__(self, lstm_layers, w):
        super(Encoder, self).__init__()
        self.cnn1 = tf.keras.layers.Conv1D()
        self.cnn2 = tf.keras.layers.Conv1D()
        self.max_pool = tf.keras.layers.MaxPooling1D(inputs=self.cnn2, strides=2)
        self.LSTM = tf.keras.layers.LSTM(w)
        self.bidi_LSTM = tf.keras.layers.Bidirectional(self.LSTM)
        self.lstm_layers = lstm_layers
        self.lstm = []
        for i in range(lstm_layers):
            self.lstm.append(self.bidi_LSTM())

        
    def call(self, x, hidden):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.max_pool(x)
        output = self.plstm3(x)
        return output, hidden
       