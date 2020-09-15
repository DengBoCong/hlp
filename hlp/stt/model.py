# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:50:12 2020
@author: 彭康
"""

#模型搭建
#step1：1Conv1D -> 1BN -> 1bi_gru -> 1BN -> 1dense
import tensorflow as tf

class DP2(tf.keras.Model):
    def __init__(self,filters,kernel_size,strides,gru_units, dense_units):
        super(DP2,self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                activation="relu"
                )
        self.bn1 = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001
                )
        self.bi_gru1 = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                        gru_units,
                        return_sequences=True
                        )
                )
        self.bn2 = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001
                )
        self.ds1 = tf.keras.layers.Dense(dense_units)
    
    def call(self,inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.bi_gru1(x)
        x = self.bn2(x)
        x = self.ds1(x)
        return x



