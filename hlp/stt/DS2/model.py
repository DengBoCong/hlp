# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:50:12 2020
@author: 彭康
"""

#模型搭建
#step1：1-3 Conv1D -> 1BN -> 1-3 bi_gru -> 1BN -> 1dense
import tensorflow as tf

#函数式构建DS2模型
def DS2_func(n_mfcc=20,conv_layers=1,filters=256,kernel_size=11,strides=2,bi_gru_layers=1,gru_units=256, dense_units=30):
    inputs=tf.keras.Input(shape=(None,None,n_mfcc))
    x=inputs
    for l in range(conv_layers):
        x = tf.keras.layers.Conv1D(
                filters=filters,
                name="conv{}".format(l+1),
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                strides=strides
                )(x)
    x=tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001
                )(x)
    for l in range(bi_gru_layers):
        x = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                        gru_units,
                        name="bi_gru{}".format(l+1),
                        return_sequences=True,
                        activation="relu"
                        ),
                merge_mode="sum"
                )(x)
    x=tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001
                )(x)
    outputs=tf.keras.layers.Dense(dense_units,activation="softmax")(x)
    return tf.keras.Model(inputs,outputs)

#子类化构建DS2模型
class DS2(tf.keras.Model):
    #dense_units=num_classes
    def __init__(self,n_mfcc=20,conv_layers=1,filters=256,kernel_size=11,strides=2,bi_gru_layers=1,gru_units=256, dense_units=30):
        super(DS2,self).__init__()
        self.conv_layers=conv_layers
        self.conv = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                activation="relu",
                input_shape=(None,None,n_mfcc)
                )
        self.bi_gru_layers=bi_gru_layers
        self.bi_gru = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                        gru_units,
                        activation="relu",
                        return_sequences=True
                        ),
                merge_mode="sum"
                )
        self.bn = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001
                )
        self.ds = tf.keras.layers.Dense(dense_units,activation="softmax")
    
    def call(self,inputs):
        x=inputs
        for i in range(self.conv_layers):
            x = self.conv(x)
        x = self.bn(x)
        for i in range(self.bi_gru_layers):
            x = self.bi_gru(x)
        x = self.bn(x)
        x = self.ds(x)
        return x
