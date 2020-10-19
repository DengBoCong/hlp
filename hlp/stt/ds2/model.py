# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:50:12 2020
@author: 彭康
"""

#模型搭建
#step1：1-3 Conv1D -> 1BN -> 1-3 bi_gru -> 1BN -> 1dense
import tensorflow as tf
import config


#子类化构建DS2模型
class DS2(tf.keras.Model):
    #dense_units=num_classes
    def __init__(
        self,
        dense_units,
        n_mfcc=config.configs_other()["n_mfcc"],
        conv_layers=config.configs_model()["conv_layers"],
        filters=config.configs_model()["conv_filters"],
        kernel_size=config.configs_model()["conv_kernel_size"],
        strides=config.configs_model()["conv_strides"],
        bi_gru_layers=config.configs_model()["bi_gru_layers"],
        gru_units=config.configs_model()["gru_units"]
        ):
        super(DS2,self).__init__()
        self.conv_layers=conv_layers
        self.conv = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
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
        for _ in range(self.conv_layers):
            x = self.conv(x)
        x = self.bn(x)
        for _ in range(self.bi_gru_layers):
            x = self.bi_gru(x)
        x = self.bn(x)
        x = self.ds(x)
        return x

if __name__ == "__main__":
    pass