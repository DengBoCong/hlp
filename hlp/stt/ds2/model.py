# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:50:12 2020
@author: 彭康
"""

#模型搭建
#step1：1-3 Conv1D -> 1BN -> 1-3 bi_gru -> 1BN -> 1dense
import tensorflow as tf
from utils import get_config_model


#子类化构建DS2模型
class DS2(tf.keras.Model):
    #dense_units=num_classes
    def __init__(
        self,
        n_mfcc,
        conv_layers,
        filters,
        kernel_size,
        strides,
        bi_gru_layers,
        gru_units,
        dense_units
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

def init_ds2():
    configs_model = get_config_model()
    n_mfcc=configs_model[0]
    conv_layers=configs_model[1]
    filters=configs_model[2]
    kernel_size=configs_model[3]
    strides=configs_model[4]
    bi_gru_layers=configs_model[5]
    gru_units=configs_model[6]
    dense_units=configs_model[7]
    return DS2(n_mfcc,conv_layers,filters,kernel_size,strides,bi_gru_layers,gru_units,dense_units)


if __name__ == "__main__":
    pass