# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:50:12 2020
@author: 彭康
"""

#模型搭建
#step1：1-3 Conv1D -> 1BN -> 1-3 bi_gru -> 1BN -> 1dense
import tensorflow as tf
from utils import get_config


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

def get_ds2_model():
    configs = get_config()
    n_mfcc = configs["other"]["n_mfcc"]
    conv_layers = configs["model"]["conv_layers"]
    filters = configs["model"]["conv_filters"]
    kernel_size = configs["model"]["conv_kernel_size"]
    strides = configs["model"]["conv_strides"]
    bi_gru_layers = configs["model"]["bi_gru_layers"]
    gru_units = configs["model"]["gru_units"]
    dense_units = configs["model"]["dense_units"]
    return DS2(n_mfcc,conv_layers,filters,kernel_size,strides,bi_gru_layers,gru_units,dense_units)

#基于模型预测得到的序列list并通过字典集来进行解码处理
def decode_output(seq, index_word):
    configs = get_config()
    mode = configs["preprocess"]["text_process_mode"]
    if mode == "cn":
        return decode_output_ch_sentence(seq, index_word)
    elif mode == "en_word":
        return decode_output_en_sentence_word(seq, index_word)
    else:
        return decode_output_en_sentence_char(seq, index_word)

def decode_output_ch_sentence(seq, index_word):
    result = ""
    for i in seq:
        if i >= 1 and i <= len(index_word):
            word = index_word[str(i)]
            if word != "<start>":
                if word != "<end>":
                    result += word
                else:
                    return result
    return result

def decode_output_en_sentence_word(seq,index_word):
    result = ""
    for i in seq:
        if i >= 1 and i <= (len(index_word)):
            word = index_word[str(i)]
            if word != "<start>":
                if word != "<end>":
                    result += word+" "
                else:
                    return result
    return result

def decode_output_en_sentence_char(seq,index_word):
    result = ""
    for i in seq:
        if i >= 1 and i <= (len(index_word)):
            word = index_word[str(i)]
            if word != "<start>":
                if word != "<end>":
                    if word !="<space>":
                        result += word
                    else:
                        word += " "
                else:
                    return result
    return result


if __name__ == "__main__":
    pass
