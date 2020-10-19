# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:05:42 2020

@author: 彭康
"""

import os
import random

import config
import tensorflow as tf
from utils import text_to_int_sequence, wav_to_mfcc, text_process, get_index_and_char_map


def data_process(
        data_path,
        batch_size,
        if_train_or_test,  # train和test的返回数据不一样
        n_mfcc=config.configs_other()["n_mfcc"]
):
    files = os.listdir(data_path)  # 得到文件夹下的所有文件名称
    # 除去最后一个文本txt的所有音频文件
    audio_nums = len(files) - 1
    if batch_size > audio_nums:
        batch_size = audio_nums
    # 构建一个batch_size长度的随机整数list,且无序(防止测试数据重复)
    file_list_num = random.sample(range(audio_nums), batch_size)
    # 对应数据文件夹下的文本list
    text_index = len(files) - 1
    with open(data_path + '/' + files[text_index], "r") as f:
        text_list = f.readlines()

    mfccs_list = []
    labels_str_list = []

    for i in file_list_num:
        filepath = data_path + '/' + files[i]
        mfcc = wav_to_mfcc(n_mfcc=n_mfcc, wav_path=filepath)
        mfccs_list.append(mfcc)
        # 文本的读取
        str = text_process(text_list[i])
        labels_str_list.append(str)
    mfccs_numpy = tf.keras.preprocessing.sequence.pad_sequences(mfccs_list, padding='post', dtype='float32')
    inputs = tf.convert_to_tensor(mfccs_numpy)
    if if_train_or_test == 'test':
        return inputs, labels_str_list
    else:
        labels_list = []
        label_length_list = []
        # 构建数据集对象
        char_map = get_index_and_char_map()[1]
        for i in range(len(labels_str_list)):
            labels_list.append(text_to_int_sequence(labels_str_list[i], char_map))
            label_length_list.append([len(labels_str_list[i])])
        labels_numpy = tf.keras.preprocessing.sequence.pad_sequences(labels_list, padding='post')
        labels = tf.convert_to_tensor(labels_numpy)
        label_length = tf.convert_to_tensor(label_length_list)
        return inputs, labels, label_length


if __name__ == "__main__":
    pass
