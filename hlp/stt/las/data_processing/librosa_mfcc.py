# -*- coding: utf-8 -*-
"""
formatted
Created on Fri Oct 16 19:15:23 2020

@author: 九童
librosa提取mfcc

"""

import os
import librosa
import tensorflow as tf


def mfcc_extract(path, n_mfcc):
    y, sr = librosa.load(path=path)
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=n_mfcc).transpose(1, 0).tolist()
    return mfcc


def wav_to_mfcc(path, n_mfcc, max_length = 36):
    print("开始处理语音数据......")
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    mfccs = []

    for file in files:  # 遍历文件夹
        position = path + '\\' + file  
        bool = file.endswith(".wav")
        if bool:
            mfcc = mfcc_extract(position, n_mfcc)
            mfccs.append(mfcc)

    mfccs = tf.keras.preprocessing.sequence.pad_sequences(mfccs, padding='post', dtype='float32', maxlen=max_length)
    return mfccs
