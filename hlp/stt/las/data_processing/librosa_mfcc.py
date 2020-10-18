# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 19:15:23 2020

@author: 九童
librosa提取mfcc
"""

import os

import librosa
import tensorflow as tf


def mfcc_extract(path):
    y, sr = librosa.load(path=path)
    # 提取mfcc(返回list(timestep,n_mfcc))
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=20).transpose(1, 0).tolist()
    return mfcc


def wav_to_mfcc(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    mfccs = []

    for file in files:  # 遍历文件夹
        position = path + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
        mfcc = mfcc_extract(position)
        mfccs.append(mfcc)

    mfccs = tf.keras.preprocessing.sequence.pad_sequences(mfccs, padding='post', dtype=float)
    print('====mfccs.shape = {}'.format(mfccs.shape))  # (100, 93, 39)
    return mfccs


'''
path='.\\wav\\BAC009S0002W0123.wav'
y,sr = librosa.load(path=path)
mfccs = librosa.feature.mfcc(y=y,n_mfcc=20)
print(mfccs.shape)
#mfccs 
#(20, 259) 259 = 时间步（wav的份数） n_mfcc=20 每个时间步的特征数

path='.\\wav\\BAC009S0002W0122.wav'
y,sr = librosa.load(path=path)
mfccs = librosa.feature.mfcc(y=y,n_mfcc=20)
print(mfccs.shape)
'''
