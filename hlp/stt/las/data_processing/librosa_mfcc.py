# -*- coding: utf-8 -*-
"""
formatted
Created on Fri Oct 16 19:15:23 2020

@author: 九童
librosa提取mfcc

"""

import librosa
import tensorflow as tf


def mfcc_extract(path, n_mfcc):
    y, sr = librosa.load(path=path)
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=n_mfcc).transpose(1, 0).tolist()
    return mfcc


def wav_to_mfcc(path_list, audio_feature_type, max_input_length, n_mfcc=39):
    mfccs = []
    for path in path_list:
        bool = path.endswith(".wav")
        if bool:
            mfcc = mfcc_extract(path, n_mfcc)
            mfccs.append(mfcc)
    mfccs = tf.keras.preprocessing.sequence.pad_sequences(mfccs, padding='post', dtype='float32',
                                                          maxlen=max_input_length)
    return mfccs
