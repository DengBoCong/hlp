# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:05:42 2020

@author: 彭康
"""

import os
from utils import wav_to_mfcc,text_to_int_sequence
import tensorflow as tf
import random

#批处理数据并返回模型的inputs和labels
def data_process(data_path,batch_size=36,n_mfcc=20):
    files = os.listdir(data_path) #得到文件夹下的所有文件名称
    #除去最后一个文本txt的所有音频文件
    audio_nums = len(files)-1
    if batch_size>audio_nums:
        batch_size=audio_nums
    #构建一个batch_size长度的随机整数list
    file_list_num = sorted(random.sample(range(audio_nums),batch_size))
    #对应数据文件夹下的文本list
    text_index = len(files)-1
    text_list=open(data_path+'/'+files[text_index],"r").readlines()
    mfccs_list = []
    labels_list = []
    for i in file_list_num:
        filepath = data_path+'/'+ files[i]
        print(filepath)
        #得到了(timestep,n_mfcc)的mfcc list,转成list是为了后面的填充。
        mfcc = wav_to_mfcc(n_mfcc=n_mfcc,wav_path=filepath).transpose(1,0).tolist()
        mfccs_list.append(mfcc)
        seq_list=text_to_int_sequence(text_list[i][12:len(text_list[i])-1].lower())
        labels_list.append(seq_list)
    mfccs_numpy = tf.keras.preprocessing.sequence.pad_sequences(mfccs_list,padding='post')
    inputs = tf.convert_to_tensor(mfccs_numpy)
    labels_numpy = tf.keras.preprocessing.sequence.pad_sequences(labels_list,padding='post')
    labels = tf.convert_to_tensor(labels_numpy)
    print(inputs.shape,labels.shape)
    
    return inputs,labels

if __name__=="__main__":
    path = "./train-clean-5/LibriSpeech/train-clean-5/19/198"
    inputs,labels=data_process(path)
