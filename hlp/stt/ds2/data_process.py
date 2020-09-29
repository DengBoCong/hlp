# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:05:42 2020

@author: 彭康
"""

import os
from utils import wav_to_mfcc,text_to_int_sequence
import tensorflow as tf
import random
import config

#批处理train数据并返回模型的inputs和labels
def data_process1(
    data_path,
    batch_size=config.configs_train["batch_size"],
    n_mfcc=config.configs_other["n_mfcc"]
    ):
    files = os.listdir(data_path) #得到文件夹下的所有文件名称
    #除去最后一个文本txt的所有音频文件
    audio_nums = len(files)-1
    if batch_size>audio_nums:
        batch_size=audio_nums
    #构建一个batch_size长度的随机整数list,且无序(防止训练数据重复)
    file_list_num = random.sample(range(audio_nums),batch_size)
    #对应数据文件夹下的文本list
    text_index = len(files)-1
    text_list=open(data_path+'/'+files[text_index],"r").readlines()
    mfccs_list = []
    labels_list = []
    
    label_length_list=[]
    
    for i in file_list_num:
        filepath = data_path+'/'+ files[i]
        #得到了(timestep,n_mfcc)的mfcc list
        mfcc = wav_to_mfcc(n_mfcc=n_mfcc,wav_path=filepath)
        mfccs_list.append(mfcc)
        #文本格式从第12个开始才是正式字符串，后面还切断了一个回车符
        str=text_list[i][12:len(text_list[i])-1].lower()
        seq_list=text_to_int_sequence(str)
        labels_list.append(seq_list)
        label_length_list.append([len(str)])
    #将内部list不同长度按最大长度填充
    mfccs_numpy = tf.keras.preprocessing.sequence.pad_sequences(mfccs_list,padding='post',dtype='float32')
    inputs = tf.convert_to_tensor(mfccs_numpy)
    labels_numpy = tf.keras.preprocessing.sequence.pad_sequences(labels_list,padding='post')
    labels = tf.convert_to_tensor(labels_numpy)
    #每个lebel的真实长度
    label_length=tf.convert_to_tensor(label_length_list)
    return inputs,labels,label_length

#test数据集加载
def data_process2(
    data_path,
    batch_size=config.configs_test["batch_size"],
    n_mfcc=config.configs_other["n_mfcc"]
    ):
    files = os.listdir(data_path) #得到文件夹下的所有文件名称
    #除去最后一个文本txt的所有音频文件
    audio_nums = len(files)-1
    if batch_size>audio_nums:
        batch_size=audio_nums
    #构建一个batch_size长度的随机整数list,且无序(防止训练数据重复)
    file_list_num = random.sample(range(audio_nums),batch_size)
    #对应数据文件夹下的文本list
    text_index = len(files)-1
    text_list=open(data_path+'/'+files[text_index],"r").readlines()
    mfccs_list = []
    labels_list = []

    for i in file_list_num:
        filepath = data_path +'/'+files[i]
        mfcc = wav_to_mfcc(n_mfcc=n_mfcc,wav_path=filepath)
        mfccs_list.append(mfcc)
        str=text_list[i][12:len(text_list[i])-1].lower()
        labels_list.append(str)
    
    return mfccs_list,labels_list



if __name__=="__main__":
    pass