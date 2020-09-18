# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:34:04 2020

@author: 九童
"""
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
from model import minilas
import mfcc_extract
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import numpy as np
import os
import io
import time
import scipy.io.wavfile





#将文件夹中的wav文件转换为mfcc语音特征
path = "wav" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
mfccs = []

for file in files: #遍历文件夹
    position = path+'\\'+ file #构造绝对路径，"\\"，其中一个'\'为转义符
    #sample_rate = 16000，每秒采样16000次
    sample_rate, signal = scipy.io.wavfile.read(position) 
    mfcc = mfcc_extract.MFCC(sample_rate,signal)
    mfccs.append(mfcc)
mfccs = tf.keras.preprocessing.sequence.pad_sequences(mfccs,
                                                         padding='post')
mfccs.shape


# 中文语音识别语料文件
path_to_file = "text.txt"

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    ch_sentences = [l.split(' ')[1:]  for l in lines[:num_examples]]
    return  ch_sentences

ch = create_dataset(path_to_file, None)

def tokenize(texts):
  tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')  # 无过滤字符
  tokenizer.fit_on_texts(texts)  

  sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列

  sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                         padding='post')

  return sequences, tokenizer

ch_seqs, ch_tokenizer = tokenize(ch)

print('中文词典大小：', len(ch_tokenizer.word_index))  # 中文字典大小

def max_length(texts):
    return max(len(t) for t in texts)

def load_dataset(path, num_examples=None):
    # 创建清理过的输入输出对
    targ_lang = create_dataset(path, num_examples)
    input_tensor = mfccs
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    #target_tensor = tf.convert_to_tensor(target_tensor)    
    return input_tensor, target_tensor, targ_lang_tokenizer

# 尝试实验不同大小的数据集
num_examples = 100
input_tensor, target_tensor, targ_lang = load_dataset(path_to_file, num_examples)

print(input_tensor.shape, target_tensor.shape)

# 计算目标张量的最大长度 （max_length）
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
print(max_length_targ, max_length_inp)

# 采用 90 - 10 的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)

# 显示长度
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


#创建一个 tf.data 数据集

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 10
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch)
print(example_target_batch)
print(example_input_batch.shape, example_target_batch.shape)
'''
调试中
model = LAS(256, 39, 15)

model.compile(loss="mse", optimizer="adam")
#x_1 = np.random.random((1, 550, 256))

model.fit(dataset)

model = LAS(256, 39, 16)
model.compile(loss="mse", optimizer="adam")

# x_1 should have shape (Batch-size, timesteps, f_1)
x_1 = np.random.random((1, 550, 256))

# x_2 should have shape (Batch-size, no_prev_tokens, No_tokens). The token vector should be one-hot encoded.
x_2 = np.zeros((1,12,15))
for n in range(12):
  x_2[0, n, np.random.randint(1, 15)] = 1

# By passing x_1 and x_2 the model will predict the 12th token 
# given by the spectogram and the prev predicted tokens
x_1 = example_input_batch
#x_2 = tf.shape(tf.expand_dims(example_target_batch, 1))
output = model.predict([x_1, x_2])

output.shape(1,12,16)
'''