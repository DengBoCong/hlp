# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:20:25 2020

@author: 九童
"""
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
from model import minilas
from data.mfcc_extract import MFCC
import tensorflow as tf
import re
import numpy as np
import os
import io
import time
import scipy.io.wavfile



#将文件夹中的wav文件转换为mfcc语音特征
path = ".\\wav" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
mfccs = []

for file in files: #遍历文件夹
    position = path+'\\'+ file #构造绝对路径，"\\"，其中一个'\'为转义符
    sample_rate, signal = scipy.io.wavfile.read(position) 
    mfcc = MFCC(sample_rate,signal)
    mfccs.append(mfcc)
mfccs = tf.keras.preprocessing.sequence.pad_sequences(mfccs,padding='post')

# 中文语音识别语料文件
path_to_file = ".\\text.txt"

def preprocess_ch_sentence(s):
   
    s = [c for c in s]
    s = ' '.join(s)
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    s = s.strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'
    return s

def create_input_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    
    ch_sentences = [l.split(' ')[1:]  for l in lines[:num_examples]]
    ch_sentences = [''.join(word) for word in ch_sentences]
    
    ch_sentences = [preprocess_ch_sentence(s) for s in ch_sentences]
    return  ch_sentences

def tokenize(texts):
  tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')  # 无过滤字符
  tokenizer.fit_on_texts(texts)  
  sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
  sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                         padding='post')

  return sequences, tokenizer



def max_length(texts):
    return max(len(t) for t in texts)

def load_dataset(path, num_examples=None):
    # 创建清理过的输入输出对
    targ_lang = create_input_dataset(path, num_examples)    
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return target_tensor, targ_lang_tokenizer

# 尝试实验不同大小的数据集
num_examples = 100
input_tensor = mfccs
target_tensor, targ_lang_tokenizer = load_dataset(path_to_file, num_examples)
target_tensor.shape

#one_hot,一个字是一个456维的向量，共455个字
tar_oh=tf.keras.utils.to_categorical(list(targ_lang_tokenizer.word_index.values()),num_classes=len(targ_lang_tokenizer.word_index) + 1 )
print(tar_oh.shape)

def tensor_to_onehot(tensor):
    tensor = tensor.tolist()
    for _,sentence in enumerate(tensor):
        for index,word in enumerate(sentence):
            word = tf.keras.utils.to_categorical(word,num_classes=len(targ_lang_tokenizer.word_index) + 1 )             
            sentence[index] = word    
    return tensor
            

target_tensor = tensor_to_onehot(target_tensor)
target_tensor =np.array(target_tensor).astype(int)
target_tensor.shape#(100, 26, 456)
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

# 采用 90 - 10 的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)
input_tensor_train.shape#((90, 93, 39))

#创建一个 tf.data 数据集

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 1
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
vocab_inp_size = 94#len(inp_lang.word_index) + 1  # 含填充的0
vocab_tar_size = len(targ_lang_tokenizer.word_index) + 1  # 含填充的0

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

loss_object = tf.keras.losses.CategoricalCrossentropy (
    from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam()
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))  # 填充位为0，掩蔽
  real = tf.expand_dims(real, 1)
  real = tf.convert_to_tensor(real)
  pred = tf.convert_to_tensor(pred)
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

model = minilas.LAS(256, 39, 456)

checkpoint_dir = './lastraining_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=model)

#@tf.function
def train_step(inputx_1, targetx_2):
  loss = 0
  
  with tf.GradientTape() as tape:
    
    dec_input=tf.keras.utils.to_categorical([targ_lang_tokenizer.word_index['<start>']],num_classes=len(targ_lang_tokenizer.word_index) + 1 )    
    dec_input = tf.expand_dims(dec_input, 1)
    dec_input = np.array(dec_input).astype(int)
    dec_input = tf.convert_to_tensor(dec_input)
    inputx_1 = tf.convert_to_tensor(inputx_1)
    targetx_2 = tf.convert_to_tensor(targetx_2)
    
    
    # 教师强制 - 将目标词作为下一个输入
    for t in range(1, targetx_2.shape[1]):
      predictions = model([inputx_1, dec_input])
      loss += loss_function(targetx_2[:, t], predictions)  # 根据预测计算损失
      # 使用教师强制，下一步输入符号是训练集中对应目标符号
      dec_input = targetx_2[:, t]
      dec_input = tf.expand_dims(dec_input, 1)
  batch_loss = (loss / int(targetx_2.shape[1]))
  variables = model.trainable_variables
  gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
  optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数

  return batch_loss  


if __name__ == "__main__":
    
    
    #model.compile(loss="mse", optimizer="adam")
    EPOCHS = 2

    for epoch in range(EPOCHS):
     start = time.time()
     total_loss = 0
     for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
      batch_start = time.time()
      x_1 = inp
      x_2 = targ
      #print('batch = {}'.format(batch))
      #print('x_1.shape = {}'.format(x_1.shape))#(1, 93, 39)
      #print('x_2.shape = {}'.format(x_2.shape))#(1, 26, 456)
      
      batch_loss = train_step(x_1, x_2)  # 训练一个批次，返回批损失
      
      total_loss += batch_loss
     
     
      
      if batch % 2 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))
        print('Time taken for 2 batches {} sec\n'.format(time.time() - batch_start))
        
     # 每 2 个周期（epoch），保存（检查点）一次模型
     #if (epoch + 1) % 2 == 0:
     checkpoint.save(file_prefix = checkpoint_prefix)
     print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
     print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
     
'''
文字输出部分还在调试中
'''     
     
     
        
     
  






     
        
     
  





