# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:34:04 2020

@author: 九童
"""
# !/usr/bin/env Python
# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
from hlp.stt.las.model import minilas
from hlp.stt.las import recognition_evaluate
from hlp.stt.las.data_processing import preprocess_ch
from hlp.stt.las.data_processing import mfcc_extract
import tensorflow as tf
import numpy as np
import os
import time


#wav文件
path = ".\\data\\wav" 

# 中文语音识别语料文件
path_to_file = ".\\data\\text.txt"

# 尝试实验不同大小的数据集
num_examples = 100
input_tensor = mfcc_extract.wav_to_mfcc(path)
target_tensor, targ_lang_tokenizer = preprocess_ch.load_dataset(path_to_file, num_examples)


# 计算目标张量的最大长度 （max_length）
max_length_targ = preprocess_ch.max_length(target_tensor)
max_length_inp = preprocess_ch.max_length(input_tensor)

#创建一个 tf.data 数据集
def create_dataset(input_x,target_y):
  BUFFER_SIZE = len(input_x)
  BATCH_SIZE = 1
  steps_per_epoch = len(input_x)//BATCH_SIZE
  dataset = tf.data.Dataset.from_tensor_slices((input_x, target_y)).shuffle(BUFFER_SIZE)
  dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) 
  return steps_per_epoch,dataset

steps_per_epoch,dataset = create_dataset(input_tensor,target_tensor)



def loss_function(real, pred):
  loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
  mask = tf.math.logical_not(tf.math.equal(tf.argmax(real[0]).numpy(), len(targ_lang_tokenizer.word_index)))  # 填充位，掩蔽  
  real = tf.expand_dims(real, 1)  
  real = tf.convert_to_tensor(real)
  pred = tf.convert_to_tensor(pred)
  loss_ = loss_object(real, pred)  
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask  
  return tf.reduce_mean(loss_)




optimizer = tf.keras.optimizers.Adam()
model = minilas.LAS(256, 39, len(targ_lang_tokenizer.word_index)+1)
checkpoint_dir = './lastraining_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=model)  


#@tf.function
def train_step(inputx_1, targetx_2):
  loss = 0
  
  with tf.GradientTape() as tape:
    
    # 解码器输入符号
    '''
    dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
    dec_input = tf.keras.preprocessing.sequence.pad_sequences(dec_input,maxlen = 455,padding='post')
    dec_input = tf.expand_dims(dec_input, 1)
    print('dec_input = {}'.format(dec_input))
    print('dec_input.shape = {}'.format(dec_input.shape))
    '''
    dec_input=tf.keras.utils.to_categorical([targ_lang_tokenizer.word_index['<start>']-1],num_classes=len(targ_lang_tokenizer.word_index)+1)    
    dec_input = tf.expand_dims(dec_input, 1)
    dec_input = np.array(dec_input).astype(int)
    dec_input = tf.convert_to_tensor(dec_input)
    inputx_1 = tf.convert_to_tensor(inputx_1)
    targetx_2 = tf.convert_to_tensor(targetx_2)
    #print('dec_input = {}'.format(dec_input))
    #print('dec_input.shape = {}'.format(dec_input.shape))
    
    #print('targetx_2.shape = {}'.format(targetx_2.shape))#(1, 26, 456)
    
    predicted_result = ''
    target_result = ''
    # 教师强制 - 将目标词作为下一个输入
    for t in range(1, targetx_2.shape[1]):
      # 将编码器输出 （enc_output） 传送至解码器，解码
      #print('t = {}'.format(t))
      #print('inp = {}'.format(inputx_1.shape))
      #print('dec_input = {}'.format(dec_input))
      
      predictions = model([inputx_1, dec_input])
      
      predicted_id = tf.argmax(predictions[0][0]).numpy() + 1  # 贪婪解码，取最大
      
      
      result = targ_lang_tokenizer.index_word[predicted_id] + ' '  # 目标句子
      predicted_result +=result
      
      
      target_id = tf.argmax(targetx_2[:, t][0]).numpy() + 1
      if(target_id == len(targ_lang_tokenizer.word_index) +1 ):
       target_result +=  ' '  # 目标句子
      else :
       target_result += targ_lang_tokenizer.index_word[target_id] + ' '  # 目标句子 
          
      #print('predictions = {}'.format(predictions.shape))#(1, 1, 456)
      #print('targetx_2[:,:,t].shape = {}'.format(targetx_2[:, t].shape))#(1, 456)
      loss += loss_function(targetx_2[:, t], predictions)  # 根据预测计算损失
      # 使用教师强制，下一步输入符号是训练集中对应目标符号
      dec_input = targetx_2[:, t]
      
      dec_input = tf.expand_dims(dec_input, 1)
      #print('teacher_dec_input = {}'.format(dec_input))
  print('====predicted_result = {}'.format(predicted_result))
  print('====target_result = {}'.format(target_result))  
  batch_loss = (loss / int(targetx_2.shape[1]))
  #print('batch_loss = {}'.format(batch_loss))
  variables = model.trainable_variables 
  #print('variables = {}')
  gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
  #print('gradients = {}')
  optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数

  return batch_loss   


# 恢复检查点目录 （checkpoint_dir） 中最新的检查点
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

if __name__ == "__main__":
    
    
    #model.compile(loss="mse", optimizer="adam")
    EPOCHS = 500

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
        
     # 每 10 个周期（epoch），保存（检查点）一次模型
     if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
     print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
     print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
     
     

#用wav文件语音识别出中文
#recognition_evaluate.speech_recognition('.\\data\\wav_test\\BAC009S0002W0122.wav',max_length_inp,max_length_targ,targ_lang_tokenizer,model)     












