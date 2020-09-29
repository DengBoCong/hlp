# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:34:04 2020

@author: 九童
"""
# !/usr/bin/env Python
# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
from hlp.stt.las.model import minilas
import mfcc_extract
import tensorflow as tf
import re
import numpy as np
import os
import io
import time
import scipy.io.wavfile
#from data import temp
#现在的问题是，每个文字张量的最后会大量补0 
#如【1 13 145 177 5 44 2 0 0 0 .....0 0 0】
#然后进行独热编码 0-1就变成-1，而-1对应的独热编码会变成【0 0....0 1] 也就是字典里的最后一个字，这样会有影响



#将文件夹中的wav文件转换为mfcc语音特征
path = ".\\wav" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
mfccs = []

for file in files: #遍历文件夹
    position = path+'\\'+ file #构造绝对路径，"\\"，其中一个'\'为转义符
    sample_rate, signal = scipy.io.wavfile.read(position) 
    mfcc = mfcc_extract.MFCC(sample_rate,signal)
    #mfcc = temp.mfcc_extract(position)
    mfccs.append(mfcc)
mfccs = tf.keras.preprocessing.sequence.pad_sequences(mfccs,padding='post',dtype = float)
print('====mfccs.shape = {}'.format(mfccs.shape))#(100, 93, 39)
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

#ch_sentences = create_input_dataset(path_to_file,10)
def tokenize(texts):
  tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')  # 无过滤字符
  tokenizer.fit_on_texts(texts)  

  sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列

  sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                         padding='post',value = len(tokenizer.word_index)+1 )
  
  print('=====len(tokenizer.word_index) = {}'.format(len(tokenizer.word_index)))
  return sequences, tokenizer



def max_length(texts):
    return max(len(t) for t in texts)

def load_dataset(path, num_examples=None):
    # 创建清理过的输入输出对
    targ_lang = create_input_dataset(path, num_examples)
    
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    #target_tensor = tf.convert_to_tensor(target_tensor)    
    return target_tensor, targ_lang_tokenizer

# 尝试实验不同大小的数据集
num_examples = 100
input_tensor = mfccs
target_tensor, targ_lang_tokenizer = load_dataset(path_to_file, num_examples)
target_tensor.shape
targ_lang_tokenizer.word_index
#one_hot,一个字是一个455维的向量，共455个字 
#这样不行啊，一个是字典是1-455，共455个token
#one-hot要从0开始编码
#那我我就只能让字典和one-hot的对应方式改掉，
#one-hot中的0的独热编码代表字典里的1，454的独热编码代表字典里的455
#onehot_index = [i - 1 for i in list(targ_lang_tokenizer.word_index.values())] 
#tar_oh=tf.keras.utils.to_categorical(onehot_index,num_classes=len(targ_lang_tokenizer.word_index) )
#print(tar_oh.shape)


#target_tensor后置补0，0-1 = -1 ,独热编码会变成最后一维是1，即字典中的最后一个字，用<end>判断一句话的结束
def tensor_to_onehot(tensor):
    tensor = tensor.tolist()
    for _,sentence in enumerate(tensor):
        for index,word in enumerate(sentence):    
            word = tf.keras.utils.to_categorical(word-1,num_classes=len(targ_lang_tokenizer.word_index)+1)             
            sentence[index] = word  
    return tensor
            

target_tensor = tensor_to_onehot(target_tensor)
target_tensor =np.array(target_tensor).astype(int)
target_tensor.shape#(100, 26, 455)
#target_tensor[0]

#targ_lang_tokenizer.texts_to_sequences()
# 计算目标张量的最大长度 （max_length）
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

# 采用 90 - 10 的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)
input_tensor_train.shape#((90, 93, 39))

#创建一个 tf.data 数据集

#BUFFER_SIZE = len(input_tensor_train)
BUFFER_SIZE = len(input_tensor)
BATCH_SIZE = 1
#embedding_dim = 26
#steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
steps_per_epoch = len(input_tensor)//BATCH_SIZE
#embedding_dim = 256
#units = 512
#vocab_inp_size = 94#len(inp_lang.word_index) + 1  # 含填充的0
#vocab_tar_size = len(targ_lang_tokenizer.word_index) + 1  # 含填充的0


target_tensor.shape

#dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

loss_object = tf.keras.losses.CategoricalCrossentropy (
    from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam()
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(tf.argmax(real[0]).numpy(), len(targ_lang_tokenizer.word_index)))  # 填充位，掩蔽
  #print('tf.argmax(real[0]).numpy() = {}'.format(tf.argmax(real[0]).numpy()))
  #print('len(targ_lang_tokenizer.word_index) = {}'.format(len(targ_lang_tokenizer.word_index)))
  real = tf.expand_dims(real, 1)
  #print('real.shape = {}'.format(real.shape))#(1, 456)
  #print('pred.shape = {}'.format(pred.shape))#(1, 1, 456)
  real = tf.convert_to_tensor(real)
  pred = tf.convert_to_tensor(pred)
  loss_ = loss_object(real, pred)
  
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  #print('loss_ = {}'.format(loss_))
  return tf.reduce_mean(loss_)

#model = LAS(256, 39, 26)here
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



def evaluate(wav_path):
    
    #sentence = preprocess_en_sentence(sentence)
    sample_rate, signal = scipy.io.wavfile.read(wav_path) 
    wav_mfcc = mfcc_extract.MFCC(sample_rate,signal)

    #inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]  # token编码
    #inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                          #maxlen=max_length_inp,
                                                           #padding='post')  # 填充
    #print('====wav_mfcc.shape = {}'.format(wav_mfcc.shape))#(60, 39)
    wav_mfcc = tf.expand_dims(wav_mfcc, 0)#(1,60, 39)
    wav_mfcc = tf.keras.preprocessing.sequence.pad_sequences(wav_mfcc,maxlen = max_length_inp,padding='post',dtype = float)
    wav_mfcc = tf.convert_to_tensor(wav_mfcc)  # numpy数组转换成张量
    
    #print('====wav_mfcc.shape = {}'.format(wav_mfcc.shape))#(1,93, 39)
    result = ''  # 语音识别结果字符串

    
    dec_input=tf.keras.utils.to_categorical([targ_lang_tokenizer.word_index['<start>'] - 1],num_classes=len(targ_lang_tokenizer.word_index)+1)    
    dec_input = tf.expand_dims(dec_input, 1)
    dec_input = np.array(dec_input).astype(int)
    dec_input = tf.convert_to_tensor(dec_input)
    #print('====dec_input = {}'.format(dec_input))
    for t in range(max_length_targ):  # 逐步解码或预测
        predictions = model([wav_mfcc, dec_input])
        #print('====predictions.shape = {}'.format(predictions.shape))
        predicted_id = tf.argmax(predictions[0][0]).numpy() + 1  # 贪婪解码，取最大
        #print('====predicted_id = {}'.format(predicted_id))
        
        result += targ_lang_tokenizer.index_word[predicted_id] + ' '  # 目标句子
        #print('====result = {}'.format(result))
        if targ_lang_tokenizer.index_word[predicted_id] == '<end>':
            return result

        # 预测的 ID 被输送回模型
        dec_input = tf.keras.utils.to_categorical([predicted_id - 1],num_classes=len(targ_lang_tokenizer.word_index)+1)    
        dec_input = tf.expand_dims(dec_input, 1)         
        #print('====afterdec_input.shape = {}'.format(dec_input.shape))

    return result




def speech_recognition(wav_path):
    result = evaluate(wav_path)

    
    print('Predicted speech_recognition: {}'.format(result))

    
# 恢复检查点目录 （checkpoint_dir） 中最新的检查点
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

speech_recognition('.\\wav_test\\BAC009S0002W0123.wav')







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
     
     
     












