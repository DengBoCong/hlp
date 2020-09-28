# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:42:28 2020

@author: 彭康
"""

import tensorflow as tf
from utils import wav_to_mfcc,int_to_text_sequence,wers
from model import DS2
from data_process import data_process2

if __name__=="__main__":
    #加载模型检查点
    model=DS2()
    #加载检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoint',max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)
    
    
    """#测试数据
    test_audio_path='./19-198-0003.flac'
    #提取了单个音频的特征(batch_size,timesteps,n_mfcc)
    x_test=wav_to_mfcc(20,test_audio_path)
    x_test_input=tf.expand_dims(x_test,axis=0)
    print(x_test_input)
    y_test_pred=model(x_test_input)
    print(y_test_pred)
    output=tf.keras.backend.ctc_decode(y_pred=y_test_pred,input_length=tf.constant([y_test_pred.shape[1]]),greedy=True)
    print(output)
    out=output[0][0]
    str="".join(int_to_text_sequence(out.numpy()[0]))
    print(str)"""
    
    #评价
    test_data_path = './data/test_data'
    mfccs_list,labels_list = data_process2(test_data_path)
    originals = labels_list
    results = []
    for i in range(len(mfccs_list)):
       y_pred=model(tf.expand_dims(mfccs_list[i],axis=0))
       output=tf.keras.backend.ctc_decode(y_pred=y_pred,input_length=tf.expand_dims(y_pred.shape[1],axis=0),greedy=True)
       str = "".join(int_to_text_sequence(output[0][0].numpy()[0]))
       results.append(str)
    rates,aver=wers(originals,results)
    print("rates:",rates)
    print("aver:",aver)