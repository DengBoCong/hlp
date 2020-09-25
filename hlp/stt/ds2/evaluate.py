# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:42:28 2020

@author: 彭康
"""

import tensorflow as tf
from utils import wav_to_mfcc,int_to_text_sequence
from model import DS2

if __name__=="__main__":
    #加载模型检查点
    model=DS2()
    #加载检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoint',max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)
    
    
    #测试数据
    test_audio_path='./data/test_data/19-198-0000.flac'
    #提取了单个音频的特征(batch_size,timesteps,n_mfcc)
    x_test=wav_to_mfcc(20,test_audio_path)
    x_test_input=tf.convert_to_tensor([x_test])
    print(x_test_input)
    y_test_pred=model(x_test_input)
    print(y_test_pred)
    output=tf.keras.backend.ctc_decode(y_pred=y_test_pred,input_length=tf.constant([y_test_pred.shape[1]]),greedy=True)
    print(output)
    out=output[0][0]
    str="".join(int_to_text_sequence(out.numpy()[0]))
    print(str)
    
    #评价