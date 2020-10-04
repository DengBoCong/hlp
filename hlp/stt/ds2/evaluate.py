# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:42:28 2020

@author: 彭康
"""

import tensorflow as tf
import config
from data_process import data_process
from model import DS2
from utils import int_to_text_sequence, wav_to_mfcc, wers, lers

if __name__=="__main__":
    #加载模型检查点
    model=DS2()
    #加载检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=config.configs_checkpoint['directory'],
        max_to_keep=config.configs_checkpoint['max_to_keep']
        )
    checkpoint.restore(manager.latest_checkpoint)
    
    #评价
    test_data_path = config.configs_test["data_path"]
    batch_size = config.configs_test['batch_size']
    inputs,labels_list = data_process(
        data_path=test_data_path,
        batch_size=batch_size,
        if_train_or_test='test'
        )
    originals = labels_list
    results = []
    y_pred=model(inputs)
    output=tf.keras.backend.ctc_decode(
    y_pred=y_pred,
    input_length=tf.fill([y_pred.shape[0]],y_pred.shape[1]),
    greedy=True
    )
    results_int_list=output[0][0].numpy().tolist()
    for i in range(len(results_int_list)):
        str = "".join(int_to_text_sequence(results_int_list[i])).strip()
        results.append(str)
    rates_wers,aver_wers=wers(originals,results)
    rates_lers,aver_lers,norm_rates_lers,norm_aver_lers=lers(originals,results)
    print("wers:")
    print("rates_wers:",rates_wers)
    print("aver_wers:",aver_wers)
    print("lers:")
    print("rates_lers:",rates_lers)
    print("aver_lers:",aver_lers)
    print("norm_rates_lers:",norm_rates_lers)
    print("norm_aver_lers:",norm_aver_lers)

 