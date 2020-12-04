# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:43:50 2020

@author: 九童
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from hlp.stt.utils.metric import lers
from config import config


# 计算指标
def compute_metric(model, val_data_generator, val_batchs, val_batch_size):
    dataset_information = config.get_dataset_information()
    units = config.units
    dec_units = config.dec_units
    # 确定使用的model类型
    model_type = config.model_type
    word_index = dataset_information["word_index"]
    index_word = dataset_information["index_word"]
    max_label_length = dataset_information["max_label_length"]
    results = []
    labels_list = []
    for batch, (input_tensor, text_list) in zip(range(1, val_batchs + 1), val_data_generator):
        if model_type == "las_d_w":
            hidden = tf.zeros((val_batch_size, dec_units))
        elif model_type == "las":
            hidden = tf.zeros((val_batch_size, units))
        dec_input = tf.expand_dims([word_index['<start>']] * val_batch_size, 1)            
        result = ''  # 识别结果字符串

        for t in range(max_label_length):  # 逐步解码或预测
            predictions, _ = model(input_tensor, hidden, dec_input)
            predicted_ids = tf.argmax(predictions, 1).numpy()  # 贪婪解码，取最大 
            idx = str(predicted_ids[0])
            if index_word[idx] == '<end>':
                break
            else:
                result += index_word[idx]  # 目标句子
            # 预测的 ID 被输送回模型            
            dec_input = tf.expand_dims(predicted_ids, 1)

        results.append(result)
        labels_list.append(text_list[0])
    rates_lers, aver_lers, norm_rates_lers, norm_aver_lers = lers(labels_list, results)

    return rates_lers, aver_lers, norm_rates_lers, norm_aver_lers
