import json
import os
import tensorflow as tf

from model import decode_output

import sys
sys.path.append("..")
from utils.metric import wers, lers

# 获取配置文件
def get_config():
    with open("config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)
    return configs

# 写入配置文件
def set_config(configs, config_class, key, value):
    configs[config_class][key] = value
    with open("config.json", 'w', encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=4)

# 获取预处理得到的语料集信息
def get_dataset_information():
    with open("dataset_information.json", "r", encoding="utf-8") as f:
        dataset_information = json.load(f)
    return dataset_information

# 根据数据文件夹名获取所有的文件名，包括文本文件名和音频文件名列表
def get_all_data_path(data_path):
    # data_path是数据文件夹的路径
    files = os.listdir(data_path) # 得到数据文件夹下的所有文件名称list
    text_data_path = files.pop()
    audio_data_path_list = files
    return text_data_path, audio_data_path_list

# 计算ctc api中的参数input_length，基于https://github.com/tensorflow/models/blob/master/research/deep_speech
def compute_ctc_input_length(max_time_steps, ctc_time_steps, input_length):
    ctc_input_length = tf.cast(
        tf.multiply(
            input_length,
            ctc_time_steps
            ),
        dtype=tf.float32
        )
    return tf.cast(
            tf.math.floordiv(
                ctc_input_length,
                tf.cast(max_time_steps, dtype=tf.float32)
            ),
            dtype=tf.int32
        )

# 在valid或test计算指标
def compute_metric(model, test_data_generator, batchs, text_process_mode, index_word):
    aver_wers = 0
    aver_lers = 0
    aver_norm_lers = 0
    
    for batch, (input_tensor, input_length, text_list) in zip(range(1, batchs+1), test_data_generator):
        originals = text_list
        results = []
        y_pred = model(input_tensor)
        ctc_input_length = compute_ctc_input_length(input_tensor.shape[1], y_pred.shape[1], input_length)
        output = tf.keras.backend.ctc_decode(
            y_pred=y_pred,
            input_length=tf.reshape(ctc_input_length,[ctc_input_length.shape[0]]),
            # input_length=tf.fill([y_pred.shape[0]], y_pred.shape[1]),
            greedy=True
        )
        results_int_list = output[0][0].numpy().tolist()

        # 解码
        for i in range(len(results_int_list)):
            str = decode_output(results_int_list[i], index_word, text_process_mode).strip()
            results.append(str)
        
        # 通过wer、ler指标评价模型
        _, aver_wer = wers(originals, results)
        _, aver_ler, _, norm_aver_ler = lers(originals, results)
        
        aver_wers += aver_wer
        aver_lers += aver_ler
        aver_norm_lers += norm_aver_ler
    
    return aver_wers/batchs, aver_lers/batchs, aver_norm_lers/batchs

def earlyStopCheck(array):
    last = array[-1]
    rest = array[:-1]
    # 最后一个错误率比所有的大则返回True
    if all(i <= last for i in rest):
        return True
    else:
        return False


if __name__ == "__main__":
    pass