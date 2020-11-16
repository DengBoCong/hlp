# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:23:56 2020

@author: 九童
"""
import os
import json

json_path = os.path.join(os.path.dirname(__file__), 'config.json')  # 配置文件路径


# 写入配置文件
def set_config_json(configs, config_class, key, value):
    configs[config_class][key] = value
    with open(json_path, 'w', encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=4)

def get_config_json(config_file):
    with open(config_file, 'r', encoding="utf-8") as file:
        return json.load(file)
    
# 获取预处理得到的语料集信息
def get_dataset_information():
    with open("./dataset_information.json", "r", encoding="utf-8") as f:
        dataset_information = json.load(f)
    return dataset_information
    
conf = get_config_json(json_path)

# 对各变量赋值
epochs = conf["train"]['epochs'] # 训练epochs数
train_wav_path = conf["train"]['wav_path'] # 训练语音数据路径
train_label_path = conf["train"]['label_path'] # 训练标签数据路径
train_batch_size = conf["train"]['batch_size'] # 训练batch_size
num_examples = conf["train"]['num_examples'] # 训练wav文件数
test_wav_path = conf["test"]['wav_path'] # 测试语音数据路径
test_label_path = conf["test"]['label_path'] # 测试标签数据路径
test_num = conf["test"]['test_num'] # 测试wav文件数
test_batch_size = conf["test"]['batch_size'] # 测试batch_size
embedding_dim = conf["model"]['embedding_dim'] # 模型LSTM层维数
units = conf["model"]['units'] # 隐藏层状态单元数
checkpoint_dir = conf["checkpoint"]['directory'] # 检查点保存路径
checkpoint_prefix = conf["checkpoint"]['prefix'] # 检查点前缀
CHUNK = conf["recognition"]['CHUNK'] 
FORMAT = conf["recognition"]['FORMAT'] # 录音格式
CHANNELS = conf["recognition"]['CHANNELS'] # 声道数，单声道或双声道
RATE = conf["recognition"]['RATE'] # 声音采样率
file_path = conf["recognition"]['file_path'] # 录音文件路径
file_name = conf["recognition"]['file_name'] # 录音文件名
dataset_information_path = conf["data"]['dataset_information_path']
dataset_name = conf["data"]['dataset_name'] # 数据集名称
n_mfcc = conf["data"]['n_mfcc'] # 声音mfcc特征数
audio_feature_type = conf["data"]['audio_feature_type'] # 声音特征类型



