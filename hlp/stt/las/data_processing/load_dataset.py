# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:35:04 2020

@author: 九童
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
from hlp.stt.las.config import config
from hlp.stt.las.data_processing import preprocess_text
from hlp.stt.utils import features


# 基于dataset中的audio_data_path_list和text_list来加载train或test数据
def build_train_data(audio_data_path_list, text_list):
    process_text_list = []
    # 处理文本数据，格式变成如<start> z e r o <end>
    for text in text_list:
        process_text_list.append(preprocess_text.preprocess_en_sentence(text))
    
    text_int_sequences, tokenizer = preprocess_text.tokenize(process_text_list)
    # 获取音频和文本的最大length，从而进行数据补齐
    max_input_length = get_max_audio_length(audio_data_path_list, config.audio_feature_type)
    max_label_length = max_length(text_int_sequences)

    # 将数据集的相关信息写入dataset_information.json文件
    dataset_information_path = config.dataset_information_path

    dataset_information = {}
    dataset_information["vocab_tar_size"] = len(tokenizer.index_word) + 1
    dataset_information["max_input_length"] = max_input_length
    dataset_information["max_label_length"] = max_label_length
    dataset_information["index_word"] = tokenizer.index_word
    dataset_information["word_index"] = tokenizer.word_index

    with open(dataset_information_path, 'w', encoding="utf-8") as f:
        json.dump(dataset_information, f, ensure_ascii=False, indent=4)
    vocab_tar_size = dataset_information["vocab_tar_size"]
    label_length_list = [[len(text_int)] for text_int in text_int_sequences]
    return audio_data_path_list, text_int_sequences, label_length_list, vocab_tar_size


# 加载number语料，返回语音文件list和对应文本字符串list
def load_dataset_number(wav_path, label_path, num_examples=None):
    # 获取number语料中训练集语音路径和文本的列表
    text_data_path = label_path
    files = os.listdir(wav_path)  # 得到数据文件夹下的所有文件名称list    
    audio_path_list = files

    # 获取语音路径list和文本list
    audio_data_path_list = [wav_path + "\\" + audio_path for audio_path in audio_path_list[:num_examples]]
    text_list = get_text_list(text_data_path)[:num_examples]
    return audio_data_path_list, text_list


# 加载数据
def load_data(dataset_name, wav_path, label_path, train_or_test, num_examples):
    # 基于某种语料获取其中语音路径和文本的list
    if dataset_name == "number":
        audio_data_path_list, text_list = load_dataset_number(wav_path, label_path, num_examples)

    # 训练则对数据进行预处理，测试则直接返回
    if train_or_test == "train":
        return build_train_data(audio_data_path_list, text_list)
    elif train_or_test == "test":
        return audio_data_path_list, text_list


# 读取文本文件，处理原始语料
def get_text_list(text_path):
    text_list = []
    with open(text_path, "r") as f:
        sentence_list = f.readlines()
    for sentence in sentence_list:
        text_list.append(sentence.strip().split("\t", 1)[1].lower())
    return text_list


def max_length(texts):
    return max(len(t) for t in texts)


# 获取最长的音频length(timesteps)
def get_max_audio_length(audio_data_path_list, audio_feature_type):
    max_audio_length = 0
    for audio_path in audio_data_path_list:
        audio_feature = features.wav_to_feature(audio_path, audio_feature_type)
        if (audio_feature):
            max_audio_length = max(max_audio_length, len(audio_feature))

    return max_audio_length


# 获取预处理得到的语料集信息
def get_dataset_information():
    with open("./dataset_information.json", "r", encoding="utf-8") as f:
        dataset_information = json.load(f)
    return dataset_information
