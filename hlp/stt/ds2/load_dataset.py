from utils import get_config, get_all_data_path, get_word_index, set_config
import os
import numpy as np
from audio_process import get_audio_feature
from text_process import get_text_label, get_lables_list
from math import ceil
from data_preprocess import data_preprocess


#加载数据
def load_data(data_path, train_or_test, num_examples):
    configs = get_config()
    dataset_name = configs["preprocess"]["dataset_name"]
    if dataset_name == "number":
        if train_or_test == "train":
            # input_tensor, target_tensor, label_length的训练数据集生成器
            train_data_generator = load_dataset_number(data_path, train_or_test, num_examples)
            return train_data_generator
        else:
            # input_tensor, labels_list的测试数据集生成器
            test_data_generator = load_dataset_number(data_path, train_or_test, num_examples)
            return test_data_generator

#对number语料构建的数据加载方法
def load_dataset_number(data_path, train_or_test, num_examples = None):
    #加载配置文件
    configs = get_config()

    #number语料里没有文本集，故先通过文件名构建文本集
    if not os.path.exists(data_path + "/data.txt"):
        files = os.listdir(data_path)
        with open(data_path + "/data.txt", "a") as f:
            for path in files:
                f.write(path[0] + "\n")
    
    if train_or_test == "train":
        #获取word_index
        word_index = get_word_index()

        text_data_path, audio_data_path_list = get_all_data_path(data_path)
        
        BATCH_SIZE = configs["train"]["batch_size"]
        BUFFER_SIZE = len(audio_data_path_list[:num_examples])
        BATCHS = ceil(BUFFER_SIZE / BATCH_SIZE)
        
        #训练轮数一般较多
        while True:
            #batch序列打散
            order = np.arange(BATCHS)
            np.random.shuffle(order)
            for i in order:
                yield BATCHS, get_train_data_generator(data_path, text_data_path, audio_data_path_list, i, BATCH_SIZE, word_index)
    else:
        text_data_path, audio_data_path_list = get_all_data_path(data_path)

        BATCH_SIZE = configs["test"]["batch_size"]
        BUFFER_SIZE = len(audio_data_path_list[:num_examples])
        BATCHS = ceil(BUFFER_SIZE / BATCH_SIZE)

        order = np.arange(BATCHS)
        np.random.shuffle(order)
        for i in order:
            yield BATCHS, get_test_data_generator(data_path, text_data_path, audio_data_path_list, i, BATCH_SIZE)

def get_train_data_generator(data_path, text_data_path, audio_data_path_list, i, BATCH_SIZE, word_index):
    input_tensor = get_audio_feature(data_path, audio_data_path_list[i*BATCH_SIZE : (i+1)*BATCH_SIZE])
    target_tensor, target_length = get_text_label(data_path, text_data_path, i, BATCH_SIZE, word_index)
    return input_tensor, target_tensor, target_length

def get_test_data_generator(data_path, text_data_path, audio_data_path_list, i, BATCH_SIZE):
    input_tensor = get_audio_feature(data_path, audio_data_path_list[i*BATCH_SIZE : (i+1)*BATCH_SIZE])
    labels_list = get_lables_list(data_path, text_data_path, i, BATCH_SIZE)
    return input_tensor, labels_list