import os
from text_process import tokenize, text_row_process, get_text_data
from audio_process import get_audio_feature
import tensorflow as tf
from utils import get_config, set_config
import json


#对number语料构建的数据加载方法
def load_dataset_number(data_path, train_or_test, num_examples = None):
    #number语料里没有文本集，故先通过文件名构建文本集
    if not os.path.exists(data_path + "/data.txt"):
        files = os.listdir(data_path)
        with open(data_path + "/data.txt", "a") as f:
            for path in files:
                f.write(path[0] + "\n")
    #训练集
    if train_or_test == "train":
        #构建数据集
        text_data_path, audio_data_path_list = get_all_data_path(data_path)
        mfccs_list, sentences_list = create_dataset(
            data_path,
            text_data_path,
            audio_data_path_list,
            num_examples
            )
        mfccs_numpy = tf.keras.preprocessing.sequence.pad_sequences(
            mfccs_list,
            padding='post',
            dtype='float32'
            )
        input_tensor = tf.convert_to_tensor(mfccs_numpy)
        target_sequence, sentences_length_list, target_tokenizer = tokenize(sentences_list)
        target_tensor = tf.convert_to_tensor(target_sequence)
        target_length = tf.convert_to_tensor(sentences_length_list)

        #保存index_word到json文件
        configs = get_config()
        index_word_path = configs["other"]["index_word_path"]
        with open(index_word_path, 'w', encoding="utf-8") as f:
            json.dump(target_tokenizer.index_word, f, ensure_ascii=False, indent=4)
        
        #基于训练数据将输入序列的长度以及模型dense层所需的单元数(即token集合数量)写入配置文件
        set_config("preprocess", "max_inputs_len", input_tensor.shape[1])
        set_config("model", "dense_units", len(target_tokenizer.index_word) + 2)
        return input_tensor, target_tensor, target_length
    #测试集
    else:
        configs = get_config()
        text_data_path, audio_data_path_list = get_all_data_path(data_path)
        mfccs_list = get_audio_feature(data_path, audio_data_path_list, num_examples)

        #测试集所需的label字符串list
        labels_list = []
        with open(data_path + "/" + text_data_path, "r") as f:
            sen_list = f.readlines()
        for sentence in sen_list[:num_examples]:
            labels_list.append(text_row_process(sentence))
        
        #截取所需的样本集合
        mfccs_list = mfccs_list[:num_examples]
        labels_list = labels_list[:num_examples]
        mfccs_numpy = tf.keras.preprocessing.sequence.pad_sequences(
                mfccs_list,
                maxlen=configs["preprocess"]["max_inputs_len"],
                padding='post',
                dtype='float32'
                )
        input_tensor = tf.convert_to_tensor(mfccs_numpy)
        return input_tensor, labels_list

#根据数据文件夹名获取所有的文件名，包括文本文件名和音频文件名列表
def get_all_data_path(data_path):
    #data_path是数据文件夹的路径
    files = os.listdir(data_path) #得到数据文件夹下的所有文件名称list
    text_data_path = files.pop()
    audio_data_path_list = files
    return text_data_path, audio_data_path_list

#分别获取语音文件和文本文件
def create_dataset(data_path,text_data_path,audio_data_path_list,num_examples):
    mfccs_list = get_audio_feature(data_path,audio_data_path_list,num_examples)
    sentences_list = get_text_data(data_path,text_data_path,num_examples)
    return mfccs_list, sentences_list