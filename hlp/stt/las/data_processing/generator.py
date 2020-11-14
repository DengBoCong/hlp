# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:27:45 2020

@author: 九童
"""
import numpy as np
import tensorflow as tf
from hlp.stt.utils import features
from hlp.stt.las.data_processing import preprocess_text



# 数据生成器
def data_generator(data, train_or_test, batchs, batch_size, audio_feature_type, max_input_length, max_label_length):
    if train_or_test == "train":
        audio_data_path_list, text_int_sequences, label_length_list, _ = data
        print("audio_data_path_list ===== {}".format(len(audio_data_path_list)))
        print("text_int_sequences ===== {}".format(len(text_int_sequences)))
        print("label_length_list ===== {}".format(len(label_length_list)))
        print("batchs ===== {}".format(batchs))
        # generator只能进行一次生成，故需要while True来进行多个epoch的数据生成
        while True:
            # 每epoch将所有数据进行一次shuffle
            order = np.random.choice(len(audio_data_path_list), len(audio_data_path_list), replace=False)
            audio_data_path_list = [audio_data_path_list[i] for i in order]
            text_int_sequences = [text_int_sequences[i] for i in order]
            label_length_list = [label_length_list[i] for i in order]

            for idx in range(batchs):
                batch_input_tensor = get_input_tensor(
                    audio_data_path_list[idx * batch_size: (idx + 1) * batch_size],
                    audio_feature_type,
                    max_input_length
                )

                batch_label_tensor = preprocess_text.get_text_label(
                    text_int_sequences[idx * batch_size: (idx + 1) * batch_size],
                    max_label_length
                )
                batch_label_length = tf.convert_to_tensor(label_length_list[idx * batch_size: (idx + 1) * batch_size])

                yield batch_input_tensor, batch_label_tensor, batch_label_length
    
    elif train_or_test == "test":
        audio_data_path_list, text_list = data

        for idx in range(batchs):
            batch_input_tensor = get_input_tensor(
                audio_data_path_list[idx*batch_size : (idx+1)*batch_size],
                audio_feature_type,
                max_input_length
                )
            batch_text_list = text_list[idx*batch_size : (idx+1)*batch_size]

            #测试集只需要文本串list
            yield batch_input_tensor, batch_text_list

    
# 基于语音路径序列，处理成模型的输入tensor
def get_input_tensor(audio_data_path_list, audio_feature_type, maxlen):
    audio_feature_list = []
    for audio_path in audio_data_path_list:
        audio_feature = features.wav_to_feature(audio_path, audio_feature_type)
        audio_feature_list.append(audio_feature)

    audio_feature_numpy = tf.keras.preprocessing.sequence.pad_sequences(
        audio_feature_list,
        maxlen=maxlen,
        padding='post',
        dtype='float32'
        )
    input_tensor = tf.convert_to_tensor(audio_feature_numpy)

    return input_tensor
