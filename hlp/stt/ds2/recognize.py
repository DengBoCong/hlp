import tensorflow as tf
import numpy as np
from model import DS2, decode_output
from util import get_config, get_dataset_information

from data_process.audio_process import record

import sys
sys.path.append("..")
from utils.features import wav_to_feature


if __name__ == "__main__":
    configs = get_config()
    dataset_information = get_dataset_information()

    # 获取模型配置，加载模型
    conv_layers = configs["model"]["conv_layers"]
    filters = configs["model"]["conv_filters"]
    kernel_size = configs["model"]["conv_kernel_size"]
    strides = configs["model"]["conv_strides"]
    bi_gru_layers = configs["model"]["bi_gru_layers"]
    gru_units = configs["model"]["gru_units"]
    fc_units = configs["model"]["fc_units"]
    dense_units = dataset_information["vocab_size"] + 2

    model = DS2(conv_layers, filters, kernel_size, strides, bi_gru_layers, gru_units, fc_units, dense_units)
    
    # 加载模型检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=configs["checkpoint"]['directory'],
        max_to_keep=configs["checkpoint"]['max_to_keep']
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    # 加载预测、解码所需的参数
    record_path = "./record.wav"
    audio_feature_type = configs["other"]["audio_feature_type"]
    index_word = dataset_information["index_word"]
    mode = configs["preprocess"]["text_process_mode"]
    max_input_length = dataset_information["max_input_length"]
    
    while True:
        try:
            record_duration = int(input("请设定录音时长(秒, <=0则结束):"))
        except BaseException:
            print("录音时长只能为int数值")
        else:
            if record_duration <= 0:
                break
            # 录音
            record(record_path, record_duration)

            # 加载录音数据并预测
            # record_path = "./1088-134318-0010.flac"
            x_test = wav_to_feature(record_path, audio_feature_type)
            x_test_input = tf.keras.preprocessing.sequence.pad_sequences(
                    [x_test],
                    padding='post',
                    maxlen=max_input_length,
                    dtype='float32'
                    )
            y_test_pred = model(x_test_input)
            output = tf.keras.backend.ctc_decode(
                y_pred=y_test_pred,
                input_length=tf.constant([y_test_pred.shape[1]]),
                greedy=True
            )
            
            # 解码
            str = decode_output(output[0][0].numpy()[0], index_word, mode)
            print("Output:" + str)
         