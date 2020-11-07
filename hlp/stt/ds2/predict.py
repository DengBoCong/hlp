import tensorflow as tf
from audio_process import record, wav_to_feature
from model import DS2, decode_output
from util import get_config, get_dataset_information
import numpy as np

if __name__ == "__main__":
    configs = get_config()
    dataset_information = get_dataset_information()

    # 录音
    record_path = configs["record"]["record_path"]
    record(record_path)

    # 获取模型配置，加载模型
    conv_layers = configs["model"]["conv_layers"]
    filters = configs["model"]["conv_filters"]
    kernel_size = configs["model"]["conv_kernel_size"]
    strides = configs["model"]["conv_strides"]
    bi_gru_layers = configs["model"]["bi_gru_layers"]
    gru_units = configs["model"]["gru_units"]
    dense_units = dataset_information["dense_units"]
    model = DS2(conv_layers, filters, kernel_size, strides, bi_gru_layers, gru_units, dense_units)
    
    # 加载模型检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=configs["checkpoint"]['directory'],
        max_to_keep=configs["checkpoint"]['max_to_keep']
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    # 加载录音数据并预测
    # audio_path = "./data/LibriSpeech/train-clean-5/1088/134315/1088-134315-0000.flac"
    audio_path = configs["record"]["record_path"]
    audio_feature_type = configs["other"]["audio_feature_type"]
    x_test = wav_to_feature(audio_path, audio_feature_type)
    x_test_input = tf.keras.preprocessing.sequence.pad_sequences(
            [x_test],
            padding='post',
            maxlen=dataset_information["max_input_length"],
            dtype='float32'
            )
    y_test_pred = model(x_test_input)
    output = tf.keras.backend.ctc_decode(
        y_pred=y_test_pred,
        input_length=tf.constant([y_test_pred.shape[1]]),
        greedy=True
    )
    
    # 解码
    index_word = dataset_information["index_word"]
    mode = configs["preprocess"]["text_process_mode"]
    str = decode_output(output[0][0].numpy()[0], index_word, mode)
    print("Output:" + str)
