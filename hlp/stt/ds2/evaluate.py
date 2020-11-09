import tensorflow as tf
from math import ceil
from model import DS2, decode_output
from util import get_config, get_dataset_information

from data_process.load_dataset import load_data
from data_process.generator import data_generator

import sys
sys.path.append("..")
from utils.metric import wers, lers

if __name__ == "__main__":
    configs = get_config()
    dataset_information = get_dataset_information()
    
    # 获取模型配置，加载模型
    filters = configs["model"]["conv_filters"]
    kernel_size = configs["model"]["conv_kernel_size"]
    strides = configs["model"]["conv_strides"]
    gru_units = configs["model"]["gru_units"]
    fc_units = configs["model"]["fc_units"]
    dense_units = dataset_information["vocab_size"] + 2

    model = DS2(filters, kernel_size, strides, gru_units, fc_units, dense_units)

    # 加载模型检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=configs["checkpoint"]['directory'],
        max_to_keep=configs["checkpoint"]['max_to_keep']
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    test_data_path = configs["test"]["data_path"]
    num_examples = configs["test"]["num_examples"]
    dataset_name = configs["preprocess"]["dataset_name"]
    batch_size = configs["test"]["batch_size"]
    text_row_style = configs["preprocess"]["text_row_style"]
    mode = configs["preprocess"]["text_process_mode"]
    audio_feature_type = configs["other"]["audio_feature_type"]

    # 加载测试集数据生成器
    test_data = load_data(dataset_name, test_data_path, text_row_style, "test", num_examples)
    batchs = ceil(len(test_data[0]) / batch_size)
    test_data_generator = data_generator(
        test_data,
        "test",
        batchs,
        batch_size,
        audio_feature_type,
        dataset_information["max_input_length"],
        dataset_information["max_label_length"]
    )

    aver_wers = 0
    aver_lers = 0
    aver_norm_lers = 0
    
    # 获取index_word
    index_word = dataset_information["index_word"]

    for batch, (input_tensor, labels_list) in zip(range(1, batchs+1), test_data_generator):
        originals = labels_list
        results = []
        y_pred = model(input_tensor)
        output = tf.keras.backend.ctc_decode(
            y_pred=y_pred,
            input_length=tf.fill([y_pred.shape[0]], y_pred.shape[1]),
            greedy=True
        )
        results_int_list = output[0][0].numpy().tolist()

        # 解码
        for i in range(len(results_int_list)):
            str = decode_output(results_int_list[i], index_word, mode).strip()
            results.append(str)
        
        # 通过wer、ler指标评价模型
        rates_wer, aver_wer = wers(originals, results)
        rates_ler, aver_ler, norm_rates_ler, norm_aver_ler = lers(originals, results)
        aver_wers += aver_wer
        aver_lers += aver_ler
        aver_norm_lers += norm_aver_ler
    
    # 英文单词为token，则计算wer指标，其他(如中文、英文字符，计算ler指标)
    if configs["preprocess"]["text_process_mode"] == "en_word":
        print("WER:")
        print("aver:", aver_wers/batchs)
    else:
        print("LER:")
        print("aver:", aver_lers/batchs)
        print("aver_norm:", aver_norm_lers/batchs)
