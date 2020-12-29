# -*- coding: utf-8 -*-
import os

import tensorflow as tf

from hlp.stt.las.config import config
from hlp.stt.las.model import plas, las
from hlp.stt.utils import load_dataset
from hlp.stt.utils.generator import test_generator
from hlp.stt.utils.utils import lers
from hlp.utils import beamsearch

if __name__ == "__main__":

    # 测试集数据存放路径，包括音频文件路径和文本标签文件路径
    data_path = config.test_data_path
    
    # 尝试实验不同大小的数据集
    test_num = config.test_num

    # 每一步mfcc所取得特征数
    n_mfcc = config.n_mfcc

    # 确定使用的model类型
    model_type = config.model_type

    embedding_dim = config.embedding_dim
    units = config.units
    cnn1_filters = config.cnn1_filters
    cnn1_kernel_size = config.cnn1_kernel_size
    cnn2_filters = config.cnn2_filters
    cnn2_kernel_size = config.cnn2_kernel_size
    max_pool_strides = config.max_pool_strides
    max_pool_size = config.max_pool_size
    d = config.d
    w = config.w
    emb_dim = config.emb_dim
    dec_units = config.dec_units
    batch_size = config.test_batch_size
    dataset_name = config.dataset_name
    audio_feature_type = config.audio_feature_type
    num_examples = config.test_num

    print("获取训练语料信息......")
    dataset_information = config.get_dataset_info()
    test_vocab_tar_size = dataset_information["vocab_tar_size"]
    optimizer = tf.keras.optimizers.Adam()

    # 选择模型类型
    if model_type == "las":
        model = plas.PLAS(test_vocab_tar_size, embedding_dim, units, batch_size)
    elif model_type == "las_d_w":
        model = las.LAS(test_vocab_tar_size,
                        cnn1_filters,
                        cnn1_kernel_size,
                        cnn2_filters,
                        cnn2_kernel_size,
                        max_pool_strides,
                        max_pool_size,
                        d,
                        w,
                        emb_dim,
                        dec_units,
                        batch_size)
    # 检查点
    checkpoint_dir = config.checkpoint_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, config.checkpoint_prefix)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # 恢复检查点目录 （checkpoint_dir） 中最新的检查点
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    results = []
    labels_list = []

    # 加载测试集数据生成器
    test_data = load_dataset.load_data(dataset_name, data_path, num_examples)
    batchs = len(test_data[0]) // batch_size
    print("构建数据生成器......")
    test_data_generator = test_generator(test_data,
                                         batchs,
                                         batch_size,
                                         audio_feature_type,
                                         dataset_information["max_input_length"])

    word_index = dataset_information["word_index"]
    index_word = dataset_information["index_word"]
    max_label_length = dataset_information["max_label_length"]
    beam_search_container = beamsearch.BeamSearchDecoder(
        beam_size=config.beam_size,
        min_score=0)

    for batch, (inp, _, targ) in zip(range(1, batchs + 1), test_data_generator):
        hidden = model.initialize_hidden_state()
        dec_input = tf.expand_dims([word_index['<start>']] * batch_size, 1)
        beam_search_container.reset(dec_inputs=dec_input)
        decoder_input = beam_search_container.get_candidates()
        result = ''  # 识别结果字符串

        for t in range(max_label_length):  # 逐步解码或预测
            decoder_input = decoder_input[:, -1:]
            predictions, dec_hidden = model(inp, hidden, decoder_input, len(beam_search_container))
            beam_search_container.expand(predictions=predictions, end_sign=word_index['<end>'])
            if beam_search_container.beam_size == 0:
                break
            decoder_input = beam_search_container.get_candidates()
        beam_search_result = beam_search_container.get_result()
        beam_search_result = tf.squeeze(beam_search_result)
        for i in range(len(beam_search_result)):
            idx = str(beam_search_result[i].numpy())
            if index_word[idx] == '<end>':
                break
            elif index_word[idx] != '<start>':
                result += index_word[idx]

        results.append(result)
        labels_list.append(targ[0])
    norm_rates_lers, norm_aver_lers = lers(labels_list, results)

    print("平均字母错误率: ", norm_aver_lers)
