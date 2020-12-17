# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import time

import tensorflow as tf

from hlp.stt.las.config import config
from hlp.stt.las.model import plas, las
from hlp.stt.las.util import compute_metric
from hlp.stt.utils import load_dataset
from hlp.stt.utils.audio_process import max_audio_length
from hlp.stt.utils.generator import train_generator, test_generator
from hlp.stt.utils.text_process import split_sentences, get_max_label_length, tokenize
from hlp.stt.utils.text_process import vectorize_texts
from hlp.utils.optimizers import loss_func_mask


def train_step(x_audio, y_label, enc_hidden, word_index, model, las_optimizer, train_batch_size):
    loss = 0

    with tf.GradientTape() as tape:
        x_audio = tf.convert_to_tensor(x_audio)
        y_label = tf.convert_to_tensor(y_label)

        # 解码器输入符号
        dec_input = tf.expand_dims([word_index['<start>']] * train_batch_size, 1)

        # 导师驱动 - 将目标词作为下一个输入
        for t in range(1, y_label.shape[1]):
            print(t)
            # 将编码器输出 （enc_output） 传送至解码器，解码
            predictions, _ = model(x_audio, enc_hidden, dec_input)
            loss += loss_func_mask(y_label[:, t], predictions)  # 根据预测计算损失

            # 使用导师驱动，下一步输入符号是训练集中对应目标符号
            dec_input = y_label[:, t]
            dec_input = tf.expand_dims(dec_input, 1)

    batch_loss = (loss / int(y_label.shape[1]))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    las_optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    return batch_loss


if __name__ == "__main__":
    # 训练集数据存放路径，包括音频文件路径和文本标签文件路径
    data_path = config.train_data_path
    # 尝试实验不同大小的数据集
    num_examples = config.num_examples
    # 确定使用的model类型
    model_type = config.model_type

    embedding_dim = config.embedding_dim
    units = config.units
    d = config.d
    w = config.w
    emb_dim = config.emb_dim
    dec_units = config.dec_units
    train_batch_size = config.train_batch_size
    dataset_name = config.dataset_name
    audio_feature_type = config.audio_feature_type
    text_process_mode = config.text_process_mode
    validation_data = config.validation_data

    print("加载训练数据......")
    train_wav_path_list, train_label_list = load_dataset.load_data(dataset_name,
                                                                   data_path,
                                                                   num_examples)

    print("数据预处理......")
    splitted_text_list = split_sentences(train_label_list, text_process_mode)

    # 将文本处理成对应的token数字序列
    text_int_sequences, tokenizer = tokenize(splitted_text_list)
    max_input_length = max_audio_length(train_wav_path_list, audio_feature_type)
    max_label_length = get_max_label_length(text_int_sequences)

    print("保存数据集信息...")
    ds_info_path = config.dataset_info_path
    dataset_info = {}
    dataset_info["vocab_tar_size"] = len(tokenizer.index_word) + 1
    dataset_info["max_input_length"] = max_input_length
    dataset_info["max_label_length"] = max_label_length
    dataset_info["index_word"] = tokenizer.index_word
    dataset_info["word_index"] = tokenizer.word_index

    with open(ds_info_path, 'w', encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)

    dataset_info = config.get_dataset_info()
    word_index = dataset_info["word_index"]
    max_input_length = dataset_info["max_input_length"]
    max_label_length = dataset_info["max_label_length"]

    # 若validation_data为真，则有验证数据集，val_wav_path非空，则从文件路径中加载
    # 若validation_data为真，则有验证数据集，val_wav_path为空，则将训练数据按比例划分一部分为验证数据
    # 若validation_data为假，则没有验证数据集
    if validation_data:
        val_data_path = config.val_data_path
        if val_data_path:
            validation_size = config.validation_size
            val_wav_path_list, val_label_list = load_dataset.load_data(dataset_name, val_data_path,
                                                                       validation_size)
        else:
            validation_percent = config.validation_percent
            index = len(train_wav_path_list) * validation_percent // 100
            val_wav_path_list, val_label_list = train_wav_path_list[-index:], train_label_list[-index:]
            train_wav_path_list, train_label_list = train_wav_path_list[:-index], train_label_list[:-index]
        val_data = (val_wav_path_list, val_label_list)

        print("构建验证数据生成器......")
        val_batch_size = config.val_batch_size
        val_batchs = len(val_wav_path_list) // val_batch_size
        val_data_generator = test_generator(val_data,
                                            val_batchs,
                                            val_batch_size,
                                            audio_feature_type,
                                            max_input_length
                                            )

    # 构建train_data
    train_text_int_sequences_list = vectorize_texts(train_label_list, text_process_mode, word_index)
    train_data = (train_wav_path_list, train_text_int_sequences_list)
    batchs = len(train_wav_path_list) // train_batch_size
    print("构建训练数据生成器......")
    train_data_generator = train_generator(train_data,
                                           batchs,
                                           train_batch_size,
                                           audio_feature_type,
                                           max_input_length,
                                           max_label_length
                                           )

    vocab_tar_size = dataset_info["vocab_tar_size"]
    optimizer = tf.keras.optimizers.Adam()

    # 选择模型类型
    if model_type == "las":
        model = plas.PLAS(vocab_tar_size, embedding_dim, units, train_batch_size)
    elif model_type == "las_d_w":
        model = las.LAS(vocab_tar_size, d, w, emb_dim, dec_units, train_batch_size)

    # 检查点
    checkpoint_dir = config.checkpoint_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, config.checkpoint_prefix)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=checkpoint_dir,
        max_to_keep=config.max_to_keep
    )
    checkpoint_keep_interval = config.checkpoint_keep_interval
    if manager.latest_checkpoint:
        print("恢复检查点目录 （checkpoint_dir） 中最新的检查点......")
        checkpoint.restore(manager.latest_checkpoint)

    print("开始训练...")
    EPOCHS = config.epochs
    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = model.initialize_hidden_state()
        total_loss = 0
        batch_start = time.time()
        for batch, (x_audio, y_label, _, _) in zip(range(1, batchs + 1), train_data_generator):
            batch_loss = train_step(x_audio, y_label, enc_hidden, word_index,
                                    model,
                                    optimizer,
                                    train_batch_size)  # 训练一个批次，返回批损失
            total_loss += batch_loss

            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch + 1,
                                                         batch_loss.numpy()))
            print('Time taken for 1 batch {} sec\n'.format(time.time() - batch_start))
            batch_start = time.time()

        # 每 checkpoint_keep_interval 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % checkpoint_keep_interval == 0:
            manager.save()

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / batchs))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        # 验证
        if validation_data:
            norm_rates_lers, norm_aver_lers = compute_metric(model, val_data_generator,
                                                             val_batchs, val_batch_size)
            print("平均字母错误率: ", norm_aver_lers)
