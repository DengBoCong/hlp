# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:34:04 2020
formatted
@author: 九童
"""
# !/usr/bin/env Python
# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
from model import las
from model.las_d_w import las_d_w
from config import config
from data_processing import load_dataset
from data_processing.generator import data_generator, val_generator
from util import compute_metric


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 填充位为0，掩蔽
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def train_step(inputx_1, targetx_2, enc_hidden, word_index, model, las_optimizer, train_batch_size):
    loss = 0

    with tf.GradientTape() as tape:
        inputx_1 = tf.convert_to_tensor(inputx_1)
        targetx_2 = tf.convert_to_tensor(targetx_2)

        # 解码器输入符号
        dec_input = tf.expand_dims([word_index['<start>']] * train_batch_size, 1)

        # 教师强制 - 将目标词作为下一个输入
        for t in range(1, targetx_2.shape[1]):
            # 将编码器输出 （enc_output） 传送至解码器，解码
            predictions, _ = model(inputx_1, enc_hidden, dec_input)
            loss += loss_function(targetx_2[:, t], predictions)  # 根据预测计算损失

            # 使用教师强制，下一步输入符号是训练集中对应目标符号
            dec_input = targetx_2[:, t]
            dec_input = tf.expand_dims(dec_input, 1)

    batch_loss = (loss / int(targetx_2.shape[1]))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    las_optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    return batch_loss


if __name__ == "__main__":
    # wav文件
    wav_path = config.train_wav_path
    # 标签文件
    label_path = config.train_label_path
    # 尝试实验不同大小的数据集
    num_examples = config.num_examples
    # 每一步mfcc所取得特征数
    n_mfcc = config.n_mfcc

    embedding_dim = config.embedding_dim
    units = config.units
    train_batch_size = config.train_batch_size
    dataset_name = config.dataset_name
    audio_feature_type = config.audio_feature_type
    print("加载训练数据......")
    train_wav_path_list, train_label_list = load_dataset.load_dataset_number(wav_path,
                                                                             label_path,
                                                                             num_examples)

    # 构建训练数据
    audio_data_path_list, text_int_sequences, _, _ = load_dataset.build_train_data(train_wav_path_list,
                                                                                   train_label_list)
    train_data = (audio_data_path_list, text_int_sequences)

    print("获取训练语料信息......")
    dataset_information = config.get_dataset_information()
    vocab_tar_size = dataset_information["vocab_tar_size"]
    batchs = len(audio_data_path_list) // train_batch_size
    optimizer = tf.keras.optimizers.Adam()
    #model = las.las_model(vocab_tar_size, embedding_dim, units, train_batch_size)
    d = 2
    w = 256
    embedding_dim = 256
    dec_units = 256
    model = las_d_w.las_d_w_model(vocab_tar_size, d, w, embedding_dim, dec_units, train_batch_size)
    print("构建训练数据生成器......")
    train_data_generator = data_generator(
        train_data,
        "train",
        batchs,
        train_batch_size,
        audio_feature_type,
        dataset_information["max_input_length"],
        dataset_information["max_label_length"]
    )
    validation_data = config.validation_data
    val_wav_path = config.val_wav_path
    val_label_path = config.val_label_path
    
    # 若validation_data为真，则有验证数据集，val_wav_path非空，则从文件路径中加载
    # 若validation_data为真，则有验证数据集，val_wav_path为空，则将训练数据按比例划分一部分为验证数据
    # 若validation_data为假，则没有验证数据集
    if validation_data:
        if val_wav_path:
            validation_size = config.validation_size
            val_wav_path_list, val_label_list = load_dataset.load_dataset_number(val_wav_path, val_label_path,
                                                                                 validation_size)
        else:
            validation_percent = config.validation_percent
            index = len(train_wav_path_list) * validation_percent // 100
            val_wav_path_list, val_label_list = train_wav_path_list[-index:], train_label_list[-index:]
            train_wav_list, train_label_list = train_wav_path_list[:-index], train_label_list[:-index]
        # 构建验证数据
        val_data = (val_wav_path_list, val_label_list)

        print("构建验证数据生成器......")
        val_batch_size = config.val_batch_size
        val_batchs = len(val_wav_path_list) // val_batch_size
        val_data_generator = val_generator(
            val_data,
            val_batchs,
            val_batch_size,
            audio_feature_type,
            dataset_information["max_input_length"]
        )

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
    print("恢复检查点目录 （checkpoint_dir） 中最新的检查点......")
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    EPOCHS = config.epochs

    word_index = dataset_information["word_index"]

    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = model.initialize_hidden_state()
        total_loss = 0
        batch_start = time.time()
        for batch, (inp, targ, target_length) in zip(range(1, batchs + 1), train_data_generator):
            x_1 = inp
            x_2 = targ
            batch_loss = train_step(x_1, x_2, enc_hidden, word_index,
                                    model,
                                    optimizer,
                                    train_batch_size)  # 训练一个批次，返回批损失
            total_loss += batch_loss

            if (batch + 1) % 2 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch + 1,
                                                             batch_loss.numpy()))
                print('Time taken for 2 batches {} sec\n'.format(time.time() - batch_start))
                batch_start = time.time()
        # 每 checkpoint_keep_interval 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % checkpoint_keep_interval == 0:
            manager.save()

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / batchs))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        # 验证
        if validation_data:
            rates_lers, aver_lers, norm_rates_lers, norm_aver_lers = compute_metric(model, val_data_generator,
                                                                                    val_batchs, val_batch_size)
            print("字母错误率: ")
            print("每条语音字母错误数: ", rates_lers)
            print("所有语音平均字母错误数: ", aver_lers)
            print("每条语音字母错误率，错误字母数/标签字母数: ", norm_rates_lers)
            print("所有语音平均字母错误率: ", norm_aver_lers)
