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
import time
import tensorflow as tf
from hlp.stt.las.data_processing import librosa_mfcc
from hlp.stt.las.data_processing import preprocess_text
from hlp.stt.las.model import encoder, decoder



# 创建一个 tf.data 数据集
def create_dataset(input_x, target_y, BATCH_SIZE):
    BUFFER_SIZE = len(input_x)
    steps_per_epoch = len(input_x) // BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((input_x, target_y)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return steps_per_epoch, dataset


def loss_function(real, pred, targ_tokenizer):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 填充位为0，掩蔽
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)



def train_step(inputx_1, targetx_2, enc_hidden, targ_tokenizer, encoder, decoder, las_optimizer, BATCH_SIZE):
    loss = 0
    with tf.GradientTape() as tape:
        inputx_1 = tf.convert_to_tensor(inputx_1)
        targetx_2 = tf.convert_to_tensor(targetx_2)
        enc_output, enc_hidden = encoder(inputx_1, enc_hidden)  # 前向计算，编码
        dec_hidden = enc_hidden  # 编码器状态作为解码器初始状态？
        # 解码器输入符号
        dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        predicted_result = ''
        target_result = ''
        # 教师强制 - 将目标词作为下一个输入
        for t in range(1, targetx_2.shape[1]):
            # 将编码器输出 （enc_output） 传送至解码器，解码
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            predicted_id = tf.argmax(predictions[0]).numpy()  # 贪婪解码，取最大      
            if predicted_id == 0:
                result = ' '  # 目标句子
            else:
                result = targ_tokenizer.index_word[predicted_id] + ' '  # 目标句子
            predicted_result += result
            target_id = targetx_2[:, t].numpy()
            # 如果是填充位0
            if target_id[0] == 0:
                target_result += ' '
            else:
                target_result += targ_tokenizer.index_word[target_id[0]] + ' '  # 目标句子
            loss += loss_function(targetx_2[:, t], predictions, targ_tokenizer)  # 根据预测计算损失
            # 使用教师强制，下一步输入符号是训练集中对应目标符号
            dec_input = targetx_2[:, t]
            dec_input = tf.expand_dims(dec_input, 1)

    batch_loss = (loss / int(targetx_2.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    las_optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    return batch_loss, las_optimizer


if __name__ == "__main__":
    # wav文件
    path = ".\\data\\wav_train"
    # 语音识别语料文件
    path_to_file = ".\\data\\data_train.txt"
    # 尝试实验不同大小的数据集
    num_examples = 1928
    # 每一步mfcc所取得特征数
    n_mfcc = 20
    input_tensor = librosa_mfcc.wav_to_mfcc(path, n_mfcc)
    target_tensor, targ_lang_tokenizer = preprocess_text.load_dataset(path_to_file, num_examples)
    embedding_dim = 256
    units = 512
    BATCH_SIZE = 64
    vocab_tar_size = len(targ_lang_tokenizer.word_index) + 1  # 含填充的0
    # 计算目标张量的最大长度 （max_length）
    max_length_targ = preprocess_text.max_length(target_tensor)
    max_length_inp = preprocess_text.max_length(input_tensor)
    steps_per_epoch, dataset = create_dataset(input_tensor, target_tensor, BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()
    encoder = encoder.Encoder(n_mfcc, embedding_dim, units, BATCH_SIZE)
    decoder = decoder.Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    # 检查点
    checkpoint_dir = './lastraining_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder,
                                     decoder=decoder)
    # 恢复检查点目录 （checkpoint_dir） 中最新的检查点
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    EPOCHS = 12

    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_start = time.time()
            x_1 = inp
            x_2 = targ
            batch_loss, optimizer = train_step(x_1, x_2, enc_hidden, targ_lang_tokenizer, encoder, decoder, optimizer,
                                               BATCH_SIZE)  # 训练一个批次，返回批损失
            total_loss += batch_loss

            if batch % 20 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
                print('Time taken for 20 batches {} sec\n'.format(time.time() - batch_start))

        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
