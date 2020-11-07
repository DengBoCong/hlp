# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:47:11 2020

@author: 彭康
"""
import time
from math import ceil
import tensorflow as tf
from generator import data_generator
from load_dataset import load_data
from model import DS2
from util import get_config, get_dataset_information


def train_step(input_tensor, target_tensor, target_length, optimizer, model):
    #单次训练
    with tf.GradientTape() as tape:
        y_pred=model(input_tensor)
        y_true=target_tensor
        input_length=tf.fill([y_pred.shape[0],1],y_pred.shape[1])
        loss=tf.keras.backend.ctc_batch_cost(
            y_true=y_true,
            y_pred=y_pred,
            input_length=input_length,
            label_length=target_length
            )
        """
        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        print(loss)
        mask = tf.cast(mask, dtype=loss.dtype)
        print(mask)
        loss *= mask
        print(loss)
        """
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer, train_data_generator, batchs, epochs):
    #加载检查点
    configs = get_config()
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=configs["checkpoint"]['directory'],
        max_to_keep=configs["checkpoint"]['max_to_keep']
        )
    save_interval = configs["checkpoint"]["save_interval"]
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
    
    #迭代训练
    for epoch in range(1, epochs+1):
        #epoch损失
        total_loss = 0
        epoch_start = time.time()
        print("Epoch %d/%d" % (epoch, epochs))
        for batch, (input_tensor, target_tensor, target_length) in zip(range(1,batchs+1), train_data_generator):
            
            print(input_tensor.shape,target_tensor.shape, target_length.shape)
            batch_start = time.time()
            batch_loss = train_step(input_tensor, target_tensor, target_length, optimizer, model)
            total_loss += batch_loss
            batch_end = time.time()
            
            # 打印批处理的信息
            print("Batch %d/%d" % (batch, batchs))
            print("batch_time: %dms - batch_loss: %.4f" % ((batch_end - batch_start)*1000, batch_loss))

        epoch_end = time.time()
        # 打印epoch的信息
        print("batchs: %d - epoch_time: %ds %dms/batch - loss: %.4f" % (batchs, epoch_end - epoch_start, (epoch_end-epoch_start)*1000/batchs, total_loss/batchs))
        
        # 按配置json文件里的save_interval的数值来保存检查点
        if epoch % save_interval == 0:
            manager.save()


if __name__ == "__main__":
    configs = get_config()

    epochs = configs["train"]["train_epochs"]
    data_path = configs["train"]["data_path"]
    num_examples = configs["train"]["num_examples"]
    dataset_name = configs["preprocess"]["dataset_name"]
    batch_size = configs["train"]["batch_size"]
    text_row_style = configs["preprocess"]["text_row_style"]
    audio_feature_type = configs["other"]["audio_feature_type"]

    # 加载数据并预处理
    train_data = load_data(dataset_name, data_path, text_row_style, "train", num_examples)
    batchs = ceil(len(train_data[0]) / batch_size)

    # 获取训练语料信息
    dataset_information = get_dataset_information()
    
    # 构建数据生成器
    train_data_generator = data_generator(
        train_data,
        "train",
        batchs,
        batch_size,
        audio_feature_type,
        dataset_information["max_input_length"],
        dataset_information["max_label_length"]
    )

    # 加载模型
    conv_layers = configs["model"]["conv_layers"]
    filters = configs["model"]["conv_filters"]
    kernel_size = configs["model"]["conv_kernel_size"]
    strides = configs["model"]["conv_strides"]
    bi_gru_layers = configs["model"]["bi_gru_layers"]
    gru_units = configs["model"]["gru_units"]
    
    dense_units = dataset_information["dense_units"]

    model = DS2(conv_layers, filters, kernel_size, strides, bi_gru_layers, gru_units, dense_units)
    optimizer = tf.keras.optimizers.Adam()
    #训练
    train(model, optimizer, train_data_generator, batchs, epochs)
    