# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:47:11 2020

@author: 彭康
"""
import time
from math import ceil
import tensorflow as tf
from load_dataset import load_data
from model import get_ds2_model
from utils import get_config


def train_step(input_tensor, target_tensor, target_length, optimizer, model):
    #单次训练
    with tf.GradientTape() as tape:
        y_pred=model(input_tensor)
        #print(y_pred)
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

def train(model, optimizer, train_data_generator, epochs):
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
        batch = 1
        for batchs, (input_tensor, target_tensor, target_length) in train_data_generator:
            batch_start = time.time()
            batch_loss = train_step(input_tensor, target_tensor, target_length, optimizer, model)
            total_loss += batch_loss
            batch_end = time.time()
            
            # 打印批处理的信息
            print("Batch %d/%d" % (batch, batchs))
            print("batch_time: %dms - batch_loss: %.4f" % ((batch_end - batch_start)*1000, batch_loss))
            if batch < batchs:
                batch += 1
            else:
                break

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
    #加载数据生成器
    train_data_generator = load_data(data_path, "train", num_examples)
    
    model = get_ds2_model()
    optimizer = tf.keras.optimizers.Adam()
    #训练
    train(model, optimizer, train_data_generator, epochs)
