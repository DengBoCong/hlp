# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:47:11 2020

@author: 彭康
"""
import tensorflow as tf
import time
from data_preprocess import load_dataset_train
import json
from utils import get_index_word,get_config
from model import init_ds2


def train_step(inputs,labels,label_length,optimizer,model):
    #单次迭代
    with tf.GradientTape() as tape:
        y_pred=model(inputs)
        #print(y_pred)
        y_true=labels
        input_length=tf.fill([y_pred.shape[0],1],y_pred.shape[1])
        loss=tf.keras.backend.ctc_batch_cost(
            y_true=y_true,
            y_pred=y_pred,
            input_length=input_length,
            label_length=label_length
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

def train(model,optimizer,inputs,labels,label_length,epochs):
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
    BUFFER_SIZE = len(inputs)
    BATCH_SIZE = configs["train"]["batch_size"]
    batchs = BUFFER_SIZE//BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((inputs,labels,label_length)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    for epoch in range(1,epochs+1):
        epoch_start = time.time()
        total_loss = 0
        print("Epoch %d/%d" % (epoch,epochs))
        for (batch, (input_batch, target_batch, target_length_batch)) in enumerate(dataset.take(batchs)):
            batch_start = time.time()
            batch_loss = train_step(input_batch,target_batch,target_length_batch,optimizer,model)
            total_loss += batch_loss
            batch_end = time.time()
            batch_time = batch_start - batch_end
            print("Batch %d/%d" % (batch+1,batchs))
            print("batch_time: %ds - batch_loss: %.4f" % (batch_time,batch_loss))
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print("batchs: %d - epoch_time: %ds %dms/batch - loss: %.4f" % (batchs,epoch_time,epoch_time*1000/batchs,total_loss/batchs))
        if epoch % save_interval==0:
            manager.save()


if __name__ == "__main__":
    configs = get_config()
    epochs=configs["train"]["train_epochs"]
    data_path=configs["train"]["data_path"]
    num_examples = configs["train"]["num_examples"]
    input_tensor,target_tensor,target_length = load_dataset_train(data_path,num_examples)
    print(input_tensor.shape)
    index_word = get_index_word()
    model=init_ds2()
    """
    # 采用 90 - 10 的比例切分训练集和验证集
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)
    """
    optimizer = tf.keras.optimizers.Adam()
    train(model,optimizer,input_tensor,target_tensor,target_length,epochs)