# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:47:11 2020

@author: 彭康
"""
import data_process
import tensorflow as tf
from model import DS2


def train_sample(inputs,labels,label_length,optimizer, model):
    #单次迭代
    with tf.GradientTape() as tape:
        y_pred=model(inputs)
        y_true=labels
        input_length=tf.fill([y_pred.shape[0],1],y_pred.shape[1])
        loss=tf.keras.backend.ctc_batch_cost(y_true=y_true, y_pred=y_pred, input_length=input_length, label_length=label_length)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer,inputs,labels,label_length, epochs):
    #trian的迭代
    for epoch in range(1,epochs+1):
        loss = train_sample(inputs,labels,label_length,optimizer,model)
        #5次保存一个检查点并输出一个loss
        if epoch%5==0:
            checkpoint = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoint',max_to_keep=5)
            manager.save()
            print('Epoch {}, Loss: {}'.format(epoch, loss))


if __name__ == "__main__":
    #是否为初次训练
    is_first_train=False
    if is_first_train:
        model=DS2()
        
    else:
        model=DS2()
        #加载检查点
        checkpoint = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoint',max_to_keep=5)
        checkpoint.restore(manager.latest_checkpoint)
    
    epochs=1
    data_path="./data/train_data"
    inputs,labels,label_length=data_process.data_process1(data_path=data_path)
    optimizer = tf.keras.optimizers.Adam()
    train(model, optimizer, inputs, labels,label_length, epochs)
    
    #保存检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoint',max_to_keep=5)
    manager.save()
