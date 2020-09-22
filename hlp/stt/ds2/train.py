# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:47:11 2020

@author: 彭康
"""

import tensorflow as tf
from utils import CTCLoss,WordAccuracy,wav_to_mfcc,text_to_int_sequence,sparse_tuple_from,int_to_text_sequence
from model import DS2


def train_sample(x, y, optimizer, model):
    #单次迭代，而且里边的input_length和label_length的处理有瑕疵，只是为了单样本设计的
    with tf.GradientTape() as tape:
               
        y_true=y
        y_pred=model(x)
        input_length=tf.constant([[y_pred.shape[1]]])
        label_length=tf.constant([[y_true.shape[1]]])
        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer, X, Y, epochs):
    #trian的迭代
    for step in range(1, epochs):
        loss = train_sample(X, Y, optimizer, model)
        print('Epoch {}, Loss: {}'.format(step, loss))


if __name__ == "__main__":
    epochs=1
    model=DS2(256,11,2,256,30)
    #提取了单个音频的特征(batch_size,timesteps,n_mfcc)，但只是一个list
    x=wav_to_mfcc(20,'./sample.wav')
    print("x:",x)
    x=tf.expand_dims(x, axis=0)
    x=tf.transpose(x,perm=[0,2,1])
    print(x.shape)
    
    #将文本数据转为(batch_size,vector_for_seq_length)的向量
    str='MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'.lower()
    y=text_to_int_sequence(str)
    y=tf.expand_dims(tf.convert_to_tensor(y), axis=0)
    print(y)
    """
    #稀疏矩阵，目前不需要
    tuple=sparse_tuple_from(y)
    y=tf.sparse.SparseTensor(tuple[0],tuple[1],tuple[2])
    print(y)
    """
    optimizer = tf.keras.optimizers.Adam()
    train(model, optimizer, x, y, epochs) 
    model.save('saved_model/my_model') 
    y=model.predict(x)
    output=tf.keras.backend.ctc_decode(y_pred=y,input_length=tf.constant([y.shape[1]]),greedy=True)
    out=output[0][0]
    str="".join(int_to_text_sequence(out.numpy()[0]))
    print(str)
    
    
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=CTCLoss(), metrics=[WordAccuracy()])
    
    history=model.fit(x=x,y=y,epochs=10)
    
    
    optimizer = tf.keras.optimizers.Adam()
    train(model, optimizer, x, y, 2) #先只跑一轮
    
    # getting the ctc output
    ctc_output = model(x)
    ctc_output = tf.nn.log_softmax(ctc_output)
    
    print(ctc_output)
    
    # greedy decoding
    space_token = ' '
    end_token = '>'
    blank_token = '%'
    alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]
    output_text = ''
    for timestep in ctc_output[0]:
        output_text += alphabet[tf.math.argmax(timestep)]
    print(output_text)
    print('\n\nNote: Applying a good decoder on this output will give you readable output')
    """