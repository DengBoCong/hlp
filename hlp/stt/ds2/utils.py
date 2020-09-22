# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:15:16 2020

@author: 彭康
"""

import tensorflow as tf
import librosa
import char_index_map
import numpy as np

#音频的处理
def wav_to_mfcc(n_mfcc,wav_path):
    #加载音频
    y, sr = librosa.load(wav_path,sr=None)
    #提取mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def text_to_int_sequence(text):
    """ Use a character map and convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_index_map.char_map['<SPACE>']
        else:
            ch = char_index_map.char_map[c]
        int_sequence.append(ch)
    return int_sequence

def int_to_text_sequence(seq):
    """ Use a index map and convert int to a text sequence
        >>> from utils import int_to_text_sequence
        >>> a = [2,22,10,11,21,2,13,11,6,1,21,2,8,20,17]
        >>> b = int_to_text_sequence(a)
    """
    text_sequence = []
    for c in seq:
        if c == 28: #ctc/pad char
            ch = ''
        else:
            ch = char_index_map.index_map[c]
        text_sequence.append(ch)
    #"".join(text_sequence)就会转成字符串
    return text_sequence 

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = [] # 位置
    values = [] # 具体的值

    for n, seq in enumerate(sequences): # sequences是一个二维list
        indices.extend(zip([n]*len(seq), range(len(seq)))) # 生成所有值的坐标，不管是不是0，都存下来
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64) # shape的行就是seqs的个数，列就是最长的那个seq的长度

    return indices, values, shape


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, logits_time_major=False, blank_index=-1, 
                  name='ctc_loss'):
        super().__init__(name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        logit_length = tf.fill([tf.shape(y_pred)[0]],tf.shape(y_pred)[1])
        label_length = tf.fill([tf.shape(y_true)[0]],tf.shape(y_true)[1])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index)
        return tf.reduce_mean(loss)
    
class WordAccuracy(tf.keras.metrics.Metric):
    """
    Calculate the word accuracy between y_true and y_pred.
    """
    def __init__(self, name='word_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
                
    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0]
        max_width = tf.maximum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])        
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        y_true = self.to_dense(y_true, [batch_size, max_width])
        y_pred = self.to_dense(decoded[0], [batch_size, max_width])
        num_errors = tf.math.reduce_any(
            tf.math.not_equal(y_true, y_pred), axis=1)
        num_errors = tf.cast(num_errors, tf.float32)
        num_errors = tf.reduce_sum(num_errors)
        batch_size = tf.cast(batch_size, tf.float32)
        self.total.assign_add(batch_size)
        self.count.assign_add(batch_size - num_errors)

    def to_dense(self, tensor, shape):
        tensor = tf.sparse.reset_shape(tensor, shape)
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor = tf.cast(tensor, tf.float32)
        return tensor

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)