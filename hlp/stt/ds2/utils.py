# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:15:16 2020

@author: 彭康
"""

import os
import time
import wave

import librosa
import numpy as np
import pyaudio
import tensorflow as tf

import char_index_map
import config


#音频的处理
def wav_to_mfcc(n_mfcc,wav_path):
    #加载音频
    y, sr = librosa.load(wav_path,sr=None)
    #提取mfcc(返回list(timestep,n_mfcc))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).transpose(1,0).tolist()
    return mfcc

def text_to_int_sequence(text):
    #字符序列list转成数字label list
    int_sequence = []
    for ch in text:
        if ch == ' ':
            c = char_index_map.char_map['<SPACE>']
        else:
            c = char_index_map.char_map[ch]
        int_sequence.append(c)
    return int_sequence


def int_to_text_sequence(seq):
    #数字label list转成字符序列list
    text_sequence = []
    for c in seq:
        if c>=1 and c<=(len(char_index_map.index_map)):
            ch = char_index_map.index_map[c]
        else:
            ch = ''
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
        
#输入的两个参数均是字符串的list,是wer计算的入口
def wers(originals, results):
    count = len(originals)
    try:
        assert count > 0
    except:
        print(originals)
        raise("ERROR assert count>0 - looks like data is missing")
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = _wer(originals[i], results[i])
        mean = mean + rate
        rates.append(rate)
    return rates, mean / float(count)

def _wer(original, result):
    r"""
    The WER is defined as the editing/Levenshtein distance on word level
    divided by the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    # The WER ist calculated on word (and NOT on character) level.
    # Therefore we split the strings into words first:
    original = original.split()
    result = result.split()
    return _levenshtein(original, result) / float(len(original))

def lers(originals, results):
    count = len(originals)
    assert count > 0
    rates = []
    norm_rates = []

    mean = 0.0
    norm_mean = 0.0

    assert count == len(results)
    for i in range(count):
        rate = _levenshtein(originals[i], results[i])
        mean = mean + rate

        normrate = (float(rate) / len(originals[i]))

        norm_mean = norm_mean + normrate

        rates.append(rate)
        norm_rates.append(round(normrate, 4))

    return rates, (mean / float(count)), norm_rates, (norm_mean/float(count))

def _levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

#获取麦克风录音并保存在filepath中
def record(file_path=config.configs_record["record_path"]):
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 1                # 声道数
        RATE = 16000               # 采样率
        RECORD_SECONDS = config.configs_record["record_times"]        #录音时长
        WAVE_OUTPUT_FILENAME = file_path
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print("开始录音：请在%d秒内输入语音:",RECORD_SECONDS)
        frames = []
        for i in range(1,int(RATE / CHUNK * RECORD_SECONDS)+1):
            data = stream.read(CHUNK)
            frames.append(data)
            if (i % (RATE / CHUNK)) == 0:
                print('\r%s%d%s' % ("剩余",int(RECORD_SECONDS-(i//(RATE / CHUNK))),"秒"),end="")
        print("\n录音结束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

if __name__ == "__main__":
    """
    #测试一下录音方法
    record()
    """