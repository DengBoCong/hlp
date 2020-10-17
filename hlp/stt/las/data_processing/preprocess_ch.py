# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:20:05 2020

@author: 九童

中文语料预处理
"""
import re
import io
import tensorflow as tf
import numpy as np


def preprocess_ch_sentence(s):
    s = [c for c in s]
    s = ' '.join(s)
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    s = s.strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'
    return s


def create_input_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    ch_sentences = [l.split(' ')[1:] for l in lines[:num_examples]]
    ch_sentences = [''.join(word) for word in ch_sentences]

    ch_sentences = [preprocess_ch_sentence(s) for s in ch_sentences]
    return ch_sentences


def tokenize(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')  # 无过滤字符
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列

    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                              padding='post', value=len(tokenizer.word_index) + 1)

    print('=====len(tokenizer.word_index) = {}'.format(len(tokenizer.word_index)))
    return sequences, tokenizer


def max_length(texts):
    return max(len(t) for t in texts)


def tensor_to_onehot(tensor, tokenizer):
    tensor = tensor.tolist()
    for _, sentence in enumerate(tensor):
        for index, word in enumerate(sentence):
            word = tf.keras.utils.to_categorical(word - 1, num_classes=len(tokenizer.word_index) + 1)
            sentence[index] = word
    tensor = np.array(tensor).astype(int)
    return tensor


def load_dataset(path, num_examples=None):
    # 创建清理过的输入输出对
    targ_lang = create_input_dataset(path, num_examples)

    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    # target_tensor = tf.convert_to_tensor(target_tensor)
    target_tensor = tensor_to_onehot(target_tensor, targ_lang_tokenizer)
    return target_tensor, targ_lang_tokenizer


