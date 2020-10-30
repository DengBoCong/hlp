# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:20:05 2020
formatted
@author: 九童

文字语料预处理
"""
import io
import re
import numpy as np
import tensorflow as tf


def preprocess_ch_sentence(s):
    s = [c for c in s]
    s = ' '.join(s)
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格
    s = s.strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'

    return s


# 对英文句子：小写化，切分句子，添加开始和结束标记
def preprocess_en_sentence(s):
    s = s.lower().strip()
    s = ' '.join(s)
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)

    s = s.strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'

    return s


def create_input_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    en_sentences = [l.split('\t')[1] for l in lines[:num_examples]]
    en_sentences = [preprocess_en_sentence(s) for s in en_sentences]

    return en_sentences


def tokenize(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')  # 无过滤字符
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                              padding='post')
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
    print("开始处理文本标签数据......")
    targ_lang = create_input_dataset(path, num_examples)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return target_tensor, targ_lang_tokenizer
