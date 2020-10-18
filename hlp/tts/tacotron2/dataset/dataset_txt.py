from __future__ import absolute_import, division, print_function, unicode_literals

import io
import re

import tensorflow as tf


def preprocess_en_sentence(s):
    s = s.lower().strip()
    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格
    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    s = s.strip()
    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'
    return s


def create_dataset(path, num_examples):
    # lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    en_sentences = [l.split('\t')[0] for l in lines[:num_examples]]
    en_sentences = [preprocess_en_sentence(s) for s in en_sentences]
    return en_sentences


def tokenize(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # 无过滤字符
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                              padding='post')
    return sequences, tokenizer


def Dataset_txt(path_to_file):
    en = create_dataset(path_to_file, None)
    en_seqs, en_tokenizer = tokenize(en)
    return en_seqs, en_tokenizer
