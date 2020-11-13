# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:20:05 2020
formatted
@author: 九童
文字语料预处理
"""
import io
import re
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


def create_input_text(path, num_examples):
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


# 非初次训练时，基于word_index和处理好的文本list得到数字list
def get_text_int_sequences(process_text_list, word_index):
    text_int_sequences = []
    for process_text in process_text_list:
        text_int_sequences.append(text_to_int_sequence(process_text, word_index))
    return text_int_sequences


# process_text转token序列
def text_to_int_sequence(process_text, word_index):
    int_sequence = []
    for c in process_text.split(" "):
        int_sequence.append(int(word_index[c]))
    return int_sequence


def load_text_data(path, num_examples=None):
    # 创建清理过的输入输出对    
    print("开始处理文本标签数据......")
    targ_lang = create_input_text(path, num_examples)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    print("文本标签数据处理完毕！")
    return target_tensor


# 基于原始text和其label_length的整形数字list来补齐label_tensor
def get_text_label(text_int_sequences, max_label_length):
    label_tensor_numpy = tf.keras.preprocessing.sequence.pad_sequences(
        text_int_sequences,
        maxlen=max_label_length,
        padding='post'
    )
    label_tensor = tf.convert_to_tensor(label_tensor_numpy)
    return label_tensor
