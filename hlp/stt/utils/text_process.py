'''
Author: PengKang6
Description: 文本的处理相关方法
'''

import re

import tensorflow as tf


def split_sentence(line, mode):
    '''此方法依据文本是中文文本还是英文文本，若为英文文本是按字符切分还是按单词切分

    :param text: 语料文件中每行对应文本
    :param mode: 语料文本的切分方法
    :return: 切分后的文本
    '''
    if mode.lower() == "cn":
        return _split_sentence_cn(line)
    elif mode.lower() == "en_word":
        return _split_sentence_en_word(line)
    elif mode.lower() == "en_char":
        return _split_sentence_en_char(line)


# 获取最长的label_length
def get_max_label_length(text_int_sequences):
    return max(len(seq) for seq in text_int_sequences)


def build_text_int_sequences(sentences, mode, word_index):
    # 基于文本按照某种mode切分文本
    splitted_sentences = split_sentences(sentences, mode)

    # 基于预处理时dataset_information中写入的word_index构建文本整形序列list
    text_int_sequences_list = get_text_int_sequences(splitted_sentences, word_index)

    return text_int_sequences_list


# 基于word_index和切割好的文本list得到数字序列list
def get_text_int_sequences(splitted_sentences, word_index):
    text_int_sequences = []
    for process_text in splitted_sentences:
        text_int_sequences.append(text_to_int_sequence(process_text, word_index))
    return text_int_sequences


# 对单行文本进行process_text转token整形序列
def text_to_int_sequence(process_text, word_index):
    int_sequence = []
    for c in process_text.split(" "):
        int_sequence.append(int(word_index[c]))
    return int_sequence


def split_sentences(sentences, mode):
    text_list = []
    for text in sentences:
        text_list.append(split_sentence(text, mode))
    return text_list


def get_label_and_length(text_int_sequences_list, max_label_length):
    target_length_list = []
    for text_int_sequence in text_int_sequences_list:
        target_length_list.append([len(text_int_sequence)])
    target_tensor_numpy = tf.keras.preprocessing.sequence.pad_sequences(
        text_int_sequences_list,
        maxlen=max_label_length,
        padding='post'
    )
    target_length = tf.convert_to_tensor(target_length_list)
    return target_tensor_numpy, target_length


def _split_sentence_en_word(s):
    s = s.lower().strip()
    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    s = s.strip()
    return s


def _split_sentence_en_char(s):
    s = s.lower().strip()

    result = ""
    for i in s:
        if i == " ":
            result += "<space> "
        else:
            result += i + " "
    return result.strip()


def _split_sentence_cn(s):
    s = s.lower().strip()

    s = [c for c in s]
    s = ' '.join(s)
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格
    s = s.strip()

    return s
