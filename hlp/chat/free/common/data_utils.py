import io
import os
import json
import jieba
import numpy as np
import tensorflow as tf
from pathlib import Path
import config.get_config as _config


def preprocess_sentence(start_sign, end_sign, w):
    """
    用于给句子首尾添加start和end
    :param w:
    :return: 合成之后的句子
    """
    w = start_sign + ' ' + w + ' ' + end_sign
    return w


def preprocess_request(sentence, token):
    sentence = " ".join(jieba.cut(sentence))
    sentence = preprocess_sentence(sentence, _config.start_sign, _config.end_sign)
    inputs = [token.get(i, 3) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=_config.max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    dec_input = tf.expand_dims([token[_config.start_sign]], 0)

    return inputs, dec_input


def create_dataset(path, num_examples, start_sign, end_sign):
    """
    用于将分词文本读入内存，并整理成问答对
    :param path:
    :param num_examples:
    :return: 整理好的问答对
    """
    is_exist = Path(path)
    if not is_exist.exists():
        file = open(path, 'w', encoding='utf-8')
        file.write('吃饭 了 吗' + '\t' + '吃 了')
        file.close()
    size = os.path.getsize(path)
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')
    if num_examples == 0:
        word_pairs = [[preprocess_sentence(start_sign, end_sign, w) for w in l.split('\t')] for l in lines]
    else:
        word_pairs = [[preprocess_sentence(start_sign, end_sign, w) for w in l.split('\t')] for l in
                      lines[:num_examples]]

    return zip(*word_pairs)


def max_length(tensor):
    """
    :param tensor:
    :return: 列表中最大的长度
    """
    return max(len(t) for t in tensor)


def read_data(path, num_examples, start_sign, end_sign):
    """
    读取数据，将input和target进行分词后返回
    :param path: Tokenizer文本路径
    :param num_examples: 最大序列长度
    :return: input_tensor, target_tensor, lang_tokenizer
    """
    input_lang, target_lang = create_dataset(path, num_examples, start_sign, end_sign)
    input_tensor, target_tensor, lang_tokenizer = tokenize(input_lang, target_lang)
    return input_tensor, target_tensor, lang_tokenizer


def tokenize(input_lang, target_lang):
    """
    分词方法，使用Keras API中的Tokenizer进行分词操作
    :param input_lang: 输入
    :param target_lang: 目标
    :return: input_tensor, target_tensor, lang_tokenizer
    """
    lang = np.hstack((input_lang, target_lang))
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    input_tensor = lang_tokenizer.texts_to_sequences(input_lang)
    target_tensor = lang_tokenizer.texts_to_sequences(target_lang)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=_config.max_length_inp,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=_config.max_length_inp,
                                                                  padding='post')

    return input_tensor, target_tensor, lang_tokenizer


def create_padding_mask(input):
    """
    对input中的padding单位进行mask
    :param input:
    :return:
    """
    mask = tf.cast(tf.math.equal(input, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(input):
    seq_len = tf.shape(input)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(input)
    return tf.maximum(look_ahead_mask, padding_mask)


def load_data(dict_fn, data_fn, start_sign, end_sign, checkpoint_dir, max_train_data_size=0):
    """
    数据加载方法，含四个元素的元组，包括如下：
    :return:input_tensor, input_token, target_tensor, target_token
    """
    input_tensor, target_tensor, lang_tokenizer = read_data(data_fn, max_train_data_size, start_sign, end_sign)

    with open(dict_fn, 'w', encoding='utf-8') as file:
        file.write(json.dumps(lang_tokenizer.word_index, indent=4, ensure_ascii=False))

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).cache().shuffle(
        _config.BUFFER_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(_config.BATCH_SIZE, drop_remainder=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    steps_per_epoch = len(input_tensor) // _config.BATCH_SIZE

    return input_tensor, target_tensor, lang_tokenizer, dataset, steps_per_epoch, checkpoint_prefix


def load_token_dict(dict_fn):
    """
    加载字典方法
    :return:input_token, target_token
    """
    with open(dict_fn, 'r', encoding='utf-8') as file:
        token = json.load(file)

    return token


def sequences_to_texts(sequences, token_dict):
    """
    将序列转换成text
    """
    inv = {}
    for key, value in token_dict.items():
        inv[value] = key

    result = []
    for text in sequences:
        temp = ''
        for token in text:
            temp = temp + ' ' + inv[token]
        result.append(temp)
    return result
