import io
import os
import tensorflow as tf
import config.get_config as _config
from pathlib import Path


def preprocess_sentence(w):
    """
    用于给句子首尾添加start和end
    :param w:
    :return: 合成之后的句子
    """
    w = 'start ' + w + ' end'
    return w


def create_dataset(path, num_examples):
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
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


def max_length(tensor):
    """
    :param tensor:
    :return: 列表中最大的长度
    """
    return max(len(t) for t in tensor)


def read_data(path, num_examples):
    """
    读取数据，将input和target进行分词后返回
    :param path:
    :param num_examples:
    :return: input_tensor, input_token, target_tensor, target_token
    """
    input_lang, target_lang = create_dataset(path, num_examples)
    input_tensor, input_token = tokenize(input_lang)
    target_tensor, target_token = tokenize(target_lang)
    return input_tensor, input_token, target_tensor, target_token


def tokenize(lang):
    """
    分词方法，使用Keras API中的Tokenizer进行分词操作
    :param lang:
    :return: tensor, lang_tokenizer
    """
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=_config.max_length_inp, padding='post')

    return tensor, lang_tokenizer


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


def load_dataset():
    """
    数据加载方法，含四个元素的元组，包括如下：
    :return:input_tensor, input_token, target_tensor, target_token
    """
    return read_data(_config.data, _config.max_train_data_size)


class MyDataset():
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        print('')

    def __len__(self):
        return len(self.data)
    # 抽象类
