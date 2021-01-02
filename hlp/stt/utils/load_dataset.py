import os
import numpy as np
import tensorflow as tf


def load_data(train_data_path: str, batch_size: int, buffer_size: int, valid_data_split: float = 0.0,
              valid_data_path: str = "", train_length_path: str = "", valid_length_path: str = "",
              max_train_data_size: int = 0, max_valid_data_size: int = 0):
    """
    加载训练验证数据方法，验证数据的优先级为：验证数据文件>从训练集划分验证集
    :param train_data_path: 文本数据路径
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_path: 验证数据文本路径
    :param train_length_path: 训练样本长度保存路径
    :param valid_length_path: 验证样本长度保存路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :return: 返回train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch
    """
    if not os.path.exists(train_data_path):
        print("加载的训练验证数据文件不存在，请先执行pre_treat模式后重试")
        exit(0)

    print("正在加载训练数据...")
    train_audio_data_path, train_sentence_data_path, train_length_data = \
        read_data(data_path=train_data_path, length_path=train_length_path, num_examples=max_train_data_size)

    valid_flag = True  # 是否开启验证标记
    valid_steps_per_epoch = 0

    # 根据是否传入验证数据文件，切分验证数据
    if valid_data_path != "":
        print("正在加载验证数据...")
        valid_audio_data_path, valid_sentence_data_path, valid_length_data = \
            read_data(data_path=valid_data_path, length_path=valid_length_path, num_examples=max_valid_data_size)
    elif valid_data_split != 0.0:
        print("从训练数据中划分验证数据...")
        train_size = int(len(train_audio_data_path) * (1.0 - valid_data_split))
        valid_audio_data_path = train_audio_data_path[train_size:]
        valid_sentence_data_path = train_sentence_data_path[train_size:]
        valid_length_data = train_length_data[train_size:]
        train_audio_data_path = train_audio_data_path[:train_size]
        train_sentence_data_path = train_sentence_data_path[:train_size]
        train_length_data = train_length_data[:train_size]
    else:
        valid_flag = False

    train_dataset = _to_dataset(data=(train_audio_data_path, train_sentence_data_path, train_length_data),
                                batch_size=batch_size, buffer_size=buffer_size)
    steps_per_epoch = len(train_sentence_data_path) // batch_size

    if valid_flag:
        valid_dataset = _to_dataset(data=(valid_audio_data_path, valid_sentence_data_path, valid_length_data),
                                    batch_size=batch_size, buffer_size=buffer_size)
        valid_steps_per_epoch = len(valid_sentence_data_path) // batch_size
    else:
        valid_dataset = None

    print("训练验证数据加载完毕")
    return train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch


def _to_dataset(data: tuple, batch_size: int, buffer_size: int):
    """
    将data封装成tf.data.Dataset
    :param data: 要封装的数据元组
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :return: dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices(data). \
        cache().shuffle(buffer_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(_process_audio_sentence_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def read_data(data_path: str, length_path: str, num_examples: int):
    """
    :param data_path: 需要读取整理的数据文件路径
    :param length_path: 样本长度保存路径
    :param num_examples: 读取的数据量大小
    :return: 返回读取的音频特征数据路径和句子数据
    """
    audio_data_path = []
    sentence_data_path = []
    with open(data_path, 'r', encoding="utf-8") as data_file:
        lines = data_file.read().strip().split('\n')
        if num_examples != 0:
            lines = lines[:num_examples]

        for line in lines:
            line = line.strip().strip("\n").replace("/", " ").split("\t")
            audio_data_path.append(line[0])
            sentence_data_path.append(line[1])

    length_data = np.load(length_path)

    return audio_data_path, sentence_data_path, length_data


def read_npy_file(filename):
    """
    专门用于匹配dataset的map读取文件的方法
    :param filename: 传入的文件名张量
    :return: 返回读取的数据
    """
    return np.load(filename.numpy().decode())


def _process_audio_sentence_pairs(audio_data_path: tf.Tensor, sentence_data_path: tf.Tensor, length: tf.Tensor):
    """
    用于处理音频句子对，将其转化为张量
    :param audio_data_path: 音频特征数据保存文件
    :param sentence_data_path: 音频句子
    :param length: 样本长度
    :return: audio_feature, sentence
    """
    [audio_feature] = tf.py_function(read_npy_file, [audio_data_path], [tf.float32])
    [sentence] = tf.py_function(read_npy_file, [sentence_data_path], [tf.int32])

    return audio_feature, sentence, length
