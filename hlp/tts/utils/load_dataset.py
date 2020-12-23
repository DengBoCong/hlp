import os
import numpy as np
import tensorflow as tf
from hlp.tts.utils.text_preprocess import text_to_sequence_phoneme


def load_data(train_data_path: str, max_len: int, vocab_size: int, batch_size: int, buffer_size: int,
              tokenized_type: str = "phoneme", dict_path: str = "", valid_data_split: float = 0.0,
              valid_data_path: str = "", max_train_data_size: int = 0, max_valid_data_size: int = 0):
    """
    加载训练验证数据方法，非phoneme的方法将会保存字典
    验证数据的优先级为：验证数据文件>从训练集划分验证集
    :param train_data_path: 文本数据路径
    :param max_len: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param tokenized_type: 分词类型，默认按音素分词，模式：phoneme(音素)/word(单词)/char(字符)
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :return: 返回train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch
    """
    if not os.path.exists(train_data_path):
        print("加载的训练验证数据文件不存在，请先执行pre_treat模式后重试")
        exit(0)

    print("正在加载训练数据...")
    train_audio_data_pair, train_sentence_data = read_data(data_path=train_data_path, num_examples=max_train_data_size)

    valid_flag = True  # 是否开启验证标记
    valid_steps_per_epoch = 0

    # 根据是否传入验证数据文件，切分验证数据
    if valid_data_path != "":
        print("正在加载验证数据...")
        valid_audio_data_pair, valid_sentence_data = read_data(data_path=valid_data_path,
                                                               num_examples=max_valid_data_size)
    elif valid_data_split != 0.0:
        print("从训练数据中划分验证数据...")
        train_size = int(len(train_audio_data_pair) * (1.0 - valid_data_split))
        valid_audio_data_pair = train_audio_data_pair[train_size:]
        valid_sentence_data = train_sentence_data[train_size:]
        train_audio_data_pair = train_audio_data_pair[:train_size]
        train_sentence_data = train_sentence_data[:train_size]
    else:
        print("没有验证数据.")
        valid_flag = False

    # 根据分词类型进行序列转换
    if tokenized_type == "phoneme":
        train_sentence_sequences = text_to_sequence_phoneme(texts=train_sentence_data, max_len=max_len)
        if valid_flag:
            valid_sentence_sequences = text_to_sequence_phoneme(texts=valid_sentence_data, max_len=max_len)
    else:
        if dict_path == "":
            print("请在加载数据时，传入字典保存路径")
            exit(0)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token="<unk>", num_words=vocab_size)
        tokenizer.fit_on_texts(train_sentence_data)
        train_sentence_sequences = tokenizer.texts_to_sequences(train_sentence_data)
        train_sentence_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_sentence_sequences,
                                                                                 max_len=max_len, padding="post")
        with open(dict_path, 'w', encoding="utf-8") as dict_file:
            dict_file.write(tokenizer.to_json())

        if valid_flag:
            valid_sentence_sequences = tokenizer.texts_to_sequences(valid_sentence_data)
            valid_sentence_sequences = tf.keras.preprocessing.sequence.pad_sequences(valid_sentence_sequences,
                                                                                     max_len=max_len, padding="post")

    train_dataset = _to_dataset(data=(train_audio_data_pair, train_sentence_sequences),
                                batch_size=batch_size, buffer_size=buffer_size)
    if valid_flag:
        valid_dataset = _to_dataset(data=(valid_audio_data_pair, valid_sentence_sequences),
                                    batch_size=batch_size, buffer_size=buffer_size)
        valid_steps_per_epoch = len(valid_sentence_sequences) // batch_size
    else:
        valid_dataset = None

    steps_per_epoch = len(train_sentence_sequences) // batch_size

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


def read_data(data_path: str, num_examples: int):
    """
    :param data_path: 需要读取整理的数据文件路径
    :param num_examples: 读取的数据量大小
    :return: 返回读取的音频数据对和句子数据
    """
    audio_data_pair = []
    sentence_data = []
    with open(data_path, 'r', encoding="utf-8") as data_file:
        lines = data_file.read().strip().split('\n')
        if num_examples != 0:
            lines = lines[:num_examples]

        for line in lines:
            line = line.strip().strip("\n").replace("/", " ").split("\t")
            sentence_data.append(line[-1])
            line.pop(-1)
            audio_data_pair.append(line)

    return audio_data_pair, sentence_data


def read_npy_file(filename):
    """
    专门用于匹配dataset的map读取文件的方法
    :param filename: 传入的文件名张量
    :return: 返回读取的数据
    """
    data = np.load(filename.numpy().decode())
    return data.astype(np.float32)


def _process_audio_sentence_pairs(audio_data_pair: tf.Tensor, sentence: tf.Tensor):
    """
    用于处理音频句子对，将其转化为张量
    :param audio_data_pair: 音频相关数据对，mel、mag、stop_token保存文件
    :param sentence: 音频句子对
    :return: mel, mag, stop_token, sentence
    """
    [mel, ] = tf.py_function(read_npy_file, [audio_data_pair[0]], [tf.float32, ])
    [stop_token, ] = tf.py_function(read_npy_file, [audio_data_pair[2]], [tf.float32, ])

    return mel, stop_token, sentence

