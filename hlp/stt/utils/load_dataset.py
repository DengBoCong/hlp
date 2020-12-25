import os
import numpy as np
import tensorflow as tf
from hlp.stt.utils.text_process import tokenize_and_encode


def load_data(train_data_path: str, max_len: int, vocab_size: int, batch_size: int, buffer_size: int,
              dict_path: str = "", valid_data_split: float = 0.0, valid_data_path: str = "",
              max_train_data_size: int = 0, max_valid_data_size: int = 0, unk_token: str = "<unk>"):
    """
    加载训练验证数据方法，验证数据的优先级为：验证数据文件>从训练集划分验证集
    :param train_data_path: 文本数据路径
    :param max_len: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :param unk_token: 未登录词
    :return: 返回train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch
    """
    if not os.path.exists(train_data_path):
        print("加载的训练验证数据文件不存在，请先执行pre_treat模式后重试")
        exit(0)

    print("正在加载训练数据...")
    train_audio_data_path, train_sentence_data = read_data(data_path=train_data_path, num_examples=max_train_data_size)

    valid_flag = True  # 是否开启验证标记
    valid_steps_per_epoch = 0

    # 根据是否传入验证数据文件，切分验证数据
    if valid_data_path != "":
        print("正在加载验证数据...")
        valid_audio_data_path, valid_sentence_data = read_data(data_path=valid_data_path,
                                                               num_examples=max_valid_data_size)
    elif valid_data_split != 0.0:
        print("从训练数据中划分验证数据...")
        train_size = int(len(train_audio_data_path) * (1.0 - valid_data_split))
        valid_audio_data_path = train_audio_data_path[train_size:]
        valid_sentence_data = train_sentence_data[train_size:]
        train_audio_data_path = train_audio_data_path[:train_size]
        train_sentence_data = train_sentence_data[:train_size]
    else:
        print("没有验证数据.")
        valid_flag = False

    if dict_path == "":
        print("请在加载数据时，传入字典保存路径")
        exit(0)
    train_sentence_sequences, tokenizer = tokenize_and_encode(texts=train_sentence_data,
                                                              max_len=max_len, num_words=vocab_size,
                                                              dict_path=dict_path, unk_token=unk_token)
    train_dataset = _to_dataset(data=(train_audio_data_path, train_sentence_sequences),
                                batch_size=batch_size, buffer_size=buffer_size)
    steps_per_epoch = len(train_sentence_sequences) // batch_size

    if valid_flag:
        valid_sentence_sequences = tokenizer.texts_to_sequences(valid_sentence_data)
        valid_sentence_sequences = tf.keras.preprocessing.sequence.pad_sequences(valid_sentence_sequences,
                                                                                 maxlen=max_len, padding="post")
        valid_dataset = _to_dataset(data=(valid_audio_data_path, valid_sentence_sequences),
                                    batch_size=batch_size, buffer_size=buffer_size)
        valid_steps_per_epoch = len(valid_sentence_sequences) // batch_size
    else:
        valid_dataset = None

    print("训练验证数据加载完毕")
    return tokenizer, train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch


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
    :return: 返回读取的音频特征数据路径和句子数据
    """
    audio_data_path = []
    sentence_data = []
    with open(data_path, 'r', encoding="utf-8") as data_file:
        lines = data_file.read().strip().split('\n')
        if num_examples != 0:
            lines = lines[:num_examples]

        for line in lines:
            line = line.strip().strip("\n").replace("/", " ").split("\t")
            audio_data_path.append(line[0])
            sentence_data.append(line[1])

    return audio_data_path, sentence_data


def read_npy_file(filename):
    """
    专门用于匹配dataset的map读取文件的方法
    :param filename: 传入的文件名张量
    :return: 返回读取的数据
    """
    data = np.load(filename.numpy().decode())
    return data.astype(np.float32)


def _process_audio_sentence_pairs(audio_data_path: tf.Tensor, sentence: tf.Tensor):
    """
    用于处理音频句子对，将其转化为张量
    :param audio_data_path: 音频特征数据保存文件
    :param sentence: 音频句子
    :return: audio_feature, sentence
    """
    [audio_feature] = tf.py_function(read_npy_file, [audio_data_path], [tf.float32])

    return audio_feature, sentence

############################################################
############################################################

# def load_data(dataset_name, data_path, num_examples):
#     """
#     加载数据集的语音文件名和转写列表
#     :param dataset_name: 数据集名字
#     :param data_path: 数据集路径
#     :param num_examples: 数据量
#     :return: 语音文件路径list和对应转写文本list
#     """
#     if dataset_name.lower() == "librispeech":
#         audio_data_path_list, text_list = get_data_librispeech(data_path, num_examples)
#     elif dataset_name.lower() == "thchs30":
#         audio_data_path_list, text_list = get_data_thchs30(data_path, num_examples)
#     elif dataset_name.lower() == "number":
#         audio_data_path_list, text_list = get_data_number(data_path, num_examples)
#
#     return audio_data_path_list, text_list
#
#
# def _get_text(line, colum_sep=" "):
#     """
#     获得语音转写文本
#     :param line: 可能包括语音文件和语音转写
#     :param colum_sep: 可能的语音文件名和转写文本间的分隔符
#     :return: 音频对应的转写文本
#     """
#     if colum_sep is None:
#         return line.strip().lower()
#
#     return line.strip().split(colum_sep, 1)[1].lower()
#
#
# def _get_text_list(text_path, colum_sep=" "):
#     """
#     从标注文件中获得所有转写
#     :param text_path: 标注文件路径
#     :param colum_sep: 语音文件和转写间的分隔符
#     :return: 转写列表
#     """
#     text_list = []
#     with open(text_path, "r") as f:
#         sentence_list = f.readlines()
#     for sentence in sentence_list:
#         text_list.append(_get_text(sentence, colum_sep))
#     return text_list
#
#
# def get_data_librispeech(data_path, num_examples=None):
#     """
#     获得librispeech数据集的语音文件和转写列表
#     :param data_path: librispeech数据集路径
#     :param num_examples: 最大语音文件数
#     :return: (语音文件列表，转写列表)
#     """
#     data_folder_list = []
#     folders_first = os.listdir(data_path)
#     for folder_first in folders_first:
#         folders_second = os.listdir(os.path.join(data_path, folder_first))
#         for folder_second in folders_second:
#             data_folder_list.append(os.path.join(data_path, folder_first, folder_second))
#
#     audio_data_path_list = []
#     text_list = []
#
#     for data_folder in data_folder_list:
#         # 获取某个数据文件夹下所有文件路径
#         files = os.listdir(data_folder)
#         audio_path_list = []
#         for file in files:
#             if os.path.splitext(file)[1] == ".flac":
#                 audio_path_list.append(file)
#             else:
#                 text_data_path = file
#
#         # 获取语音路径list和文本list
#         audio_data_path_list.extend([os.path.join(data_folder, audio_path) for audio_path in audio_path_list])
#         text_list.extend(_get_text_list(os.path.join(data_folder, text_data_path)))
#
#     return audio_data_path_list[:num_examples], text_list[:num_examples]
#
#
# def get_data_thchs30(data_path, num_examples=None, transcript_row=0):
#     """
#     获得thchs数据集的语音文件和转写列表
#     :param transcript_row: 语音转写行，0表示中文词，1中文拼音，2声母韵母
#     :param data_path: thchs数据集路径
#     :param num_examples: 最大语音文件数
#     :return: (语音文件列表，转写列表)
#     """
#     thchs30_dataset_dir = os.path.dirname(data_path)
#
#     files = os.listdir(data_path)
#     audio_data_path_list = []
#     text_list = []
#     for file in files:
#         if os.path.splitext(file)[1] == ".wav":
#             # 音频文件
#             audio_path = os.path.join(data_path, file)
#             audio_data_path_list.append(audio_path)
#
#             # 对应的文本
#             with open(os.path.join(thchs30_dataset_dir, "data", file + ".trn"), encoding='UTF-8') as fl:
#                 txt_content = fl.readlines()
#             text = txt_content[transcript_row].strip()
#             text_list.append(text)
#
#     return audio_data_path_list[:num_examples], text_list[:num_examples]
#
#
# def get_data_number(data_path, num_examples=None):
#     """
#     获得number数据集的语音文件和转写列表
#     :param data_path: number数据集路径
#     :param num_examples: 最大语音文件数
#     :return: (语音文件列表，转写列表)
#     """
#     wav_path = data_path[0]
#     text_data_path = data_path[1]
#     files = os.listdir(wav_path)
#     audio_path_list = files
#     audio_data_path_list = [wav_path + "\\" + audio_path for audio_path in audio_path_list[:num_examples]]
#     text_list = _get_text_list(text_data_path, colum_sep="\t")[:num_examples]
#     return audio_data_path_list, text_list


# if __name__ == "__main__":
#     dir_thchs30 = '../data/data_thchs30/train'
#     audio_fils, texts = get_data_thchs30(dir_thchs30)
#     print(audio_fils)
#     print(texts)
#
#     dir_librispeech = '../data/LibriSpeech/train-clean-5'
#     audio_fils, texts = get_data_librispeech(dir_librispeech)
#     print(audio_fils)
#     print(texts)
