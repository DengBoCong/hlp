import os
import numpy as np
import tensorflow as tf
from hlp.stt.utils.audio_process import wav_to_feature
from hlp.stt.utils.text_process import tokenize_and_encode


def dispatch_pre_treat_func(func_type: str, data_path: str, dataset_infos_file: str, max_time_step: int,
                            vocab_size: int, max_sentence_length: int, spectrum_data_dir: str, start_sign: str,
                            end_sign: str, unk_sign: str, dict_path: str = "", audio_feature_type: str = "mfcc",
                            save_length_path: str = "", transcript_row: int = 0, max_treat_data_size: int = 0,
                            is_train: bool = True):
    """
    预处理方法分发匹配
    :param func_type: 预处理方法类型
    :param data_path: 数据存放目录路径
    :param dataset_infos_file: 保存处理之后的数据路径
    :param vocab_size: 词汇大小
    :param max_time_step: 最大音频补齐长度
    :param max_sentence_length: 文本序列最大长度
    :param save_length_path: 保存样本长度文件路径
    :param spectrum_data_dir: 保存处理后的音频特征数据目录
    :param start_sign: 句子开始标记
    :param end_sign: 句子结束标记
    :param unk_token: 未登录词
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param audio_feature_type: 特征类型
    :param transcript_row: 使用文本数据中的第几行，第一行文字，第二行拼音，第三行音节
    :param max_treat_data_size: 最大处理数据，若为0，则全部数据
    :param is_train: 处理的是否是训练数据
    :return: 无返回值
    """
    operation = {
        "thchs30": lambda: preprocess_thchs30_speech_raw_data(
            data_path=data_path, max_time_step=max_time_step, dataset_infos_file=dataset_infos_file,
            spectrum_data_dir=spectrum_data_dir, audio_feature_type=audio_feature_type, unk_sign=unk_sign,
            transcript_row=transcript_row, is_train=is_train, save_length_path=save_length_path,
            max_sentence_length=max_sentence_length, max_treat_data_size=max_treat_data_size,
            vocab_size=vocab_size, dict_path=dict_path, start_sign=start_sign, end_sign=end_sign),
        "librispeech": lambda: preprocess_librispeech_speech_raw_data(
            data_path=data_path, max_time_step=max_time_step, dataset_infos_file=dataset_infos_file,
            spectrum_data_dir=spectrum_data_dir, audio_feature_type=audio_feature_type, start_sign=start_sign,
            save_length_path=save_length_path, max_sentence_length=max_sentence_length, end_sign=end_sign,
            max_treat_data_size=max_treat_data_size, vocab_size=vocab_size, dict_path=dict_path, unk_sign=unk_sign)
    }

    operation.get(func_type, "thchs30")()


def preprocess_thchs30_speech_raw_data(data_path: str, dataset_infos_file: str, max_time_step: int,
                                       spectrum_data_dir: str, max_sentence_length: int, vocab_size: int,
                                       audio_feature_type: str = "mfcc", save_length_path: str = "",
                                       is_train: bool = True, transcript_row: int = 0, start_sign: str = "<start>",
                                       dict_path: str = "", end_sign: str = "<end>", unk_sign: str = "<unk>",
                                       max_treat_data_size: int = 0):
    """
    用于处理thchs30数据集的方法，将数据整理为<音频地址, 句子>的
    形式，这样方便后续进行分批读取
    :param data_path: 数据存放目录路径
    :param dataset_infos_file: 保存处理之后的数据路径
    :param max_time_step: 最大音频补齐长度
    :param max_sentence_length: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param spectrum_data_dir: 保存处理后的音频特征数据目录
    :param audio_feature_type: 特征类型
    :param save_length_path: 保存样本长度文件路径
    :param is_train: 处理的是否是训练数据
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param transcript_row: 使用文本数据中的第几行，第一行文字，第二行拼音，第三行音节
    :param start_sign: 句子开始标记
    :param end_sign: 句子结束标记
    :param unk_sign: 未登录词
    :param max_treat_data_size: 最大处理数据，若为0，则全部数据
    :return: 无返回值
    """
    _check_and_create_file(data_path, spectrum_data_dir)

    count = 0
    len_list = []
    text_list = []
    text_file_path_list = []
    data_list = os.listdir(data_path)
    data_fir = data_path[:data_path.find("30")] + "30\\data\\"
    with open(dataset_infos_file, 'w', encoding='utf-8') as ds_infos_file:
        for data_name in data_list:
            if os.path.splitext(data_name)[1] == ".wav":
                len_pair = []
                # 音频文件
                audio_path = data_path + data_name
                # 对应的文本
                text_file_name = data_path + data_name + ".trn"
                if not os.path.exists(text_file_name):
                    print("{}文本数据不完整，请检查后重试".format(text_file_name))
                    exit(0)
                with open(text_file_name, 'r', encoding='utf-8') as text_file:
                    texts = text_file.readlines()
                text = texts[0].strip().strip("\n")[8:]

                with open(data_fir + text, 'r', encoding='utf-8') as text_file:
                    texts = text_file.readlines()
                text = texts[transcript_row].strip()
                text = start_sign + " " + text + " " + end_sign
                len_pair.append(len(text.split(" ")))
                text_list.append(text)

                audio_feature_file = spectrum_data_dir + data_name + ".npy"
                text_token_file = spectrum_data_dir + data_name + "text.npy"
                audio_feature = wav_to_feature(audio_path, audio_feature_type)
                len_pair.append(vocab_size if audio_feature.shape[0] > vocab_size else audio_feature.shape[0])
                text_file_path_list.append(text_token_file)

                audio_feature = tf.keras.preprocessing.sequence.pad_sequences([audio_feature], maxlen=max_time_step,
                                                                              dtype="float32", padding="post")
                audio_feature = tf.squeeze(audio_feature, axis=0)

                np.save(file=audio_feature_file, arr=audio_feature)
                ds_infos_file.write(audio_feature_file + '\t' + text_token_file + "\n")

                len_list.append(len_pair)
                count += 1
                print('\r已处理并写入音频条数：{}'.format(count), flush=True, end='')
                if max_treat_data_size == count:
                    break

    _treat_sentence_and_length(text_list, text_file_path_list, len_list, max_sentence_length,
                               vocab_size, save_length_path, is_train, dict_path, unk_sign)

    print("\n数据处理完毕，共计{}对语音句子数据".format(count))


def preprocess_librispeech_speech_raw_data(data_path: str, dataset_infos_file: str, max_time_step: int,
                                           spectrum_data_dir: str, max_sentence_length: int, vocab_size: int,
                                           save_length_path: str = "", start_sign: str = "<start>",
                                           end_sign: str = "<end>", unk_sign: str = "<unk>", dict_path: str = "",
                                           is_train: bool = True, audio_feature_type: str = "mfcc",
                                           max_treat_data_size: int = 0):
    """
    用于处理librispeech数据集的方法，将数据整理为<音频地址, 句子>的
    形式，这样方便后续进行分批读取
    :param data_path: 数据存放目录路径
    :param dataset_infos_file: 保存处理之后的数据路径
    :param max_time_step: 最大音频补齐长度
    :param save_length_path: 保存样本长度文件路径
    :param max_sentence_length: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param spectrum_data_dir: 保存处理后的音频特征数据目录
    :param start_sign: 句子开始标记
    :param end_sign: 句子结束标记
    :param unk_sign: 未登录词
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param is_train: 处理的是否是训练数据
    :param audio_feature_type: 特征类型
    :param max_treat_data_size: 最大处理数据，若为0，则全部数据
    :return: 无返回值
    """
    _check_and_create_file(data_path, spectrum_data_dir)

    count = 0
    len_list = []
    text_list = []
    text_file_path_list = []
    with open(dataset_infos_file, 'w', encoding='utf-8') as ds_infos_file:
        first_folders = os.listdir(data_path)
        for first_folder in first_folders:
            second_folders = os.listdir(os.path.join(data_path, first_folder))
            for second_folder in second_folders:
                second_dir = os.path.join(data_path, first_folder, second_folder)

                with open(os.path.join(data_path, first_folder, second_folder,
                                       first_folder + "-" + second_folder + ".trans.txt"),
                          "r", encoding="utf-8") as trans_file:
                    for line in trans_file:
                        line = line.strip("\n").strip()

                        if line == "":
                            continue
                        len_pair = []
                        line = line.split(" ", 1)

                        audio_path = os.path.join(second_dir, line[0] + ".flac")
                        audio_feature_file = spectrum_data_dir + line[0] + ".npy"
                        text_token_file = spectrum_data_dir + line[0] + "text.npy"
                        text_file_path_list.append(text_token_file)

                        text = start_sign + " " + line[1].lower() + " " + end_sign
                        len_pair.append(len(text.split(" ")))
                        text_list.append(text)

                        audio_feature = wav_to_feature(audio_path, audio_feature_type)
                        len_pair.append(vocab_size if audio_feature.shape[0] > vocab_size else audio_feature.shape[0])

                        audio_feature = tf.keras.preprocessing.sequence.pad_sequences(
                            [audio_feature], maxlen=max_time_step, dtype="float32", padding="post")
                        audio_feature = tf.squeeze(audio_feature, axis=0)

                        np.save(file=audio_feature_file, arr=audio_feature)
                        ds_infos_file.write(audio_feature_file + "\t" + spectrum_data_dir + line[0] + "text.npy" + "\n")

                        count += 1
                        len_list.append(len_pair)
                        print('\r已处理并写入音频条数：{}'.format(count), flush=True, end='')
                        if max_treat_data_size == count:
                            break

    _treat_sentence_and_length(text_list, text_file_path_list, len_list, max_sentence_length,
                               vocab_size, save_length_path, is_train, dict_path, unk_sign)

    print("\n数据处理完毕，共计{}对语音句子数据".format(count))


def _treat_sentence_and_length(text_list: list, text_file_path_list: list, len_list: list,
                               max_sentence_length: int, vocab_size: int, save_length_path: str = "",
                               is_train: bool = True, dict_path: str = "", unk_token: str = "<unk>"):
    """
    用于整合Librispeech和thchs30预处理方法的尾部，针对句子和长度文件的处理抽取
    :param text_list: 文本列表
    :param text_file_path_list: 保存文本文件名的列表
    :param len_list: 长度列表
    :param max_sentence_length: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param save_length_path: 保存样本长度文件路径
    :param is_train: 处理的是否是训练数据
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param unk_token: 未登录词
    :
    """
    print("\n正在处理数据集的句子数据")
    if not is_train:
        with open(dict_path, 'r', encoding='utf-8') as dict_file:
            json_string = dict_file.read().strip().strip("\n")
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        sentence_sequences = tokenizer.texts_to_sequences(text_list)
        sentence_sequences = tf.keras.preprocessing.sequence.pad_sequences(sentence_sequences,
                                                                           maxlen=max_sentence_length, padding="post")
    else:
        sentence_sequences, _ = tokenize_and_encode(texts=text_list, max_len=max_sentence_length,
                                                    num_words=vocab_size, dict_path=dict_path, unk_token=unk_token)

    count = 0
    for (_, (text, text_file)) in enumerate(zip(sentence_sequences, text_file_path_list)):
        np.save(file=text_file, arr=text)
        count += 1
        print('\r已处理并写入句子条数：{}'.format(count), flush=True, end='')

    if save_length_path is not "":
        np.save(file=save_length_path, arr=len_list)


def _check_and_create_file(data_path: str, spectrum_data_dir: str):
    """
    在预处理数据之前，检查各文件是否齐全，目录不齐全则创建
    :param data_path: 数据存放目录路径
    :param spectrum_data_dir: 保存处理后的音频特征数据目录
    :return: 无返回值
    """
    if not os.path.exists(data_path):
        print("librispeech数据集不存在，请检查重试")
        exit(0)

    if not os.path.exists(spectrum_data_dir):
        os.makedirs(spectrum_data_dir)
