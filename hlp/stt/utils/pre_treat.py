import os
import numpy as np
import tensorflow as tf
from hlp.stt.utils.audio_process import wav_to_feature


def dispatch_pre_treat_func(func_type: str, data_path: str, dataset_infos_file: str, max_length: int,
                            spectrum_data_dir: str, audio_feature_type: str = "mfcc",
                            transcript_row: int = 0):
    """
    预处理方法分发匹配
    :param func_type: 预处理方法类型
    :param data_path: 数据存放目录路径
    :param dataset_infos_file: 保存处理之后的数据路径
    :param max_length: 最大音频补齐长度
    :param spectrum_data_dir: 保存处理后的音频特征数据目录
    :param audio_feature_type: 特征类型
    :param transcript_row: 使用文本数据中的第几行，第一行文字，第二行拼音，第三行音节
    :return: 无返回值
    """
    operation = {
        "thchs30": lambda: preprocess_thchs30_speech_raw_data(data_path=data_path, max_length=max_length,
                                                              dataset_infos_file=dataset_infos_file,
                                                              spectrum_data_dir=spectrum_data_dir,
                                                              audio_feature_type=audio_feature_type,
                                                              transcript_row=transcript_row),
        "librispeech": lambda: preprocess_librispeech_speech_raw_data(data_path=data_path, max_length=max_length,
                                                                      dataset_infos_file=dataset_infos_file,
                                                                      spectrum_data_dir=spectrum_data_dir,
                                                                      audio_feature_type=audio_feature_type)
    }

    operation.get(func_type, "thchs30")()


def preprocess_thchs30_speech_raw_data(data_path: str, dataset_infos_file: str, max_length: int,
                                       spectrum_data_dir: str, audio_feature_type: str = "mfcc",
                                       transcript_row: int = 0, start_sign: str = "<start>", end_sign: str = "<end>"):
    """
    用于处理thchs30数据集的方法，将数据整理为<音频地址, 句子>的
    形式，这样方便后续进行分批读取
    :param data_path: 数据存放目录路径
    :param dataset_infos_file: 保存处理之后的数据路径
    :param max_length: 最大音频补齐长度
    :param spectrum_data_dir: 保存处理后的音频特征数据目录
    :param audio_feature_type: 特征类型
    :param transcript_row: 使用文本数据中的第几行，第一行文字，第二行拼音，第三行音节
    :param start_sign: 句子开始标记
    :param end_sign: 句子结束标记
    :return: 无返回值
    """
    if not os.path.exists(data_path):
        print("thchs30数据集不存在，请检查重试")
        exit(0)
    data_list = os.listdir(data_path)

    if not os.path.exists(spectrum_data_dir):
        os.makedirs(spectrum_data_dir)

    count = 0
    with open(dataset_infos_file, 'w', encoding='utf-8') as ds_infos_file:
        for data_name in data_list:
            if os.path.splitext(data_name)[1] == ".wav":
                # 音频文件
                audio_path = data_path + data_name
                # 对应的文本
                text_file_name = data_path + data_name + ".trn"
                if not os.path.exists(text_file_name):
                    print("{}文本数据不完整，请检查后重试".format(text_file_name))
                    exit(0)
                with open(text_file_name, 'r', encoding='utf-8') as text_file:
                    texts = text_file.readlines()
                text = texts[transcript_row].strip()
                text = start_sign + " " + text + " " + end_sign

                audio_feature_file = spectrum_data_dir + data_name + ".npy"
                audio_feature = wav_to_feature(audio_path, audio_feature_type)
                audio_feature = tf.expand_dims(audio_feature, axis=0)
                audio_feature = tf.keras.preprocessing.sequence.pad_sequences(audio_feature, maxlen=max_length,
                                                                              dtype="float32", padding="post")
                audio_feature = tf.squeeze(audio_feature, axis=0)

                np.save(file=audio_feature_file, arr=audio_feature)
                ds_infos_file.write(audio_feature_file + '\t' + text + "\n")

                count += 1
                print('\r已处理音频句子对数：{}'.format(count), flush=True, end='')

    print("\n数据处理完毕，共计{}条语音数据".format(count))


def preprocess_librispeech_speech_raw_data(data_path: str, dataset_infos_file: str, max_length: int,
                                           spectrum_data_dir: str, audio_feature_type: str = "mfcc"):
    """
    用于处理librispeech数据集的方法，将数据整理为<音频地址, 句子>的
    形式，这样方便后续进行分批读取
    :param data_path: 数据存放目录路径
    :param dataset_infos_file: 保存处理之后的数据路径
    :param max_length: 最大音频补齐长度
    :param spectrum_data_dir: 保存处理后的音频特征数据目录
    :param audio_feature_type: 特征类型
    :return: 无返回值
    """
    if not os.path.exists(data_path):
        print("thchs30数据集不存在，请检查重试")
        exit(0)

    if not os.path.exists(spectrum_data_dir):
        os.makedirs(spectrum_data_dir)

    count = 0
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
                        line = line.split(" ", 1)

                        audio_path = os.path.join(second_dir, line[0] + ".flac")

                        audio_feature_file = spectrum_data_dir + line[0] + ".npy"
                        audio_feature = wav_to_feature(audio_path, audio_feature_type)
                        audio_feature = tf.keras.preprocessing.sequence.pad_sequences(audio_feature, maxlen=max_length,
                                                                                      dtype="float32", padding="post")
                        np.save(file=audio_feature_file, arr=audio_feature)
                        ds_infos_file.write(audio_feature_file + "\t" + line[1] + "\n")

                        count += 1
                        print('\r已处理音频句子对数：{}'.format(count), flush=True, end='')

    print("\n数据处理完毕，共计{}条语音数据".format(count))
