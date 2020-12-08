import os
import posixpath

from text_process import get_text_list


# 根据数据文件夹名获取所有的文件名，包括文本文件名和音频文件名列表
def _get_text_audio_files(data_path):
    # data_path是数据文件夹的路径
    files = os.listdir(data_path)  # 得到数据文件夹下的所有文件名称list
    text_data_path = files.pop()
    audio_data_path_list = files
    return text_data_path, audio_data_path_list


# 获得语音文件名和转写列表
def load_data(dataset_name, data_path, text_row_style, num_examples):
    if dataset_name.lower() == "librispeech":
        audio_data_path_list, text_list = _get_data_librispeech(data_path, text_row_style, num_examples)
    elif dataset_name.lower() == "thchs30":
        audio_data_path_list, text_list = _get_data_thchs30(data_path, num_examples)

    return audio_data_path_list, text_list


def _get_data_librispeech(data_path, text_row_style, num_examples=None):
    # 获取librispeech数据文件下(train或test)所有的数据folder
    data_folder_list = []
    folders_first = os.listdir(data_path)
    for folder_first in folders_first:
        folders_second = os.listdir(posixpath.join(data_path, folder_first))
        for folder_second in folders_second:
            data_folder_list.append(posixpath.join(data_path, folder_first, folder_second))

    audio_data_path_list = []
    text_list = []

    # 基于每个数据文件夹进行语音路径和文本的获取
    for data_folder in data_folder_list:
        # 获取number语料中训练集语音路径和文本的列表
        text_data_path, audio_path_list = _get_text_audio_files(data_folder)

        # 获取语音路径list和文本list
        audio_data_path_list.extend([posixpath.join(data_folder, audio_path) for audio_path in audio_path_list])
        text_list.extend(get_text_list(posixpath.join(data_folder, text_data_path), text_row_style))

    audio_data_path_list = audio_data_path_list[:num_examples]
    text_list = text_list[:num_examples]
    return audio_data_path_list, text_list


def _get_data_thchs30(data_path, num_examples=None):
    # 音频文件绝对路径
    thchs30_dataset_path = os.path.dirname(data_path)
    files = os.listdir(data_path)[:2 * num_examples]
    audio_data_path_list = []
    text_list = []
    for f in files:
        if os.path.splitext(f)[1] == ".wav":
            # 音频文件
            audio_path = posixpath.join(data_path, f)
            audio_data_path_list.append(audio_path)

            # 对应的文本
            with open(posixpath.join(thchs30_dataset_path, "data", f + ".trn"), encoding='UTF-8') as fl:
                txt_content = fl.readlines()
            text = "".join(txt_content[0].strip().split(" "))
            text_list.append(text)

    return audio_data_path_list, text_list


if __name__ == "__main__":
    pass
