'''
Author: PengKang6
Description: 加载数据集, 获取音频路径list和转写文本list
'''
import os

from text_process import get_text_list


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
        folders_second = os.listdir(os.path.join(data_path, folder_first))
        for folder_second in folders_second:
            data_folder_list.append(os.path.join(data_path, folder_first, folder_second))

    audio_data_path_list = []
    text_list = []

    # 基于每个数据文件夹进行语音路径和文本的获取
    for data_folder in data_folder_list:
        # 获取某个数据文件夹下所有文件路径
        files = os.listdir(data_folder)
        audio_path_list = []
        for file in files:
            if os.path.splitext(file)[1] == ".flac":
                audio_path_list.append(file)
            else:
                text_data_path = file

        # 获取语音路径list和文本list
        audio_data_path_list.extend([os.path.join(data_folder, audio_path) for audio_path in audio_path_list])
        text_list.extend(get_text_list(os.path.join(data_folder, text_data_path), text_row_style))

    return audio_data_path_list[:num_examples], text_list[:num_examples]


def _get_data_thchs30(data_path, num_examples=None):
    # 语料文件夹路径
    thchs30_dataset_dir = os.path.dirname(data_path)

    files = os.listdir(data_path)
    audio_data_path_list = []
    text_list = []
    for file in files:
        if os.path.splitext(file)[1] == ".wav":
            # 音频文件
            audio_path = os.path.join(data_path, file)
            audio_data_path_list.append(audio_path)

            # 对应的文本
            with open(os.path.join(thchs30_dataset_dir, "data", file + ".trn"), encoding='UTF-8') as fl:
                txt_content = fl.readlines()
            text = "".join(txt_content[0].strip().split(" "))
            text_list.append(text)

    return audio_data_path_list[:num_examples], text_list[:num_examples]


if __name__ == "__main__":
    pass
