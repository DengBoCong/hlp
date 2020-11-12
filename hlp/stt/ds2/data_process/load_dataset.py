import os
from util import get_all_data_path

from data_process.text_process import get_text_list


#加载数据
def load_data(dataset_name, data_path, text_row_style, num_examples):
    # 基于某种语料获取其中语音路径和文本的list
    if dataset_name.lower() == "number":
        audio_data_path_list, text_list = load_dataset_number(data_path, text_row_style, num_examples)
    elif dataset_name.lower() == "librispeech":
        audio_data_path_list, text_list = load_dataset_librispeech(data_path, text_row_style, num_examples)
    
    return audio_data_path_list, text_list
    
# 加载number语料，返回语音文件list和对应文本字符串list
def load_dataset_number(data_path, text_row_style, num_examples = None):
    # number语料里没有文本集，故先通过audio文件名构建文本文件
    if not os.path.exists(data_path + "/text.txt"):
        files = os.listdir(data_path)
        with open(data_path + "/text.txt", "a") as f:
            for path in files:
                f.write(path.split(".")[0] +" "+ path[0] + "\n")

    # 获取number语料中训练集语音路径和文本的列表
    text_data_path, audio_path_list = get_all_data_path(data_path)
        
    # 获取语音路径list和文本list
    audio_data_path_list = [data_path + "/" + audio_path for audio_path in audio_path_list[:num_examples]]
    text_list = get_text_list(data_path + "/" + text_data_path, text_row_style)[:num_examples]
    return audio_data_path_list, text_list

# 加载librispeech语料
def load_dataset_librispeech(data_path, text_row_style, num_examples = None):
    # 获取librispeech数据文件下(train或test)所有的数据folder
    data_folder_list = []
    folders_first = os.listdir(data_path)
    for folder_first in folders_first:
        folders_second = os.listdir(data_path + "/" + folder_first)
        for folder_second in folders_second:
            data_folder_list.append(data_path + "/" + folder_first + "/" +folder_second)

    audio_data_path_list = []
    text_list = []

    # 基于每个数据文件夹进行语音路径和文本的获取
    for data_folder in data_folder_list:
        # 获取number语料中训练集语音路径和文本的列表
        text_data_path, audio_path_list = get_all_data_path(data_folder)
        
        # 获取语音路径list和文本list
        audio_data_path_list.extend([data_folder + "/" + audio_path for audio_path in audio_path_list])
        text_list.extend(get_text_list(data_folder + "/" + text_data_path, text_row_style))

    audio_data_path_list = audio_data_path_list[:num_examples]
    text_list = text_list[:num_examples]
    return audio_data_path_list, text_list


if __name__ == "__main__":
    pass