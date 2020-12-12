
import os


# 获得语音文件名和转写列表
def load_data(dataset_name, data_path, num_examples):
    """加载数据集

    :param dataset_name: 数据集名字
    :param data_path: 数据集路径
    :param num_examples: 数据量
    :return: 语音文件路径list和对应转写文本list
    """
    if dataset_name.lower() == "librispeech":
        audio_data_path_list, text_list = get_data_librispeech(data_path, num_examples)
    elif dataset_name.lower() == "thchs30":
        audio_data_path_list, text_list = get_data_thchs30(data_path, num_examples)
    elif dataset_name.lower() == "number":
        audio_data_path_list, text_list = get_data_number(data_path, num_examples)

    return audio_data_path_list, text_list


def get_text(line, colum_sep=" "):
    """基于数据文本规则的行获取

    :param line: 语料文件中每行索引及其对应文本
    :param colum_sep: 可能的语音文件名和转写文本间的分隔符
    :return: 音频对应的转写文本
    """
    if colum_sep is None:
        return line.strip().lower()

    return line.strip().split(colum_sep, 1)[1].lower()


# 读取文本文件，并基于某种row_style来处理原始语料
def get_text_list(text_path, colum_sep=" "):
    text_list = []
    with open(text_path, "r") as f:
        sentence_list = f.readlines()
    for sentence in sentence_list:
        text_list.append(get_text(sentence, colum_sep))
    return text_list


def get_data_librispeech(data_path, num_examples=None):
    """ 获得librispeech数据集的语音文件和转写列表

    :param data_path: librispeech数据集路径
    :param num_examples: 最大语音文件数
    :return: (语音文件列表，转写列表)
    """
    data_folder_list = []
    folders_first = os.listdir(data_path)
    for folder_first in folders_first:
        folders_second = os.listdir(os.path.join(data_path, folder_first))
        for folder_second in folders_second:
            data_folder_list.append(os.path.join(data_path, folder_first, folder_second))

    audio_data_path_list = []
    text_list = []

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
        text_list.extend(get_text_list(os.path.join(data_folder, text_data_path)))

    return audio_data_path_list[:num_examples], text_list[:num_examples]


def get_data_thchs30(data_path, num_examples=None):
    """ 获得thchs数据集的语音文件和转写列表

    :param data_path: thchs数据集路径
    :param num_examples: 最大语音文件数
    :return: (语音文件列表，转写列表)
    """
    thchs30_dataset_dir = os.path.dirname(data_path)

    files = os.listdir(data_path)
    audio_data_path_list = []
    text_list = []
    row_num = 1  # 第0行是汉字序列，1行是完整拼音序列，2是分离拼音序列
    for file in files:
        if os.path.splitext(file)[1] == ".wav":
            # 音频文件
            audio_path = os.path.join(data_path, file)
            audio_data_path_list.append(audio_path)

            # 对应的文本
            with open(os.path.join(thchs30_dataset_dir, "data", file + ".trn"), encoding='UTF-8') as fl:
                txt_content = fl.readlines()
            text = txt_content[row_num].strip()
            text_list.append(text)

    return audio_data_path_list[:num_examples], text_list[:num_examples]


def get_data_number(data_path, num_examples=None):
    """ 获得number数据集的语音文件和转写列表

    :param data_path: number数据集路径
    :param num_examples: 最大语音文件数
    :return: (语音文件列表，转写列表)
    """
    wav_path = data_path[0]
    text_data_path = data_path[1]
    files = os.listdir(wav_path)    
    audio_path_list = files
    audio_data_path_list = [wav_path + "\\" + audio_path for audio_path in audio_path_list[:num_examples]]
    text_list = get_text_list(text_data_path, colum_sep = "\t")[:num_examples]
    return audio_data_path_list, text_list

if __name__ == "__main__":
    dir_thchs30 = '../data/data_thchs30/train'
    audio_fils, texts = get_data_thchs30(dir_thchs30)
    print(audio_fils)
    print(texts)

    dir_librispeech = '../data/LibriSpeech/train-clean-5'
    audio_fils, texts = get_data_librispeech(dir_librispeech)
    print(audio_fils)
    print(texts)


