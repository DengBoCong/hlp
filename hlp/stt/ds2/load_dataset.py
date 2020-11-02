from utils import get_config, get_all_data_path, set_config
import os
from text_process import get_text_list, get_process_text_list, tokenize, get_text_int_sequences, get_max_label_length, get_word_index
import json
from audio_process import get_max_audio_length

#加载数据
def load_data(dataset_name, data_path, train_or_test, num_examples):
    # 基于某种语料获取其中语音路径和文本的list
    if dataset_name == "number":
        audio_data_path_list, text_list = load_dataset_number(data_path, num_examples)
    elif dataset_name == "LibriSpeech" or dataset_name == "librispeech":
        audio_data_path_list, text_list = load_dataset_librispeech(data_path, num_examples)
    
    # 返回构建好的训练或测试所需数据
    return build_data(audio_data_path_list, text_list, train_or_test)

# 加载number语料，返回语音文件list和对应文本字符串list
def load_dataset_number(data_path, num_examples = None):
    # 获取配置文件
    configs = get_config()

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
    text_list = get_text_list(data_path + "/" + text_data_path, configs["preprocess"]["text_raw_style"])[:num_examples]
    return audio_data_path_list, text_list

# 加载librispeech语料
def load_dataset_librispeech(data_path, num_examples = None):
    # 获取配置文件
    configs = get_config()

    # 获取librispeech数据文件下(train或test)所有的最终数据folder
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
        text_list.extend(get_text_list(data_folder + "/" + text_data_path, configs["preprocess"]["text_raw_style"]))

    audio_data_path_list = audio_data_path_list[:num_examples]
    text_list = text_list[:num_examples]
    return audio_data_path_list, text_list

# 基于dataset中的audio_data_path_list和text_list来加载train或test数据
def build_data(audio_data_path_list, text_list, train_or_test):
    configs = get_config()

    if train_or_test == "train":
        # 基于文本按照某种mode切分文本
        mode = configs["preprocess"]["text_process_mode"]
        process_text_list = get_process_text_list(text_list, mode)
        
        if configs["train"]["if_is_first_train"]:
            text_int_sequences, tokenizer = tokenize(process_text_list)

            # 获取音频和文本的最大length，从而进行数据补齐
            max_input_length = get_max_audio_length(audio_data_path_list, configs["other"]["n_mfcc"])
            max_label_length = get_max_label_length(text_int_sequences)

            # 若为初次训练则将构建的字典集合写入json文件
            index_word_path = configs["other"]["index_word_path"]
            with open(index_word_path, 'w', encoding="utf-8") as f:
                json.dump(tokenizer.index_word, f, ensure_ascii=False, indent=4)
            word_index_path = configs["other"]["word_index_path"]
            with open(word_index_path, 'w', encoding="utf-8") as f:
                json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=4)
            
            set_config("model", "dense_units", len(tokenizer.index_word)+2 )
            set_config("preprocess", "max_input_length", max_input_length)
            set_config("preprocess", "max_label_length", max_label_length)
            # 将是否为初次加载改为false
            set_config("train", "if_is_first_train", False)

        else:
            # 不是初次训练就基于初次训练时写入的word_index构建文本
            word_index = get_word_index()
            text_int_sequences = get_text_int_sequences(process_text_list, word_index)
        
        label_length_list = [[len(text_int)] for text_int in text_int_sequences]
        return audio_data_path_list, text_int_sequences, label_length_list
    else:
        # 测试模块只需要文本字符串list即可
        return audio_data_path_list, text_list

if __name__ == "__main__":
    pass