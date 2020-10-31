import json
import os


#获取配置文件
def get_config():
    with open("config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)
    return configs

#写入配置文件
def set_config(config_class, key, value):
    configs = get_config()
    configs[config_class][key] = value
    with open("config.json", 'w', encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=4)

#获取index_word
def get_index_word():
    configs = get_config()
    index_word_path = configs["other"]["index_word_path"]
    with open(index_word_path, "r", encoding="utf-8") as f:
        index_word = json.load(f)
    return index_word

#获取word_index
def get_word_index():
    configs = get_config()
    word_index_path = configs["other"]["word_index_path"]
    with open(word_index_path, "r", encoding="utf-8") as f:
        word_index = json.load(f)
    return word_index

#根据数据文件夹名获取所有的文件名，包括文本文件名和音频文件名列表
def get_all_data_path(data_path):
    #data_path是数据文件夹的路径
    files = os.listdir(data_path) #得到数据文件夹下的所有文件名称list
    text_data_path = files.pop()
    audio_data_path_list = files
    return text_data_path, audio_data_path_list


if __name__ == "__main__":
    pass